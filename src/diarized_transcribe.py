"""
WhisperS2T + NeMo Sortformer Diarization
講者分離 + 語音轉文字 整合程式 (含 OpenCC 繁體輸出)

流程:
  音檔 -> WhisperS2T (轉文字) -> NeMo Sortformer (講者分離)
       -> 時間戳對齊 -> OpenCC 繁體化 -> 輸出 [時間] SPEAKER_XX: 文字
"""

import os
import json
import tempfile
import subprocess

import torch

import whisper_s2t
from nemo.collections.asr.models import SortformerEncLabelModel
from opencc import OpenCC

# s2twp = 簡體 -> 繁體 (台灣正體, 含慣用詞轉換)
cc = OpenCC("s2twp")


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def preprocess_audio(input_path: str) -> str:
    """把任意音檔轉成 16kHz 單聲道 wav。"""
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", out_path,
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def sec_to_hms(seconds: float) -> str:
    """把秒數轉成 HH:MM:SS 格式。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# 1) 語音轉文字 (WhisperS2T)
# ------------------------------------------------------------------
def transcribe(audio_path: str, language: str = "zh"):
    """回傳 segment 層級帶時間戳的轉錄結果。"""
    model = whisper_s2t.load_model(
        model_identifier="large-v3",   # 可改 base / small / medium / large-v3
        backend="CTranslate2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16",
    )

    out = model.transcribe_with_vad(
        [audio_path],
        lang_codes=[language],
        tasks=["transcribe"],
        initial_prompts=[None],
        batch_size=16,
    )

    segments = []
    for seg in out[0]:
        segments.append({
            "start": float(seg["start_time"]),
            "end":   float(seg["end_time"]),
            "text":  seg["text"].strip(),
        })
    return segments


# ------------------------------------------------------------------
# 2) 講者分離 (NeMo Sortformer / Nemotron Diarization)
# ------------------------------------------------------------------
def diarize(audio_path: str):
    """回傳 [(start, end, speaker_label), ...]。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_sortformer_4spk-v1"
    ).to(device)
    diar_model.eval()

    predicted = diar_model.diarize(audio=audio_path, batch_size=1)

    segments = []
    for line in predicted[0]:
        # 格式通常為: "start_sec end_sec speaker_X"
        parts = line.split()
        start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        segments.append((start, end, spk))
    return segments


# ------------------------------------------------------------------
# 3) 對齊: 把每句文字指派給重疊最多的講者
# ------------------------------------------------------------------
def assign_speakers(asr_segments, diar_segments):
    results = []
    for seg in asr_segments:
        best_spk, best_overlap = "UNKNOWN", 0.0
        for d_start, d_end, spk in diar_segments:
            overlap = max(0.0, min(seg["end"], d_end) - max(seg["start"], d_start))
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, spk
        results.append({**seg, "speaker": best_spk})
    return results


# ------------------------------------------------------------------
# 4) 格式化輸出 (含 OpenCC 繁體轉換)
# ------------------------------------------------------------------
def format_transcript(final_segments) -> str:
    lines = []
    for seg in final_segments:
        start = sec_to_hms(seg["start"])
        end = sec_to_hms(seg["end"])
        speaker = seg["speaker"]
        text = cc.convert(seg["text"].strip())
        lines.append(f"[{start} -> {end}] {speaker}: {text}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def run(input_audio: str, language: str = "zh",
        out_txt: str = "transcript.txt", out_json: str = "result.json"):
    print(f"[1/5] 預處理音檔: {input_audio}")
    wav = preprocess_audio(input_audio)

    try:
        print("[2/5] 語音轉文字 (WhisperS2T)...")
        asr_segments = transcribe(wav, language=language)

        print("[3/5] 講者分離 (NeMo Sortformer)...")
        diar_segments = diarize(wav)

        print("[4/5] 對齊講者與文字...")
        final = assign_speakers(asr_segments, diar_segments)

        print("[5/5] OpenCC 繁體轉換並格式化輸出...")
        transcript = format_transcript(final)
    finally:
        if os.path.exists(wav):
            os.remove(wav)

    print("\n================ 逐字稿 ================")
    print(transcript)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n已輸出至 {out_txt} 與 {out_json}")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WhisperS2T + NeMo 講者分離")
    parser.add_argument("audio", help="輸入音檔 (wav/mp3/m4a...)")
    parser.add_argument("--lang", default="zh", help="語言代碼, 例如 zh / en")
    parser.add_argument("--out", default="transcript.txt", help="輸出文字路徑")
    parser.add_argument("--json", default="result.json", help="輸出 JSON 路徑")
    args = parser.parse_args()

    run(args.audio, language=args.lang, out_txt=args.out, out_json=args.json)
