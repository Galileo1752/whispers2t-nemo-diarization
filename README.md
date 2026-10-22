# WhisperS2T + NeMo Sortformer 講者分離

結合 [WhisperS2T](https://github.com/shashikg/WhisperS2T)（語音轉文字）與 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) Sortformer Diarization（講者分離）的整合工具，並使用 **OpenCC** 將輸出轉為繁體中文。

## 功能

1. **WhisperS2T** 把音檔轉成逐句文字（含時間戳）
2. **NeMo Sortformer** 判斷每個時間段是哪位講者
3. 時間戳對齊，指派講者
4. **OpenCC**（`s2twp`）轉繁體中文
5. 輸出格式：`[HH:MM:SS -> HH:MM:SS] SPEAKER_XX: 文字`

## 輸出範例

```
[00:00:00 -> 00:00:42] SPEAKER_01: 歡迎收看英文聖經時間，今天的經文在列王紀上三章七到九節...
[00:00:42 -> 00:02:21] SPEAKER_02: 現在，主啊，我的神，您將您的僕人立為王...
```

## 環境需求

- Python 3.10+
- NVIDIA GPU + CUDA（建議 8GB+ VRAM）
- ffmpeg

## 安裝

```bash
# 1) 音訊處理工具
sudo apt-get update && sudo apt-get install -y ffmpeg

# 2) PyTorch（請依你的 CUDA 版本調整，見 https://pytorch.org）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) 其餘套件
pip install -r requirements.txt
```

## 使用方式

```bash
python src/diarized_transcribe.py my_meeting.mp3 --lang zh --out transcript.txt --json result.json
```

參數：

| 參數 | 說明 | 預設 |
|------|------|------|
| `audio` | 輸入音檔 (wav/mp3/m4a...) | （必填） |
| `--lang` | 語言代碼，例如 `zh` / `en` | `zh` |
| `--out` | 輸出文字檔路徑 | `transcript.txt` |
| `--json` | 輸出 JSON 路徑 | `result.json` |

## 注意事項

| 項目 | 說明 |
|------|------|
| **模型名稱** | NeMo diarization 模型版本更新頻繁。若 `nvidia/diar_sortformer_4spk-v1` 不可用，請到 [NVIDIA NGC](https://catalog.ngc.nvidia.com/) 或 [HuggingFace nvidia](https://huggingface.co/nvidia) 查最新名稱。 |
| **講者數量** | Sortformer `4spk` 版本最多支援 4 位講者；超過需換模型或改用 MSDD pipeline。 |
| **API 細節** | `whisper_s2t` 與 NeMo `diarize()` 的回傳格式可能隨版本微調，執行時可先 `print()` 原始輸出確認欄位名稱再調整解析。 |
| **GPU 記憶體** | `large-v3` + diarization 同時載入約需 8GB+ VRAM。不足時可改用 `medium`。 |

## 授權

MIT License
