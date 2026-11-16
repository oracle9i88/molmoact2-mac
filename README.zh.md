# 在 Mac 上跑 MolmoAct2（Apple Silicon，PyTorch MPS）

[English](README.md)

MolmoAct2 的官方文件寫「需要 NVIDIA 顯卡」，其實不需要。SO-100/101 這份權重
[`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot)
**不用改模型程式一行**就能在 Mac 顯卡上跑：每段 30 步動作 0.82 秒，輸出和 CPU 全精度參考逐位一致。

這個倉庫不是移植版，只有一頁說明、兩個必須注意的坑，和幾個用來準備權重、驗證你自己機器的小腳本。

## 實測數字

Mac Studio，M4 Max，64 GB，macOS 27.2.1 · Python 3.12 · lerobot 0.7.0 · torch 2.15.0 · transformers 5.22.2

固定的假輸入（`smoke_test.py`），一段 30 步動作，熱機後：

| 設備／精度 | 耗時 | 與 CPU 全精度的差（度） | 顯示記憶體 |
|---|---|---|---|
| **Mac 顯卡，全精度 float32** | **0.82 秒** | **最大 0.0000，平均 0.0000** | 20.5 GiB |
| Mac 顯卡，半精度 bfloat16 | 0.80 秒 | 最大 26.5，平均 9.8，**結果是錯的** | 11.3 GiB |
| CPU，全精度 | 3.1 秒 | 參考本身 | – |
| CPU，半精度 | 15.5 秒 | 最大 0.30，平均 0.06 | – |

真實錄製資料開環回放（`replay_test.py`，`Beegbrain/pick_lemon_and_drop_in_bowl` 的 3 段錄製，
36 個起點 × 30 步，Mac 顯卡全精度）：

| | 模型 | 「原地不動」對照 |
|---|---|---|
| 全部關節平均差 | 2.13° | 11.00° |
| 第 1 步／第 30 步 | 1.2°／3.0° | 4.0°／17.0° |
| 耗時（含前後處理） | 0.84 秒 | |

這份資料是底座模型說明頁自己的示例，很可能在訓練集裡。回放只證明整條流程在 Mac 上接對了
（畫面格式、關節坐標換算、歸一化、顯卡計算），不能說明模型在你的桌面上表現如何。

## 兩個必須注意的坑

**1. Mac 顯卡上一律用全精度 float32，不要用半精度 bfloat16。**
LeRobot 設定裡預設是 bfloat16。它在 Mac 顯卡上不報錯、不警告，但同樣的輸入算出的動作平均偏
約 10°，最大偏 26°。同樣的半精度在 CPU 上沒問題（只差 0.06°），所以是 MPS 後端的問題，不是精度
本身的問題。具體是哪個運算出錯我們還沒查到。它也不更快。接上真臂時，這個錯看起來會像「模型不行」，
而不是「數算錯了」。

**2. 官方的 `config.json` 用 lerobot 0.7.0 載入會報錯。**
裡面有四個欄位後來改名或刪除了，載入時會停在 `draccus.utils.DecodingError`。`prepare.py` 會修好
這個檔案並保留原件。其中 `model_dtype` 是改名為 `dtype`（保留 `float32`），不是刪掉：設定裡沒有
`dtype` 的話，lerobot 會退回預設的 bfloat16，正好掉進第 1 個坑。

另外：**不需要下載底座模型。** LeRobot 會先載入 `allenai/MolmoAct2-SO100_101`（又一個 21.8 GB），
再用 LeRobot 權重把每個張量都覆蓋掉。兩份檔案是同樣的 1295 個張量，只差一個 `model.` 名字前綴。
`prepare.py` 只改寫檔案頭，就能在本機生成底座檔案。

## 需求

- Apple Silicon Mac。我們在 64 GB 上測的。顯示記憶體佔用約 21 GiB，但**載入過程中**行程記憶體峰值
  約 41 GiB（權重會在記憶體裡過不止一遍）。32／36／48 GB 的機器沒測過，32 GB 預計會大量換頁或失敗。
- 約 45 GB 可用硬碟：權重 21.8 GB，加生成的底座檔案 21.8 GB。

## 步驟

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "lerobot[molmoact2]==0.7.0" pyarrow opencv-python
```

```bash
hf download lerobot/MolmoAct2-SO100_101-LeRobot --local-dir models/ckpt
```

```bash
hf download allenai/MolmoAct2-SO100_101 --local-dir models/base --exclude "*.safetensors" "*.py"
```

```bash
python prepare.py models/ckpt models/base
```

```bash
python smoke_test.py --ckpt models/ckpt --base models/base
```

假輸入下第 1 步應該是 `[4.78, -65.89, 70.87, 38.89, -0.25, 8.71]`。數字不一樣就說明環境有問題。

可選的真實資料回放（下載約 35 MB；國內可先設 `HF_ENDPOINT=https://hf-mirror.com`）：

```bash
./fetch_episodes.sh data/pick_lemon
```

```bash
python replay_test.py --ckpt models/ckpt --base models/base --data data/pick_lemon
```

## 在自己的程式裡用

```python
from molmoact2_mac import load

policy, pre, post, cfg = load("models/ckpt", "models/base")        # Mac 顯卡，全精度
obs = {
    "observation.images.cam0": img0,      # 浮點張量，CHW，0..1
    "observation.images.cam1": img1,
    "observation.state": joints_deg,      # 6 個關節角，LeRobot SO-101 校準坐標，單位度
    "task": "pick the yellow lemon and drop it in the red bowl",
}
actions = post(policy.predict_action_chunk(pre(obs)))   # (1, 30, 6)，絕對關節目標，單位度
```

## 沒有覆蓋的

- 還沒接真臂測過，也沒做閉環評測。
- 沒在模型沒見過的資料上測過。
- Mac 顯卡半精度出錯的根因未知。
- 沒試過在 Mac 上微調。

任何學習出來的策略上真機之前：在模型之外限制速度和工作空間，急停放在手邊，並遵守模型說明頁的安全章節。

## 授權和致謝

本倉庫腳本：Apache-2.0。不重新分發任何權重。
MolmoAct2（程式和權重）由 Allen Institute for AI 發布，Apache-2.0，按 Ai2 的 Responsible Use
Guidelines 用於研究和教育：[allenai/molmoact2](https://github.com/allenai/molmoact2)。
LeRobot 由 Hugging Face 發布，Apache-2.0。回放資料來自
[Beegbrain/pick_lemon_and_drop_in_bowl](https://huggingface.co/datasets/Beegbrain/pick_lemon_and_drop_in_bowl)。
