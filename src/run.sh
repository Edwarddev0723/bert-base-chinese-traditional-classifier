#!/bin/zsh

echo "=== 0. 建立並啟動 Python 虛擬環境 ==="
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# echo "=== 1. 安裝必要套件 ==="
# pip install -U pip
# pip install -U transformers datasets opencc-python-reimplemented scikit-learn pandas matplotlib seaborn
# pip install -U torch torchvision torchaudio
# pip install -U accelerate
# pip install -U wandb

# echo "\n=== 2. 資料集準備 ==="
# python data_prepare.py

echo "\n=== 3. 開始訓練 ==="
python train.py

# echo "\n=== 4. 模型評估 ==="
# python evaluate.py
# 
# echo "\n=== 5. 測試推論 ==="
# python test_inference.py
# 
# echo "\n=== 6. (可選) 推送模型到 Hugging Face Hub ==="
# echo "如需推送，請執行：python push_to_hub.py"