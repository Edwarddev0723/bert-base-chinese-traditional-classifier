# BERT Chinese Traditional-Simplified Classifier

A modular pipeline for labeling Chinese text as **Simplified**, **Traditional**, or **Hybrid**. The project provides command line tools for dataset creation, tokenization, and BERT fine‑tuning.

---

## Features

- **Configurable dataset builder** (`data_prepare.py`)
- **Tokenization & train/val split utility** (`tokenizer_util.py`)
- **Robust BERT training CLI** (`train.py`)
- Evaluation and inference helpers
- **Interactive notebook demo** (`bert_test_notebook.ipynb`)

---

## Project Structure

```
bert-base-chinese-traditional-classifier/
├── assets/                # images and diagrams
├── src/                   # CLI tools and utilities
│   ├── data_prepare.py
│   ├── tokenizer_util.py
│   ├── train.py
│   ├── evaluate.py
│   ├── push_to_hub.py
│   └── test_inference.py
├── requirements.txt
├── bert_test_notebook.ipynb
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### 1. Build the Dataset

Use the parameter driven dataset builder. CLI flags or a YAML/JSON config may override any hyper‑parameter.

```bash
python data_prepare.py \
    --rows 200000 \
    --out_dir data_cache_v2 \
    --min_zh_ratio 0.2
# or
python data_prepare.py --config configs/cci3.yaml
```

### 2. Tokenize & Split

```bash
python tokenizer_util.py \
  --input fineweb.parquet \
  --out_dir data_cache_v2 \
  --model ckiplab/bert-base-chinese \
  --max_len 256 --stride 128 --test_size 0.2
```

### 3. Train the Model

```bash
python train.py -d data_cache_v2 -o ckpt --rows 5000 \
  --project myproj --run debug-5k
```

### 4. Evaluate & Inference

Run `evaluate.py` or `test_inference.py` for reporting and prediction demos.

### 5. Upload to Hugging Face

```python
from src.push_to_hub import push_model
push_model(repo_id="renhehuang/bert-base-chinese-traditional-classifier-v3", model_dir="./model_ckpt")
```

### 6. Interactive Notebook

Open `bert_test_notebook.ipynb` to run through the full pipeline in a Jupyter environment.
Or you can just go to the Colab and [try the pipeline.](https://colab.research.google.com/drive/1OSF3l-n60sHW0Z0kuwEpD_0zjKXxOk3Q?usp=sharing)

