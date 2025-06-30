# BERT Chinese Traditional-Simplified Classifier

A robust, modular pipeline for classifying Chinese text as **Simplified**, **Traditional**, or **Hybrid** using BERT-based sequence models.

**[Demo Model on Hugging Face →](https://huggingface.co/renhehuang/bert-base-chinese-traditional-classifier-v3)**

---

## Features



---

## Project Structure

```
bert-chinese-classifier/
│

```

---

## Installation

```bash

```

---

## Quick Start

### 1. Training Pipeline

```python

```

Or simply run:

```bash

```

---



### 3. Uploading Your Own Model to Hugging Face

After training, push to Hugging Face with:

```python
from src.upload_hf import upload_to_hf

upload_to_hf(
    model_dir="./model_ckpt",
    repo_id="renhehuang/bert-base-chinese-traditional-classifier-v3",
    private=False
)
```

