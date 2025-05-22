# BERT Chinese Traditional-Simplified Classifier

A robust, modular pipeline for classifying Chinese text as **Simplified**, **Traditional**, or **Hybrid** using BERT-based sequence models.

**[Demo Model on Hugging Face →](https://huggingface.co/renhehuang/bert-base-chinese-traditional-classifier-v3)**

---

## Features

- **Automatic sampling** from major open Chinese web corpora
- **Hybrid text generation** by segmenting and mixing simplified/traditional characters
- **Customizable, object-oriented pipeline** for end-to-end data, model, and analysis workflow
- **Classification for three classes**: Simplified, Traditional, Hybrid
- **Extensive error analysis** and visualization
- **Easy model sharing via Hugging Face Hub**

---

## Project Structure

```
bert-chinese-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_fun.py
├── src/
│   ├── pipeline.py
│   └── upload_hf.py
├── main.py
├── tests/
│   └── test_pipeline.py
└── examples/
    └── train_and_eval_example.ipynb
```

---

## Installation

```bash
git clone https://github.com/yourusername/bert-chinese-classifier.git
cd bert-chinese-classifier
pip install -r requirements.txt
```

---

## Quick Start

### 1. Training Pipeline

```python
from data_fun import prepare_dataset
from src.pipeline import TextClassifierPipeline

pipeline = TextClassifierPipeline()
pipeline.prepare_data(prepare_dataset_func=prepare_dataset)
pipeline.setup_model()
pipeline.train()
y_true, y_pred, cm, report = pipeline.evaluate()
pipeline.visualize_confusion_matrix(cm)
pipeline.error_analysis(y_true, y_pred)
```

Or simply run:

```bash
python main.py
```

---

### 2. Inference with the Public Model

You can **directly use the public model on Hugging Face**:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="renhehuang/bert-base-chinese-traditional-classifier-v3",
    tokenizer="renhehuang/bert-base-chinese-traditional-classifier-v3"
)
print(classifier("你好中國"))
```
- Output example:
    ```
    [{'label': 'LABEL_1', 'score': 0.997...}]
    ```
    - `LABEL_0`: Simplified
    - `LABEL_1`: Traditional
    - `LABEL_2`: Hybrid

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
- You can then see and share your model at:  
  [https://huggingface.co/renhehuang/bert-base-chinese-traditional-classifier-v3](https://huggingface.co/renhehuang/bert-base-chinese-traditional-classifier-v3)

---

## Dataset Preparation Details

- **sample_streaming**: Efficient sampling from huge datasets (streaming mode)
- **sample_unique_streaming**: No overlap between category samples
- **random_hybrid_segments_distributed**: Robust hybrid (mixed) data synthesis
- **split_long_text**: Chunk long texts to fit model input
- **prepare_dataset**: Integrates above for one-click dataset prep

---

## Model Training & Customization

- Based on [ckiplab/bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese)
- Three-way classification:  
    - `label=0`: Simplified  
    - `label=1`: Traditional  
    - `label=2`: Hybrid  
- Training parameters fully customizable in `TextClassifierPipeline`

---

## Visualization & Error Analysis

- Prints classification report and confusion matrix
- Displays class-wise error statistics for targeted improvement
- Automatic CJK font detection for matplotlib visualizations

---

## Requirements

- `transformers`
- `datasets`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `opencc`
- `huggingface_hub`

See `requirements.txt` for details.

---

## License

MIT License

---
