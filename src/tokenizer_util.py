# tokenizer_util.py
"""
tokenizer 初始化與資料集 tokenization
"""
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import multiprocessing as mp

NUM_PROC  = mp.cpu_count()  # 或你實際計算的最佳值
# 1. 檢查原始欄位
# 例：['text', 'label', 'id', 'source']

# 2. 定義 tokenize_fn（只回傳 3 個欄位）
def tokenize_fn(examples):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")
    texts = examples["text"]
    labs  = examples["label"]

    input_ids_batch = []
    attn_mask_batch = []
    label_batch     = []

    for text, lab in zip(texts, labs):
        if len(text) <= 256:
            out = tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_attention_mask=True,
                return_token_type_ids=False,
                padding=False
            )
            input_ids_batch.append(out["input_ids"])
            attn_mask_batch.append(out["attention_mask"])
            label_batch.append(lab)
        else:
            out = tokenizer(
                text,
                truncation=True,
                max_length=256,
                stride=128,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                padding=False
            )
            mapping = out.pop("overflow_to_sample_mapping")
            chunk_labels = [lab] * len(mapping)

            input_ids_batch.extend(out["input_ids"])
            attn_mask_batch.extend(out["attention_mask"])
            label_batch.extend(chunk_labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attn_mask_batch,
        "labels": label_batch
    }

def split_and_tokenize(dataset, tokenizer, test_size=0.2, random_state=42):
    df = dataset.to_pandas()
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True))
    })
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    encoded_dataset = raw_datasets.map(
    tokenize_fn,
    batched=True,
    batch_size=512,                   # 視記憶體與 CPU 能力調整
    num_proc=NUM_PROC,                # 多進程
    remove_columns=["text", "label"]
)
    return encoded_dataset, df_train, df_val

if __name__ == "__main__":
    from data_prepare import prepare_dataset
    dataset, _ = prepare_dataset()
    encoded_dataset, df_train, df_val = split_and_tokenize(dataset, get_tokenizer())
    print(encoded_dataset)
    print(df_train.head())
    print(df_val.head())
