# tokenizer_util.py
"""
tokenizer 初始化與資料集 tokenization
"""
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def get_tokenizer(model_name="ckiplab/bert-base-chinese"):
    return AutoTokenizer.from_pretrained(model_name)

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
    encoded_dataset = raw_datasets.map(tokenize, batched=True)
    return encoded_dataset, df_train, df_val

if __name__ == "__main__":
    from data_prepare import prepare_dataset
    dataset, _ = prepare_dataset()
    tokenizer = get_tokenizer()
    encoded_dataset, df_train, df_val = split_and_tokenize(dataset, tokenizer)
    print(encoded_dataset)
    print(df_train.head())
    print(df_val.head())
