import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

class TextClassifierPipeline:
    def __init__(
        self,
        model_name="ckiplab/bert-base-chinese",
        num_labels=3,
        hidden_dropout_prob=0.3,
        train_batch_size=16,
        eval_batch_size=64,
        max_length=512,
        epochs=3,
        learning_rate=2e-5,
        output_dir="./model_ckpt"
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        self.encoded_dataset = None
        self.target_names = ["簡體", "繁體", "混合"]

    def prepare_data(self, prepare_dataset_func):
        df = prepare_dataset_func().to_pandas()
        df_train, df_val = train_test_split(
            df,
            test_size=0.2,
            stratify=df['label'],
            random_state=42
        )
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
            "validation": Dataset.from_pandas(df_val.reset_index(drop=True))
        })

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        encoded_dataset = raw_datasets.map(tokenize, batched=True)
        # 確保 label 為 int
        encoded_dataset = encoded_dataset.map(lambda x: {"label": int(x["label"])})
        self.encoded_dataset = encoded_dataset

    def setup_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.hidden_dropout_prob
        )

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            lr_scheduler_type="linear",
            weight_decay=0.1,
            warmup_ratio=0.06,
            save_strategy="steps",
            save_steps=1000,
            logging_strategy="steps",
            logging_steps=200,
            eval_strategy="steps",
            eval_steps=1000,
            save_total_limit=2,
            fp16=True,
            seed=42,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=2,
            disable_tqdm=False,
            max_grad_norm=1.0,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["validation"],
            tokenizer=self.tokenizer,
        )
        self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def evaluate(self, dataset_split="validation"):
        eval_dataset = self.encoded_dataset[dataset_split]
        preds_output = self.trainer.predict(eval_dataset)
        y_true = preds_output.label_ids
        y_pred = np.argmax(preds_output.predictions, axis=1)
        report = classification_report(
            y_true, y_pred, target_names=self.target_names, digits=4)
        print("=== Classification Report ===")
        print(report)
        cm = confusion_matrix(y_true, y_pred)
        print("=== Confusion Matrix ===")
        print(cm)
        return y_true, y_pred, cm, report

    def visualize_confusion_matrix(self, cm):
        font_path = ""
        for candidate in ['NotoSansCJK-TC-Regular.otf', 'SimHei.ttf', 'Microsoft JhengHei.ttf']:
            fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            for f in fonts:
                if candidate in f:
                    font_path = f
                    break
            if font_path:
                break

        plt.rcParams['font.sans-serif'] = [
            'Microsoft JhengHei', 'Noto Sans CJK TC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.target_names,
            yticklabels=self.target_names
        )
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')
        plt.title('Confusion Matrix')
        plt.show()

    def error_analysis(self, y_true, y_pred):
        df_check = pd.DataFrame({'true': y_true, 'pred': y_pred})
        print("=== 預測類別分布 ===")
        print(df_check['pred'].value_counts().sort_index())
        print("=== 真實類別分布 ===")
        print(df_check['true'].value_counts().sort_index())
        for true_label in range(self.num_labels):
            misclassified = df_check[(df_check['true'] == true_label) & (df_check['pred'] != true_label)]
            print(f"--- 真實類別 {self.target_names[true_label]} 被誤判分布 ---")
            print(misclassified['pred'].value_counts())

# 用法範例
# from data_fun import prepare_dataset
pipeline = TextClassifierPipeline()

pipeline.prepare_data(prepare_dataset_func=prepare_dataset)
pipeline.setup_model()
pipeline.train()
y_true, y_pred, cm, report = pipeline.evaluate()
pipeline.visualize_confusion_matrix(cm)
pipeline.error_analysis(y_true, y_pred)
