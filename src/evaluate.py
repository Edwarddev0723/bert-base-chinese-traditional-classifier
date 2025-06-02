# evaluate.py
"""
模型評估、報表與混淆矩陣
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import main as train_main

def evaluate(trainer, encoded_dataset):
    eval_dataset = encoded_dataset["validation"]
    preds_output = trainer.predict(eval_dataset)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)
    target_names = ["簡體", "繁體", "混合"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("=== Classification Report ===")
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    print("=== Confusion Matrix ===")
    print(cm)
    # 中文字型自動尋找
    from matplotlib import font_manager
    font_path = ""
    for candidate in ['NotoSansCJK-TC-Regular.otf', 'SimHei.ttf', 'Microsoft JhengHei.ttf']:
        fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        for f in fonts:
            if candidate in f:
                font_path = f
                break
        if font_path:
            break
    if not font_path:
        print("⚠️ 未找到合適中文字型，請手動下載並設置 font_path。")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('預測類別')
    plt.ylabel('真實類別')
    plt.title('Confusion Matrix')
    plt.show()
    return report, cm

def error_analysis(y_true, y_pred, target_names):
    import pandas as pd
    df_check = pd.DataFrame({'true': y_true, 'pred': y_pred})
    print("=== 預測類別分布 ===")
    print(df_check['pred'].value_counts().sort_index())
    print("=== 真實類別分布 ===")
    print(df_check['true'].value_counts().sort_index())
    for true_label in range(3):
        misclassified = df_check[(df_check['true'] == true_label) & (df_check['pred'] != true_label)]
        print(f"--- 真實類別 {target_names[true_label]} 被誤判分布 ---")
        print(misclassified['pred'].value_counts())

if __name__ == "__main__":
    trainer, encoded_dataset = train_main()
    eval_dataset = encoded_dataset["validation"]
    preds_output = trainer.predict(eval_dataset)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)
    target_names = ["簡體", "繁體", "混合"]
    evaluate(trainer, encoded_dataset)
    error_analysis(y_true, y_pred, target_names)
