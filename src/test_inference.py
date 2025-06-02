# test_inference.py
"""
測試資料推論與結果統計
"""
import random
import pandas as pd
from transformers import pipeline

def build_test_df():
    trad_samples = [
        "這是一個繁體中文的測試句子。", "歡迎來到台灣！", "請問您需要什麼協助？",
        "我們今天學習了很多知識。", "天氣很好，適合出門散步。", "你喜歡吃什麼台灣小吃？",
        "這本書很有趣，推薦你看看。", "捷運系統非常方便。", "週末一起去看電影好嗎？", "我在圖書館讀書。"
    ]
    simp_samples = [
        "这是一个简体中文的测试句子。", "欢迎来到中国！", "请问您需要什么帮助？",
        "我们今天学到了很多知识。", "天气很好，适合出去散步。", "你喜欢吃什么中国小吃？",
        "这本书很有趣，推荐你看看。", "地铁系统非常方便。", "周末一起去看电影好吗？", "我在图书馆学习。"
    ]
    def random_samples(source, n):
        return [random.choice(source) for _ in range(n)]
    def mixed_samples(trad_source, simp_source, n, trad_ratio):
        n_trad = int(n * trad_ratio)
        n_simp = n - n_trad
        return random_samples(trad_source, n_trad) + random_samples(simp_source, n_simp)
    trad_data = random_samples(trad_samples, 50)
    simp_data = random_samples(simp_samples, 50)
    mix1_data = mixed_samples(trad_samples, simp_samples, 50, trad_ratio=0.7)
    random.shuffle(mix1_data)
    mix2_data = mixed_samples(trad_samples, simp_samples, 50, trad_ratio=0.3)
    random.shuffle(mix2_data)
    test_df = pd.DataFrame({
        "group": ["trad"]*50 + ["simp"]*50 + ["mix1"]*50 + ["mix2"]*50,
        "text": trad_data + simp_data + mix1_data + mix2_data
    })
    return test_df

def run_inference(model_dir="./model_ckpt"):
    test_df = build_test_df()
    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=0)
    results = [classifier(t)[0] for t in test_df["text"]]
    test_df["pred_label"] = [r["label"] for r in results]
    test_df["score"] = [r["score"] for r in results]
    summary = test_df.groupby("group")["pred_label"].value_counts().unstack(fill_value=0)
    print("分類結果分布：")
    print(summary)
    score_stats = test_df.groupby("group")["score"].agg(["mean", "std", "min", "max"])
    print("\n信心分數統計：")
    print(score_stats)
    return test_df

if __name__ == "__main__":
    run_inference()
