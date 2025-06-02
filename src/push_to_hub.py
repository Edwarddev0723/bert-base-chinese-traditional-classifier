# push_to_hub.py
"""
模型推送到 Hugging Face Hub
"""
from huggingface_hub import notebook_login, create_repo, upload_folder

def push_model(repo_id="renhehuang/bert-base-chinese-traditional-classifier-v3", model_dir="./model_ckpt"):
    notebook_login()
    create_repo(repo_id, private=False)
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message="Initial commit"
    )
    print(f"模型已推送到 Hugging Face Hub: {repo_id}")

if __name__ == "__main__":
    push_model()
