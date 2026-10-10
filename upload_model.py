from huggingface_hub import HfApi, create_repo

MODEL_FOLDER = "splitsnap_model"
MODEL_NAME = "splitsnap-donut"

api = HfApi()

# Uses the account you logged into with:
# hf auth login
username = api.whoami()["name"]

repo_id = f"{username}/{MODEL_NAME}"

print("Hugging Face account:", username)
print("Model repository:", repo_id)

# Create the model repository if it doesn't already exist
create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

print("\nUploading checkpoint...")

api.upload_folder(
    folder_path=MODEL_FOLDER,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload final SplitSnap Donut checkpoint",
)

# Save the repo ID so app.py can read it
with open(
    "hf_model_repo.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(repo_id)

print("\nUPLOAD COMPLETE")
print("Model repo:", repo_id)
print("Saved repo ID to hf_model_repo.txt")