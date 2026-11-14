from huggingface_hub import HfApi

REPO_ID = "rakesh-bishnoi29/structuraai-dataset"

# Path to your extracted dataset folder
DATASET_PATH = r"D:\DEV\Laxsi Labs Hacathon\Dataset\Invoice_Dataset"

api = HfApi()

# Create dataset repository if it doesn't exist
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True
)

# Upload complete folder, preserving subdirectories
api.upload_folder(
    folder_path=DATASET_PATH,
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Uploaded StructuraAI invoice dataset"
)

print("Dataset uploaded successfully!")
print(f"https://huggingface.co/datasets/{REPO_ID}")
