import pandas as pd
import boto3
import os
import subprocess
import argparse
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--baseline_path", type=str, required=True, help="Path to baseline dataset")
parser.add_argument("--new_batch_path", type=str, required=True, help="Path to new incoming batch")
parser.add_argument("--bucket_name", type=str, required=True, help="Name of S3 bucket for model upload")
parser.add_argument("--model_prefix", type=str, default="models/", help="S3 path prefix for model storage")
args = parser.parse_args()

baseline_path = args.baseline_path
new_batch_path = args.new_batch_path
bucket_name = args.bucket_name
model_prefix = args.model_prefix

# -------------------------
# Load Datasets
# -------------------------
print("📥 Loading baseline and new batch datasets...")
baseline = pd.read_csv(baseline_path)
new_batch = pd.read_csv(new_batch_path)
print(f"Baseline shape: {baseline.shape}")
print(f"New batch shape: {new_batch.shape}")

# -------------------------
# Drift Detection
# -------------------------
print("🔍 Running data drift detection...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline, current_data=new_batch)
report.save_html("drift_detection_report.html")
print("📄 Drift report saved: drift_detection_report.html")

result = report.as_dict()
drift_detected = result['metrics'][0]['result']['dataset_drift']

# -------------------------
# Retraining on Drift
# -------------------------
if drift_detected:
    print("⚠️ Drift detected! Proceeding with retraining...")

    # Combine baseline + new batch
    combined_df = pd.concat([baseline, new_batch]).drop_duplicates().reset_index(drop=True)
    combined_path = "combined_data.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"✅ Combined dataset saved: {combined_path}")

    # Call training script
    subprocess.run(["python", "train.py", "--dataset", combined_path], check=True)
    print("✅ Model retrained.")

    # -------------------------
    # Upload to S3
    # -------------------------
    model_file = "fraud_model.pth"
    if os.path.exists(model_file):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        s3_key = f"{model_prefix}fraud_model_{timestamp}.pth"

        s3 = boto3.client("s3")
        s3.upload_file(model_file, bucket_name, s3_key)
        print(f"☁️ Model uploaded to s3://{bucket_name}/{s3_key}")
    else:
        print("❗Model file not found. Skipping upload.")
        exit(1)

    # -------------------------
    # Trigger Deployment
    # -------------------------
    print("🚀 Triggering redeployment via GitHub Actions...")
    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    branch = "main"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # Create a dummy commit file
with open("trigger.txt", "w") as f:
    f.write("trigger deployment\n")

    subprocess.run(["git", "config", "--global", "user.email", "bot@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "GitHub Actions Bot"])
    subprocess.run(["git", "add", "trigger.txt"])
    subprocess.run(["git", "commit", "-m", "Trigger ECS deployment"])
    subprocess.run(["git", "push", f"https://x-access-token:{token}@github.com/{repo}.git", branch]

    print("✅ Redeployment triggered.")

else:
    print("✅ No drift detected. No retraining needed.")
