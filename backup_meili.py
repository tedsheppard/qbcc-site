import requests
import os
import time
import sys
from google.cloud import storage

# --- Configuration ---
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# This is the directory INSIDE your Render instance where MeiliSearch saves snapshots.
# It should correspond to the disk mount point for MeiliSearch.
SNAPSHOT_DIR = "/meili_data/snapshots" 

def get_gcs_client():
    """
    Creates a GCS client by reading JSON credentials from an environment variable.
    """
    gcs_credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
    if not gcs_credentials_json:
        print("FATAL: GCS_CREDENTIALS_JSON environment variable not found.", file=sys.stderr)
        return None

    temp_credentials_path = "/tmp/gcs_credentials.json"
    try:
        with open(temp_credentials_path, "w") as f:
            f.write(gcs_credentials_json)
        client = storage.Client.from_service_account_json(temp_credentials_path)
        os.remove(temp_credentials_path) # Clean up the temp file
        return client
    except Exception as e:
        print(f"FATAL: Failed to create GCS client. Error: {e}", file=sys.stderr)
        return None

def create_and_upload_backup():
    """
    Triggers a MeiliSearch snapshot, waits for completion, and uploads it to GCS.
    """
    # --- Step 1: Basic Configuration Checks ---
    if not all([MEILI_URL, GCS_BUCKET_NAME]):
        print("FATAL: Missing required environment variables (MEILI_URL, GCS_BUCKET_NAME).")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}

    # --- Step 2: Trigger the MeiliSearch Snapshot ---
    print("-> Step 1: Triggering MeiliSearch snapshot...")
    try:
        start_res = requests.post(f"{MEILI_URL}/snapshots", headers=headers, timeout=10)
        start_res.raise_for_status()
        task = start_res.json()
        task_uid = task.get('taskUid')
        if not task_uid:
            raise Exception("MeiliSearch did not return a task UID to track.")
        print(f"   ... Snapshot creation started. Task UID: {task_uid}")
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not trigger snapshot. Is MeiliSearch running at {MEILI_URL}?")
        print(f"Error: {e}")
        sys.exit(1)

    # --- Step 3: Wait for the Snapshot to Complete ---
    print("-> Step 2: Waiting for snapshot to complete (this may take a moment)...")
    while True:
        try:
            time.sleep(2)
            task_res = requests.get(f"{MEILI_URL}/tasks/{task_uid}", headers=headers, timeout=10)
            task_res.raise_for_status()
            task_status = task_res.json()

            if task_status.get('status') == 'succeeded':
                print("   ... Snapshot task succeeded.")
                break
            if task_status.get('status') in ['failed', 'canceled']:
                error_details = task_status.get('error', {})
                raise Exception(f"Snapshot failed: {error_details.get('message', 'No details')}")
        except requests.exceptions.RequestException as e:
            print(f"FATAL: Could not get task status. Error: {e}")
            sys.exit(1)

    # --- Step 4: Find the Newest Snapshot File ---
    print(f"-> Step 3: Searching for the newest snapshot file in '{SNAPSHOT_DIR}'...")
    try:
        if not os.path.isdir(SNAPSHOT_DIR):
             raise Exception(f"Snapshot directory '{SNAPSHOT_DIR}' not found.")
        
        snapshot_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.snapshot')]
        if not snapshot_files:
            raise Exception("No snapshot files were found in the directory.")
            
        latest_snapshot = max(snapshot_files, key=lambda f: os.path.getmtime(os.path.join(SNAPSHOT_DIR, f)))
        local_snapshot_path = os.path.join(SNAPSHOT_DIR, latest_snapshot)
        print(f"   ... Found latest snapshot: {latest_snapshot}")
    except Exception as e:
        print(f"FATAL: Error finding snapshot file. {e}")
        sys.exit(1)
        
    # --- Step 5: Upload to Google Cloud Storage ---
    print(f"-> Step 4: Uploading '{latest_snapshot}' to Google Cloud Storage...")
    try:
        storage_client = get_gcs_client()
        if not storage_client:
            sys.exit(1) # get_gcs_client will print the error

        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        gcs_path = f"meili_backups/{latest_snapshot}"
        blob = bucket.blob(gcs_path)
        
        blob.upload_from_filename(local_snapshot_path)
        print(f"   ... Successfully uploaded to GCS bucket '{GCS_BUCKET_NAME}' at '{gcs_path}'.")
    except Exception as e:
        print(f"FATAL: Failed to upload to GCS. Error: {e}")
        sys.exit(1)

    # --- Step 6: Clean Up Local File ---
    print(f"-> Step 5: Cleaning up local snapshot file...")
    try:
        os.remove(local_snapshot_path)
        print(f"   ... Removed '{local_snapshot_path}'.")
    except Exception as e:
        print(f"Warning: Could not remove local snapshot file. Error: {e}")

    print("\nBackup complete! Your MeiliSearch instance has been safely backed up to GCS.")

if __name__ == "__main__":
    create_and_upload_backup()
