import requests
import os
import time
import sys
from google.cloud import storage
import argparse
from urllib.parse import urlparse
from datetime import timedelta

# --- Configuration ---
MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

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

def get_meili_data_path(headers: dict) -> str:
    """
    Asks the MeiliSearch API for its stats to determine its data directory.
    """
    print("-> Step 1: Auto-detecting MeiliSearch data path...")
    try:
        stats_res = requests.get(f"{MEILI_URL}/stats", headers=headers, timeout=10)
        stats_res.raise_for_status()
        db_path = stats_res.json().get('databasePath')
        if not db_path:
            raise Exception("Could not find 'databasePath' in MeiliSearch stats.")
        
        # The data path is the directory containing the 'data.ms' file
        data_dir = os.path.dirname(db_path)
        print(f"   ... MeiliSearch data directory detected: {data_dir}")
        return data_dir
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not get stats from MeiliSearch to find data path. Is it running?")
        print(f"Error: {e}")
        sys.exit(1)


def create_and_download_backup():
    """
    Triggers a MeiliSearch snapshot, uploads it to GCS, and provides a direct download link.
    """
    if not all([MEILI_URL, GCS_BUCKET_NAME]):
        print("FATAL: Missing required environment variables (MEILI_URL, GCS_BUCKET_NAME).")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {MEILI_KEY}"} if MEILI_KEY else {}

    # Step 1: Find where MeiliSearch stores its data
    meili_data_dir = get_meili_data_path(headers)
    snapshot_dir = os.path.join(meili_data_dir, "snapshots")

    # Step 2: Trigger the snapshot
    print("-> Step 2: Triggering MeiliSearch snapshot...")
    try:
        start_res = requests.post(f"{MEILI_URL}/snapshots", headers=headers, timeout=10)
        start_res.raise_for_status()
        task_uid = start_res.json().get('taskUid')
        if not task_uid:
            raise Exception("MeiliSearch did not return a task UID.")
        print(f"   ... Snapshot creation started. Task UID: {task_uid}")
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not trigger snapshot. Error: {e}")
        sys.exit(1)

    # Step 3: Wait for completion
    print("-> Step 3: Waiting for snapshot to complete...")
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
                raise Exception(f"Snapshot failed: {task_status.get('error', {}).get('message', 'No details')}")
        except requests.exceptions.RequestException as e:
            print(f"FATAL: Could not get task status. Error: {e}")
            sys.exit(1)

    # Step 4: Find the new snapshot file in the auto-detected directory
    print(f"-> Step 4: Searching for the newest snapshot file in '{snapshot_dir}'...")
    try:
        if not os.path.isdir(snapshot_dir):
             raise Exception(f"Auto-detected snapshot directory '{snapshot_dir}' does not exist. This may indicate a problem with your MeiliSearch disk setup on Render.")
        
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith('.snapshot')]
        if not snapshot_files:
            raise Exception("No snapshot files were found.")
            
        latest_snapshot = max(snapshot_files, key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f)))
        local_snapshot_path = os.path.join(snapshot_dir, latest_snapshot)
        print(f"   ... Found latest snapshot: {latest_snapshot}")
    except Exception as e:
        print(f"FATAL: Error finding snapshot file. {e}")
        sys.exit(1)
        
    # Step 5: Upload to GCS
    print(f"-> Step 5: Uploading '{latest_snapshot}' to Google Cloud Storage...")
    gcs_path = f"meili_backups/{latest_snapshot}"
    try:
        storage_client = get_gcs_client()
        if not storage_client: sys.exit(1)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_snapshot_path)
        print(f"   ... Successfully uploaded to GCS.")
    except Exception as e:
        print(f"FATAL: Failed to upload to GCS. Error: {e}")
        sys.exit(1)

    # Step 6: Generate a signed URL for local download
    print(f"-> Step 6: Generating secure download link...")
    try:
        download_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15), # Link will be valid for 15 minutes
            method="GET",
        )
        print("\n" + "="*70)
        print("  BACKUP COMPLETE & READY FOR LOCAL DOWNLOAD")
        print("="*70)
        print("\n  Your secure download link is (valid for 15 minutes):")
        print(f"\n  {download_url}\n")
        print("  Click the link above to download the backup to your computer.")
        print("="*70)
    except Exception as e:
        print(f"Warning: Could not generate download link, but backup is safe in GCS at '{gcs_path}'.")
        print(f"Error: {e}")

    # Step 7: Clean up local file
    try:
        os.remove(local_snapshot_path)
        print(f"\n-> Step 7: Cleaned up local snapshot file on the server.")
    except Exception as e:
        print(f"Warning: Could not remove local snapshot file. Error: {e}")

if __name__ == "__main__":
    create_and_download_backup()

