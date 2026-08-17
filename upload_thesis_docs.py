"""
upload_thesis_docs.py - Uploads the compiled Word thesis documents to the new GitHub repo
"""

import os
import base64
import subprocess
import urllib.request
import json

REPO_OWNER = "saadamirlive-blip"
REPO_NAME = "rl-adaptive-alert-correlation"

def get_token():
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def upload_file(filepath, rel_path, token):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{rel_path}"
    with open(filepath, "rb") as f:
        content_bytes = f.read()
    b64_content = base64.b64encode(content_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Antigravity-Thesis-Deployer",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    sha = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            existing = json.loads(resp.read().decode("utf-8"))
            sha = existing.get("sha")
    except Exception:
        pass

    payload = {
        "message": f"Upload {rel_path}",
        "content": b64_content,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    req = urllib.request.Request(url, headers=headers, data=json.dumps(payload).encode("utf-8"), method="PUT")
    with urllib.request.urlopen(req) as resp:
        print(f"[+] Successfully uploaded {rel_path}")

if __name__ == "__main__":
    token = get_token()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ["MS_Thesis_Haaziq_Rasool.docx", "MS_Thesis_Log_Correlation_RL.docx"]:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            upload_file(fpath, fname, token)
