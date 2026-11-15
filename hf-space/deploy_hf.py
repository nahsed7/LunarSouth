"""Deploy LunarSouth to a free Hugging Face Docker Space (permanent URL, runs on HF servers).

    1.  CLPS_mission_planner-main/.venv/bin/hf auth login        (paste a *Write* token from huggingface.co/settings/tokens)
    2.  CLPS_mission_planner-main/.venv/bin/python hf-space/deploy_hf.py [space-name]      (default: lunarsouth)

Re-run step 2 any time to push updates. Only changed files are uploaded."""
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

APP = Path(__file__).resolve().parent.parent / "CLPS_mission_planner-main"
HF = Path(__file__).resolve().parent
name = sys.argv[1] if len(sys.argv) > 1 else "lunarsouth"
FRONT = """---
title: LunarSouth CLPS Mission Planner
emoji: 🌑
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
short_description: Lunar south-pole power, comms and mission planner
---

"""
api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/{name}"
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
IGNORE = [".venv/**", "**/__pycache__/**", "*.pyc", "outputs/**", "*.zip", "datakit (1).py", "app_v2_pre_polewide.py",
          "data/discovered_sites.prev.csv", "keep_public.sh", "start_public.sh", "setup_named_tunnel.sh",
          "README.md", "Dockerfile", ".dockerignore", "requirements-deploy.txt"]
print("uploading app + data (first time ~560 MB, later only changes) ...")
api.upload_folder(folder_path=str(APP), repo_id=repo, repo_type="space", ignore_patterns=IGNORE,
                  commit_message="LunarSouth app + data")
readme = (FRONT + (APP / "README.md").read_text()).encode()
api.create_commit(repo, repo_type="space", commit_message="Space config (Docker, port 7860)", operations=[
    CommitOperationAdd("Dockerfile", str(HF / "Dockerfile")),
    CommitOperationAdd("requirements-hf.txt", str(HF / "requirements-hf.txt")),
    CommitOperationAdd("README.md", readme)])
sub = f"{user}-{name}".lower().replace("_", "-").replace(".", "-")
print(f"\nBuilding now (5-10 min). Watch the logs: https://huggingface.co/spaces/{repo}?logs=build")
print(f"SPACE PAGE : https://huggingface.co/spaces/{repo}")
print(f"DIRECT APP : https://{sub}.hf.space")
