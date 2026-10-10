"""Download the public Roboflow Universe datasets (latest version, YOLOv8 format)."""
import os

from dotenv import load_dotenv
from roboflow import Roboflow

from config import RAW_ROBOFLOW, ROBOFLOW_DATASETS, ROOT


def version_number(v):
    return int(str(v.version).split("/")[-1])


def main():
    load_dotenv(ROOT / ".env")
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (see .env.example)")

    rf = Roboflow(api_key=key)
    for workspace, project_id in ROBOFLOW_DATASETS:
        project = rf.workspace(workspace).project(project_id)
        if project.type != "object-detection":
            print(f"skip {workspace}/{project_id}: {project.type}")
            continue
        latest = max(project.versions(), key=version_number)
        out = RAW_ROBOFLOW / project_id
        print(f"{workspace}/{project_id} v{version_number(latest)} -> {out}")
        latest.download("yolov8", location=str(out), overwrite=False)


if __name__ == "__main__":
    main()
