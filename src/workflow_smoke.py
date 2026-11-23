"""Verify a deployed source checkout and write a small workflow result."""

import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    detectors = ("ocrapt", "velox", "orthrus", "magic", "kairos")
    required = [root / "pidsmaker/main.py", root / "pyproject.toml"]
    required.extend(root / "config" / (name + ".yml") for name in detectors)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing source files: " + ", ".join(missing))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    output = Path(os.environ.get("RUN_DIR", root.parent))
    summary = {
        "status": "succeeded",
        "check": "source deployment and remote execution",
        "commit": commit,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "detector_configs_found": list(detectors),
        "training_performed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
