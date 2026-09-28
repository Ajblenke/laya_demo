"""Save each run to `data/` so the notebook can replay the last one without a network."""

import json
import shutil
from pathlib import Path

from .records import RunResult

DEFAULT_DATA_DIR = Path("data")
LAST_RUN = "last_run.json"


def save_run(run: RunResult, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Write the run to `data/runs/<timestamp>.json` and refresh `data/last_run.json`."""
    runs = Path(data_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = run.meta["started_at"][:19].replace(":", "").replace("-", "")
    path = runs / f"run-{stamp}.json"
    path.write_text(json.dumps(run.to_dict(), indent=1))
    shutil.copyfile(path, Path(data_dir) / LAST_RUN)
    return path


def load_run(path: Path | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> RunResult:
    """Load a saved run, by default the most recent one."""
    path = Path(path) if path else Path(data_dir) / LAST_RUN
    if not path.exists():
        raise FileNotFoundError(f"No cached run at {path}. Run `uv run laya-demo` once while online.")
    return RunResult.from_dict(json.loads(path.read_text()))
