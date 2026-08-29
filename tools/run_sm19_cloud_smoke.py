#!/usr/bin/env python3
"""Run the two-user Recovered SM19 cloud smoke test reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import llvmlite
import numba
import numpy as np
import psutil
from huggingface_hub import hf_hub_download

REPOSITORY = "open-spaced-repetition/anki-revlogs-10k"
REVISION = "75299740cff05894ef42d7ad990666691efdd2da"
ALGORITHM = "Recovered-SM19-Again1"
REPO_ROOT = Path(__file__).resolve().parents[1]

USERS: dict[int, dict[str, Any]] = {
    6701: {
        "size_bytes": 48_919_254,
        "sha256": "1b5e6daa8f7ae3d5dda1d3a05183ef1bde84f957e0e0577d4679519c4d98493c",
        "result": {
            "metrics": {
                "RMSE": 0.29698,
                "LogLoss": 0.33413,
                "RMSE(bins)": 0.073017,
                "smECE": 0.048437,
                "AUC": 0.638562,
                "precision@90": 0.947244,
                "recall@90": 0.546459,
                "ICI": 0.05675,
                "MBE": -0.035859,
            },
            "user": 6701,
            "size": 1_611_815,
        },
    },
    6810: {
        "size_bytes": 75_136_980,
        "sha256": "14ab7abf775d146010a489a77235621a335387202fb85d482b1376d944a42530",
        "result": {
            "metrics": {
                "RMSE": 0.337439,
                "LogLoss": 0.406514,
                "RMSE(bins)": 0.083638,
                "smECE": 0.050147,
                "AUC": 0.659859,
                "precision@90": 0.9294,
                "recall@90": 0.405098,
                "ICI": 0.05614,
                "MBE": -0.026391,
            },
            "user": 6810,
            "size": 1_939_325,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write(path, payload)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows
    )
    atomic_write(path, payload)


def validate_input(path: Path, user_id: int) -> None:
    expected = USERS[user_id]
    actual_size = path.stat().st_size
    if actual_size != expected["size_bytes"]:
        raise ValueError(
            f"user {user_id} parquet size {actual_size} != {expected['size_bytes']}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected["sha256"]:
        raise ValueError(
            f"user {user_id} parquet SHA-256 {actual_sha} != {expected['sha256']}"
        )


def source_file(source_root: Path | None, download_root: Path, user_id: int) -> Path:
    filename = f"revlogs/user_id={user_id}/data.parquet"
    if source_root is not None:
        path = source_root / filename
    else:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required because the pinned dataset is gated"
            )
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                repo_type="dataset",
                revision=REVISION,
                filename=filename,
                local_dir=download_root,
                token=token,
            )
        )
    validate_input(path, user_id)
    return path


def materialize_subset(source: Path, root: Path, user_id: int) -> Path:
    target = root / "revlogs" / f"user_id={user_id}" / "data.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_input(target, user_id)
        return root
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)
    validate_input(target, user_id)
    return root


def process_tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            continue
    return total


def load_single_result(path: Path, user_id: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"user {user_id} produced no result file")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"user {user_id} produced {len(lines)} result rows")
    result = json.loads(lines[0])
    expected = USERS[user_id]["result"]
    if result != expected:
        raise RuntimeError(
            f"user {user_id} result mismatch:\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"actual={json.dumps(result, sort_keys=True)}"
        )
    return result


def run_user(
    user_id: int, source: Path, work_dir: Path, python: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = work_dir / "runs" / f"user-{user_id}-{time.time_ns()}"
    attempt.mkdir(parents=True)
    data_root = materialize_subset(source, attempt / "dataset", user_id)
    stdout_path = attempt / "stdout.log"
    stderr_path = attempt / "stderr.log"
    command = [
        str(python),
        str(REPO_ROOT / "script.py"),
        "--algo",
        ALGORITHM,
        "--data",
        str(data_root),
        "--partitions",
        "none",
        "--processes",
        "1",
        "--n_splits",
        "5",
        "--max_seq_len",
        "64",
        "--torch_num_threads",
        "1",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SRSB_TIMING"] = "1"
    started = time.perf_counter()
    peak_rss = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=attempt,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        while process.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            time.sleep(0.25)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    peak_rss = max(peak_rss, process_tree_rss(process.pid))
    if return_code != 0:
        raise RuntimeError(
            f"user {user_id} benchmark exited {return_code}; see {stderr_path}"
        )
    result_path = attempt / "result" / f"{ALGORITHM}.jsonl"
    result = load_single_result(result_path, user_id)
    metadata = {
        "attempt": str(attempt),
        "command": command,
        "elapsed_seconds": elapsed,
        "peak_process_tree_rss_bytes": peak_rss,
        "result_sha256": sha256_file(result_path),
        "stderr_sha256": sha256_file(stderr_path),
        "stdout_sha256": sha256_file(stdout_path),
    }
    return result, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / ".sm19-cloud-smoke",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        help="Use an existing verified dataset root instead of HF_TOKEN download",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download and verify the pinned inputs without running the benchmark",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "smoke-state.json"
    result_path = work_dir / "smoke-result.jsonl"
    source_data_root = (
        args.source_data_root.resolve() if args.source_data_root else None
    )
    sources = {
        user_id: source_file(
            source_data_root,
            work_dir / "download",
            user_id,
        )
        for user_id in USERS
    }
    if args.prepare_only:
        prepared_root = source_data_root or (work_dir / "download")
        print(
            json.dumps(
                {
                    "dataset": REPOSITORY,
                    "revision": REVISION,
                    "source_data_root": str(prepared_root),
                    "status": "prepared",
                    "users": sorted(sources),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    results: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "algorithm": ALGORITHM,
        "dataset": REPOSITORY,
        "revision": REVISION,
        "environment": {
            "cpu_count": os.cpu_count(),
            "llvmlite": llvmlite.__version__,
            "machine": platform.machine(),
            "memory_total_bytes": psutil.virtual_memory().total,
            "numba": numba.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": {},
        "status": "running",
    }
    atomic_write_json(state_path, state)
    try:
        for user_id in USERS:
            result, metadata = run_user(
                user_id, sources[user_id], work_dir, args.python.absolute()
            )
            results.append(result)
            results.sort(key=lambda row: row["user"])
            state["runs"][str(user_id)] = metadata
            atomic_write_jsonl(result_path, results)
            atomic_write_json(state_path, state)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(state_path, state)
        raise

    state["status"] = "complete"
    state["result_sha256"] = sha256_file(result_path)
    atomic_write_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
