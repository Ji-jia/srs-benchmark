import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_sm19_cloud_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_sm19_cloud_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SM19CloudSmokeTests(unittest.TestCase):
    def test_atomic_jsonl_has_one_line_per_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            rows = [{"user": 2}, {"user": 7}]
            smoke.atomic_write_jsonl(path, rows)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"user": 2}\n{"user": 7}\n',
            )

    def test_materialized_subset_is_verified(self) -> None:
        payload = b"parquet-test-payload"
        expected = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "result": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            source.write_bytes(payload)
            with patch.dict(smoke.USERS, {7: expected}, clear=True):
                dataset = smoke.materialize_subset(source, root / "subset", 7)
                target = dataset / "revlogs" / "user_id=7" / "data.parquet"
                self.assertEqual(target.read_bytes(), payload)

    def test_prepare_only_downloads_and_verifies_without_running(self) -> None:
        payload = b"verified-cloud-input"
        expected = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "result": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "downloaded.parquet"
            downloaded.write_bytes(payload)
            args = SimpleNamespace(
                prepare_only=True,
                python=Path(sys.executable),
                source_data_root=None,
                work_dir=root / "work",
            )
            with (
                patch.dict(smoke.USERS, {7: expected}, clear=True),
                patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
                patch.object(smoke, "parse_args", return_value=args),
                patch.object(
                    smoke,
                    "hf_hub_download",
                    return_value=str(downloaded),
                ) as download,
                patch.object(smoke, "run_user") as run_user,
            ):
                self.assertEqual(smoke.main(), 0)

            download.assert_called_once_with(
                repo_id=smoke.REPOSITORY,
                repo_type="dataset",
                revision=smoke.REVISION,
                filename="revlogs/user_id=7/data.parquet",
                local_dir=root / "work" / "download",
                token="test-token",
            )
            run_user.assert_not_called()
            self.assertFalse((root / "work" / "smoke-state.json").exists())
            self.assertFalse((root / "work" / "smoke-result.jsonl").exists())

    def test_python_executable_is_made_absolute_without_resolving(self) -> None:
        python = Path("venv-python-symlink")
        expected_python = python.absolute()
        original_resolve = Path.resolve

        def reject_python_resolve(
            path: Path, *resolve_args: object, **resolve_kwargs: object
        ) -> Path:
            if path == python:
                raise AssertionError("the Python symlink must not be resolved")
            return original_resolve(path, *resolve_args, **resolve_kwargs)

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                prepare_only=False,
                python=python,
                source_data_root=None,
                work_dir=Path(directory),
            )
            with (
                patch.dict(smoke.USERS, {7: {}}, clear=True),
                patch.object(smoke, "parse_args", return_value=args),
                patch.object(smoke, "source_file", return_value=Path("source")),
                patch.object(
                    smoke,
                    "run_user",
                    return_value=({"user": 7}, {}),
                ) as run_user,
                patch.object(Path, "resolve", reject_python_resolve),
            ):
                self.assertEqual(smoke.main(), 0)

            self.assertEqual(run_user.call_args.args[3], expected_python)

    def test_result_must_match_all_expected_fields(self) -> None:
        expected_result = {"metrics": {"AUC": 0.5}, "user": 7, "size": 9}
        expected = {"size_bytes": 1, "sha256": "unused", "result": expected_result}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text(json.dumps(expected_result) + "\n", encoding="utf-8")
            with patch.dict(smoke.USERS, {7: expected}, clear=True):
                self.assertEqual(smoke.load_single_result(path, 7), expected_result)
                path.write_text(
                    json.dumps({**expected_result, "size": 10}) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "result mismatch"):
                    smoke.load_single_result(path, 7)


if __name__ == "__main__":
    unittest.main()
