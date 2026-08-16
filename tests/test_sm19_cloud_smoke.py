import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
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
