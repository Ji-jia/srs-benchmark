from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

import numpy as np

from models import recovered_sm19
from models.recovered_sm19 import RecoveredSM19


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _history(length: int, seed: int) -> list[recovered_sm19.HistoryRecord]:
    rng = np.random.default_rng(seed)
    records = []
    repetitions = 1
    lapses = 0
    for index in range(length):
        grade = int(rng.integers(0, 6))
        if grade >= 3:
            repetitions += 1
        else:
            repetitions = 1
            lapses += 1
        records.append(
            recovered_sm19.HistoryRecord(
                elapsed_days=float(np.exp(rng.uniform(-3.0, 9.0))),
                grade=grade,
                repetitions=repetitions,
                lapses=lapses,
            )
        )
    return records


class RecoveredSM19NumbaTests(unittest.TestCase):
    def test_nopython_kernel_matches_python_candidate_search_bitwise(self) -> None:
        model = RecoveredSM19(1)
        for case, length in enumerate((12, 13, 16, 32, 61, 95, 1435)):
            history = _history(length, 0x5A19 + case)
            with patch.object(
                recovered_sm19,
                "NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY",
                10_000,
            ):
                reference = model._initial_difficulty_for_stability(history)
            with patch.object(
                recovered_sm19,
                "NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY",
                12,
            ):
                candidate = model._initial_difficulty_for_stability(history)
            self.assertEqual(_float_bits(candidate), _float_bits(reference))
            if length == 1435:
                self.assertEqual(candidate, 0.5)

        dispatcher = recovered_sm19._initial_difficulty_for_stability_kernel
        self.assertTrue(dispatcher.nopython_signatures)
        self.assertTrue(dispatcher.targetoptions["nopython"])
        self.assertFalse(dispatcher.targetoptions["fastmath"])
        self.assertEqual(type(dispatcher._cache).__name__, "NullCache")

    def test_full_search_kernel_matches_python_search_bitwise(self) -> None:
        model = RecoveredSM19(1)
        cases = (
            (2, 0.0, False),
            (3, 1.0, True),
            (4, 7.0 / 19.0, False),
            (8, 0.5, True),
            (12, 3.0 / 19.0, True),
            (32, 11.0 / 19.0, False),
            (95, 17.0 / 19.0, True),
            (144, 1.0, True),
            (1435, 0.5, True),
        )
        for case, (length, difficulty, allow_up) in enumerate(cases):
            history = _history(length, 0x5EA2C + case)
            elapsed = np.fromiter(
                (record.elapsed_days for record in history),
                dtype=np.float64,
                count=length,
            )
            grade = np.fromiter(
                (record.grade for record in history),
                dtype=np.int64,
                count=length,
            )
            reference = model._search_initial_stability(
                history,
                difficulty,
                allow_up=allow_up,
            )
            candidate = float(
                recovered_sm19._initial_stability_search_kernel(
                    elapsed,
                    grade,
                    difficulty,
                    allow_up,
                )
            )
            self.assertEqual(_float_bits(candidate), _float_bits(reference))

        dispatcher = recovered_sm19._initial_stability_search_kernel
        self.assertTrue(dispatcher.nopython_signatures)
        self.assertTrue(dispatcher.targetoptions["nopython"])
        self.assertFalse(dispatcher.targetoptions["fastmath"])
        self.assertEqual(type(dispatcher._cache).__name__, "NullCache")

    def test_threshold_is_only_a_bitwise_equal_performance_branch(self) -> None:
        model = RecoveredSM19(1)
        compiled = recovered_sm19._initial_difficulty_for_stability_kernel

        short_history = _history(11, 11)
        with patch.object(
            recovered_sm19,
            "_initial_difficulty_for_stability_kernel",
            wraps=compiled,
        ) as kernel:
            model._initial_difficulty_for_stability(short_history)
            kernel.assert_not_called()

        threshold_history = _history(12, 12)
        with patch.object(
            recovered_sm19,
            "NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY",
            10_000,
        ):
            reference = model._initial_difficulty_for_stability(threshold_history)
        with patch.object(
            recovered_sm19,
            "_initial_difficulty_for_stability_kernel",
            wraps=compiled,
        ) as kernel:
            candidate = model._initial_difficulty_for_stability(threshold_history)
            kernel.assert_called_once()
        self.assertEqual(_float_bits(candidate), _float_bits(reference))

        search_compiled = recovered_sm19._initial_stability_search_kernel
        single_history = _history(1, 21)
        with patch.object(
            recovered_sm19,
            "_initial_stability_search_kernel",
            wraps=search_compiled,
        ) as search_kernel:
            model._initial_stability(single_history)
            search_kernel.assert_not_called()

        two_record_history = _history(2, 22)
        with patch.object(
            recovered_sm19,
            "NUMBA_INITIAL_STABILITY_SEARCH_MIN_HISTORY",
            10_000,
        ):
            reference = model._initial_stability(two_record_history)
        with patch.object(
            recovered_sm19,
            "_initial_stability_search_kernel",
            wraps=search_compiled,
        ) as search_kernel:
            candidate = model._initial_stability(two_record_history)
            search_kernel.assert_called_once()
        self.assertEqual(_float_bits(candidate), _float_bits(reference))

    def test_long_history_converts_to_numpy_only_once_for_both_kernels(self) -> None:
        model = RecoveredSM19(1)
        history = _history(12, 0xC0A1E5CE)
        with patch.object(
            recovered_sm19.np,
            "fromiter",
            wraps=np.fromiter,
        ) as fromiter:
            model._initial_stability(history)
        self.assertEqual(fromiter.call_count, 2)

    def test_numpy_stability_mirror_stays_bitwise_synced_after_every_commit(
        self,
    ) -> None:
        model = RecoveredSM19(1)
        for index in range(160):
            card_id = f"card-{(index * 7) % 13}"
            elapsed = float((index * 37) % 401 + 1)
            if index % 9 == 0:
                elapsed += 0.5
            rating = (index * 11 + 3) % 4 + 1
            model.predict(card_id, elapsed)
            model.commit(card_id, elapsed, rating)

            canonical = np.asarray(
                model.learning_state.stability_increase,
                dtype=np.float64,
            )
            np.testing.assert_array_equal(
                canonical.view(np.uint64),
                model._stability_increase_np.view(np.uint64),
            )


if __name__ == "__main__":
    unittest.main()
