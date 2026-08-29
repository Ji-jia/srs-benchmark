from __future__ import annotations

import hashlib
import inspect
import math
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from models import recovered_sm19
from models.recovered_sm19 import RecoveredSM19


class RecoveredSM19CausalityTests(unittest.TestCase):
    @staticmethod
    def _run_card(ratings: list[int], elapsed_days: list[float]) -> list[float]:
        model = RecoveredSM19(1)
        predictions: list[float] = []
        for elapsed, rating in zip(elapsed_days, ratings, strict=True):
            predictions.append(model.predict("card", elapsed))
            model.commit("card", elapsed, rating)
        return predictions

    def test_predict_signature_cannot_receive_current_outcome(self) -> None:
        parameters = list(inspect.signature(RecoveredSM19.predict).parameters)
        self.assertEqual(parameters, ["self", "card_id", "elapsed_days"])
        self.assertNotIn("rating", parameters)
        self.assertNotIn("y", parameters)

    def test_first_review_probability_is_finite_and_bounded(self) -> None:
        probability = RecoveredSM19(1).predict("new-card", 17)
        self.assertTrue(math.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_predict_is_read_only_and_commit_learns_current_review_once(self) -> None:
        model = RecoveredSM19(1)
        before = model.totals()

        first_probability = model.predict("card", 17.0)
        repeated_probability = model.predict("card", 17.0)
        self.assertEqual(first_probability, repeated_probability)
        self.assertEqual(model.totals(), before)
        self.assertEqual(model.history_for("card"), ())

        model.commit("card", 17.0, 4)
        after = model.totals()
        self.assertEqual(after.recall_cases, before.recall_cases + 1)
        self.assertEqual(after.first_review_cases, before.first_review_cases + 1)
        self.assertEqual(after.history_records, before.history_records + 1)

        with self.assertRaises(RuntimeError):
            model.commit("card", 17.0, 4)

    def test_current_or_future_grade_cannot_change_current_probability(self) -> None:
        elapsed = [5.0, 11.0, 23.0, 47.0]
        failed_third = self._run_card([4, 3, 1, 3], elapsed)
        passed_third = self._run_card([4, 3, 4, 3], elapsed)

        # The two runs have identical strict pasts through the third prediction.
        # Only the grade committed after that prediction differs.
        np.testing.assert_allclose(
            failed_third[:3], passed_third[:3], rtol=0.0, atol=0.0
        )

        # Once that grade is in the strict past, it must be allowed to affect the
        # next prediction. This also guards against an accidentally stateless model.
        self.assertGreater(abs(failed_third[3] - passed_third[3]), 1e-12)

    def test_all_probabilities_are_finite_and_bounded_for_interleaved_cards(
        self,
    ) -> None:
        model = RecoveredSM19(1)
        reviews = [
            ("A", 1.0, 4),
            ("B", 35.0, 1),
            ("A", 3.0, 3),
            ("B", 2.0, 4),
            ("A", 120.0, 1),
            ("B", 8.0, 2),
            ("A", 0.5, 4),
            ("B", 3650.0, 3),
        ]

        predictions = []
        for card_id, elapsed, rating in reviews:
            probability = model.predict(card_id, elapsed)
            predictions.append(probability)
            model.commit(card_id, elapsed, rating)

        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((np.asarray(predictions) >= 0.0).all())
        self.assertTrue((np.asarray(predictions) <= 1.0).all())

    def test_again_mapping_is_explicit_and_rejects_every_other_value(self) -> None:
        for grade in (0, 1, 2):
            model = RecoveredSM19(grade)
            self.assertEqual(model.again_grade, grade)
            for rating, expected in ((1, grade), (2, 3), (3, 4), (4, 5)):
                card_id = f"card-{rating}"
                model.predict(card_id, 2.0)
                model.commit(card_id, 2.0, rating)
                self.assertEqual(model.history_for(card_id)[0].grade, expected)

        for invalid in (-1, 3, True, 1.0, "1", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RecoveredSM19(invalid)  # type: ignore[arg-type]

    def test_commit_rejects_non_integer_anki_ratings(self) -> None:
        for invalid in (True, 1.0, 2.0, "1", None):
            model = RecoveredSM19(1)
            model.predict("card", 2.0)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                model.commit("card", 2.0, invalid)

    def test_again_0_1_2_are_numerically_identical_recovered_failure_paths(
        self,
    ) -> None:
        reviews = [
            ("A", 1.0, 4),
            ("B", 2.0, 1),
            ("A", 5.0, 3),
            ("B", 7.0, 4),
            ("A", 13.0, 1),
            ("B", 19.0, 2),
            ("A", 37.0, 4),
            ("B", 61.0, 1),
        ]
        runs = []
        for again_grade in (0, 1, 2):
            model = RecoveredSM19(again_grade)
            predictions = []
            for card_id, elapsed, rating in reviews:
                predictions.append(model.predict(card_id, elapsed))
                model.commit(card_id, elapsed, rating)
            runs.append((predictions, model.learning_state, model.totals()))

        for candidate in runs[1:]:
            np.testing.assert_array_equal(candidate[0], runs[0][0])
            self.assertEqual(candidate[1], runs[0][1])
            self.assertEqual(candidate[2], runs[0][2])

    def test_lapse_counter_uses_native_uint16_domain(self) -> None:
        self.assertEqual(recovered_sm19._lapse_category(255), 20)
        self.assertEqual(recovered_sm19._lapse_category(256), 20)
        self.assertEqual(recovered_sm19._lapse_category(0xFFFF), 20)
        for invalid in (-1, 0x10000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recovered_sm19._lapse_category(invalid)

        model = RecoveredSM19(1)
        model._history_by_card["card"] = [recovered_sm19.HistoryRecord(1.0, 1, 1, 255)]
        before = model.learning_state.post_lapse_cases.copy()

        model.predict("card", 1.0)
        model.commit("card", 1.0, 1)

        self.assertEqual(model.history_for("card")[-1].lapses, 256)
        changed = [
            index
            for index, (old, new) in enumerate(
                zip(before, model.learning_state.post_lapse_cases, strict=True)
            )
            if old != new
        ]
        self.assertEqual(len(changed), 1)
        lapse_category = (
            changed[0] // recovered_sm19.NATIVE_FIRST_INTERVAL_DAY_SLOTS
        ) % recovered_sm19.NATIVE_CATEGORY_SLOTS + 1
        self.assertEqual(lapse_category, 20)

        max_model = RecoveredSM19(1)
        max_model._history_by_card["card"] = [
            recovered_sm19.HistoryRecord(1.0, 1, 1, 0xFFFE)
        ]
        max_model.predict("card", 1.0)
        max_model.commit("card", 1.0, 1)
        self.assertEqual(max_model.history_for("card")[-1].lapses, 0xFFFF)

        max_model.predict("card", 1.0)
        with self.assertRaisesRegex(ValueError, "review counters exceed"):
            max_model.commit("card", 1.0, 1)


class RecoveredSM19CacheTests(unittest.TestCase):
    def test_pre_cache_golden_stream_is_bitwise_identical(self) -> None:
        model = RecoveredSM19(1)
        predictions: list[float] = []
        for index in range(240):
            remainder = index % 5
            if remainder in (0, 1, 2):
                card_id = "heavy-A"
            elif remainder == 3:
                card_id = f"single-{index // 5}"
            else:
                card_id = "heavy-B"
            elapsed_days = float((index * 37 + len(card_id) * 11) % 173 + 1)
            if index % 7 == 0:
                elapsed_days += 0.5
            rating = (index * 13 + len(card_id)) % 4 + 1
            predictions.append(model.predict(card_id, elapsed_days))
            model.commit(card_id, elapsed_days, rating)

        digest = hashlib.sha256()
        for probability in predictions:
            digest.update(struct.pack(">d", probability))
        state = model.learning_state
        integer_arrays = (
            state.recall_cases,
            state.recall_success,
            state.stability_increase_cases,
        )
        for values in integer_arrays:
            for value in values:
                digest.update(struct.pack(">q", value))
        for value in state.stability_increase:
            digest.update(struct.pack(">d", value))
        integer_arrays = (
            state.post_lapse_cases,
            state.post_lapse_success,
            state.first_review_cases,
            state.first_review_success,
        )
        for values in integer_arrays:
            for value in values:
                digest.update(struct.pack(">q", value))

        self.assertEqual(
            digest.hexdigest(),
            "764d43a75423039e12ae5312bcde965825a32d7cd6cda3e947f61e99828ef2b4",
        )

    def test_single_record_initial_stability_cache_uses_elapsed_and_success(
        self,
    ) -> None:
        model = RecoveredSM19(1)
        success = [recovered_sm19.HistoryRecord(11.5, 3, 2, 0)]
        same_key = [recovered_sm19.HistoryRecord(11.5, 5, 99, 7)]
        failure = [recovered_sm19.HistoryRecord(11.5, 1, 1, 1)]

        with patch.object(
            model,
            "_search_initial_stability",
            wraps=model._search_initial_stability,
        ) as search:
            expected = model._initial_stability(success)
            cached = model._initial_stability(same_key)
            model._initial_stability(failure)

        self.assertEqual(struct.pack(">d", expected), struct.pack(">d", cached))
        self.assertEqual(search.call_count, 2)
        self.assertEqual(
            list(model._single_initial_stability_cache),
            [(11.5, True), (11.5, False)],
        )

        model._single_initial_stability_cache.clear()
        with patch.object(
            recovered_sm19,
            "SINGLE_INITIAL_STABILITY_CACHE_LIMIT",
            2,
        ):
            for elapsed_days in (1.0, 2.0, 3.0):
                model._initial_stability(
                    [recovered_sm19.HistoryRecord(elapsed_days, 1, 1, 1)]
                )
        self.assertEqual(len(model._single_initial_stability_cache), 2)
        self.assertNotIn((1.0, False), model._single_initial_stability_cache)

    def test_derived_fits_invalidate_only_their_relevant_rows(self) -> None:
        model = RecoveredSM19(1)
        relevant_calibration = model._calibration_fit(0.0, 2.0)
        unrelated_calibration = model._calibration_fit(1.0 / 19.0, 3.0)
        self.assertIs(model._calibration_fit(0.0, 2.0), relevant_calibration)

        model._post_lapse_stability(3, 2)
        model._post_lapse_stability(4, 3)
        first_interval_fit = model._first_interval_fit()
        self.assertIs(model._first_interval_fit(), first_interval_fit)

        previous_state = recovered_sm19.ReplayState(
            difficulty=0.0,
            stability=2.0,
            previous_observed_stability=-1.0,
            previous_pre_review_stability_category=0,
            repetitions=1,
            lapses=0,
        )
        record = recovered_sm19.HistoryRecord(
            elapsed_days=5.0,
            grade=4,
            repetitions=2,
            lapses=2,
        )
        transition = recovered_sm19._Transition(
            state=previous_state,
            raw_retrievability=0.5,
            calibrated_retrievability=0.5,
            observed_stability=2.0,
            observed_ratio=-1.0,
            difficulty_category_before=1,
            stability_category_before=1,
            retrievability_category_before=3,
            difficulty_category_after=1,
        )
        model._learn_current(previous_state, record, transition)

        self.assertNotIn((1, 1), model._calibration_fit_cache)
        self.assertIs(model._calibration_fit_cache[(2, 2)], unrelated_calibration)
        self.assertNotIn((3, 2), model._post_lapse_stability_cache)
        self.assertIn((4, 3), model._post_lapse_stability_cache)
        self.assertIsNone(model._first_interval_fit_cache)


class RecoveredSM19ProcessorTests(unittest.TestCase):
    def test_only_explicit_again_variants_are_registered(self) -> None:
        from typing import get_args

        from config import ModelName, load_config
        from features.factory import FEATURE_ENGINEER_REGISTRY
        from model_processors import RECOVERED_SM19_AGAIN_GRADES

        expected = {
            "Recovered-SM19-Again0": 0,
            "Recovered-SM19-Again1": 1,
            "Recovered-SM19-Again2": 2,
        }
        self.assertEqual(RECOVERED_SM19_AGAIN_GRADES, expected)
        self.assertNotIn("Recovered-SM19", get_args(ModelName))
        self.assertNotIn("Recovered-SM19", FEATURE_ENGINEER_REGISTRY)
        with self.assertRaisesRegex(ValueError, "Model name 'Recovered-SM19'"):
            load_config(["--algo", "Recovered-SM19"])
        for model_name in expected:
            self.assertIn(model_name, get_args(ModelName))
            self.assertIn(model_name, FEATURE_ENGINEER_REGISTRY)

    def test_processor_constructs_each_explicit_again_grade(self) -> None:
        import pandas as pd

        from model_processors import process_recovered_sm19

        dataset = pd.DataFrame(
            {
                "review_th": range(1, 7),
                "card_id": ["A"] * 6,
                "delta_t": [1.0] * 6,
                "rating": [1, 2, 3, 4, 1, 4],
                "sm19_score": [True] * 6,
                "sm19_score_y": [0, 1, 1, 1, 0, 1],
                "sm19_score_i": range(2, 8),
                "sm19_score_rmse_bins_lapse": [0, 1, 1, 1, 2, 2],
            }
        )
        constructed: list[int] = []

        class RecordingModel:
            def __init__(self, again_grade: int) -> None:
                constructed.append(again_grade)

            def predict(self, card_id: object, elapsed_days: float) -> float:
                return 0.5

            def commit(
                self, card_id: object, elapsed_days: float, anki_rating: int
            ) -> None:
                pass

        with (
            patch("model_processors.RecoveredSM19", RecordingModel),
            patch("model_processors.save_evaluation_file"),
            patch("model_processors.evaluate", return_value=({}, None)),
        ):
            for grade in (0, 1, 2):
                config = SimpleNamespace(
                    n_splits=2,
                    model_name=f"Recovered-SM19-Again{grade}",
                    get_evaluation_file_name=lambda grade=grade: (
                        f"Recovered-SM19-Again{grade}"
                    ),
                )
                process_recovered_sm19(1, dataset, config)

        self.assertEqual(constructed, [0, 1, 2])

    def test_time_series_score_mask_is_union_of_all_test_folds(self) -> None:
        from sklearn.model_selection import TimeSeriesSplit

        from model_processors import time_series_test_mask

        score_count = 11
        n_splits = 3
        expected = np.zeros(score_count, dtype=bool)
        for _, test_index in TimeSeriesSplit(n_splits=n_splits).split(
            np.arange(score_count)
        ):
            expected[test_index] = True

        actual = time_series_test_mask(score_count, n_splits)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(bool))

    def test_processor_refuses_to_reorder_causal_events(self) -> None:
        import pandas as pd

        from model_processors import process_recovered_sm19

        dataset = pd.DataFrame(
            {
                "review_th": [1, 3, 2, 4, 5, 6],
                "card_id": ["A"] * 6,
                "delta_t": [1.0] * 6,
                "rating": [4] * 6,
                "sm19_score": [True] * 6,
                "sm19_score_y": [1] * 6,
                "sm19_score_i": range(2, 8),
                "sm19_score_rmse_bins_lapse": [0] * 6,
            }
        )
        config = SimpleNamespace(n_splits=2)

        with self.assertRaisesRegex(ValueError, "will not reorder"):
            process_recovered_sm19(1, dataset, config)

    def test_two_cards_replay_once_in_global_order_without_fold_reset(self) -> None:
        import pandas as pd

        from model_processors import process_recovered_sm19

        # Rows 3 and 6 are replay-only. They are deliberately retained to verify
        # that fold selection controls scoring, not the chronological state stream.
        dataset = pd.DataFrame(
            {
                "review_th": range(1, 9),
                "card_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "delta_t": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0],
                "rating": [4, 1, 3, 4, 1, 2, 4, 3],
                "sm19_score": [True, True, False, True, True, False, True, True],
                "sm19_score_y": [1, 0, np.nan, 1, 0, np.nan, 1, 1],
                "sm19_score_i": [2, 2, np.nan, 3, 3, np.nan, 4, 4],
                "sm19_score_rmse_bins_lapse": [
                    0,
                    0,
                    np.nan,
                    1,
                    0,
                    np.nan,
                    1,
                    1,
                ],
            }
        )
        events: list[tuple[object, ...]] = []
        instances: list[object] = []

        class RecordingModel:
            def __init__(self, again_grade: int) -> None:
                events.append(("again_grade", again_grade))
                instances.append(self)

            # Intentionally no rating/y argument: passing the current outcome to
            # prediction would make this regression fail with TypeError.
            def predict(self, card_id: object, elapsed_days: float) -> float:
                events.append(("predict", card_id, elapsed_days))
                return 0.2 + 0.01 * len(events)

            def commit(
                self, card_id: object, elapsed_days: float, anki_rating: int
            ) -> None:
                events.append(("commit", card_id, elapsed_days, anki_rating))

        config = SimpleNamespace(
            n_splits=2,
            model_name="Recovered-SM19-Again2",
            get_evaluation_file_name=lambda: "Recovered-SM19-Again2",
        )

        with (
            patch("model_processors.RecoveredSM19", RecordingModel),
            patch("model_processors.save_evaluation_file") as save_mock,
            patch(
                "model_processors.evaluate", return_value=({"ok": True}, None)
            ) as evaluate_mock,
        ):
            stats, raw = process_recovered_sm19(7, dataset, config)

        self.assertEqual(stats, {"ok": True})
        self.assertIsNone(raw)
        self.assertEqual(
            len(instances), 1, "the model must not reset at fold boundaries"
        )

        expected_events: list[tuple[object, ...]] = [("again_grade", 2)]
        for row in dataset.itertuples(index=False):
            expected_events.append(("predict", row.card_id, row.delta_t))
            expected_events.append(("commit", row.card_id, row.delta_t, row.rating))
        self.assertEqual(events, expected_events)

        # Six score-eligible rows with two folds yield the final four eligible
        # rows as the union: global reviews 4, 5, 7, and 8.
        self.assertEqual(save_mock.call_count, 1)
        saved = save_mock.call_args.args[1]
        self.assertEqual(saved["review_th"].tolist(), [4, 5, 7, 8])
        self.assertTrue(np.isfinite(saved["p"]).all())
        self.assertTrue(((saved["p"] >= 0.0) & (saved["p"] <= 1.0)).all())

        self.assertEqual(evaluate_mock.call_count, 1)
        evaluated_y, evaluated_p, evaluated_df = evaluate_mock.call_args.args[:3]
        self.assertEqual(list(evaluated_y), [1, 0, 1, 1])
        self.assertEqual(list(evaluated_p), saved["p"].tolist())
        self.assertEqual(evaluated_df["review_th"].tolist(), [4, 5, 7, 8])


class RecoveredSM19FeatureStreamTests(unittest.TestCase):
    @staticmethod
    def _standard_protocol_config() -> SimpleNamespace:
        return SimpleNamespace(
            include_short_term=False,
            use_secs_intervals=False,
            equalize_test_with_non_secs=False,
            train_equals_test=False,
            two_buttons=False,
            max_seq_len=64,
        )

    @staticmethod
    def _scoreable_raw():
        import pandas as pd

        cards = [f"card-{index}" for index in range(24)]
        all_cards = [*cards, "outlier-card"]
        rows: list[dict[str, object]] = []

        def append_phase(
            elapsed_days: float,
            ratings: list[int],
            *,
            outlier_elapsed_days: float | None = None,
        ) -> None:
            for index, (card_id, rating) in enumerate(
                zip(all_cards, ratings, strict=True)
            ):
                interval = (
                    outlier_elapsed_days
                    if index == len(cards) and outlier_elapsed_days is not None
                    else elapsed_days
                )
                rows.append(
                    {
                        "card_id": card_id,
                        "day_offset": len(rows),
                        "rating": rating,
                        "elapsed_days": interval,
                        "elapsed_seconds": interval * 86400,
                    }
                )

        append_phase(-1.0, [4] * len(all_cards))
        append_phase(
            1.0,
            [*[1 if index % 2 == 0 else 4 for index in range(24)], 1],
            outlier_elapsed_days=999.0,
        )
        append_phase(
            3.0,
            [*[2 if index % 3 == 0 else 3 for index in range(24)], 4],
        )
        append_phase(
            9.0,
            [*[1 if index % 5 == 0 else 4 for index in range(24)], 1],
        )
        for _ in range(126):
            rows.append(
                {
                    "card_id": cards[0],
                    "day_offset": len(rows),
                    "rating": 4,
                    "elapsed_days": 1.0,
                    "elapsed_seconds": 86400.0,
                }
            )
        rows.extend(
            (
                {
                    "card_id": cards[1],
                    "day_offset": len(rows),
                    "rating": 1,
                    "elapsed_days": 0.0,
                    "elapsed_seconds": 60.0,
                },
                {
                    "card_id": cards[2],
                    "day_offset": len(rows) + 1,
                    "rating": 5,
                    "elapsed_days": 5.0,
                    "elapsed_seconds": 5.0 * 86400,
                },
            )
        )
        frame = pd.DataFrame(rows)
        frame["state"] = 2
        frame["duration"] = 3000
        return frame

    def test_score_fast_path_is_exact_and_skips_unused_histories(self) -> None:
        import pandas as pd

        from features.base import BaseFeatureEngineer
        from features.recovered_sm19 import RecoveredSM19FeatureEngineer

        raw = self._scoreable_raw()
        config = self._standard_protocol_config()
        reference_engineer = RecoveredSM19FeatureEngineer(config)
        reference = BaseFeatureEngineer.create_features(
            reference_engineer,
            raw.copy(),
        )

        normal_card_count = 24
        heavy_card_score_count = 124
        expected = pd.DataFrame(
            {
                "review_th": [
                    *range(26, 50),
                    *range(51, 75),
                    *range(76, 100),
                    *range(101, 225),
                ],
                "y": [
                    *[0 if index % 2 == 0 else 1 for index in range(24)],
                    *[1] * normal_card_count,
                    *[0 if index % 5 == 0 else 1 for index in range(24)],
                    *[1] * heavy_card_score_count,
                ],
                "i": [
                    *[2] * normal_card_count,
                    *[3] * normal_card_count,
                    *[4] * normal_card_count,
                    *range(5, 129),
                ],
                "rmse_bins_lapse": [
                    *[0] * normal_card_count,
                    *[1 if index % 2 == 0 else 0 for index in range(24)],
                    *[1 if index % 2 == 0 else 0 for index in range(24)],
                    *[2] * heavy_card_score_count,
                ],
            }
        ).astype(float)
        score_columns = ["review_th", "y", "i", "rmse_bins_lapse"]
        pd.testing.assert_frame_equal(
            reference.loc[:, score_columns].reset_index(drop=True),
            expected,
            check_exact=True,
        )

        fast_engineer = RecoveredSM19FeatureEngineer(config)
        with (
            patch.object(
                BaseFeatureEngineer,
                "create_features",
                side_effect=AssertionError("must not call the full base pipeline"),
            ),
            patch.object(
                fast_engineer,
                "_compute_histories",
                side_effect=AssertionError("must not build unused histories"),
            ),
        ):
            replay = fast_engineer.create_features(raw)

        self.assertEqual(replay["review_th"].tolist(), list(range(26, 227)))
        self.assertEqual(
            replay.columns.tolist(),
            [
                "card_id",
                "review_th",
                "rating",
                "elapsed_days",
                "delta_t",
                "sm19_score_y",
                "sm19_score_i",
                "sm19_score_rmse_bins_lapse",
                "sm19_score",
            ],
        )
        self.assertEqual(replay["sm19_score"].dtype, np.dtype(bool))
        self.assertEqual(
            replay.loc[~replay["sm19_score"], "review_th"].tolist(),
            [50, 75, 100, 225, 226],
        )
        expected_markers = pd.DataFrame({"review_th": range(26, 227)})
        for source, target in (
            ("sm19_score_y", "y"),
            ("sm19_score_i", "i"),
            ("sm19_score_rmse_bins_lapse", "rmse_bins_lapse"),
        ):
            expected_markers[source] = expected_markers["review_th"].map(
                dict(
                    zip(
                        expected["review_th"].astype(int),
                        expected[target],
                        strict=True,
                    )
                )
            )
        expected_markers.insert(
            1,
            "sm19_score",
            expected_markers["sm19_score_y"].notna(),
        )
        pd.testing.assert_frame_equal(
            replay.loc[
                :,
                [
                    "review_th",
                    "sm19_score",
                    "sm19_score_y",
                    "sm19_score_i",
                    "sm19_score_rmse_bins_lapse",
                ],
            ],
            expected_markers,
            check_exact=True,
        )
        self.assertTrue(
            {"last_rating", "r_history", "t_history"}.isdisjoint(replay.columns)
        )

    def test_user_4371_like_empty_score_path_keeps_common_error(self) -> None:
        import pandas as pd

        from features.base import BaseFeatureEngineer
        from features.recovered_sm19 import RecoveredSM19FeatureEngineer

        cards = [f"card-{index}" for index in range(6)]
        rows = [
            {
                "card_id": card_id,
                "day_offset": index,
                "rating": 4,
                "elapsed_days": -1.0,
                "elapsed_seconds": -1.0,
            }
            for index, card_id in enumerate(cards)
        ]
        rows.extend(
            {
                "card_id": card_id,
                "day_offset": len(cards) + index,
                "rating": 4,
                "elapsed_days": float((1, 2, 7)[index % 3]),
                "elapsed_seconds": float((1, 2, 7)[index % 3]) * 86400,
            }
            for index, card_id in enumerate(cards)
        )
        raw = pd.DataFrame(rows)
        config = self._standard_protocol_config()
        error = "^No data after handling outliers and non-continuous rows$"

        with self.assertRaisesRegex(ValueError, error):
            BaseFeatureEngineer.create_features(
                RecoveredSM19FeatureEngineer(config),
                raw.copy(),
            )

        fast_engineer = RecoveredSM19FeatureEngineer(config)
        with (
            patch.object(
                fast_engineer,
                "_compute_histories",
                side_effect=AssertionError("must not build unused histories"),
            ),
            self.assertRaisesRegex(ValueError, error),
        ):
            fast_engineer.create_features(raw)

    def test_source_review_order_must_be_unique_and_chronological(self) -> None:
        import pandas as pd

        from features.recovered_sm19 import RecoveredSM19FeatureEngineer

        raw = pd.DataFrame(
            {
                "card_id": ["A", "B", "A"],
                "review_th": [1, 3, 2],
                "delta_t": [1.0, 2.0, 3.0],
                "elapsed_days": [1.0, 2.0, 3.0],
                "rating": [4, 3, 1],
                "day_offset": [0, 1, 2],
            }
        )
        score_rows = pd.DataFrame(
            {
                "review_th": [1],
                "y": [1],
                "i": [2],
                "rmse_bins_lapse": [0],
            }
        )
        preprocessed = raw.copy()
        preprocessed["i"] = [1, 1, 2]

        engineer = RecoveredSM19FeatureEngineer(self._standard_protocol_config())
        with (
            patch.object(
                engineer,
                "_common_preprocessing",
                return_value=preprocessed,
            ),
            patch.object(
                engineer,
                "_common_postprocessing",
                return_value=score_rows,
            ),
            self.assertRaisesRegex(ValueError, "chronological"),
        ):
            engineer.create_features(raw)

    def test_replay_stream_is_not_truncated_by_score_sequence_cap(self) -> None:
        import pandas as pd

        from features.recovered_sm19 import RecoveredSM19FeatureEngineer

        raw = pd.DataFrame(
            {
                "card_id": ["card"] * 130,
                "delta_t": [1.0] * 130,
                "elapsed_days": [1.0] * 130,
                "rating": [4] * 130,
                "day_offset": list(range(130)),
            }
        )
        score_rows = pd.DataFrame(
            {
                "review_th": [1],
                "y": [1],
                "i": [2],
                "rmse_bins_lapse": [0],
            }
        )
        preprocessed = raw.copy()
        preprocessed["review_th"] = range(1, len(preprocessed) + 1)
        preprocessed["i"] = range(1, len(preprocessed) + 1)
        engineer = RecoveredSM19FeatureEngineer(self._standard_protocol_config())
        with (
            patch.object(
                engineer,
                "_common_preprocessing",
                return_value=preprocessed,
            ),
            patch.object(
                engineer,
                "_common_postprocessing",
                return_value=score_rows,
            ),
        ):
            replay = engineer.create_features(raw)

        self.assertEqual(len(replay), 130)
        self.assertEqual(replay["review_th"].tolist(), list(range(1, 131)))
        self.assertEqual(int(replay["sm19_score"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
