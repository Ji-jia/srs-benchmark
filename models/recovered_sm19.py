"""Causal online adapter for the recovered SuperMemo 19 DSR algorithm.

The benchmark owns one instance per user.  Item D/S values are deliberately
not cached: every prediction replays the card's strict-past History against the
current, read-only shared Algorithm 17 model.  ``commit`` is the only method
that accepts the current rating; it trains the shared model exactly once and
only then appends the current History record.

This module contains the recovered numerical leaves needed by that contract.
It does not import the older standalone recovery bundle because that bundle's
v14 convenience scheduler predates the recovered History reducer, the 1.4
FirstInterval day axis, and failure-side observed-stability roll-forward.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit

FORGETTING_CONSTANT = -math.log(0.9)
STABILITY_BUCKET_POWER = math.log(6000.0) / math.log(20.0)

ACTIVE_CATEGORY_COUNT = 20
NATIVE_CATEGORY_SLOTS = 21
NATIVE_CUBE_LENGTH = NATIVE_CATEGORY_SLOTS**3
FIRST_INTERVAL_DAY_COUNT = 35
NATIVE_FIRST_INTERVAL_DAY_SLOTS = 36
NATIVE_POST_LAPSE_LENGTH = (
    NATIVE_CATEGORY_SLOTS * NATIVE_CATEGORY_SLOTS * NATIVE_FIRST_INTERVAL_DAY_SLOTS
)

INITIAL_DIFFICULTY = 0.5
INITIAL_STABILITY_SEARCH_SEED = 4.0
FIRST_INTERVAL_EMPTY_STABILITY = 3.0

SINGLE_INITIAL_STABILITY_CACHE_LIMIT = 4096
NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY = 12
NUMBA_INITIAL_STABILITY_SEARCH_MIN_HISTORY = 2


@njit(cache=False, fastmath=False)
def _initial_difficulty_for_stability_kernel(
    elapsed: np.ndarray,
    grade: np.ndarray,
    stability_increase: np.ndarray,
) -> float:
    """Evaluate the 20 native difficulty candidates in strict binary64 order."""

    scores = np.empty(ACTIVE_CATEGORY_COUNT, dtype=np.float64)
    for d_offset in range(ACTIVE_CATEGORY_COUNT):
        stability = INITIAL_STABILITY_SEARCH_SEED
        weighted_error = 0.0
        total_weight = 0.0
        weight = 1.0
        for record_index in range(elapsed.shape[0]):
            if stability > -1.0:
                stability = min(max(stability, 0.7), 44530.0)
            exponent = -FORGETTING_CONSTANT * elapsed[record_index] / stability
            raw = math.exp(min(max(exponent, -38.0), 38.0))
            error = raw - (1.0 if grade[record_index] >= 3 else 0.0)
            weight *= 2.0
            weighted_error += error * weight
            total_weight += weight
            if grade[record_index] >= 3:
                normalized = stability
                if stability > -1.0:
                    normalized = min(max(stability, 0.7), 44530.0)
                shifted = max(normalized - 2.0, 0.0)
                s_category = round(shifted ** (1.0 / STABILITY_BUCKET_POWER)) + 1
                s_category = min(max(s_category, 1), ACTIVE_CATEGORY_COUNT)
                r_category = round(20.0**raw)
                r_category = min(max(r_category, 1), ACTIVE_CATEGORY_COUNT)
                cube_index = (
                    d_offset * NATIVE_CATEGORY_SLOTS**2
                    + (s_category - 1) * NATIVE_CATEGORY_SLOTS
                    + r_category
                    - 1
                )
                stability *= stability_increase[cube_index]
            else:
                stability = INITIAL_STABILITY_SEARCH_SEED
        scores[d_offset] = weighted_error / total_weight

    best_low = 0
    best_score = math.inf
    for index in range(ACTIVE_CATEGORY_COUNT):
        score = scores[index]
        if score < best_score:
            best_score = score
            best_low = index
    best_high = ACTIVE_CATEGORY_COUNT - 1
    best_score = math.inf
    for index in range(ACTIVE_CATEGORY_COUNT - 1, -1, -1):
        score = scores[index]
        if score < best_score:
            best_score = score
            best_high = index
    return ((best_low / 19.0) + (best_high / 19.0)) / 2.0


@njit(cache=False, fastmath=False)
def _initial_stability_search_kernel(
    elapsed: np.ndarray,
    grade: np.ndarray,
    difficulty: float,
    allow_up: bool,
) -> float:
    """Run the complete native candidate search in strict binary64 order."""

    stability = INITIAL_STABILITY_SEARCH_SEED
    while True:
        center = 0.0
        lower = 0.0
        upper = 0.0
        for objective_index in range(3):
            if objective_index == 0:
                candidate = stability
            elif objective_index == 1:
                candidate = 0.99 * stability
            else:
                candidate = 1.01 * stability

            replay_stability = candidate
            squared_error = 0.0
            for record_index in range(elapsed.shape[0]):
                normalized = replay_stability
                if normalized > -1.0:
                    normalized = min(max(normalized, 0.7), 44530.0)
                exponent = -FORGETTING_CONSTANT * elapsed[record_index] / normalized
                raw = math.exp(min(max(exponent, -38.0), 38.0))
                error = raw - (1.0 if grade[record_index] >= 3 else 0.0)
                squared_error += error * error

                if grade[record_index] >= 3:
                    maximum_growth = 15.0 - 12.0 * difficulty
                    stability_exponent = -0.08 - 0.27 * difficulty
                    stability_multiplier = (
                        normalized**stability_exponent * (maximum_growth - 1.0) + 1.0
                    )
                    retrievability_exponent = min(
                        2.25 - 2.0 * difficulty,
                        600.0,
                    )
                    ratio_exponent = -retrievability_exponent * raw
                    ratio = (
                        math.exp(min(max(ratio_exponent, -38.0), 38.0))
                        * stability_multiplier
                    )
                    ratio = min(max(ratio, 0.8), 20.0)
                    replay_stability = normalized * ratio
                else:
                    replay_stability = candidate

            objective = (
                math.sqrt(squared_error / elapsed.shape[0]) if elapsed.shape[0] else 0.0
            )
            if objective_index == 0:
                center = objective
            elif objective_index == 1:
                lower = objective
            else:
                upper = objective

        if lower < center:
            stability *= 0.99
        elif upper < center:
            stability *= 1.01
            if not allow_up:
                break
        else:
            break
        if stability < 0.7 or stability > 100.0:
            break
    return stability


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _bounded_exp(exponent: float) -> float:
    return math.exp(_clamp(exponent, -38.0, 38.0))


def _delphi_round(value: float) -> int:
    """Delphi's default round-to-nearest-even System.Round."""

    return round(value)


def _saturating_weight(cases: float, prior: float) -> float:
    return cases / (cases + prior)


def _normalize_stability(stability: float) -> float:
    if stability > -1.0:
        return _clamp(stability, 0.7, 44530.0)
    return stability


def _difficulty_category(difficulty: float) -> int:
    if not 0.0 <= difficulty <= 1.0:
        return 10
    return _delphi_round(19.0 * difficulty) + 1


def _difficulty_category_center(category: int) -> float:
    if not 1 <= category <= ACTIVE_CATEGORY_COUNT:
        raise ValueError("difficulty category must be in 1..20")
    return (category - 1) / 19.0


def _stability_category(stability: float) -> int:
    normalized = _normalize_stability(stability)
    shifted = max(normalized - 2.0, 0.0)
    category = _delphi_round(shifted ** (1.0 / STABILITY_BUCKET_POWER)) + 1
    return min(max(category, 1), ACTIVE_CATEGORY_COUNT)


def _stability_category_center(category: int) -> float:
    if not 1 <= category <= ACTIVE_CATEGORY_COUNT:
        raise ValueError("stability category must be in 1..20")
    return (category - 1) ** STABILITY_BUCKET_POWER + 2.0


def _retrievability_category(retrievability: float) -> int:
    category = _delphi_round(20.0**retrievability)
    return min(max(category, 1), ACTIVE_CATEGORY_COUNT)


def _retrievability_category_center(category: int) -> float:
    if not 1 <= category <= ACTIVE_CATEGORY_COUNT:
        raise ValueError("retrievability category must be in 1..20")
    return math.log(float(category)) / math.log(20.0)


def _lapse_category(lapses: int) -> int:
    if not 0 <= lapses <= 0xFFFF:
        raise ValueError("lapses must fit the native UInt16")
    return min(max(lapses, 1), ACTIVE_CATEGORY_COUNT)


def _cube_index(
    difficulty_category: int,
    stability_category: int,
    retrievability_category: int,
) -> int:
    for name, category in (
        ("difficulty", difficulty_category),
        ("stability", stability_category),
        ("retrievability", retrievability_category),
    ):
        if not 1 <= category <= ACTIVE_CATEGORY_COUNT:
            raise ValueError(f"{name} category must be in 1..20")
    return (
        (difficulty_category - 1) * NATIVE_CATEGORY_SLOTS**2
        + (stability_category - 1) * NATIVE_CATEGORY_SLOTS
        + retrievability_category
        - 1
    )


def _post_lapse_index(
    retrievability_category: int,
    lapse_category: int,
    day_category: int,
) -> int:
    if not 1 <= retrievability_category <= ACTIVE_CATEGORY_COUNT:
        raise ValueError("retrievability category must be in 1..20")
    if not 1 <= lapse_category <= ACTIVE_CATEGORY_COUNT:
        raise ValueError("lapse category must be in 1..20")
    if not 1 <= day_category <= FIRST_INTERVAL_DAY_COUNT:
        raise ValueError("day category must be in 1..35")
    return (
        (retrievability_category - 1)
        * NATIVE_CATEGORY_SLOTS
        * NATIVE_FIRST_INTERVAL_DAY_SLOTS
        + (lapse_category - 1) * NATIVE_FIRST_INTERVAL_DAY_SLOTS
        + day_category
        - 1
    )


def _first_interval_day_category(elapsed_days: float) -> int:
    day = _delphi_round(elapsed_days)
    if not 0 <= day <= 0xFFFF:
        raise ValueError("rounded elapsed_days must fit the native UInt16")
    day = max(day, 1)
    logarithmic = _delphi_round(math.log(day) / math.log(1.4)) + 12
    return min(logarithmic, day, FIRST_INTERVAL_DAY_COUNT)


def _first_interval_day_axis(category: int) -> int:
    if not 1 <= category <= FIRST_INTERVAL_DAY_COUNT:
        raise ValueError("day category must be in 1..35")
    exponential_day = math.trunc(1.4 ** (category - 12)) + 1
    return max(category, exponential_day)


def _weighted_linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
    case_counts: Sequence[float],
) -> tuple[float, float]:
    if not (len(x_values) == len(y_values) == len(case_counts)):
        raise ValueError("regression arrays must have equal lengths")
    if not x_values:
        raise ValueError("regression requires at least one point")

    weights = [_delphi_round(value) + 1.0e-5 for value in case_counts]
    total_weight = sum(weights)
    mean_x = sum(x * weight for x, weight in zip(x_values, weights)) / total_weight
    mean_y = sum(y * weight for y, weight in zip(y_values, weights)) / total_weight
    mean_x2 = sum(x * x * weight for x, weight in zip(x_values, weights)) / total_weight
    mean_xy = (
        sum(x * y * weight for x, y, weight in zip(x_values, y_values, weights))
        / total_weight
    )
    variance = mean_x2 - mean_x * mean_x
    slope = (mean_xy - mean_x * mean_y) / variance
    return slope, mean_y - mean_x * slope


@dataclass(frozen=True)
class _FirstIntervalCurve:
    stability: float
    slope: float
    intercept: float
    cases: int


def _fit_first_interval_curve(
    remembered_counts: Sequence[int],
    case_counts: Sequence[int],
    *,
    default_stability: float,
) -> _FirstIntervalCurve:
    if not (
        len(remembered_counts) == len(case_counts) == NATIVE_FIRST_INTERVAL_DAY_SLOTS
    ):
        raise ValueError("FirstInterval curves require native 36-slot arrays")

    total_remembered = 0
    total_cases = 0
    ratios: list[float] = []
    cases: list[float] = []
    for category in range(1, FIRST_INTERVAL_DAY_COUNT + 1):
        remembered = remembered_counts[category]
        count = case_counts[category]
        if count <= 0:
            ratios.append(0.1)
        else:
            ratios.append(max(remembered / count, 0.1))
            total_remembered += _delphi_round(float(remembered))
            total_cases += _delphi_round(float(count))
        cases.append(float(count))

    if total_cases == 0:
        return _FirstIntervalCurve(
            stability=default_stability,
            slope=-0.07,
            intercept=-0.01,
            cases=0,
        )

    average_recall = _clamp(total_remembered / total_cases, 0.1, 0.9999)
    x_values = [
        math.log(float(_first_interval_day_axis(category)))
        for category in range(1, FIRST_INTERVAL_DAY_COUNT + 1)
    ]
    y_values = [math.log(ratio) for ratio in ratios]
    slope, intercept = _weighted_linear_regression(x_values, y_values, cases)

    upper_intercept = math.log(0.9999)
    lower_intercept = math.log(0.1)
    target_log_recall = math.log(0.9)
    if intercept > upper_intercept:
        slope *= (target_log_recall - upper_intercept) / (target_log_recall - intercept)
        intercept = upper_intercept
    intercept = max(intercept, lower_intercept)
    if slope > -0.001:
        slope = -0.001
        intercept = math.log(average_recall)
    slope = max(slope, -3.0)

    curve_stability = _bounded_exp((target_log_recall - intercept) / slope)
    blend = _saturating_weight(float(total_cases), 50.0)
    stability = curve_stability * blend + default_stability * (1.0 - blend)
    return _FirstIntervalCurve(stability, slope, intercept, total_cases)


def _theoretical_success_ratio(
    retrievability: float,
    old_stability: float,
    new_difficulty: float,
) -> float:
    maximum_growth = 15.0 - 12.0 * new_difficulty
    stability_exponent = -0.08 - 0.27 * new_difficulty
    stability_multiplier = (
        old_stability**stability_exponent * (maximum_growth - 1.0) + 1.0
    )
    retrievability_exponent = min(2.25 - 2.0 * new_difficulty, 600.0)
    ratio = (
        _bounded_exp(-retrievability_exponent * retrievability) * stability_multiplier
    )
    return _clamp(ratio, 0.8, 20.0)


def _observed_stability_update(
    grade: int,
    elapsed_days: float,
    old_stability: float,
    inferred_stability: float,
    calibration_cases: float,
) -> float:
    observed_weight = 0.01
    if grade >= 3:
        if elapsed_days >= old_stability:
            ratio = elapsed_days / old_stability
            observed_weight = 100.0 * _saturating_weight(ratio, 3.0)
        current_weight = 1.25 * _saturating_weight(calibration_cases, 333.0)
    else:
        if elapsed_days <= old_stability:
            ratio = 1.0 - elapsed_days / old_stability
            observed_weight = 200.0 * _saturating_weight(ratio, 1.0)
        current_weight = 1.0

    inferred_weight = 100.0 + current_weight
    old_weight = 175.0
    return (
        inferred_stability * inferred_weight
        + old_stability * old_weight
        + elapsed_days * observed_weight
    ) / (inferred_weight + old_weight + observed_weight)


def _success_stability(
    matrix_cases: int,
    theoretical_ratio: float,
    matrix_ratio: float,
    elapsed_days: float,
    observed_stability: float,
) -> float:
    upper_bound = max(
        elapsed_days * matrix_ratio,
        elapsed_days * theoretical_ratio,
    )
    if upper_bound < observed_stability:
        upper_bound = observed_stability + min(1.0, elapsed_days)
    matrix_weight = 10.0 * _saturating_weight(float(matrix_cases), 300.0)
    blended_ratio = (matrix_weight * matrix_ratio + theoretical_ratio) / (
        matrix_weight + 1.0
    )
    return _clamp(observed_stability * blended_ratio, 0.1, upper_bound)


def _update_difficulty(
    grade: int,
    repetitions_after_review: int,
    lapses_after_review: int,
    calibrated_retrievability: float,
    old_difficulty: float,
) -> float:
    target = 0.0 if grade <= 2 else 1.0
    prediction_error = target - calibrated_retrievability
    desired_difficulty = (0.1 - prediction_error) * 9.0
    rate = 0.2
    if repetitions_after_review == 2 and lapses_after_review == 0:
        rate = 0.7
    if target == 1.0 and desired_difficulty > old_difficulty:
        rate /= 2.0
    if target == 0.0 and desired_difficulty < old_difficulty:
        rate /= 2.0
    return _clamp(
        rate * desired_difficulty + (1.0 - rate) * old_difficulty,
        0.0,
        1.0,
    )


@dataclass
class Algorithm17State:
    """Shared per-user SM19 learning arrays in their native padded shapes."""

    recall_cases: list[int]
    recall_success: list[int]
    stability_increase_cases: list[int]
    stability_increase: list[float]
    post_lapse_cases: list[int]
    post_lapse_success: list[int]
    first_review_cases: list[int]
    first_review_success: list[int]

    @classmethod
    def defaults(cls) -> Algorithm17State:
        stability_increase = [0.0] * NATIVE_CUBE_LENGTH
        for d_category in range(1, ACTIVE_CATEGORY_COUNT + 1):
            difficulty = _difficulty_category_center(d_category)
            for s_category in range(1, ACTIVE_CATEGORY_COUNT + 1):
                stability = _stability_category_center(s_category)
                for r_category in range(1, ACTIVE_CATEGORY_COUNT + 1):
                    retrievability = _retrievability_category_center(r_category)
                    index = _cube_index(d_category, s_category, r_category)
                    stability_increase[index] = _theoretical_success_ratio(
                        retrievability,
                        stability,
                        difficulty,
                    )
        return cls(
            recall_cases=[0] * NATIVE_CUBE_LENGTH,
            recall_success=[0] * NATIVE_CUBE_LENGTH,
            stability_increase_cases=[0] * NATIVE_CUBE_LENGTH,
            stability_increase=stability_increase,
            post_lapse_cases=[0] * NATIVE_POST_LAPSE_LENGTH,
            post_lapse_success=[0] * NATIVE_POST_LAPSE_LENGTH,
            first_review_cases=[0] * NATIVE_FIRST_INTERVAL_DAY_SLOTS,
            first_review_success=[0] * NATIVE_FIRST_INTERVAL_DAY_SLOTS,
        )


@dataclass(frozen=True)
class HistoryRecord:
    """One real positive-interval review; counters are post-review values."""

    elapsed_days: float
    grade: int
    repetitions: int
    lapses: int


@dataclass(frozen=True)
class ReplayState:
    difficulty: float
    stability: float
    previous_observed_stability: float
    previous_pre_review_stability_category: int
    repetitions: int
    lapses: int


@dataclass(frozen=True)
class ModelTotals:
    recall_cases: int
    stability_increase_cases: int
    post_lapse_cases: int
    first_review_cases: int
    history_records: int


@dataclass(frozen=True)
class _CalibrationFit:
    cases: int
    slope: float
    intercept: float

    def calibrated(self, predicted_retrievability: float) -> float:
        fitted = _clamp(
            _bounded_exp(self.slope * predicted_retrievability + self.intercept),
            0.005,
            0.995,
        )
        weight = _saturating_weight(float(self.cases), 3.5)
        return weight * fitted + (1.0 - weight) * predicted_retrievability


@dataclass(frozen=True)
class _Transition:
    state: ReplayState
    raw_retrievability: float
    calibrated_retrievability: float
    observed_stability: float
    observed_ratio: float
    difficulty_category_before: int
    stability_category_before: int
    retrievability_category_before: int
    difficulty_category_after: int


@dataclass(frozen=True)
class _PendingPrediction:
    card_id: Hashable
    elapsed_days: float
    probability: float


class RecoveredSM19:
    """One user's causal, shared-model Recovered SM19 benchmark instance."""

    def __init__(self, again_grade: int) -> None:
        if type(again_grade) is not int or again_grade not in (0, 1, 2):
            raise ValueError("again_grade must be exactly 0, 1, or 2")
        self._again_grade = again_grade
        self._model = Algorithm17State.defaults()
        self._stability_increase_np = np.asarray(
            self._model.stability_increase,
            dtype=np.float64,
        )
        self._history_by_card: dict[Hashable, list[HistoryRecord]] = {}
        self._pending: _PendingPrediction | None = None
        self._single_initial_stability_cache: OrderedDict[tuple[float, bool], float] = (
            OrderedDict()
        )
        self._calibration_fit_cache: dict[tuple[int, int], _CalibrationFit] = {}
        self._post_lapse_stability_cache: dict[tuple[int, int], float] = {}
        self._first_interval_fit_cache: _FirstIntervalCurve | None = None

    @property
    def again_grade(self) -> int:
        return self._again_grade

    @property
    def learning_state(self) -> Algorithm17State:
        """Expose the native-shaped state for audit/tests; callers must not mutate it."""

        return self._model

    def history_for(self, card_id: Hashable) -> tuple[HistoryRecord, ...]:
        return tuple(self._history_by_card.get(card_id, ()))

    def totals(self) -> ModelTotals:
        return ModelTotals(
            recall_cases=sum(self._model.recall_cases),
            stability_increase_cases=sum(self._model.stability_increase_cases),
            post_lapse_cases=sum(self._model.post_lapse_cases),
            first_review_cases=sum(self._model.first_review_cases),
            history_records=sum(map(len, self._history_by_card.values())),
        )

    def predict(self, card_id, elapsed_days):
        """Predict before seeing the current review's rating or binary outcome."""

        card_key = self._validate_card_id(card_id)
        elapsed = self._validate_elapsed(elapsed_days)
        if self._pending is not None:
            if (
                self._pending.card_id == card_key
                and self._pending.elapsed_days == elapsed
            ):
                return self._pending.probability
            raise RuntimeError("commit the pending review before predicting another")

        history = self._history_by_card.get(card_key, ())
        if not history:
            probability = self._first_review_probability(elapsed)
        else:
            state = self._replay(history, learn_last=False)
            raw = self._raw_retrievability(elapsed, state.stability)
            fit = self._calibration_fit(state.difficulty, state.stability)
            probability = fit.calibrated(raw)

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("Recovered SM19 produced an invalid probability")
        self._pending = _PendingPrediction(card_key, elapsed, probability)
        return probability

    def commit(self, card_id, elapsed_days, anki_rating):
        """Consume one current grade, train once, then append its History row."""

        card_key = self._validate_card_id(card_id)
        elapsed = self._validate_elapsed(elapsed_days)
        if type(anki_rating) is not int or anki_rating not in (1, 2, 3, 4):
            raise ValueError("anki_rating must be one of 1, 2, 3, 4")
        rating = anki_rating
        if self._pending is None:
            raise RuntimeError("predict must be called before commit")
        if self._pending.card_id != card_key or self._pending.elapsed_days != elapsed:
            raise RuntimeError("commit does not match the pending prediction")

        history = self._history_by_card.setdefault(card_key, [])
        if history:
            previous_repetitions = history[-1].repetitions
            previous_lapses = history[-1].lapses
        else:
            # Native History starts with an implicit synthetic acquisition row.
            previous_repetitions = 1
            previous_lapses = 0

        grade = self._again_grade if rating == 1 else rating + 1
        if grade >= 3:
            repetitions = previous_repetitions + 1
            lapses = previous_lapses
        else:
            repetitions = 1
            lapses = previous_lapses + 1
        if repetitions > 0xFFFF or lapses > 0xFFFF:
            raise ValueError("review counters exceed the recovered native domain")

        current = HistoryRecord(elapsed, grade, repetitions, lapses)
        full_history = (*history, current)

        # The replay is pure for every strict-past record.  Only its final
        # transition mutates the shared Algorithm 17 arrays, exactly once.
        self._replay(full_history, learn_last=True)
        history.append(current)
        self._pending = None

    @staticmethod
    def _validate_card_id(card_id: object) -> Hashable:
        try:
            hash(card_id)
        except TypeError as error:
            raise TypeError("card_id must be hashable") from error
        return card_id  # type: ignore[return-value]

    @staticmethod
    def _validate_elapsed(elapsed_days: object) -> float:
        try:
            elapsed = float(elapsed_days)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("elapsed_days must be a finite positive number") from error
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("elapsed_days must be a finite positive number")
        return elapsed

    def _first_review_probability(self, elapsed_days: float) -> float:
        fit = self._first_interval_fit()
        category = _first_interval_day_category(elapsed_days)
        representative_day = _first_interval_day_axis(category)
        cases = self._model.first_review_cases[category]
        if cases > 0:
            empirical = self._model.first_review_success[category] / cases
        else:
            empirical = 0.9
        fitted = _bounded_exp(
            fit.intercept + fit.slope * math.log(float(representative_day))
        )
        weight = _saturating_weight(float(cases), 40.0)
        return _clamp(
            weight * empirical + (1.0 - weight) * fitted,
            0.0,
            1.0,
        )

    @staticmethod
    def _raw_retrievability(elapsed_days: float, stability: float) -> float:
        stability = _normalize_stability(stability)
        return _clamp(
            _bounded_exp(-FORGETTING_CONSTANT * elapsed_days / stability),
            0.005,
            0.995,
        )

    def _calibration_fit(
        self,
        difficulty: float,
        stability: float,
    ) -> _CalibrationFit:
        d_category = _difficulty_category(difficulty)
        s_category = _stability_category(stability)
        cache_key = (d_category, s_category)
        cached = self._calibration_fit_cache.get(cache_key)
        if cached is not None:
            return cached

        x_values: list[float] = []
        y_values: list[float] = []
        case_counts: list[float] = []
        total_cases = 0
        for r_category in range(1, ACTIVE_CATEGORY_COUNT + 1):
            retrievability = _retrievability_category_center(r_category)
            index = _cube_index(d_category, s_category, r_category)
            cases = self._model.recall_cases[index]
            remembered = self._model.recall_success[index]
            recall = 0.64 * retrievability + 0.3 if cases <= 0 else remembered / cases
            x_values.append(retrievability)
            y_values.append(math.log(recall) if recall > 0.0 else 0.0)
            case_counts.append(float(cases))
            total_cases += _delphi_round(float(cases))
        slope, intercept = _weighted_linear_regression(
            x_values,
            y_values,
            case_counts,
        )
        fit = _CalibrationFit(total_cases, slope, intercept)
        self._calibration_fit_cache[cache_key] = fit
        return fit

    def _initial_difficulty_for_stability(
        self,
        history: Sequence[HistoryRecord],
    ) -> float:
        if len(history) >= NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY:
            elapsed = np.fromiter(
                (record.elapsed_days for record in history),
                dtype=np.float64,
                count=len(history),
            )
            grade = np.fromiter(
                (record.grade for record in history),
                dtype=np.int64,
                count=len(history),
            )
            return float(
                _initial_difficulty_for_stability_kernel(
                    elapsed,
                    grade,
                    self._stability_increase_np,
                )
            )

        scores: list[float] = []
        for d_category in range(1, ACTIVE_CATEGORY_COUNT + 1):
            stability = INITIAL_STABILITY_SEARCH_SEED
            weighted_error = 0.0
            total_weight = 0.0
            weight = 1.0
            for record in history:
                stability = _normalize_stability(stability)
                raw = _bounded_exp(
                    -FORGETTING_CONSTANT * record.elapsed_days / stability
                )
                error = raw - (1.0 if record.grade >= 3 else 0.0)
                weight *= 2.0
                weighted_error += error * weight
                total_weight += weight
                if record.grade >= 3:
                    index = _cube_index(
                        d_category,
                        _stability_category(stability),
                        _retrievability_category(raw),
                    )
                    stability *= self._model.stability_increase[index]
                else:
                    stability = INITIAL_STABILITY_SEARCH_SEED
            scores.append(weighted_error / total_weight)

        best_low = 0
        best_score = math.inf
        for index, score in enumerate(scores):
            if score < best_score:
                best_score = score
                best_low = index
        best_high = len(scores) - 1
        best_score = math.inf
        for index in range(len(scores) - 1, -1, -1):
            score = scores[index]
            if score < best_score:
                best_score = score
                best_high = index
        return (
            _difficulty_category_center(best_low + 1)
            + _difficulty_category_center(best_high + 1)
        ) / 2.0

    def _initial_stability_objective(
        self,
        history: Sequence[HistoryRecord],
        difficulty: float,
        candidate: float,
    ) -> float:
        stability = candidate
        squared_error = 0.0
        for record in history:
            stability = _normalize_stability(stability)
            raw = _bounded_exp(-FORGETTING_CONSTANT * record.elapsed_days / stability)
            error = raw - (1.0 if record.grade >= 3 else 0.0)
            squared_error += error * error
            if record.grade >= 3:
                stability *= _theoretical_success_ratio(
                    raw,
                    stability,
                    difficulty,
                )
            else:
                stability = candidate
        return math.sqrt(squared_error / len(history)) if history else 0.0

    def _search_initial_stability(
        self,
        history: Sequence[HistoryRecord],
        difficulty: float,
        *,
        allow_up: bool,
    ) -> float:
        stability = INITIAL_STABILITY_SEARCH_SEED
        while True:
            center = self._initial_stability_objective(
                history,
                difficulty,
                stability,
            )
            lower = self._initial_stability_objective(
                history,
                difficulty,
                0.99 * stability,
            )
            upper = self._initial_stability_objective(
                history,
                difficulty,
                1.01 * stability,
            )
            if lower < center:
                stability *= 0.99
            elif upper < center:
                stability *= 1.01
                if not allow_up:
                    break
            else:
                break
            if stability < 0.7 or stability > 100.0:
                break
        return stability

    def _initial_stability(
        self,
        history: Sequence[HistoryRecord],
    ) -> float:
        if not history:
            curve = self._first_interval_fit()
            return _clamp(curve.stability, 0.1, 33.0)

        single_cache_key: tuple[float, bool] | None = None
        if len(history) == 1:
            first = history[0]
            single_cache_key = (first.elapsed_days, first.grade >= 3)
            cached = self._single_initial_stability_cache.pop(
                single_cache_key,
                None,
            )
            if cached is not None:
                self._single_initial_stability_cache[single_cache_key] = cached
                return cached

        first_is_success = history[0].grade >= 3
        if len(history) >= NUMBA_INITIAL_STABILITY_SEARCH_MIN_HISTORY:
            elapsed = np.fromiter(
                (record.elapsed_days for record in history),
                dtype=np.float64,
                count=len(history),
            )
            grade = np.fromiter(
                (record.grade for record in history),
                dtype=np.int64,
                count=len(history),
            )
            if len(history) >= NUMBA_INITIAL_DIFFICULTY_MIN_HISTORY:
                difficulty = float(
                    _initial_difficulty_for_stability_kernel(
                        elapsed,
                        grade,
                        self._stability_increase_np,
                    )
                )
            else:
                difficulty = self._initial_difficulty_for_stability(history)
            fitted = float(
                _initial_stability_search_kernel(
                    elapsed,
                    grade,
                    difficulty,
                    first_is_success,
                )
            )
        else:
            difficulty = self._initial_difficulty_for_stability(history)
            fitted = self._search_initial_stability(
                history,
                difficulty,
                allow_up=first_is_success,
            )
        if first_is_success:
            native_history_length = len(history) + 1
            weight = _saturating_weight(float(native_history_length), 6.0)
            stability = weight * fitted + (1.0 - weight) * INITIAL_STABILITY_SEARCH_SEED
        else:
            stability = min(fitted, INITIAL_STABILITY_SEARCH_SEED)

        if single_cache_key is not None:
            self._single_initial_stability_cache[single_cache_key] = stability
            if (
                len(self._single_initial_stability_cache)
                > SINGLE_INITIAL_STABILITY_CACHE_LIMIT
            ):
                self._single_initial_stability_cache.popitem(last=False)
        return stability

    def _first_interval_fit(self) -> _FirstIntervalCurve:
        if self._first_interval_fit_cache is None:
            self._first_interval_fit_cache = _fit_first_interval_curve(
                self._model.first_review_success,
                self._model.first_review_cases,
                default_stability=FIRST_INTERVAL_EMPTY_STABILITY,
            )
        return self._first_interval_fit_cache

    def _post_lapse_stability(
        self,
        retrievability_category: int,
        lapse_category: int,
    ) -> float:
        cache_key = (retrievability_category, lapse_category)
        cached = self._post_lapse_stability_cache.get(cache_key)
        if cached is not None:
            return cached

        cases = [0] * NATIVE_FIRST_INTERVAL_DAY_SLOTS
        remembered = [0] * NATIVE_FIRST_INTERVAL_DAY_SLOTS
        for day_category in range(1, FIRST_INTERVAL_DAY_COUNT + 1):
            index = _post_lapse_index(
                retrievability_category,
                lapse_category,
                day_category,
            )
            cases[day_category] = self._model.post_lapse_cases[index]
            remembered[day_category] = self._model.post_lapse_success[index]
        curve = _fit_first_interval_curve(
            remembered,
            cases,
            default_stability=1.0,
        )
        stability = _clamp(curve.stability, 1.0, 9.0)
        self._post_lapse_stability_cache[cache_key] = stability
        return stability

    def _transition(
        self,
        state: ReplayState,
        record: HistoryRecord,
        *,
        learn: bool,
    ) -> _Transition:
        old_difficulty = state.difficulty
        old_stability = _normalize_stability(state.stability)
        raw = self._raw_retrievability(
            record.elapsed_days,
            old_stability,
        )
        d_before = _difficulty_category(old_difficulty)
        s_before = _stability_category(old_stability)
        r_before = _retrievability_category(raw)
        calibration_fit = self._calibration_fit(
            old_difficulty,
            old_stability,
        )
        calibrated = calibration_fit.calibrated(raw)
        new_difficulty = _update_difficulty(
            record.grade,
            record.repetitions,
            record.lapses,
            calibrated,
            old_difficulty,
        )
        d_after = _difficulty_category(new_difficulty)

        inferred = _normalize_stability(
            -FORGETTING_CONSTANT * record.elapsed_days / math.log(calibrated)
        )
        observed = _observed_stability_update(
            record.grade,
            record.elapsed_days,
            old_stability,
            inferred,
            float(calibration_fit.cases),
        )
        observed_ratio = -1.0
        if state.previous_observed_stability > 0.0:
            observed_ratio = observed / state.previous_observed_stability
            if record.grade >= 3:
                observed_ratio = _clamp(observed_ratio, 0.8, 20.0)
            else:
                observed_ratio = _clamp(observed_ratio, 0.5, 2.0)

        if record.grade >= 3:
            success_index = _cube_index(d_after, s_before, r_before)
            matrix_cases = self._model.stability_increase_cases[success_index]
            matrix_ratio = self._model.stability_increase[success_index]
            theoretical = _theoretical_success_ratio(
                raw,
                old_stability,
                new_difficulty,
            )
            next_stability = _success_stability(
                matrix_cases,
                theoretical,
                matrix_ratio,
                record.elapsed_days,
                observed,
            )
        else:
            next_stability = self._post_lapse_stability(
                r_before,
                _lapse_category(record.lapses),
            )
            observed_ratio = -1.0

        next_state = ReplayState(
            difficulty=new_difficulty,
            stability=_normalize_stability(next_stability),
            # Native sub_84C410 advances this for both success and failure.
            previous_observed_stability=observed,
            previous_pre_review_stability_category=s_before,
            repetitions=record.repetitions,
            lapses=record.lapses,
        )
        transition = _Transition(
            state=next_state,
            raw_retrievability=raw,
            calibrated_retrievability=calibrated,
            observed_stability=observed,
            observed_ratio=observed_ratio,
            difficulty_category_before=d_before,
            stability_category_before=s_before,
            retrievability_category_before=r_before,
            difficulty_category_after=d_after,
        )
        if learn:
            self._learn_current(state, record, transition)
        return transition

    def _learn_current(
        self,
        previous_state: ReplayState,
        record: HistoryRecord,
        transition: _Transition,
    ) -> None:
        # All transition reads have completed.  Current-grade writes begin here.
        recall_index = _cube_index(
            transition.difficulty_category_before,
            transition.stability_category_before,
            transition.retrievability_category_before,
        )
        self._model.recall_cases[recall_index] += 1
        if record.grade >= 3:
            self._model.recall_success[recall_index] += 1
        self._calibration_fit_cache.pop(
            (
                transition.difficulty_category_before,
                transition.stability_category_before,
            ),
            None,
        )

        day_category = _first_interval_day_category(record.elapsed_days)
        if previous_state.repetitions == 1 and record.lapses > 0:
            lapse_category = _lapse_category(record.lapses)
            post_index = _post_lapse_index(
                transition.retrievability_category_before,
                lapse_category,
                day_category,
            )
            self._model.post_lapse_cases[post_index] += 1
            if record.grade >= 3:
                self._model.post_lapse_success[post_index] += 1
            self._post_lapse_stability_cache.pop(
                (
                    transition.retrievability_category_before,
                    lapse_category,
                ),
                None,
            )

        if previous_state.repetitions == 1 and previous_state.lapses == 0:
            self._model.first_review_cases[day_category] += 1
            if record.grade >= 3:
                self._model.first_review_success[day_category] += 1
            self._first_interval_fit_cache = None

        if (
            record.grade >= 3
            and transition.observed_ratio > 0.0
            and previous_state.previous_pre_review_stability_category != 0
        ):
            growth_index = _cube_index(
                transition.difficulty_category_after,
                previous_state.previous_pre_review_stability_category,
                transition.retrievability_category_before,
            )
            old_cases = self._model.stability_increase_cases[growth_index]
            old_mean = self._model.stability_increase[growth_index]
            new_cases = old_cases + 1
            self._model.stability_increase[growth_index] = (
                old_cases * old_mean + transition.observed_ratio
            ) / new_cases
            self._stability_increase_np[growth_index] = self._model.stability_increase[
                growth_index
            ]
            self._model.stability_increase_cases[growth_index] = new_cases

    def _replay(
        self,
        history: Sequence[HistoryRecord],
        *,
        learn_last: bool,
    ) -> ReplayState:
        if not history:
            raise ValueError("positive History replay requires at least one record")
        state = ReplayState(
            difficulty=INITIAL_DIFFICULTY,
            stability=self._initial_stability(history),
            previous_observed_stability=-1.0,
            previous_pre_review_stability_category=0,
            repetitions=1,
            lapses=0,
        )
        final_index = len(history) - 1
        for index, record in enumerate(history):
            transition = self._transition(
                state,
                record,
                learn=learn_last and index == final_index,
            )
            state = transition.state
        return state


__all__ = [
    "Algorithm17State",
    "HistoryRecord",
    "ModelTotals",
    "RecoveredSM19",
    "ReplayState",
]
