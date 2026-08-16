"""Feature stream for the causal Recovered SM19 benchmark adapter."""

import pandas as pd

from config import Config

from .base import BaseFeatureEngineer


class RecoveredSM19FeatureEngineer(BaseFeatureEngineer):
    """Keep every usable review for online replay and mark benchmark score rows.

    SM19 has a shared user model that changes after each review.  Rows removed by
    the benchmark's score-only outlier/continuity filters must therefore remain in
    the chronological replay stream so they can affect later predictions.
    """

    _SCORE_COLUMNS = ("y", "i", "rmse_bins_lapse")
    _SCORE_PREPROCESS_COLUMNS = (
        "card_id",
        "day_offset",
        "rating",
        "elapsed_days",
    )
    _SCORE_POSTPROCESS_COLUMNS = (
        "card_id",
        "review_th",
        "rating",
        "elapsed_days",
        "delta_t",
        "i",
    )
    _REPLAY_OUTPUT_COLUMNS = (
        "card_id",
        "review_th",
        "rating",
        "elapsed_days",
        "delta_t",
    )

    def __init__(self, config: Config):
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        unsupported = []
        if self.config.include_short_term:
            unsupported.append("--short")
        if self.config.use_secs_intervals:
            unsupported.append("--secs")
        if self.config.equalize_test_with_non_secs:
            unsupported.append("--equalize_test_with_non_secs")
        if self.config.train_equals_test:
            unsupported.append("--train_equals_test")
        if self.config.two_buttons:
            unsupported.append("--two_buttons")
        if unsupported:
            flags = ", ".join(unsupported)
            raise ValueError(
                "Recovered-SM19 supports only the standard day-interval, "
                f"four-button evaluation protocol; unsupported: {flags}"
            )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep score selection byte-for-byte on the common benchmark path.  Its
        # per-card sequence cap is an evaluation rule, however, and must not
        # truncate the user's shared online SM19 learning stream: a later,
        # unscored review can still change predictions for other cards.
        # Recovered SM19 consumes only the score metadata produced by the common
        # preprocessing/postprocessing path.  Building cumulative t/r histories
        # is both unused and quadratic in a card's review count, so skip it.
        time_input_column = "delta_t" if "delta_t" in df.columns else "elapsed_seconds"
        score_input = df.loc[
            :,
            [*self._SCORE_PREPROCESS_COLUMNS, time_input_column],
        ].copy()
        score_rows = self._common_preprocessing(score_input)
        del score_input
        score_rows = score_rows.loc[
            :,
            list(self._SCORE_POSTPROCESS_COLUMNS),
        ].copy()
        score_rows = self._common_postprocessing(score_rows)
        score_metadata = score_rows.loc[
            :,
            ["review_th", *self._SCORE_COLUMNS],
        ].rename(
            columns={column: f"sm19_score_{column}" for column in self._SCORE_COLUMNS}
        )
        del score_rows

        if "review_th" in df.columns:
            source_review_th = df["review_th"]
            if (
                source_review_th.isna().any()
                or source_review_th.duplicated().any()
                or not source_review_th.is_monotonic_increasing
            ):
                raise ValueError(
                    "Recovered-SM19 requires input review_th to be unique and "
                    "chronological; it will not reorder causal events"
                )
        replay = df.loc[
            :,
            ["card_id", "rating", "elapsed_days", time_input_column],
        ].copy()
        replay["review_th"] = range(1, replay.shape[0] + 1)
        replay.drop(
            replay[~replay["rating"].isin([1, 2, 3, 4])].index,
            inplace=True,
        )
        replay = self._process_time_intervals(replay)
        replay.drop(replay[replay["elapsed_days"] == 0].index, inplace=True)

        # Acquisition has no native scoreable R and is represented implicitly
        # by the model.  The standard day-mode experiment also excludes
        # same-day reviews from both scoring and model learning.
        replay = replay.loc[
            replay["delta_t"] > 0,
            list(self._REPLAY_OUTPUT_COLUMNS),
        ].copy()
        replay = replay.merge(
            score_metadata,
            on="review_th",
            how="left",
            sort=False,
        )
        replay["sm19_score"] = replay["sm19_score_y"].notna()
        return replay

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
