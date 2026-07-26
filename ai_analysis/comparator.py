import pandas as pd


class Comparator:
    def __init__(self, breakpoint_column: str, ai_event_signal: pd.Series, aaci_score: pd.Series):
        self.breakpoint_column = breakpoint_column
        self.ai_event_signal = ai_event_signal
        self.aaci_score = aaci_score

    def break_point_in_ai_signal(
        self,
        breakpoints: pd.Series
    ) -> pd.Series:
        """
        For each breakpoint quarter, return True if ai_event_signal
        contains that quarter and its value is 1.
        """
        return (
            breakpoints
            .map(self.ai_event_signal)
            .fillna(0)
            .eq(1)
            .astype(bool)
        )

    def break_point_with_non_zero_aaci_score(
        self,
        breakpoints: pd.Series
    ) -> pd.Series:
        """
        For each breakpoint quarter, return True if aaci_score
        contains that quarter and its value not 0.
        """
        return (
            breakpoints
            .map(self.aaci_score)
            .fillna(0)
            .ne(0)
            .astype(bool)
        )

    def compare(
        self,
        pelt_result: pd.DataFrame,
    ) -> pd.DataFrame:

        required_columns = {
            "Variable",
            self.breakpoint_column
        }

        missing_columns = required_columns.difference(
            pelt_result.columns
        )

        if missing_columns:
            raise KeyError(
                f"Missing pelt_result columns: {sorted(missing_columns)}"
            )

        if not self.ai_event_signal.index.is_unique:
            raise ValueError(
                "ai_event_signal must have a unique Quarter index."
            )

        result = pelt_result.copy()

        result["break_point_in_ai_signal"] = (
            result
            .groupby("Variable")[self.breakpoint_column]
            .transform(
                lambda breakpoints:
                self.break_point_in_ai_signal(breakpoints)
            )
        )

        result["break_point_with_non_zero_aaci_score"] = (
            result
            .groupby("Variable")[self.breakpoint_column]
            .transform(
                lambda breakpoints:
                self.break_point_with_non_zero_aaci_score(breakpoints)
            )
        )

        return result