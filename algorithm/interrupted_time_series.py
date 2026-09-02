from __future__ import annotations

import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable

import causalpy as cp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


class InterruptedTimeSeries:
    """
    Small wrapper around CausalPy ITS for the case-company quarterly data.

    ``post_ai_event`` must be indexed by Quarter and equal 0 before the first
    AI-signal quarter and 1 from that quarter onward.
    """

    DEFAULT_VARIABLES = (
        "R&D",
        "Cost of goods sold",
        "Gross Margin",
        "Adjusted EBITDA margin",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        post_ai_event: pd.Series,
        variables: Iterable[str] | None = None,
        quarter_column: str = "Quarter",
        include_quarter_seasonality: bool = True,
        minimum_pre_quarters: int = 8,
        minimum_post_quarters: int = 4,
    ) -> None:
        self.data = data.copy()
        self.post_ai_event = post_ai_event.copy()
        self.variables = list(variables or self.DEFAULT_VARIABLES)
        self.quarter_column = quarter_column
        self.include_quarter_seasonality = include_quarter_seasonality
        self.minimum_pre_quarters = minimum_pre_quarters
        self.minimum_post_quarters = minimum_post_quarters

        self.results: dict[str, cp.InterruptedTimeSeries] = {}
        self.effect_summaries: dict[str, object] = {}
        self.analysis_data: dict[str, pd.DataFrame] = {}
        self.treatment_time: pd.Timestamp | None = None

    @staticmethod
    def _quarter(value: object) -> pd.Period:
        """Parse Q4-24 or 2024Q4."""
        if isinstance(value, pd.Period):
            return value.asfreq("Q")

        text = str(value).strip().upper()
        match = re.fullmatch(r"Q([1-4])-(\d{2}|\d{4})", text)
        if match:
            quarter = int(match.group(1))
            year_text = match.group(2)
            year = int(year_text) + (2000 if len(year_text) == 2 else 0)
            return pd.Period(year=year, quarter=quarter, freq="Q")

        match = re.fullmatch(r"(\d{4})Q([1-4])", text)
        if match:
            return pd.Period(
                year=int(match.group(1)),
                quarter=int(match.group(2)),
                freq="Q",
            )

        raise ValueError(f"Invalid quarter: {value!r}")

    @staticmethod
    def _numeric(values: pd.Series) -> pd.Series:
        """Convert numbers and values such as '46,9%' to floats."""
        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce")

        return pd.to_numeric(
            values.astype("string")
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip(),
            errors="coerce",
        )

    def _prepare_signal(self) -> pd.Series:
        signal = self.post_ai_event.copy()

        if not signal.index.is_unique:
            raise ValueError("post_ai_event must have a unique Quarter index.")

        signal.index = signal.index.map(self._quarter)
        signal = pd.to_numeric(signal, errors="raise").sort_index()

        if not signal.isin([0, 1]).all():
            raise ValueError("post_ai_event must contain only 0 and 1.")
        if not signal.eq(1).any():
            raise ValueError("post_ai_event contains no AI-event quarter.")

        first_one = int(signal.eq(1).to_numpy().argmax())
        expected = pd.Series(
            [0] * first_one + [1] * (len(signal) - first_one),
            index=signal.index,
            dtype=signal.dtype,
        )
        if not signal.equals(expected):
            raise ValueError(
                "post_ai_event must be 0 before the first AI signal and "
                "remain 1 afterward."
            )
        return signal

    def _prepare_variable(
        self,
        variable: str,
        signal: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        frame = self.data[[self.quarter_column, variable]].copy()
        frame["_period"] = frame[self.quarter_column].map(self._quarter)
        frame["y"] = self._numeric(frame[variable])
        frame = (
            frame.dropna(subset=["y"])
            .sort_values("_period")
            .reset_index(drop=True)
        )

        if frame["_period"].duplicated().any():
            raise ValueError(f"{variable!r} contains duplicate quarters.")
        if not frame["_period"].astype("int64").diff().dropna().eq(1).all():
            raise ValueError(f"{variable!r} contains missing quarters.")
        if not frame["_period"].isin(signal.index).all():
            raise ValueError(
                "post_ai_event does not cover every outcome quarter."
            )

        frame["post_ai_event"] = frame["_period"].map(signal).astype(int)
        event_rows = frame.index[frame["post_ai_event"].eq(1)]
        if event_rows.empty:
            raise ValueError(f"{variable!r} has no post-AI observations.")

        first_event_row = int(event_rows[0])
        pre_count = first_event_row
        post_count = len(frame) - first_event_row
        if pre_count < self.minimum_pre_quarters:
            raise ValueError(
                f"{variable!r} has only {pre_count} pre-event quarters."
            )
        if post_count < self.minimum_post_quarters:
            raise ValueError(
                f"{variable!r} has only {post_count} post-event quarters."
            )

        frame["t"] = range(len(frame))
        frame["calendar_quarter"] = frame["_period"].map(
            lambda quarter: quarter.quarter
        )
        frame["date"] = frame["_period"].map(
            lambda quarter: quarter.to_timestamp(how="end").normalize()
        )
        frame = frame.set_index("date")

        return frame, frame.index[first_event_row]

    def run(
        self,
        *,
        plot: bool = False,
        print_summary: bool = False,
        company: str = None,
        save_output: str = None,
        alpha: float = 0.05,
    ) -> dict[str, cp.InterruptedTimeSeries]:
        """Run one CausalPy ITS analysis for each requested outcome."""
        missing = {
            self.quarter_column,
            *self.variables,
        }.difference(self.data.columns)
        if missing:
            raise KeyError(f"Missing data columns: {sorted(missing)}")

        signal = self._prepare_signal()
        formula = "y ~ 1 + t"
        if self.include_quarter_seasonality:
            formula += " + C(calendar_quarter)"

        self.results.clear()
        self.effect_summaries.clear()
        self.analysis_data.clear()
        self.treatment_time = None

        for variable in self.variables:
            frame, treatment_time = self._prepare_variable(variable, signal)

            result = cp.InterruptedTimeSeries(
                frame,
                treatment_time,
                formula=formula,
                model=LinearRegression(),
            )

            self.results[variable] = result
            self.effect_summaries[variable] = result.effect_summary(
                alpha=alpha
            )
            self.analysis_data[variable] = frame
            if self.treatment_time is None:
                self.treatment_time = treatment_time
            elif treatment_time != self.treatment_time:
                raise ValueError(
                    "Outcome variables produced different treatment times."
                )

            if print_summary:
                print(f"\n--- {variable} ---")
                result.summary(round_to=3)
                print(self.effect_summaries[variable].text)


            if plot:
                fig, ax = result.plot(show=False)
                fig.suptitle(f"CausalPy ITS: {company}: {variable}")
                fig.tight_layout()

                output_dir = Path.cwd() / "ITS_plots"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"{company}_{variable}_ITS.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                #plt.show()


            if save_output is not None:
                self.save_results(filename=save_output)

        return self.results

    def effect_table(self) -> pd.DataFrame:
        """Combine CausalPy average and cumulative effects across outcomes."""
        if not self.effect_summaries:
            raise RuntimeError("Call run() first.")

        output = []
        for variable, summary in self.effect_summaries.items():
            table = summary.table.copy().reset_index(names="Effect type")
            table.insert(0, "Variable", variable)
            output.append(table)

        return pd.concat(output, ignore_index=True)

    def save_results(
            self,
            filename: str = "interrupted_time_series_results.txt",
            round_to: int = 4
    ) -> Path:
        """
        Save model summaries, effect summaries, and the combined effect
        table to a text file in the current working directory.
        """
        if not self.results:
            raise RuntimeError(
                "Call run() before saving results."
            )

        output_path = Path.cwd() / filename

        with output_path.open(
                mode="w",
                encoding="utf-8"
        ) as output_file:

            output_file.write(
                "INTERRUPTED TIME-SERIES ANALYSIS\n"
            )
            output_file.write("=" * 80 + "\n\n")

            if self.treatment_time is not None:
                output_file.write(
                    f"Treatment time: {self.treatment_time}\n"
                )

            output_file.write(
                f"Quarter seasonality included: "
                f"{self.include_quarter_seasonality}\n"
            )
            output_file.write(
                f"Variables: {', '.join(self.variables)}\n\n"
            )

            for variable, result in self.results.items():
                output_file.write("=" * 80 + "\n")
                output_file.write(f"VARIABLE: {variable}\n")
                output_file.write("=" * 80 + "\n\n")

                # result.summary() prints instead of returning text,
                # so capture its standard output.
                summary_buffer = StringIO()

                with redirect_stdout(summary_buffer):
                    result.summary(round_to=round_to)

                output_file.write("MODEL SUMMARY\n")
                output_file.write("-" * 80 + "\n")
                output_file.write(summary_buffer.getvalue())
                output_file.write("\n")

                effect_summary = self.effect_summaries[variable]

                output_file.write("EFFECT SUMMARY\n")
                output_file.write("-" * 80 + "\n")
                output_file.write(effect_summary.text)
                output_file.write("\n\n")

                output_file.write("EFFECT TABLE\n")
                output_file.write("-" * 80 + "\n")
                output_file.write(
                    effect_summary.table
                    .round(round_to)
                    .to_string()
                )
                output_file.write("\n\n")

            output_file.write("=" * 80 + "\n")
            output_file.write("COMBINED EFFECT TABLE\n")
            output_file.write("=" * 80 + "\n\n")

            output_file.write(
                self.effect_table()
                .round(round_to)
                .to_string(index=False)
            )
            output_file.write("\n")

        return output_path
