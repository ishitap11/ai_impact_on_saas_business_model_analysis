from statsmodels.tsa.stattools import zivot_andrews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class ZivotAndrews:
    def run_zivot_andrews_break_test(column_name : str, analysis_df: pd.DataFrame):
        za_result = zivot_andrews(analysis_df[column_name], maxlag=1, regression='ct')
        result_for_col = {
            "Variable": column_name,
            "Break point Quarter": analysis_df.iloc[za_result[4]]['Quarter'],
            "Break point index": za_result[4],
            "Test Statistic": za_result[0],
            "p-value": za_result[1],
            "Critical value": za_result[2]
        }
        return result_for_col, za_result[4]

    def plot_structural_breaks(
            analysis_df,
            results,
            quarter_column="Quarter",
            ncols=2,
            figsize_per_subplot=(7, 4),
            marker="o"
    ):
        """
        Plot multiple time-series variables in one figure, with the detected
        structural break shown as a vertical line.

        Parameters
        ----------
        analysis_df : pandas.DataFrame
            Data containing Quarter and the tested variables.

        results : list[dict] or dict
            Either a list of result dictionaries or one dictionary keyed by
            variable name.

        quarter_column : str
            Name of the quarter column.

        ncols : int
            Number of subplot columns.

        figsize_per_subplot : tuple
            Width and height allocated to each subplot.

        marker : str
            Matplotlib marker used for observations.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects.
        """

        # Support either a list of dictionaries or a dictionary of results.
        if isinstance(results, dict):
            if "Variable" in results:
                results = [results]
            else:
                results = list(results.values())

        if not results:
            raise ValueError("No structural-break results were supplied.")

        nplots = len(results)
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                figsize_per_subplot[0] * ncols,
                figsize_per_subplot[1] * nrows
            ),
            squeeze=False
        )

        axes = axes.flatten()

        quarter_labels = analysis_df[quarter_column].astype(str)
        x_positions = np.arange(len(analysis_df))

        for axis, result in zip(axes, results):
            variable = result["Variable"]

            if variable not in analysis_df.columns:
                axis.set_visible(False)
                print(f"Skipped '{variable}': column not found.")
                continue

            series = pd.to_numeric(
                analysis_df[variable],
                errors="coerce"
            )

            break_quarter = str(result["Break point Quarter"])
            test_statistic = result.get("Test Statistic", np.nan)
            p_value = result.get("p-value", np.nan)

            # Find the break quarter in the complete DataFrame.
            break_matches = np.flatnonzero(
                quarter_labels.to_numpy() == break_quarter
            )

            axis.plot(
                x_positions,
                series,
                color="#1F4E78",
                linewidth=1.8,
                marker=marker,
                markersize=4,
                label=variable
            )

            if len(break_matches) > 0:
                break_position = break_matches[0]

                axis.axvline(
                    x=break_position,
                    color="#C00000",
                    linestyle="--",
                    linewidth=2,
                    label=f"Break: {break_quarter}"
                )

                # Highlight the observation at the break, if available.
                break_value = series.iloc[break_position]

                if pd.notna(break_value):
                    axis.scatter(
                        break_position,
                        break_value,
                        color="#C00000",
                        s=55,
                        zorder=5
                    )

            else:
                print(
                    f"Warning: break quarter '{break_quarter}' "
                    f"was not found for '{variable}'."
                )

            axis.set_title(variable, fontweight="bold")
            axis.set_xlabel("Quarter")
            axis.set_ylabel(variable)
            axis.grid(
                axis="y",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5
            )

            # Show a manageable number of quarter labels.
            tick_step = max(1, len(analysis_df) // 8)
            tick_positions = x_positions[::tick_step]

            axis.set_xticks(tick_positions)
            axis.set_xticklabels(
                quarter_labels.iloc[::tick_step],
                rotation=45,
                ha="right"
            )

            statistics_text = (
                f"ZA statistic: {test_statistic:.3f}\n"
                f"p-value: {p_value:.3f}"
            )

            axis.text(
                0.02,
                0.97,
                statistics_text,
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#BFBFBF",
                    "alpha": 0.85
                }
            )

            axis.legend(loc="best", fontsize=8)

        # Hide unused subplot positions.
        for axis in axes[nplots:]:
            axis.set_visible(False)

        fig.suptitle(
            "Structural Breaks in Quarterly Financial Variables",
            fontsize=16,
            fontweight="bold",
            y=1.01
        )

        fig.tight_layout()

        return fig, axes