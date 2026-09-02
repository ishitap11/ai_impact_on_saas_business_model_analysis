import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from ai_analysis import TeamviewerAIEvent, SapAIEvent
from ai_analysis.comparator import Comparator
from algorithm import ChowTest, PELT, ZivotAndrews, InterruptedTimeSeries


TMV_ANALYSIS_THESIS_VARS = [
        # "Total revenue",
        # "NRR (%)",
        # "Gross profit",
        # "Gross Margin (%)",
        # "Adjusted EBITDA",
        # "Adjusted EBITDA margin",
        # "R&D",
        # "Cost of goods sold",
        # "Sales & Marketing",
        "QoQ Total revenue",
        "QoQ NRR (%)",
        "QoQ Gross profit",
        "QoQ Gross Margin (%)",
        "QoQ Adjusted EBITDA",
        "QoQ Adjusted EBITDA margin",
        "QoQ R&D",
        "QoQ Cost of goods sold",
        "QoQ Sales & Marketing",
        "YoY Total revenue",
        "YoY NRR (%)",
        "YoY Gross profit",
        "YoY Gross Margin (%)",
        "YoY Adjusted EBITDA",
        "YoY Adjusted EBITDA margin",
        "YoY R&D",
        "YoY Cost of goods sold",
        "YoY Sales & Marketing"
    ]
SAP_ANALYSIS_THESIS_VARS = [
    # "total revenue",
    # "cloud revenue",
    # "software license revenue",
    # "Gross profit",
    # "Gross margin",
    # "Operating profit",
    # "Operating margin",
    # "R&D",
    # "Cost of goods sold",
    # "Sales and Marketing",
    "QoQ total revenue",
    "QoQ cloud revenue",
    "QoQ software license revenue",
    "QoQ Gross profit",
    "QoQ Operating profit",
    "QoQ R&D",
    "QoQ Cost of goods sold",
    "QoQ Sales and Marketing",
    "YoY total revenue",
    "YoY cloud revenue",
    "YoY software license revenue",
    "YoY Gross profit",
    "YoY Operating profit",
    "YoY R&D",
    "YoY Cost of goods sold",
    "YoY Sales and Marketing"
]


def calculate_change(
    df: pd.DataFrame,
    column_name: str,
    yoy: bool = False
) -> pd.Series:
    """
    Calculate quarter-over-quarter or year-over-year percentage change
    for a dataframe column.

    Formula:
        QoQ (%) = ((current quarter / previous quarter) - 1) * 100

        YoY (%) = ((current quarter / same quarter last year) - 1) * 100

    Parameters
    ----------
    df:
        Dataframe containing quarterly data.

    column_name:
        Name of the numeric column to calculate the percentage change for.

    yoy:
        If False (default), calculate quarter-over-quarter change.
        If True, calculate year-over-year change using the value
        from 4 quarters earlier.

    Returns
    -------
    pd.Series
        Percentage change values.

        For QoQ, the first observation will be NaN.

        For YoY, the first four observations will be NaN because
        the corresponding quarter from the previous year is unavailable.
    """
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' was not found in the dataframe."
        )

    numeric_series = pd.to_numeric(
        df[column_name],
        errors="coerce"
    )

    periods = 4 if yoy else 1

    return (
        numeric_series
        .pct_change(periods=periods, fill_method=None)
        * 100
    )

def export_dataframe_to_csv(
    df: pd.DataFrame,
    filename: str
) -> None:
    """
    Export a dataframe as a CSV file to the current working directory.

    Parameters
    ----------
    df:
        DataFrame to export.

    filename:
        Name of the output file.
        The '.csv' extension is added automatically if omitted.
    """
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    df.to_csv(
        filename,
        index=False
    )

    print(f"CSV exported successfully: {filename}")

def calculate_safe_pen(length: int):
    bic_pen = math.log(length)
    return bic_pen*2.5

def get_model_for_var(variable: str):
    ratio_vars = ["Operating margin", "Gross margin", "NRR (%)","Gross Margin (%)", "Adjusted EBITDA margin"]
    if variable in ratio_vars:
        return "rbf"
    else:
        return "l1"


def run_pelt_for_all_variables(
        df: pd.DataFrame,
        analysis_vars,
        n,
        min_size=4,
        company:str =  None,
        plot=True):
    all_results = []
    plotting_results = {}

    for variable in analysis_vars:
        data = df[n:][:]
        result, analysis_data, breakpoints = PELT.run(
            data=data,  # Considering quarters after Y2020 to disregard COVID-19 event observation on variables
            variable=variable,
            penalty=calculate_safe_pen(len(data[variable])),
            model=get_model_for_var(variable),
            min_size=min_size
        )

        all_results.append(result)
        plotting_results[variable] = {
            "data": analysis_data,
            "breakpoints": breakpoints
        }

    pelt_results = pd.concat(
        all_results,
        ignore_index=True
    )

    if pelt_results.empty:
        print("No breakpoints detected with the selected penalty.")
    elif plot:
        fig, axes = plt.subplots(9, 2, figsize=(12, 20))
        axes = axes.flatten()

        for ax, variable in zip(axes, analysis_vars):
            analysis_data = plotting_results[variable]["data"]
            breakpoints = plotting_results[variable]["breakpoints"]

            ax.plot(
                analysis_data["Quarter"],
                analysis_data[variable],
                marker="o",
                linewidth=2,
                label=variable
            )

            for position, breakpoint in enumerate(breakpoints):
                break_quarter = analysis_data.loc[breakpoint, "Quarter"]

                ax.axvline(
                    x=break_quarter,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label="PELT breakpoint" if position == 0 else None
                )

            ax.set_title(variable)
            ax.set_xlabel("Quarter")

            if "margin" in variable.lower():
                ax.set_ylabel("Percentage points")
            else:
                ax.set_ylabel("EUR million")

            ax.tick_params(axis="x", rotation=60)
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(
            "PELT structural-break detection",
            fontsize=16
        )

        fig.tight_layout()
        output_dir = Path.cwd() / "PELT_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{company}_PELT.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
    return pelt_results

def run_zivot_andrews(analysis_df: pd.DataFrame, variables):
    result = []
    for column in variables:
        test_result, break_point = ZivotAndrews.run(column, analysis_df)
        result.append(test_result)
    fig, axes = ZivotAndrews.plot_structural_breaks(analysis_df,result)
    plt.show()

    display_columns = [
        "Variable",
        "Break point Quarter",
        "Break point index",
        "Test Statistic",
        "p-value",
        "Critical value",
    ]

    results_df = pd.DataFrame(result)
    print(results_df[display_columns].round(4).to_string(index=False))

def run_chow_test(analysis_df: pd.DataFrame, analysis_thesis_variables, breakpoint: str):
    results = []

    for variable in analysis_thesis_variables:
        result, test_data = ChowTest.run(
            data=analysis_df,
            dependent_variable=variable,
            breakpoint=breakpoint,
            difference_nonstationary=False,
            alpha=0.05
        )

        results.append(result)
    return results

def run_ai_comparator(pelt_results):
    tmv_result = pelt_results[0].copy()
    sap_result = pelt_results[1].copy()

    tmv = TeamviewerAIEvent()
    tmv_ai_comp = Comparator(breakpoint_column="breakpoint", ai_event_signal=tmv.get_ai_event_signal(), aaci_score=tmv.get_aaci_score())
    print("Running AI comparator for TeamViewer...")
    print(tmv_ai_comp.compare(tmv_result).to_string(index=False))

    sap = SapAIEvent()
    sap_ai_comp = Comparator(breakpoint_column="breakpoint", ai_event_signal=sap.get_ai_event_signal(), aaci_score=sap.get_aaci_score())
    print("Running AI comparator for SAP...")
    print(sap_ai_comp.compare(sap_result).to_string(index=False))

def run_its_on_post_ai_event(data: pd.DataFrame, company: str):
    if company == "tmv":
        tmv = TeamviewerAIEvent()
        its = InterruptedTimeSeries(data, tmv.get_ai_event_signal(), TMV_ANALYSIS_THESIS_VARS)
    else:
        sap = SapAIEvent()
        its = InterruptedTimeSeries(data, sap.get_ai_event_signal(), SAP_ANALYSIS_THESIS_VARS)
    result = its.run(plot=True, print_summary=False, company=company, save_output=company+"_interrupted_time_series_results.txt")
    its.effect_table()
    return result
