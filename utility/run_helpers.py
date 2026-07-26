import math

import pandas as pd
from matplotlib import pyplot as plt

from ai_analysis import TeamviewerAIEvent, SapAIEvent
from ai_analysis.comparator import Comparator
from algorithm import ChowTest, PELT, ZivotAndrews


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
        plot=False):
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
        fig, axes = plt.subplots(5, 2, figsize=(30, 15))
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