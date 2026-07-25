import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from algorithm import PELT, ZivotAndrews

TMV_ANALYSIS_THESIS_VARS = [
        "Total revenue",
        "NRR (%)",
        "Gross profit",
        "Gross Margin (%)",
        "Adjusted EBITDA",
        "Adjusted EBITDA margin",
        "R&D",
        "Cost of goods sold",
        "Sales & Marketing"
    ]
SAP_ANALYSIS_THESIS_VARS = [
    "total revenue",
    "cloud revenue",
    "software license revenue",
    "Gross profit",
    "Gross margin",
    "Operating profit",
    "Operating margin",
    "R&D",
    "Cost of goods sold",
    "Sales and Marketing"
]

teamviewer_file_path = "TeamViewer_Long_Format_Quarterly_Dataset.csv"
sap_file_path = "SAP_Long_Format_Quarterly_Dataset.csv"


from pathlib import Path

def import_german_csv(
    file_path: str | Path
) -> pd.DataFrame:
    """
    Import the SAP CSV and convert German-formatted numeric values.

    Examples
    --------
    "2.000,5" -> 2000.5
    "5.261"   -> 5261
    "68,5%"   -> 68.5

    If percentages_as_decimals=True:
    "68,5%"   -> 0.685
    """

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str
    )

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    identifier_columns = {"Company", "Quarter"}
    numeric_columns = [
        column for column in df.columns
        if column not in identifier_columns
    ]

    for column in numeric_columns:
        values = df[column].astype("string").str.strip()
        percentage_mask = values.str.contains("%", regex=False, na=False)

        # German number conversion:
        # 1. Remove thousands separator "."
        # 2. Replace decimal comma "," with "."
        # 3. Remove percentage symbol
        values = (
            values
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
        )

        values = values.replace({
            "": pd.NA,
            "-": pd.NA,
            "–": pd.NA,
            "—": pd.NA,
            "n.a.": pd.NA,
            "N/A": pd.NA
        })

        numeric_values = pd.to_numeric(values, errors="coerce")

        df[column] = numeric_values

    # Ensure identifiers remain clean strings.
    df["Company"] = df["Company"].astype("string").str.strip()
    df["Quarter"] = df["Quarter"].astype("string").str.strip()

    return df

# Convert percentage strings such as "46,9%" into 46.9.
# R&D, COGS, and Gross Margin should already be numeric because
# read_csv(decimal=",") handles their decimal commas.
def clean_numeric_column(series):
    """
    Convert European-formatted numbers and percentage strings to float.

    Examples:
        '46,9%' -> 46.9
        '102,7' -> 102.7
        ''      -> NaN
    """
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)  # non-breaking spaces
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "-": np.nan
        }),
        errors="coerce"
    )

def run_pelt_for_all_variables(
        df: pandas.DataFrame,
        analysis_vars,
        n,
        model:str,
        penalty=3.0,
        min_size=4,
        plot=False):
    all_results = []
    plotting_results = {}

    for variable in analysis_vars:
        result, analysis_data, breakpoints = PELT.run(
            data=df[n:][:],  # Considering quarters after Y2020 to disregard COVID-19 event observation on variables
            variable=variable,
            penalty=penalty,
            model=model,
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
    print(pelt_results.to_string(index=False))
    return all_results

def run_zivot_andrews(analysis_df: pandas.DataFrame, variables):
    result = []
    for column in variables:
        test_result, break_point = ZivotAndrews.run_zivot_andrews_break_test(column, analysis_df)
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

def run_for_sap(run_full= False):
    full_df: pandas.DataFrame = import_german_csv(sap_file_path)
    expected_cols = [
    "Company",
    "Quarter",
    "total revenue",
    "cloud revenue",
    "software license revenue",
    "Gross profit",
    "Gross margin",
    "Operating profit",
    "Operating margin",
    "R&D",
    "Cost of goods sold",
    "Sales and Marketing",
    "COGS % revenue",
    "R&D % revenue",
    "Sales and marketing % of revenue"
    ]

    numeric_columns = ["total revenue",
    "cloud revenue",
    "software license revenue",
    "Gross profit",
    "Gross margin",
    "Operating profit",
    "Operating margin",
    "R&D",
    "Cost of goods sold",
    "Sales and Marketing",
    "COGS % revenue",
    "R&D % revenue",
    "Sales and marketing % of revenue"]

    df: pandas.DataFrame = full_df[expected_cols]
    print(df.head(15))

    for column in numeric_columns:
        values = df[column].astype("string").str.strip()

        df[column] = clean_numeric_column(df[column])

    df = df.reset_index(drop=True)

    run_pelt_for_all_variables(df, SAP_ANALYSIS_THESIS_VARS, 12, model="rbf", penalty=2.0, min_size=4)

    if run_full:
        run_zivot_andrews(df[12:][:], SAP_ANALYSIS_THESIS_VARS)

def run_for_teamviewer(run_full= False):
    full_df: pandas.DataFrame = pd.read_csv(
        teamviewer_file_path,
        sep=";",
        decimal=","
    )
    expected_cols = ["Quarter", "Total revenue", "Enterprise revenue", "SMB revenue", "ARR", "NRR (%)", "Gross Margin",
                     "Adjusted EBITDA", "Adjusted EBITDA margin", "R&D", "Cost of goods sold", "Sales & Marketing"]

    df: pandas.DataFrame = full_df[expected_cols]
    print(df.head(15))

    # Clean every financial variable, not only selected ones
    numeric_columns = [
        "Total revenue",
        "Enterprise revenue",
        "SMB revenue",
        "ARR",
        "NRR (%)",
        "Gross Margin",
        "Adjusted EBITDA",
        "Adjusted EBITDA margin",
        "R&D",
        "Cost of goods sold",
        "Sales & Marketing"
    ]

    for column in numeric_columns:
        df[column] = clean_numeric_column(df[column])

    df.rename(columns={"Gross Margin": "Gross profit"}, inplace=True)
    df["Gross Margin (%)"] = (df["Gross profit"] / df["Total revenue"]) * 100

    df = df.reset_index(drop=True)

    run_pelt_for_all_variables(df, TMV_ANALYSIS_THESIS_VARS,8, model="rbf", penalty=3.0, min_size=4)

    if run_full:
        run_zivot_andrews(df[8:][:], TMV_ANALYSIS_THESIS_VARS)

if __name__ == '__main__':
    run_for_sap(run_full=True)
    #run_for_teamviewer()

