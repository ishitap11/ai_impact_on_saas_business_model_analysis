import pandas as pd
from utility import import_german_csv, run_chow_test, run_pelt_for_all_variables

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

def run_for_sap(run_full= False):
    full_df: pd.DataFrame = import_german_csv(sap_file_path)
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

    df: pd.DataFrame = full_df[expected_cols]

    df = df.reset_index(drop=True)

    print("Running PELT for SAP...")
    run_pelt_for_all_variables(df, SAP_ANALYSIS_THESIS_VARS, 12, min_size=4)

    if run_full:
        print("Running Chow Test for SAP...")
        display_columns = [
            "Variable",
            "Pre-break observations",
            "Post-break observations",
            "Chow F-statistic",
            "Critical value",
            "p-value",
            "Structural break at 5%"
        ]
        result = pd.DataFrame(run_chow_test(df[12:][:], SAP_ANALYSIS_THESIS_VARS, "Q1-24"))
        print(result[display_columns].to_string(index=False))


def run_for_teamviewer(run_full= False):
    full_df: pd.DataFrame = import_german_csv(teamviewer_file_path)
    expected_cols = ["Quarter", "Total revenue", "Enterprise revenue", "SMB revenue", "ARR", "NRR (%)", "Gross Margin",
                     "Adjusted EBITDA", "Adjusted EBITDA margin", "R&D", "Cost of goods sold", "Sales & Marketing"]

    df: pd.DataFrame = full_df[expected_cols]

    df.rename(columns={"Gross Margin": "Gross profit"}, inplace=True)
    df["Gross Margin (%)"] = (df["Gross profit"] / df["Total revenue"]) * 100

    df = df.reset_index(drop=True)

    print("Running PELT for TeamViewer...")
    run_pelt_for_all_variables(df, TMV_ANALYSIS_THESIS_VARS,8, min_size=4)

    if run_full:
        print("Running Chow Test for TeamViewer...")
        display_columns = [
            "Variable",
            "Pre-break observations",
            "Post-break observations",
            "Chow F-statistic",
            "Critical value",
            "p-value",
            "Structural break at 5%"
        ]
        result = pd.DataFrame(run_chow_test(df[8:][:], TMV_ANALYSIS_THESIS_VARS, "Q4-24"))
        print(result[display_columns].to_string(index=False))

def run(full_mode: bool):
    run_for_teamviewer(run_full=full_mode)
    run_for_sap(run_full=full_mode)

if __name__ == '__main__':
    run(full_mode=True)

