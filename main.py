import pandas as pd
from utility import import_german_csv, run_pelt_for_all_variables
from utility.run_helpers import run_its_on_post_ai_event, SAP_ANALYSIS_THESIS_VARS, TMV_ANALYSIS_THESIS_VARS, \
    calculate_change, export_dataframe_to_csv

teamviewer_file_path = "TeamViewer_Long_Format_Quarterly_Dataset.csv"
sap_file_path = "SAP_Long_Format_Quarterly_Dataset.csv"

def run_for_sap(run_sbt= False, run_its=False):
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

    for col in [
        "total revenue",
        "cloud revenue",
        "software license revenue",
        "Gross profit",
        "Operating profit",
        "R&D",
        "Cost of goods sold",
        "Sales and Marketing"
    ] :
        qoq_col = "QoQ "+col
        df[qoq_col] = calculate_change(df, col)
        yoy_col = "YoY "+col
        df[yoy_col] = calculate_change(df, col, yoy=True)

    df = df.reset_index(drop=True)

    if run_sbt:
        print("Running PELT for SAP...")
        pelt_results = run_pelt_for_all_variables(df, SAP_ANALYSIS_THESIS_VARS, 12, min_size=4)
        export_dataframe_to_csv(pelt_results, filename="sap_pelt_test.csv")
        return pelt_results

    if run_its:
        print("Running ITS for SAP...")
        result = run_its_on_post_ai_event(df[12:-1][:], company="sap")
        return result
    return None


def run_for_teamviewer(run_sbt= False, run_its=False):
    full_df: pd.DataFrame = import_german_csv(teamviewer_file_path)
    expected_cols = ["Quarter", "Total revenue", "Enterprise revenue", "SMB revenue", "ARR", "NRR (%)", "Gross Margin",
                     "Adjusted EBITDA", "Adjusted EBITDA margin", "R&D", "Cost of goods sold", "Sales & Marketing"]

    df: pd.DataFrame = full_df[expected_cols]

    df.rename(columns={"Gross Margin": "Gross profit"}, inplace=True)
    df["Gross Margin (%)"] = (df["Gross profit"] / df["Total revenue"]) * 100

    for col in [
        "Total revenue",
        "NRR (%)",
        "Gross profit",
        "Gross Margin (%)",
        "Adjusted EBITDA",
        "Adjusted EBITDA margin",
        "R&D",
        "Cost of goods sold",
        "Sales & Marketing"
    ]:
        qoq_col = "QoQ "+col
        df[qoq_col] = calculate_change(df, col)
        yoy_col = "YoY "+col
        df[yoy_col] = calculate_change(df, col, yoy=True)

    df = df.reset_index(drop=True)
    if run_sbt:
        print("Running PELT for TeamViewer...")
        pelt_results = run_pelt_for_all_variables(df, TMV_ANALYSIS_THESIS_VARS, 8, min_size=4)
        export_dataframe_to_csv(pelt_results, filename="teamviewer_pelt_test.csv")
        return pelt_results
    if run_its:
        print("Running ITS for TeamViewer...")
        result = run_its_on_post_ai_event(df[8:-1][:], company="tmv")
        return result

    return None

def run(sbt_mode: bool, its_mode: bool):
    tmv_pelt_results = run_for_teamviewer(run_sbt=sbt_mode, run_its=its_mode)
    sap_pelt_results = run_for_sap(run_sbt=sbt_mode, run_its=its_mode)
    #run_ai_comparator([tmv_pelt_results, sap_pelt_results])

if __name__ == '__main__':
    run(sbt_mode=True, its_mode=False)

