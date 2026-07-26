from pathlib import Path

import pandas as pd

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