"""
library used - ruptures
source code of library and documentation available at {link:https://github.com/deepcharles/ruptures}

Literature Vault code RP-51.
"""

import ruptures as rpt
import pandas as pd

class PELT:
    def run(
            data,
            variable,
            penalty=3.0,
            model="rbf",
            min_size=4
    ):
        """
        Run PELT on one variable.

        Parameters
        ----------
        data : DataFrame
            Must contain Quarter and the selected variable.
        variable : str
            Variable on which PELT is performed.
        penalty : float
            Penalty for adding a breakpoint. Higher values produce fewer breaks.
        model : str
            "l2" detects changes in mean.
            "rbf" detects more general distributional changes.
        min_size : int
            Minimum number of observations in each regime.
            min_size=4 requires approximately one year per regime.

        Returns
        -------
        result : DataFrame containing detected breaks
        analysis_data : cleaned observations
        breakpoint_indices : raw ruptures output, excluding endpoint
        """

        analysis_data = (
            data[[variable]]
            .dropna()
            .reset_index(drop=True)
            .copy()
        )

        signal = analysis_data[variable].to_numpy(dtype=float)

        algorithm = rpt.Pelt(
            model=model,
            min_size=min_size,
            jump=1
        ).fit(signal)

        # ruptures includes len(signal) as the final endpoint
        predicted_endpoints = algorithm.predict(pen=penalty)
        breakpoint_indices = [
            index for index in predicted_endpoints
            if index < len(signal)
        ]

        break_records = []

        for breakpoint in breakpoint_indices:
            # A breakpoint b separates rows b-1 and b
            break_records.append({
                "Variable": variable,
                "Model": model,
                "Penalty": penalty,
                "Breakpoint index": breakpoint,
                "Previous regime ends": analysis_data.loc[
                    breakpoint - 1, "Quarter"
                ],
                "breakpoint": analysis_data.loc[
                    breakpoint, "Quarter"
                ]
            })

        result = pd.DataFrame(break_records)

        return result, analysis_data, breakpoint_indices