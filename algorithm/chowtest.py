import numpy as np
import statsmodels.api as sm

from scipy.stats import f
from statsmodels.tsa.stattools import adfuller


class ChowTest:

    def run(
        data,
        dependent_variable,
        breakpoint,
        difference_nonstationary=False,
        alpha=0.05
    ):
        """
        Chow test for a change in both intercept and linear time trend.

        Parameters
        ----------
        data : pandas.DataFrame
            Must contain Quarter and the dependent variable.
        dependent_variable : str
            Column to test.
        breakpoint : str
            First observation in the post-break period.
        difference_nonstationary : bool
            If True, replicate the article's ADF/differencing approach.
            If False, test the original metric in levels.
        alpha : float
            Significance level.

        Returns
        -------
        results : dict
        plotting_data : pandas.DataFrame
        """

        test_data = (
            data[["Quarter", dependent_variable]]
            .dropna()
            .reset_index(drop=True)
            .copy()
        )

        if breakpoint not in test_data["Quarter"].values:
            raise ValueError(
                f"Breakpoint {breakpoint!r} was not found in the Quarter column."
            )

        # -----------------------------------------------------
        # ADF stationarity test
        # -----------------------------------------------------

        adf_result = adfuller(
            test_data[dependent_variable],
            autolag="AIC"
        )

        adf_statistic = adf_result[0]
        adf_p_value = adf_result[1]
        transformed = False

        if difference_nonstationary and adf_p_value > alpha:
            test_data[dependent_variable] = (
                test_data[dependent_variable].diff()
            )

            test_data = test_data.dropna().reset_index(drop=True)
            transformed = True

            # Confirm stationarity after differencing
            post_difference_adf = adfuller(
                test_data[dependent_variable],
                autolag="AIC"
            )

            transformed_adf_statistic = post_difference_adf[0]
            transformed_adf_p_value = post_difference_adf[1]
        else:
            transformed_adf_statistic = np.nan
            transformed_adf_p_value = np.nan

        if breakpoint not in test_data["Quarter"].values:
            raise ValueError(
                "The breakpoint disappeared after transformation."
            )

        # Sequential quarterly time index
        test_data["Time"] = np.arange(len(test_data))

        # breakpoint is included in the post-break sample
        breakpoint_index = test_data.index[
            test_data["Quarter"].eq(breakpoint)
        ][0]

        pre_break = test_data.iloc[:breakpoint_index].copy()
        post_break = test_data.iloc[breakpoint_index:].copy()

        # Intercept plus time coefficient = two parameters
        k = 2
        n = len(test_data)
        n1 = len(pre_break)
        n2 = len(post_break)

        if n1 <= k or n2 <= k:
            raise ValueError(
                f"Insufficient observations: pre={n1}, post={n2}. "
                f"Each segment needs more than {k} observations."
            )

        # -----------------------------------------------------
        # Regression models
        # -----------------------------------------------------

        X_full = sm.add_constant(test_data["Time"])
        X_pre = sm.add_constant(pre_break["Time"])
        X_post = sm.add_constant(post_break["Time"])

        model_full = sm.OLS(
            test_data[dependent_variable],
            X_full
        ).fit()

        model_pre = sm.OLS(
            pre_break[dependent_variable],
            X_pre
        ).fit()

        model_post = sm.OLS(
            post_break[dependent_variable],
            X_post
        ).fit()

        ssr_full = model_full.ssr
        ssr_pre = model_pre.ssr
        ssr_post = model_post.ssr
        ssr_unrestricted = ssr_pre + ssr_post

        denominator_df = n1 + n2 - 2 * k

        chow_statistic = ((ssr_full - ssr_unrestricted) / k) / (ssr_unrestricted / denominator_df)

        critical_value = f.ppf(
            1 - alpha,
            dfn=k,
            dfd=denominator_df
        )

        # Survival function is more numerically stable than 1 - f.cdf()
        p_value = f.sf(
            chow_statistic,
            dfn=k,
            dfd=denominator_df
        )

        test_data["Fitted value"] = np.nan
        test_data.loc[pre_break.index, "Fitted value"] = model_pre.predict(X_pre)
        test_data.loc[post_break.index, "Fitted value"] = model_post.predict(X_post)

        results = {
            "Variable": dependent_variable,
            "Series tested": (
                "First difference" if transformed else "Level"
            ),
            "Pre-break observations": n1,
            "Post-break observations": n2,
            "Chow F-statistic": chow_statistic,
            "Critical value": critical_value,
            "p-value": p_value,
            "Structural break at 5%": p_value < alpha
        }

        return results, test_data