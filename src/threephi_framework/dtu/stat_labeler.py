import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from threephi_framework.data_extractor.data_extractor import DataExtractor
from threephi_framework.object_storage.s3_connector import S3Connector
from threephi_framework.controllers.time_series import TimeSeriesController
from threephi_framework.controllers.meta import MetaController
import threephi_framework.db.db as threephi_db



def _preprocess_smart_meters(sm_df, sm_id):

    # Convert the "timestamp" column to index
    sm_df.index = pd.to_datetime(sm_df['timestamp'])  # Ensure the index is datetime

    # Keep only the active power columns
    data = sm_df[[col for col in sm_df.columns if 'active_power_p14' in col]]

    # Remove 'active_power_p14' from column names but keep the columns
    data.columns = [col.replace('active_power_p14_', '') for col in data.columns]
    
    # remove columns with zero or very low consumption
    min_std_threshold = 1e-8
    data = data.loc[:, data.std() > min_std_threshold]

    # Convert index to UTC timezone
    data.index = pd.to_datetime(data.index, utc=True)

    # Remove periods with zero consumption across all phases (likely missing data)
    data = data[(data != 0).any(axis=1)]
    
    # Remove periods of sustained zero consumption longer than 1 weeks (assuming: 7 days * 4 recordings per hour * 24 = 672)
    zero_consumption_mask = (data == 0).all(axis=1)
    zero_consumption_groups = (zero_consumption_mask != zero_consumption_mask.shift()).cumsum()
    zero_consumption_durations = zero_consumption_mask.groupby(zero_consumption_groups).transform('sum')
    data = data[~((zero_consumption_mask) & (zero_consumption_durations >= 672))]

    # Fill remaining NaNs with zeros
    data = data.fillna(0)

    # Resample to hourly data
    sm_df = sm_df.resample("1h").mean(numeric_only=True)

    # Remove periods with zero consumption across all phases (likely missing data)
    sm_df = sm_df[(sm_df[sm_df.columns[0]] != 0) | (sm_df[sm_df.columns[1]] != 0) | (sm_df[sm_df.columns[2]] != 0)]

    return data

# Method to plot the smart meter data
def _plot_sm(sm_id, sm_df, savedir, filename=None):
    plt.clf()  # Clear any previous plot

    for col in sm_df.columns:  # Plot the different active power columns
        plt.plot(sm_df.index, sm_df[col], label=col)

    # Add labels and title and save
    plt.xlabel("Time")
    plt.ylabel("Active Power P14")
    plt.title(f"Smart Meter {sm_id} Active power" + f"( {filename})" if filename else "")
    plt.legend()
    plt.savefig(os.path.join(savedir, f"sm_{sm_id}" + f"_{filename}.png" if filename else ".png"))


# Method to get the base load and temperature bins
def _get_base_load_and_temperature_bins(sm_df, cfg):
    # Assume the base load is the minimum load of a specific temperature bin
    # Find this temperature bin for each phase
    max_bins = cfg["thresholds"]["max_bins"]
    min_bins = cfg["thresholds"]["min_bins"]
    n_bins = max_bins
    valid_bins_found = False

    phase_cols = [col for col in sm_df.columns if col.startswith('l')]  

    # Find the temperature bins such that all 24 hours of the day are represented in the lowest temperature bin
    while n_bins > min_bins and not valid_bins_found:
        try:
            # Create temperature bins
            sm_df["T_bin"] = pd.qcut(sm_df["Temperature"], n_bins, labels=range(n_bins))

            tmin_list = []
            valid_bins_found = True  # assume success until proven otherwise

            for phase_col in phase_cols:
                grouped_means = sm_df[phase_col].groupby(sm_df["T_bin"], observed=False).mean()
                tmin = grouped_means.idxmin()

                # Check all 24 hours exist in this tmin bin
                missing_hours = [
                    hours
                    for hours in range(24)
                    if sm_df[(sm_df["T_bin"] == tmin) & (sm_df.index.hour == hours)][phase_col].dropna().empty
                ]

                if missing_hours:  # If any hour is missing, this bin is not valid
                    valid_bins_found = False
                    break  # stop checking phases, try fewer bins

                tmin_list.append(tmin)

            if not valid_bins_found:
                n_bins -= 1  # Retry with fewer bins

        except Exception:
            n_bins -= 1
            continue

    if not valid_bins_found:
        logging.warning(f"Could not find a fully populated lowest temperature bin; using {n_bins} bins as fallback.")
        sm_df["T_bin"] = pd.qcut(sm_df["Temperature"], n_bins, labels=range(n_bins), duplicates="drop")
        tmin_list = [sm_df["T_bin"].value_counts().idxmax()] * len(phase_cols)  # If no valid bin found, use the most common bin as fallback

    # Create a base load profile, based on consumption at different hours of the day at the specific temperature bin
    load_base_list = []
    for phase_col, tmin in zip(phase_cols, tmin_list):
        loadb = (
            pd.Series(
                # Sample the base load for each hour and specific temperature bin
                [
                    sm_df[(sm_df["T_bin"] == tmin) & (sm_df.index.hour == hour)][phase_col].sample().values[0]
                    if not sm_df[(sm_df["T_bin"] == tmin) & (sm_df.index.hour == hour)].empty
                    else np.nan
                    for hour in sm_df.index.hour
                ],
                index=sm_df.index,
                name="load_base_" + phase_col,
            )
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )
        load_base_list.append(loadb)

    return load_base_list, tmin_list, n_bins


# Method to get the thermal priors
def _get_priors(dfl, dfb, nsamples=1000, maxstep=4000, step=40, tmin=None, nT=12):
    # Get thermal priors by sampling parameters of a truncated normal distribution
    mu_prior = np.zeros((24, nT))
    sd_prior = np.zeros((24, nT))

    selected_mu = np.array([])
    selected_sd = np.array([])

    for hour in range(24):
        for t_bin in range(nT):
            load = dfl[(dfl.index.hour == hour) & (dfb == t_bin)]
            loadb = dfl[(dfl.index.hour == hour) & (dfb == tmin)]

            threshold_index = int(nsamples / 100)
            counter = 0

            for _ in range(1000): 
                mu = np.random.uniform(0, maxstep, nsamples)
                sd = np.random.uniform(0, maxstep/4, nsamples)
                distance = np.zeros(nsamples)

                total_load, _ = np.histogram(load, bins=step, range=(0, maxstep))
                total_load = (total_load + 1.e-6) / (total_load + 1.e-6).sum() # avoid NaNs
                    
                for i in range(nsamples):
                    samples_base = np.random.choice(loadb, 1000)
                    lim_a, lim_b = (0 - mu[i]) / sd[i], (maxstep - mu[i]) / sd[i]
                    samples_thermal = truncnorm.rvs(lim_a, lim_b, loc=mu[i], scale=sd[i], size=1000)
                    
                    # Approximate the total load, by sampling from the base load and thermal load
                    approx_load, _ = np.histogram(samples_base + samples_thermal, bins=step, range=(0, maxstep))
                    approx_load = (approx_load + 1.e-6) / (approx_load + 1.e-6).sum() # avoid NaNs

                    distance[i] = (total_load * (np.log(total_load) - np.log(approx_load))).sum()

                # Select the mu and sd values that give the lowest distance
                threshold = np.sort(distance)[threshold_index]
                selected_mu = mu[distance < threshold]
                selected_sd = sd[distance < threshold]
    
                # Increase the threshold, if we cannot find any valid mu and sd values
                # Try twice for the same threshold, before increasing the threshold, to avoid increasing the threshold too quickly in case of outliers
                if counter > 2:
                    threshold_index += 1
                    counter = 0
                    continue

                if selected_mu.size == 0 or selected_sd.size == 0:
                    counter += 1
                    if counter > 2:
                        break
                    continue

                if not np.isnan(selected_mu.mean()) and not np.isnan(selected_sd.mean()):
                    break
                
                # Ensure that no more than 100 iterations are done
                threshold_index += 1
                if threshold_index >= nsamples:
                    raise ValueError("Unable to find a valid threshold without NaNs.")

                
            if selected_mu.size == 0 or selected_sd.size == 0:
                if len(load) > 0:
                    mu_prior[hour, t_bin] = np.mean(load)
                    sd_prior[hour, t_bin] = max(np.std(load), 1e-3)
                elif len(loadb) > 0:
                    mu_prior[hour, t_bin] = np.mean(loadb)
                    sd_prior[hour, t_bin] = max(np.std(loadb), 1e-3)
                else:
                    mu_prior[hour, t_bin] = 0
                    sd_prior[hour, t_bin] = 1
                continue

            
            mu_prior[hour, t_bin] = selected_mu.mean()
            sd_prior[hour, t_bin] = selected_sd.mean()

    # From the mu and sd parameters, create a list of n_maxstep values for the truncated
    # normal distribution for each hour and temperature bin
    thermal_prior = np.zeros((step, 24, nT))
    for hour in range(24):
        for t_bin in range(nT):
            for a, xa in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                lim_a, lim_b = (
                    (0 - mu_prior[hour, t_bin]) / sd_prior[hour, t_bin],
                    (maxstep - mu_prior[hour, t_bin]) / sd_prior[hour, t_bin],
                )
                thermal_prior[a, hour, t_bin] = truncnorm(
                    lim_a, lim_b, loc=mu_prior[hour, t_bin], scale=sd_prior[hour, t_bin]
                ).pdf(xa)

    # Normalize the thermal prior distributions
    thermal_prior += 1.0e-6
    thermal_prior /= thermal_prior.sum(axis=0)

    return thermal_prior


# Method to get the posterior distribution
def _get_posterior(prior, phase_col, maxstep, step, sm_df, tmin, nT=12):
    # Set up the likelihood, by counting the occurrences of the load at different hours of the day and temperature bins
    likelihood = np.zeros((step, step, 24, nT))
    for hour in range(24):
        hour_loadb, _ = np.histogram(
            sm_df[phase_col][(sm_df.index.hour == hour) & (sm_df["T_bin"] == tmin)],
            bins=step,
            range=(0, maxstep),
        )
        for t_bin in range(nT):
            for a, xa in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                for b, xb in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                    likelihood[b, a, hour, t_bin] = hour_loadb[np.clip(int((xb - xa) / 100), 0, step - 1)]

    # Set up the evidence by marginalizing through all combinations of likelihood and prior
    evidence = np.zeros((step, 24, nT))
    for hour in range(24):
        for t_bin in range(nT):
            for b, _ in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                evidence[b, hour, t_bin] = (likelihood[b, :, hour, t_bin] * prior[:, hour, t_bin]).sum()

    # Set up the posterior using Bayes' theorem
    posterior = np.zeros((step, step, 24, nT))
    for hour in range(24):
        for t_bin in range(nT):
            for a, _ in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                for b, _ in enumerate(np.linspace(0, maxstep, step, endpoint=False)):
                    posterior[a, b, hour, t_bin] = (
                        likelihood[b, a, hour, t_bin] * prior[a, hour, t_bin] / evidence[b, hour, t_bin]
                        if evidence[b, hour, t_bin] > 0
                        else 0
                    )

    # Normalize the posterior distributions
    posterior += 1.0e-6
    posterior /= posterior.sum(axis=0)

    return posterior


# Method to get the thermal prior distributions
def _get_thermal_prior(sm_df, tmin_list, n_bins):
    # Get the thermal prior distribution for each phase
    # Divide the load into steps of 100, up to the maximum load rounded up to the nearest 1000
    thermal_prior_list = []
    maxstep_list = []  # maximum load
    step_list = []  # step size

    print("Sm to _get_thermal_prior")

    phase_cols = [col for col in sm_df.columns if col.startswith('l')]

    for phase_col, tmin in zip(phase_cols, tmin_list):
        # If the sum of one phase is less than 1% of the sum of the two other phases, then we regard it as dead
        sum_phase = sm_df[phase_col].sum()
        other_phases = [p for p in phase_cols if p != phase_col]

        if sum_phase == 0 or all(sum_phase < 0.01 * sm_df[p].sum(axis=0) for p in other_phases):
            maxstep_list.append(0)
            step_list.append(0)
            thermal_prior_list.append(np.zeros((100, 24, 12)))
            continue

        # Find the maxstep and step
        maxstep = int(np.ceil(sm_df[phase_col].max() / 1000) * 1000)
        maxstep_list.append(maxstep)
        step = min(100, int(maxstep / 100))
        step_list.append(step)

        # Get a array of normalized thermal prior distributions for each hour of the day and temperature bin
        thermal_prior = _get_priors(
            sm_df[phase_col],
            sm_df["T_bin"],
            nsamples=100,
            maxstep=maxstep,
            step=step,
            tmin=tmin,
            nT=n_bins,
        )
        thermal_prior_list.append(thermal_prior)  # Add them to the thermmal prior list, one for each phase

    return thermal_prior_list, maxstep_list, step_list


# Method to get the thermal posterior distributions and sample thermal loads
def _get_thermal_posterior(thermal_prior_list, maxstep_list, step_list, sm_df, tmin_list, n_bins):
    # Get the thermal posteriors
    thermal_posterior_list = []
    phase_cols = [col for col in sm_df.columns if col.startswith('l')]
    for phase_col, tmin, thermal_prior, maxstep, step in zip(phase_cols, tmin_list, thermal_prior_list, maxstep_list, step_list):
        # maxstep = maxstep_list[phase]
        # step = step_list[phase]

        # If the phase is dead, then append a zero array
        if maxstep == 0:
            thermal_posterior_list.append(np.zeros((100, 24, n_bins)))
        else:
            # thermal_prior = thermal_prior_list[phase]
            thermal_posterior = _get_posterior(thermal_prior, phase_col, maxstep, step, sm_df, tmin, n_bins)
            thermal_posterior_list.append(thermal_posterior)

    # Sample a posterior thermal load for each phase
    post_thermal_load = []
    for phase_col, thermal_posterior, maxstep, step in zip(phase_cols, thermal_posterior_list, maxstep_list, step_list):
        # step = step_list[phase]
        # maxstep = maxstep_list[phase]

        if maxstep == 0:
            # Append a Series of zeros
            post_thermal_load.append(pd.Series(np.zeros(len(sm_df)), index=sm_df.index))
        else:
            # thermal_posterior = thermal_posterior_list[phase]
            series = pd.Series(
                [
                    np.random.choice(100 * np.arange(0, step), p=thermal_posterior[:, load, hour, t_bin])
                    for load, hour, t_bin in zip(
                        (sm_df[phase_col] / 100).astype(int).clip(0, step - 1),
                        sm_df.index.hour,
                        sm_df["T_bin"],
                        strict=False,
                    )
                ],
                index=sm_df.index,
            )
            post_thermal_load.append(series)

    return post_thermal_load

def _fit_BIC(y: np.ndarray, X: np.ndarray, compute_k_conf: bool = False) -> float:
        """
        Assumes the errors are i.i.d. This formula is used:
            BIC = n*log(RSS/n) + k*log(n)
        where:
            n = number of observations
            k = number of parameters (columns of X)
            RSS = sum of squared residuals from OLS fit

        Returns +inf if the fit is not well-defined.
        """
        try:
            # Ensure 1D y and 2D X
            y = np.asarray(y).reshape(-1)
            X = np.asarray(X)

            n = y.shape[0]
            if n < 10 or X.ndim != 2 or X.shape[0] != n:
                return np.inf

            k = X.shape[1]
            if k <= 0:
                return np.inf

            # Solve least squares: minimize ||X b - y||_2
            beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)

            # If rank deficient, the model is not identifiable -> treat as invalid
            if rank < k:
                return np.inf

            # Compute RSS (residual sum of squares)
            if residuals.size > 0:
                rss = float(residuals[0])
            else:
                # residuals can be empty in some cases. Then, compute explicitly
                y_hat = X @ beta
                err = y - y_hat
                rss = float(err @ err)

            # Guard against log(0) / negative numerical issues
            rss = max(rss, 1e-12)

            # Compute BIC
            bic = n * np.log(rss / n) + k * np.log(n)

            # If not computing confidence intervals for k, return BIC only
            if not compute_k_conf:
                return float(bic)
            else:
                df = n - k
                if df <= 0:
                    return bic, beta.astype(float), None
                sigma2_hat = rss / df
                XtX = X.T @ X
                XtX_inv = np.linalg.inv(XtX)
                var_beta = sigma2_hat * XtX_inv
                se = np.sqrt(np.diag(var_beta))

                return bic, beta.astype(float), se.astype(float)

        except Exception:
            return np.inf if not compute_k_conf else (np.inf, None, None)


def _BIC_to_model_probs(bics: dict, priors: dict) -> dict:

    """
    Convert BICs into approximate posterior model probabilities:
    p(Mi|y) ∝ prior_i * exp(-0.5 * (BIC_i -  min(BICs) ))

    where:
        Mi is the i-th model
        y is the posterior thermal load data
        prior_i is the prior probability of model Mi, assumed to be 0.5 unless specified otherwise
        BIC_i is the Bayesian Information Criterion of model Mi
        BICs is the set of BICs for all models being compared

    """

    # Filter out non-finite BICs and models with zero prior
    finite = {k: v for k, v in bics.items() if np.isfinite(v) and priors.get(k, 0.0) > 0}
    
    # If no finite BICs, return priors normalized
    if not finite:
        tot = sum(priors.values())
        return {k: (priors[k] / tot if tot > 0 else 0.0) for k in priors}

    # If some finite BICs, compute posterior probs
    bmin = min(finite.values()) # find minimum BIC
    weights = {k: float(priors[k]) * np.exp(-0.5 * (bics[k] - bmin)) for k in finite}  # Compute posterior weights
    Z = sum(weights.values()) # normalization constant
    return {k: (weights.get(k, 0.0) / Z if Z > 0 else 0.0) for k in priors}

def _compute_TDEL_confidence_BIC(
    sm_df: pd.DataFrame,
    post_thermal_load,
    temp_col: str = "Temperature",
    T_balance: float = 15.0,
    model_priors: dict | None = None,
    min_timestamps: int = 1400,
):
    """
    INFO:
        Compute temperature dependent electric load (TDEL) confidence per phase using BIC-based model comparison on
        the posterior thermal load.

        3 models are compared:
        M0: y_t ≈ x̄ + ϵ_t 
            Null model (intercept only). 
            Load can be described by a constant mean + noise, meaning there is no dependency on temperature or schedule.

        M1: y_t ≈ x̄ + k * max(0, T_balance - T_t) + ϵ_t 
            Heating-driven model (intercept + heating degree hours proxy). 
            Load can be described by a constant mean + relation to heating degree hours + noise, meaning it is dependent on outside temperature.
            The max function implies that y_t ≈ x̄ in warm periods (e.g. over the balance temperature)
            
        M2: y_t ≈ x̄ + ξ_t + ϵ_t 
            Schedule-driven model (intercept + hour-of-day dummies).
            Load can be described by a constant mean + daily schedule + noise, meaning it is dependent on time of day but not temperature.

    Caveats:
        - Model incompleteness. Models may not capture all relevant factors affecting load.
        - Low sample size results in poor confidence estimates. Therefore a minimum number of 2 months worth of 
            1-hour data (1400 timestamps) is required to perform the analysis; if not met, confidence is set to 0.0.
        - Large sample size results in overconfident estimates, as even small effects become statistically significant.
        - Data is approximated by ABC. This creates biases in the load estimates, which may affect confidence estimates.
        
    Args:
        sm_df: pd.DataFrame with datetime index, temperature column
        post_thermal_load: list of pd.Series of approximated thermal load per phase
        temp_col: str name of temperature column in sm_df
        T_balance: float balance temperature for heating degree hours proxy
        model_priors: dict with prior probabilities for models M0, M1, M2; if None, default priors are used
        min_timestamps: int minimum number of timestamps required to perform the analysis; if not met, confidence is 0.0

    Returns:
        phase_conf: list[float] length 3
        meta: dict with BICs and posterior model probs
    """
    
    # Initial guess: Each model have an equal probability of explaining the data, before seeing the data.
    if model_priors is None:
        model_priors = {"M0": 0.33, "M1": 0.33, "M2": 0.34}

    # Normalize priors
    s = sum(model_priors.values())
    if s > 0:
        model_priors = {k: v / s for k, v in model_priors.items()}

    # Define the temperature variable for the analysis
    T = sm_df[temp_col]
    
    # Define the heating degree hours (hdh) variable
    hdh = np.maximum(0.0, T_balance - T.values)

    # Identify phase columns
    phase_cols = [col for col in sm_df.columns if col.startswith('l')]

    # Define confidence list and meta dict
    phase_conf = []
    meta = {
        "T_balance": float(T_balance),
        "min_timestamps": int(min_timestamps),
        "per_phase": []
        }

    for phase_col, y_series in zip(phase_cols, post_thermal_load):
        # y_series = post_thermal_load[phase]
        # Align and drop NaNs consistently
        df = pd.DataFrame({"y": y_series, "Temperature": T, "hdh": hdh, "hour": sm_df.index.hour}, index=sm_df.index).dropna()

        # Data must at least be worth over 2 months of 15-min load data, before meaningful results can be expected
        if len(df) < min_timestamps: 
            phase_conf.append(0.0)
            meta["per_phase"].append(
                {
                    "phase": phase_col,
                    "n": len(df),
                    "reason": f"n timestamps under threshold of {min_timestamps}",
                    "bics": {"M0": None, "M1": None, "M2": None},
                    "probs": {"M0": None, "M1": None, "M2": None},
                }
            )
            continue

        y = df["y"].values

        # M0: mean only
        X0 = np.ones((len(df), 1))

        # M1: mean + HDH
        X1 = np.column_stack([np.ones(len(df)), df["hdh"].values])

        # M2: mean + hour-of-day schedule
        hour_schedule = pd.get_dummies(df['hour'].astype(int), drop_first=True)
        X2 = np.column_stack([np.ones(len(df)), hour_schedule.values])

        # Fit models and compute BICs
        BIC0 = _fit_BIC(y, X0)
        BIC1, beta, SE = _fit_BIC(y, X1, compute_k_conf=True)
        BIC2 = _fit_BIC(y, X2)

        bics = {"M0": BIC0, "M1": BIC1, "M2": BIC2}

        # Determine k_hat and its confidence for M1
        if beta is not None and SE is not None and len(beta) >= 2 and np.isfinite(SE[1]):
            k_hat = float(beta[1])
            k_se  = float(SE[1])
            k_ci95 = (float(k_hat - 1.96 * k_se), float(k_hat + 1.96 * k_se))
        else:
            k_hat = None
            k_se = None
            k_ci95 = (None, None)

        # Convert BICs -> approximate posterior over models
        priors = model_priors.copy()
        probs = _BIC_to_model_probs(bics, priors)

        # Confidence in load has temperature dependency = P(M1 | data)
        conf = float(probs.get("M1", 0.0))
        phase_conf.append(conf)

        bics_rounded = {k: float(round(v, 2)) for k, v in bics.items()}
        probs_rounded = {k: float(round(v, 6)) for k, v in probs.items()}

        meta["per_phase"].append(
            {
                "phase": phase_col,
                "n": len(df),
                "bics": bics_rounded,
                "probs": probs_rounded,
                "k_hat": k_hat,
                "k_se": k_se,
                "k_ci95": k_ci95
            }
        )

    return phase_conf, meta

# Method to calculate the MAE and relative MAE
def _calculate_mae(sm_df, post_thermal_load, loadb_list):

    phase_cols = [col for col in sm_df.columns if col.startswith('l')]
    # Compute the MAE and relative MAE between the post_thermal_load and the base load
    total_load = [post + loadb for post, loadb in zip(post_thermal_load, loadb_list)]

    mae_list = []
    maer_list = []
    for phase_col, total in zip(phase_cols, total_load):
        mae_list.append(mean_absolute_error(total, sm_df[phase_col]))
        maer_list.append(mean_absolute_error(total, sm_df[phase_col]) / sm_df[phase_col].mean())

    return mae_list, maer_list


def label_meters(sm_ids, sm_with_hp, cfg):
    try:
        data_extractor = DataExtractor(phase_measurements_dir=cfg["data_dir_path"])
        s3_connector = S3Connector(data_dir_path="phase_measurements")
        timeseries_controller = TimeSeriesController(s3_connector)
        meta_controller = MetaController(threephi_db.new_session)

        # Load temperature data
        weather_df = data_extractor.s3_connector.read_small_csv(data_extractor.s3_base + cfg["temp_data_path"], dtype={"Temperature": float})
        weather_df.index = pd.to_datetime(weather_df.iloc[:, 0], format="mixed", dayfirst=True, utc=True)
        weather_df.index.name = "DateTime"
        weather_df.drop(weather_df.columns[0], axis=1, inplace=True)
        weather_df.iloc[:, 0] = pd.to_numeric(weather_df.iloc[:, 0], errors='coerce')
        weather_df = weather_df.interpolate(method="time").bfill().ffill()
        weather_df = weather_df.reset_index()
        weather_df["DateTime"] = weather_df["DateTime"].astype("datetime64[us, UTC]")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Iterate per-phase labeling for each smart meter
    for sm_id in tqdm(sm_ids, desc="Per-phase temperature-dependent load labeling on smart meters"):
        logging.info(f"Processing smart meter {sm_id}")

        try:
            # If the sm_id does not have a heat pump, skip it
            if cfg["process_only_sm_with_hp"] and sm_id not in sm_with_hp:
                logging.info(f"Smart meter {sm_id} does not have a heat pump. Skipped.")
                continue

            # Start workflow for sm_id
            meta_controller.start_workflow(workflow=f"stat_labeling_sm_{sm_id}")

            # Load the cleaned sm data
            sm_df = timeseries_controller.get_time_series_data(meter_ids = [str(sm_id)])
            sm_df = sm_df.compute()
            sm_df = _preprocess_smart_meters(sm_df, sm_id)

            print("Sm to _preprocess_smart_meters")

            phase_cols = [col for col in sm_df.columns if col.startswith('l')]
            sm_df = sm_df[phase_cols]

            sm_df = sm_df.reset_index()
            sm_df["timestamp"] = sm_df["timestamp"].astype("datetime64[us, UTC]")

            # Join weather data
            sm_df = pd.merge_asof(
                sm_df.sort_values("timestamp"),
                weather_df.sort_values("DateTime").rename(columns={"DateTime": "timestamp"}),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("1h")
            )
            sm_df = sm_df.set_index("timestamp")
            sm_df = sm_df[[col for col in sm_df.columns if col.startswith('l')] + ["Temperature"]]

            # Get the base load and the lowest consumpting temperature bin for each phase
            loadb_list, tmin_list, n_bins = _get_base_load_and_temperature_bins(sm_df, cfg)

            print("Sm to _get_base_load_and_temperature_bins")

            sm_df["T_bin"] = pd.qcut(sm_df["Temperature"], n_bins, labels=range(n_bins)).astype(int)

            # Find the thermal priors for each phase
            thermal_prior_list, maxstep_list, step_list = _get_thermal_prior(sm_df, tmin_list, n_bins)

            # Find the thermal posterior for each phase, and sample a thermal load
            post_thermal_load = _get_thermal_posterior(
                thermal_prior_list, maxstep_list, step_list, sm_df, tmin_list, n_bins
            )

            print("Sm to _get_thermal_posterior")

            # Compute the TDEL confidence and meta information using BIC-based model comparison
            phase_labels, meta = _compute_TDEL_confidence_BIC(
                sm_df=sm_df, post_thermal_load=post_thermal_load, temp_col="Temperature", T_balance=15.0, min_timestamps=1400)

            print("Sm to _compute_TDEL_confidence_BIC")

            # Add meta results
            phase_cols = [col for col in sm_df.columns if col.startswith('l')]

            sm_phase_label = {phase_col: {"label": cfg["thresholds"]["confidence_threshold"] < round(float(label), 4),
                                          "confidence": round(float(label), 4)} 
                                          for phase_col, label in zip(phase_cols, phase_labels)}
            
            if cfg["add_meta_results"]:
                meta_dict = {"TDEL_info": [], "TDEL_meta": [], "MAE": [], "MAEr": [], "n_t_bins": ""}
                meta_dict["TDEL_info"].append(sm_phase_label)
                meta_dict["TDEL_meta"].append(meta)
                meta_dict["n_t_bins"] = n_bins
                
                # Calculate MAE and MAEr
                if loadb_list is not None and cfg["include_mae"]:
                    mae_list, maer_list = _calculate_mae(sm_df, post_thermal_load, loadb_list)
                    meta_dict["MAE"].extend(f"{phase}: {mae_list[i]:.4f}" for i, phase in enumerate(phase_cols))
                    meta_dict["MAEr"].extend(f"{phase}: {maer_list[i]:.4%}" for i, phase in enumerate(phase_cols))
            
            print(f"Smart meter {sm_id} phase labels: {sm_phase_label}")

            for phase_col, phase_data in sm_phase_label.items():
                meta_controller.insert_run_result(
                    dag_id=cfg.get("dag_id", "default_dag"),
                    run_id=cfg.get("run_id", "default_run"),
                    meter_id=int(sm_id),
                    phase=str.capitalize(phase_col),
                    label_type="Electric heating",
                    label_value=str(phase_data["label"]),
                    confidence=phase_data["confidence"],
                    topology_version=None,
                    result=meta_dict if cfg["add_meta_results"] else None,
                    source="StatLabeler",
                    node_id=None,
                    edge_id=None,
                    cable_id=None,
                )

            
            # # TODO: Return results and insert in DAG
            # # Add results to meta.run_results schema
            # for phase_col, phase_data in sm_phase_label.items():
            #     # dag_id=cfg.get("dag_id", "default_dag"),
            #     # run_id=cfg.get("run_id", "default_run"),
            #     meta_controller.insert_run_result_current_run(
            #         meter_id=int(sm_id),
            #         phase=str.capitalize(phase_col),
            #         label_type="Electric heating",
            #         label_value=str(phase_data["label"]),
            #         confidence=phase_data["confidence"],
            #         topology_version=None,
            #         result=meta_dict if cfg["add_meta_results"] else None,
            #         source="StatLabeler",
            #         node_id=None,
            #         edge_id=None,
            #         cable_id=None,
            #     )
            
            meta_controller.complete_workflow(workflow=f"stat_labeling_sm_{sm_id}")


        except Exception as e:
            logging.error(f"Error processing smart meter {sm_id}: {e}")
            continue
