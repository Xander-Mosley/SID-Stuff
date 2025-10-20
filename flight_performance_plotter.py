#!/usr/bin/env python3

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from IPython import get_ipython
from typing import Union


def _ensure_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

def _matrix_norm(matrix: np.ndarray) -> np.ndarray:
    matrix_mean = np.mean(matrix, axis=0)
    sjj = np.sum((matrix - matrix_mean) ** 2, axis=0)
    sjj_safe = np.where(sjj == 0, 1, sjj)
    matrix_norm = (matrix - matrix_mean) / np.sqrt(sjj_safe)
    return matrix_norm

def _extract_param_number(col: str) -> Union[int, float]:
    match = re.search(r'parameter_(\d+)', col)
    return int(match.group(1)) if match else float('inf')


def _sliding_mse(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        window_size: int = 6
        ) -> np.ndarray:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    sliding_mse = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            mse = np.nan
        else:
            mse = np.mean((true_window - pred_window) ** 2)
            
        sliding_mse.append(mse)
        
    return np.array(sliding_mse)

def _sliding_cod(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        window_size: int = 30
        ) -> np.ndarray:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
        
    sliding_cod = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            cod = np.nan
        else:
            ss_total = np.sum((true_window - np.mean(true_window)) ** 2)
            ss_res = np.sum((true_window - pred_window) ** 2)
            
            if ss_total == 0:
                cod = np.nan
            else:
                cod = 1 - (ss_res / ss_total)
                
        sliding_cod.append(cod)
        
    return np.array(sliding_cod)

def _sliding_adjusted_cod(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        num_predictors: int,
        window_size: int = 30
        ) -> np.ndarray:
    
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for adjusted R².")

    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    sliding_cod = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            cod = np.nan
        else:
            ss_total = np.sum((true_window - np.mean(true_window)) ** 2)
            ss_res = np.sum((true_window - pred_window) ** 2)
            
            if ss_total == 0 or (window_size - num_predictors - 1) <= 0:
                cod = np.nan
            else:
                r2 = 1 - (ss_res / ss_total)
                cod = 1 - (((1 - r2) * (window_size - 1)) / (window_size - num_predictors - 1))
                
        sliding_cod.append(cod)
    
    return np.array(sliding_cod)

def _sliding_confidence_intervals(
    true_values: np.ndarray | pd.Series | list,
    pred_values: np.ndarray | pd.Series | list,
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for confidence intervals.")

    output_ci_array = np.full(num_samples, np.nan)
    param_ci_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(true_window).any() or np.isnan(pred_window).any() or np.isnan(X).any():
            continue

        residuals = true_window - pred_window
        sigma_squared = np.sum(residuals ** 2) / (window_size - num_predictors)
        XtX_inv = np.linalg.pinv(X.T @ X)
        djj = np.diag(XtX_inv)
        param_ci_array[i, :] = 2 * np.sqrt(sigma_squared * djj)

        x_i = X[-1, :].reshape(1, -1)
        output_var = (x_i @ XtX_inv @ x_i.T).item()
        output_ci_array[i] = 2 * np.sqrt(sigma_squared * output_var)

    return output_ci_array, param_ci_array

def _sliding_vif_cod(
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for variance inflation factors.")

    cod_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        XtX = X_norm.T @ X_norm
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        
        vif = np.diag(XtX_inv)
        cod = 1 - 1 / vif
        cod_array[i, :] = cod

    return cod_array

def _sliding_svd_cond(
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for singular value decomposition.")

    cond_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        
        try:
            U, singular_values, Vt = np.linalg.svd(X_norm, full_matrices=False)
            if np.any(singular_values == 0):
                continue
            
            max_sv = np.max(singular_values)
            cond = max_sv / singular_values
            cond_array[i, :len(cond)] = cond
            
        except np.linalg.LinAlgError:
            continue

    return cond_array

def _sliding_correlation_matrix(
    predictors: np.ndarray | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_features = predictors.shape
    if num_features >= window_size:
        raise ValueError("Number of predictors must be less than window size.")
        
    corr_matrices = np.full((num_samples, num_features, num_features), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        
        corr_matrix = (X_norm.T @ X_norm)   # / (window_size - 1)
        corr_matrices[i] = corr_matrix

    return corr_matrices


def extract_model(dataframe, prefix):
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("Missing required 'timestamp' column in DataFrame.")
    if not any(col.startswith(prefix) for col in dataframe.columns):
        raise ValueError(f"No columns found for prefix '{prefix}' in provided data frame.")
    
    regressor_cols = sorted(
        [col for col in dataframe.columns if col.startswith(prefix + 'regressor_')],
        key=_extract_param_number
    )
    parameter_cols = sorted(
        [col for col in dataframe.columns if col.startswith(prefix + 'parameter_')],
        key=_extract_param_number
    )
    if len(regressor_cols) == 0:
        raise ValueError(f"Missing a '{prefix}regressor_#' column.")
    if len(parameter_cols) == 0:
        raise ValueError(f"Missing a '{prefix}parameter_#' column.")
    if len(regressor_cols) != len(parameter_cols):
        raise ValueError(
            f"Mismatched number of regressors ({len(regressor_cols)}) and parameters ({len(parameter_cols)})."
        )
    
    extracted_cols = ['timestamp', f"{prefix}measured_output"] + regressor_cols + parameter_cols
    
    return dataframe[extracted_cols]

def process_models(dataframes: list[pd.DataFrame]):
    for i, df in enumerate(dataframes):
        if df.empty:
            raise ValueError(f"DataFrame at index {i} is empty.")
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame at index {i} missing 'timestamp' column.")
        
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        regressor_cols = sorted(
            [col for col in df.columns if 'regressor_' in col],
            key=_extract_param_number
            )
        parameter_cols = sorted(
            [col for col in df.columns if 'parameter_' in col],
            key=_extract_param_number
            )
        if len(regressor_cols) == 0:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_regressor_#' column.")
        if len(parameter_cols) == 0:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_parameter_#' column.")
        if (len(regressor_cols) != len(parameter_cols)):
            raise ValueError(f"DataFrame at index {i} has a mismatched number of regressors ({len(regressor_cols)}) and parameters ({len(parameter_cols)}).")
        
        # modeled output
        regressors = df[regressor_cols].to_numpy()
        parameters = df[parameter_cols].to_numpy()
        modeled_output = np.sum(regressors * parameters, axis=1)
        modeled_output_label = f"{prefix}modeled_output"
        df.insert(loc=2, column=modeled_output_label, value=modeled_output)
        
        # modeled output confidence intervals (cis)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        modeled_output_cis, parameter_cis = _sliding_confidence_intervals(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], regressors, window_size=100)
        modeled_output_cis_label = f"{prefix}modeled_output_cis"
        df.insert(loc=3, column=modeled_output_cis_label, value=modeled_output_cis)
        
        # modeled output confidence interval percentages (cips)
        with np.errstate(divide='ignore', invalid='ignore'):
            modeled_output_cips = (df[f"{prefix}modeled_output_cis"] / df[f"{prefix}modeled_output"]).abs() * 100
            modeled_output_cips = modeled_output_cips.mask(~np.isfinite(modeled_output_cips), np.nan)
        modeled_output_cips_label = f"{prefix}modeled_output_cips"
        df.insert(loc=4, column=modeled_output_cips_label, value=modeled_output_cips)
        
        # residuals
        residuals = df[f"{prefix}measured_output"] - df[f"{prefix}modeled_output"]
        residuals_label = f"{prefix}residuals"
        df.insert(loc=5, column=residuals_label, value=residuals)
        
        # mean squared error (mse)
        mse = _sliding_mse(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=6)
        mse_label = f"{prefix}mse"
        df.insert(loc=6, column=mse_label, value=mse)
        
        # coefficient of determination (cod)
        num_regressors = len([col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)])
        # cod = _sliding_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=30)
        cod = _sliding_adjusted_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], num_regressors, window_size=100)
        cod_label = f"{prefix}cod"
        df.insert(loc=7, column=cod_label, value=cod)
        
        # regressor coefficient of determinations (cod) from variance inflation factors (vif)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cod = _sliding_vif_cod(regressors, window_size=100)
        for j in range(regressors_cod.shape[1]):
            regressors_cod_label = f"{prefix}regressor_{j+1}_cod"
            df.insert(loc=len(df.columns), column=regressors_cod_label, value=regressors_cod[:, j])
        
        # regressor condition numbers (cond) from singular value decomposition (svd)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cond = _sliding_svd_cond(regressors, window_size=100)
        for j in range(regressors_cond.shape[1]):
            regressors_cond_label = f"{prefix}regressor_{j+1}_cond"
            df.insert(loc=len(df.columns), column=regressors_cond_label, value=regressors_cond[:, j])
                             
        # paramter confidence intervals (cis)
        for j in range(parameter_cis.shape[1]):
            parameter_cis_label = f"{prefix}parameter_{j+1}_cis"
            df.insert(loc=len(df.columns), column=parameter_cis_label, value=parameter_cis[:, j])
        
        # parameter confidence interval percentages (cips)
        for j in range(parameter_cis.shape[1]):
            with np.errstate(divide='ignore', invalid='ignore'):
                parameter_cips = (df[f"{prefix}parameter_{j+1}_cis"] / df[f"{prefix}parameter_{j+1}"]).abs() * 100
                parameter_cips = parameter_cips.mask(~np.isfinite(parameter_cips), np.nan)
            parameter_cips_label = f"{prefix}parameter_{j+1}_cips"
            df.insert(loc=len(df.columns), column=parameter_cips_label, value=parameter_cips)
            
        # correlation matrix
        regressor_cols = [col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]
        regressors = df[regressor_cols]
        num_regressors = len(regressor_cols)
        correlation_matrix = _sliding_correlation_matrix(regressors, window_size=100)
        for j in range(num_regressors):
            for k in range(num_regressors):
                if j != k:
                    correlation_element = correlation_matrix[:, j, k]
                    correlation_element_label = f"{prefix}correlation_{j}_to_{k}"
                    df.insert(loc=len(df.columns), column=correlation_element_label, value=correlation_element)
        
    return dataframes


def plot_overall(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Overall Figure"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["rcout_ch5"]
    elv_def = dataframe["rcout_ch6"]
    rud_def = dataframe["rcout_ch7"]
    rol_rate = dataframe["gx"]
    pit_rate = dataframe["gy"]
    yaw_rate = dataframe["gz"]
    rol_deg = dataframe["roll_deg"]
    pit_deg = dataframe["pitch_deg"]
    yaw_deg = dataframe["yaw_deg"]
    airspeed = dataframe["airspeed"]
    altitude = dataframe["altitude"]
    
    axs[0].plot(time, ail_def, label='Aileron', color="blue", linestyle="-")
    axs[0].plot(time, elv_def, label='Elevator', color="red", linestyle="-")
    axs[0].plot(time, rud_def, label='Rudder', color="green", linestyle="-")
    
    axs[1].plot(time, rol_rate, label='Roll', color="blue", linestyle="-")
    axs[1].plot(time, pit_rate, label='Pitch', color="red", linestyle="-")
    axs[1].plot(time, yaw_rate, label='Yaw', color="green", linestyle="-")
    
    axs[2].plot(time, rol_deg, label='Roll', color="blue", linestyle="-")
    axs[2].plot(time, pit_deg, label='Pitch', color="red", linestyle="-")
    axs[2].plot(time, yaw_deg, label='Yaw', color="green", linestyle="-")
    
    axs[3].plot(time, airspeed, label='Airspeed', color="black")
    axs[4].plot(time, altitude, label='Altitude', color="black")
    
    # --- Final Formatting ---
    axs[0].set_title("Controls Over Time")
    axs[0].set_ylabel("PWM Signal\n[±1500]")
    axs[1].set_title("Rates Over Time")
    axs[1].set_ylabel("Rate\n[rad/s]")
    axs[2].set_title("Attitude Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    axs[3].set_title("Airspeed Over Time")
    axs[3].set_ylabel("Airspeed\n[deg]")
    axs[4].set_title("Altitude Over Time")
    axs[4].set_ylabel("Altitude\n[]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()

def plot_signals(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 1"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["rcout_ch5"]
    elv_def = dataframe["rcout_ch6"]
    rud_def = dataframe["rcout_ch7"]
    thrust = dataframe["rcout_ch8"]
    
    axs[0].plot(time, ail_def, label='Aileron', color="blue", linestyle="-")
    axs[1].plot(time, elv_def, label='Elevator', color="red", linestyle="-")
    axs[2].plot(time, rud_def, label='Rudder', color="green", linestyle="-")
    axs[3].plot(time, thrust, label='Thrust', color="black", linestyle="-")
    
    # --- Final Formatting ---
    axs[0].set_title("Aileron Signal Over Time")
    axs[0].set_ylabel("PWM Signal\n[±1500]")
    axs[1].set_title("Elevator Signal Over Time")
    axs[1].set_ylabel("PWM Signal\n[±1500]")
    axs[2].set_title("Rudder Signal Over Time")
    axs[2].set_ylabel("PWM Signal\n[±1500]")
    axs[3].set_title("Thrust Signal Over Time")
    axs[3].set_ylabel("PWM Signal\n[0-1500]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()

def plot_rates(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 2"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    rol_rate = dataframe["gx"]
    pit_rate = dataframe["gy"]
    yaw_rate = dataframe["gz"]
    
    axs[0].plot(time, rol_rate, label='Roll', color="blue")
    axs[1].plot(time, pit_rate, label='Pitch', color="red")
    axs[2].plot(time, yaw_rate, label='Yaw', color="green")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Rate Over Time")
    axs[0].set_ylabel("Rate\n[rad/s]")
    axs[1].set_title("Pitch Rate Over Time")
    axs[1].set_ylabel("Rate\n[rad/s]")
    axs[2].set_title("Yaw Rate Over Time")
    axs[2].set_ylabel("Rate\n[rad/s]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()

def plot_angles(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 3"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    rol_deg = dataframe["roll_deg"]
    pit_deg = dataframe["pitch_deg"]
    yaw_deg = dataframe["yaw_deg"]
    
    axs[0].plot(time, rol_deg, label='Roll', color="blue")
    axs[1].plot(time, pit_deg, label='Pitch', color="red")
    axs[2].plot(time, yaw_deg, label='Yaw', color="green")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Angle Over Time")
    axs[0].set_ylabel("Deflection\n[deg]")
    axs[1].set_title("Pitch Angle Over Time")
    axs[1].set_ylabel("Deflection\n[deg]")
    axs[2].set_title("Yaw Angle Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()

def plot_energy(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 4"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    airspeed = dataframe["airspeed"]
    altitude = dataframe["altitude"]
    
    axs[0].plot(time, airspeed, label='Airspeed', color="black")
    axs[1].plot(time, altitude, label='Altitude', color="black")
    
    # --- Final Formatting ---
    axs[0].set_title("Airspeed Over Time")
    axs[0].set_ylabel("Airspeed\n[deg]")
    axs[1].set_title("Altitude Over Time")
    axs[1].set_ylabel("Altitude\n[]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()

def plot_trajectory(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 5"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["roll_cmd"]
    elv_def = dataframe["pitch_cmd"]
    rud_def = dataframe["yaw_cmd"]
    
    axs[0].plot(time, ail_def, label='Aileron', color="blue", linestyle="-")
    axs[1].plot(time, elv_def, label='Elevator', color="red", linestyle="-")
    axs[2].plot(time, rud_def, label='Rudder', color="green", linestyle="-")
    
    # --- Final Formatting ---
    axs[0].set_title("Aileron Trajectory Over Time")
    axs[0].set_ylabel("Deflection\n[deg]")
    axs[1].set_title("Elevator Trajectory Over Time")
    axs[1].set_ylabel("Deflection\n[deg]")
    axs[2].set_title("Rudder Trajectory Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
        
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.show()



if __name__ == "__main__":
    get_ipython().run_line_magic('matplotlib', 'qt')
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # csv_path = "M:/cUAS Unclassified/125 - Mosley/OLS Model Results/synced_all_data.csv"
    # csv_path = "M:/cUAS Unclassified/125 - Mosley/OLS Model Results/ols_rol_data.csv"
    
    # models = ['ols_rol_']
    
    start_time = 26
    end_time = 38
    # plot_labels = {
    # "subtitle": "Roll Models",
    # "time": "Time [s]",
    # "measured_output": "Measured Roll\nAcceleration [deg/s²]",
    # "output_amp": "Roll Acceleration\n[deg/s²]",
    # "output_percent_confidence": "Confidence [%]",
    # "cod_amp": "R²",
    # "residuals": "Roll Acceleration\nResiduals [deg/s²]",
    # "mse": "Roll Acceleration\nSquared Error\n[(deg/s²)²]",
    # "param_1_amp": "Roll Velocity\nParameter [1/s]",
    # "param_2_amp": "Aileron Parameter\n[1/s²]",
    # "param_3_amp": "Yaw Velocity\nParameter [1/s]",
    # "param_4_amp": "Rudder Parameter\n[1/s²]",
    # "param_1_cod_amp": "Roll Velocity\nParameter's r²\n[%]",
    # "param_2_cod_amp": "Aileron\nParameter's r²\n[%]",
    # "param_3_cod_amp": "Yaw Velocity\nParameter's r²\n[%]",
    # "param_4_cod_amp": "Rudder\nParameter's r²\n[%]",
    # "param_1_cond_amp": "Roll Velocity\nParameter's\nConditioning",
    # "param_2_cond_amp": "Aileron\nParameter's\nConditioning",
    # "param_3_cond_amp": "Yaw Velocity\nParameter's\nConditioning",
    # "param_4_cond_amp": "Rudder\nParameter's\nConditioning",
    # }
