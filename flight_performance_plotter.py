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
        
def _recenter_angles(data: np.ndarray) -> np.ndarray:
    x = _ensure_numpy(data)
    recentered = np.where(x >= 0, 180 - x, -180 - x)
    return recentered

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
    ail_def = dataframe["rcout_ch1"] - 1500
    elv_def = dataframe["rcout_ch2"] - 1500
    rud_def = dataframe["rcout_ch4"] - 1500
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
    ail_def = dataframe["rcout_ch1"] - 1500
    elv_def = dataframe["rcout_ch2"] - 1500
    rud_def = dataframe["rcout_ch4"] - 1500
    thrust = dataframe["rcout_ch3"]
    
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
    axs[3].set_ylabel("PWM Signal\n[0-3000]")
    
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
    yaw_deg = _recenter_angles(dataframe["yaw_deg"])
    
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

    plot_overall(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/synced_all_data.csv"))
    plot_signals(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/rcout_data.csv"))
    plot_rates(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/imu_data.csv"))
    plot_angles(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/odometry_data.csv"))
    plot_energy(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/synced_all_data.csv"))
    plot_trajectory(pd.read_csv("M:/cUAS Unclassified/125 - Mosley/OLS Model Results/trajectory_data.csv"))
