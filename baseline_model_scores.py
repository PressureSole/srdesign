#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline Model Scores

Automatically converted from the Google Colab version.
Now processes all CSV files in the "runData" folder.
"""

import re
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
from numpy import trapz

# Regex pattern to match filenames like "Run_Data_yyyy-mm-dd_hh-mm-ss.csv"
pattern = re.compile(r"Run_Data_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv")

def calculate_refined_scores(filename, window_size=0.2):
    """
    Calculates refined fatigue, stress, and symmetry scores from treadmill data.
    Note: The stress scores are now calculated without factoring in body weight.
    
    Args:
        filename (str): Path to the CSV file.
        window_size (float, optional): Size of the sliding window for fatigue analysis (as a fraction of total data). Defaults to 0.2.

    Returns:
        dict: A dictionary containing the refined scores.
              Keys: "Right_Fatigue", "Left_Fatigue", "R_Stress", "L_Stress", "Symmetry_Score"
    """
    # Read CSV file
    data = pd.read_csv(filename)
    
    # Strip any extra whitespace from column names
    data.columns = data.columns.str.strip()
    print("Columns in file", filename, ":", data.columns.tolist())
    
    time = data['Time'].values

    # Sum sensor values for each foot region.
    R_Heel = data[['Sensor_1', 'Sensor_2', 'Sensor_3']].sum(axis=1).values
    R_Midfoot = data[['Sensor_4', 'Sensor_5', 'Sensor_6', 'Sensor_7']].sum(axis=1).values
    R_Forefoot = data[['Sensor_8', 'Sensor_9', 'Sensor_10', 'Sensor_11']].sum(axis=1).values

    L_Heel = data[['Sensor_12', 'Sensor_13', 'Sensor_14']].sum(axis=1).values
    L_Midfoot = data[['Sensor_15', 'Sensor_16', 'Sensor_17', 'Sensor_18']].sum(axis=1).values
    L_Forefoot = data[['Sensor_19', 'Sensor_20', 'Sensor_21', 'Sensor_22']].sum(axis=1).values

    R_Total = R_Heel + R_Midfoot + R_Forefoot
    L_Total = L_Heel + L_Midfoot + L_Forefoot
    total_time = time[-1] - time[0]

    # Refined Fatigue Calculation
    num_windows = int(1 / window_size)
    R_fatigue_changes = []
    L_fatigue_changes = []

    for i in range(num_windows):
        start = int(len(time) * i * window_size)
        end = int(len(time) * (i + 1) * window_size)
        window_time = time[start:end]
        window_R = R_Total[start:end]
        window_L = L_Total[start:end]

        if len(window_time) < 2:
            continue

        # Use up to 10 peaks (or fewer if the window is short)
        num_peaks = min(10, len(window_R))
        R_peak_forces = np.partition(window_R, -num_peaks)[-num_peaks:]
        L_peak_forces = np.partition(window_L, -num_peaks)[-num_peaks:]
        weights = np.arange(1, num_peaks + 1)
        
        R_weighted_peak = np.average(R_peak_forces, weights=weights)
        L_weighted_peak = np.average(L_peak_forces, weights=weights)

        if i > 0:
            R_fatigue_changes.append(R_weighted_peak - prev_R_peak)
            L_fatigue_changes.append(L_weighted_peak - prev_L_peak)

        prev_R_peak = R_weighted_peak
        prev_L_peak = L_weighted_peak

    R_Fatigue = np.mean(R_fatigue_changes) if R_fatigue_changes else 0
    L_Fatigue = np.mean(L_fatigue_changes) if L_fatigue_changes else 0

    # Refined Stress Calculation without body weight factor.
    R_peak_force = np.max(R_Total)
    L_peak_force = np.max(L_Total)

    R_impact_rate = np.max(np.diff(R_Total) / np.diff(time))
    L_impact_rate = np.max(np.diff(L_Total) / np.diff(time))

    R_Stress = R_peak_force * R_impact_rate * (total_time / 600) * 100
    L_Stress = L_peak_force * L_impact_rate * (total_time / 600) * 100

    # Symmetry Calculation
    step_differences = np.abs(R_Total - L_Total)
    Symmetry = 100 - (np.sum(step_differences) / np.sum(R_Total + L_Total) * 100)
    Symmetry = max(85, min(Symmetry, 100))

    return {
        "Right_Fatigue": R_Fatigue,
        "Left_Fatigue": L_Fatigue,
        "R_Stress": R_Stress,
        "L_Stress": L_Stress,
        "Symmetry_Score": Symmetry,
    }

if __name__ == '__main__':
    # Define the folders for run data and scores
    run_data_dir = "runData"
    scores_dir = "scores"
    os.makedirs(scores_dir, exist_ok=True)

    # Use glob to find all CSV files in the runData folder that match the naming convention.
    run_files = glob.glob(os.path.join(run_data_dir, "Run_Data_*.csv"))
    
    if not run_files:
        print("No run data files found in the runData folder.")
    else:
        # Sort files by timestamp extracted using the regex pattern (latest first)
        run_files.sort(key=lambda f: datetime.strptime(pattern.match(os.path.basename(f)).group(1), "%Y-%m-%d_%H-%M-%S"), reverse=True)
        
        for file in run_files:
            print(f"Processing file: {file}")
            scores = calculate_refined_scores(file)
            # Extract the timestamp part from the filename.
            basename = os.path.basename(file)
            timestamp_part = basename[len("Run_Data_"):-len(".csv")]
            output_filename = os.path.join(scores_dir, f"scores_{timestamp_part}.txt")
            with open(output_filename, "w") as f:
                f.write(f"R_Stress: {scores['R_Stress']:.2f}\n")
                f.write(f"L_Stress: {scores['L_Stress']:.2f}\n")
                f.write(f"Symmetry_Score: {scores['Symmetry_Score']:.2f}\n")
                f.write(f"Right_Fatigue: {scores['Right_Fatigue']:.2f}\n")
                f.write(f"Left_Fatigue: {scores['Left_Fatigue']:.2f}\n")
            print(f"Scores written to {output_filename}")
