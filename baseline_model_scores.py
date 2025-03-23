#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline Model Scores

Automatically converted from the Google Colab version.
"""

import pandas as pd
import numpy as np
from scipy.integrate import trapz

def calculate_refined_scores(filename, body_weight=700, window_size=0.2):
    """
    Calculates refined fatigue, stress, and symmetry scores from treadmill data.

    Args:
        filename (str): Path to the Excel file.
        body_weight (float, optional): Body weight in Newtons. Defaults to 700.
        window_size (float, optional): Size of the sliding window for fatigue analysis (as a fraction of total data). Defaults to 0.2.

    Returns:
        dict: A dictionary containing the refined scores.
    """

    data = pd.read_excel(filename)

    time = data['Time'].values
    R_Total = data['R_Heel'].values + data['R_Midfoot'].values + data['R_Forefoot'].values
    L_Total = data['L_Heel'].values + data['L_Midfoot'].values + data['L_Forefoot'].values
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

        R_peak_forces = np.partition(window_R, -10)[-10:]
        L_peak_forces = np.partition(window_L, -10)[-10:]

        R_weighted_peak = np.average(R_peak_forces, weights=np.arange(1, 11))
        L_weighted_peak = np.average(L_peak_forces, weights=np.arange(1, 11))

        if i > 0:
            R_fatigue_changes.append(R_weighted_peak - prev_R_peak)
            L_fatigue_changes.append(L_weighted_peak - prev_L_peak)

        prev_R_peak = R_weighted_peak
        prev_L_peak = L_weighted_peak

    R_Fatigue = np.mean(R_fatigue_changes) if R_fatigue_changes else 0
    L_Fatigue = np.mean(L_fatigue_changes) if L_fatigue_changes else 0

    # Refined Stress Calculation
    stride_count = len(R_Total) / 2
    R_peak_force = np.max(R_Total)
    L_peak_force = np.max(L_Total)

    R_impact_rate = np.max(np.diff(R_Total) / np.diff(time))
    L_impact_rate = np.max(np.diff(L_Total) / np.diff(time))

    R_Stress = (R_peak_force / body_weight) * R_impact_rate * (total_time / 600) * 100
    L_Stress = (L_peak_force / body_weight) * L_impact_rate * (total_time / 600) * 100

    Total_Stress = min(((R_Stress + L_Stress) / 2), 100)

    # Symmetry Calculation
    step_differences = np.abs(R_Total - L_Total)
    Symmetry = 100 - (np.sum(step_differences) / np.sum(R_Total + L_Total) * 100)
    Symmetry = max(85, min(Symmetry, 100))

    return {
        "Right_Fatigue": R_Fatigue,
        "Left_Fatigue": L_Fatigue,
        "Stress_Score": Total_Stress,
        "Symmetry_Score": Symmetry,
    }

if __name__ == '__main__':
    # Use the "Copy of test_lab shortening.xlsx" file from the repo.
    filename = 'Copy of test_lab shortening.xlsx'
    scores = calculate_refined_scores(filename)
    print(f"Fatigue Score (Right): {scores['Right_Fatigue']:.2f}")
    print(f"Fatigue Score (Left): {scores['Left_Fatigue']:.2f}")
    print(f"Stress Score: {scores['Stress_Score']:.2f}")
    print(f"Symmetry Score: {scores['Symmetry_Score']:.2f}")
