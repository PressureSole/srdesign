import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, savgol_filter
from github import Github

# GitHub repository and token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Ensure this is set as an environment variable
REPO_NAME = "PressureSole/srdesign"

# Connect to GitHub repository
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    print(f"Connected to repository: {repo.full_name}")
except Exception as e:
    print(f"GitHub connection error: {e}")
    exit(1)

# Get script directory and define input/output folders
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, 'runData')
output_folder = os.path.join(script_dir, 'cadence')

# Ensure folders exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Load data function
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Main cadence calculation
def calculate_cadence(df):
    try:
        time_column = df.columns[0]
        left_foot_sensors = df.columns[1:12]  # First 11 columns
        right_foot_sensors = df.columns[12:23]  # Next 11 columns

        df["Left_Total_Pressure"] = df[left_foot_sensors].sum(axis=1)
        df["Right_Total_Pressure"] = df[right_foot_sensors].sum(axis=1)

        # Apply Savitzky-Golay filter for smoothing
        window_size = 15  # Must be odd
        poly_order = 3
        df["Left_Smoothed"] = savgol_filter(df["Left_Total_Pressure"], window_size, poly_order)
        df["Right_Smoothed"] = savgol_filter(df["Right_Total_Pressure"], window_size, poly_order)

        # Detect peaks (heel strikes)
        left_peaks, _ = find_peaks(df["Left_Smoothed"], height=14, distance=10)
        right_peaks, _ = find_peaks(df["Right_Smoothed"], height=14, distance=10)

        # Compute total time
        total_time = df[time_column].iloc[-1] - df[time_column].iloc[0]  # In seconds

        # Compute cadence (steps per minute)
        left_cadence = (len(left_peaks) * 60) / total_time
        right_cadence = (len(right_peaks) * 60) / total_time
        average_cadence = round((left_cadence + right_cadence) / 2)

        print(f"Average Cadence: {average_cadence} steps/min")
        return average_cadence
    except Exception as e:
        print(f"Error calculating cadence: {e}")
        return None

# Upload .txt files to GitHub
def upload_to_github(output_folder):
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, 'r') as f:
                content = f.read()

            github_path = f'cadence/{file_name}'  # Define GitHub repo folder path

            try:
                existing_file = repo.get_contents(github_path)
                repo.update_file(github_path, f"Update {file_name}", content, existing_file.sha)
                print(f"Updated {file_name} on GitHub.")
            except Exception:
                try:
                    repo.create_file(github_path, f"Add {file_name}", content)
                    print(f"Uploaded {file_name} to GitHub.")
                except Exception as err:
                    print(f"Failed to upload {file_name}: {err}")

# Process files in input folder
def process_all_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            data = load_data(file_path)
            if data is not None:
                cadence = calculate_cadence(data)
                if cadence is not None:
                    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
                    with open(output_file, 'w') as f:
                        f.write(str(cadence))

    # Upload results to GitHub
    upload_to_github(output_folder)

# Run the script
if __name__ == '__main__':
    process_all_files(input_folder, output_folder)
