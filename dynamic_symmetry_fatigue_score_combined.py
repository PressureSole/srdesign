import zipfile
import os
import time
import glob
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata, Rbf, splprep, splev
from github import Github
from io import BytesIO

# GitHub repository and token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Set the GitHub Personal Access Token as an environment variable
REPO_NAME = "PressureSole/srdesign"

# Get the absolute path of the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output folders relative to the script's location
input_folder = os.path.join(script_dir, 'runData')
output_folder = os.path.join(script_dir, 'images')

# Ensure GitHub connection
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    print(f"Connected to repository: {repo.full_name}")
except Exception as e:
    print(f"GitHub connection error: {e}")

def load_data(file_path):
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise Exception("No CSV file found in the ZIP archive.")
                data = pd.read_csv(z.open(csv_files[0]))
                return data
        else:
            data = pd.read_csv(file_path)
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# SENSOR SETUP & FOOT OUTLINE (remains unchanged)
# -----------------------------------------------
# Sensor coordinates
sensor_coords = np.array([
    ,
    [5, 3], [7.5, 3], [5, 7.5], [7.5, 7.5], [5, 12], [7.5, 12], [8, 15.5], [5.9, 16.4], [3.2, 17.7], [6.5, 18.9], [4.2, 20.4],
    [-5, 3], [-7.5, 3], [-5, 7.5], [-7.5, 7.5], [-5, 12], [-7.5, 12], [-8, 15.5], [-5.9, 16.4], [-3.2, 17.7], [-6.5, 18.9], [-4.2, 20.4]
])

# Foot outline
foot_outline = np.array([
    [6, 0.4], [3.3, 2.2], [3.7, 7], [3, 10.8], [2.2, 15.7], [2.3, 19.8],
    [4, 23.6], [7, 22], [8.5, 19.2], [9.2, 15.2], [9, 11.5], [8.5, 6.6],
    [8.4, 2.3], [7.9, 0.9], [6, 0.4]
])

# Generate a smooth Bézier curve for the foot outline
tck, u = splprep(foot_outline.T, s=2)
u_fine = np.linspace(0, 1, 300)
smooth_foot_outline = np.array(splev(u_fine, tck)).T
left_foot_outline = smooth_foot_outline.copy()
left_foot_outline[:, 0] *= -1

# Grid setup for interpolation
min_x, max_x = np.min(left_foot_outline[:, 0]), np.max(smooth_foot_outline[:, 0])
min_y, max_y = np.min(smooth_foot_outline[:, 1]), np.max(smooth_foot_outline[:, 1])
X, Y = np.meshgrid(np.linspace(min_x, max_x, 120), np.linspace(min_y, max_y, 300))

def create_foot_path(smooth_x, smooth_y):
    return Path(np.column_stack([smooth_x, smooth_y]))

r_foot_path = create_foot_path(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1])
l_foot_path = create_foot_path(left_foot_outline[:, 0], left_foot_outline[:, 1])

def normalize(data):
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    return (data - min_val) / (max_val - min_val)

def interpolate_pressure_data(pressure_values, method):
    if method == 'rbf':
        rbf_interpolator = Rbf(sensor_coords[:, 0], sensor_coords[:, 1], pressure_values, function='linear')
        pressure_data = rbf_interpolator(X, Y)
    elif method == 'griddata':
        grid_points = np.column_stack([X.flatten(), Y.flatten()])
        pressure_data = griddata(sensor_coords, pressure_values, grid_points, method='nearest').reshape(X.shape)
    
    # Mask points outside the foot outlines
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    inside_right = r_foot_path.contains_points(grid_points).reshape(X.shape)
    inside_left = l_foot_path.contains_points(grid_points).reshape(X.shape)
    combined_pressure = np.where(inside_right | inside_left, pressure_data, np.nan)
    return combined_pressure

def get_pressure_data_for_third(timestamps, sensor_pressures, start_time, end_time):
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    relevant_pressures = np.array(sensor_pressures)[:, mask]
    avg_relevant_pressures = np.mean(relevant_pressures, axis=1)
    return avg_relevant_pressures

def generate_plots(pressure_values, method, output_prefix):
    try:
        interpolated_pressure_rbf = interpolate_pressure_data(pressure_values, method='rbf')
        interpolated_pressure_grid = interpolate_pressure_data(pressure_values, method='griddata')
    
        normalized_pressure_rbf = normalize(interpolated_pressure_rbf)
        normalized_pressure_grid = normalize(interpolated_pressure_grid)
    
        # RBF plot (Gradient View)
        fig_rbf, ax_rbf = plt.subplots(figsize=(8, 8))
        ax_rbf.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
        pressure_img_rbf = ax_rbf.imshow(normalized_pressure_rbf, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_rbf.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="black", lw=2)
        ax_rbf.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="black", lw=2)
        ax_rbf.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=50)
        #ax_rbf.set_title(f"Pressure Mapping (Gradient View) - {output_prefix}")
        cbar_rbf = fig_rbf.colorbar(pressure_img_rbf, ax=ax_rbf)
        cbar_rbf.set_label('Pressure Value (Normalized)', fontsize=14)
        
        rbf_output_file = f"images/{output_prefix}_gradient.png"
        fig_rbf.savefig(rbf_output_file, bbox_inches='tight')
        plt.close(fig_rbf)
        print(f"Saved RBF plot: {rbf_output_file}")
        
        # GridData plot (Section View)
        fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
        ax_grid.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
        pressure_img_grid = ax_grid.imshow(normalized_pressure_grid, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_grid.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="black", lw=2)
        ax_grid.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="black", lw=2)
        ax_grid.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=50)
        #ax_grid.set_title(f"Pressure Mapping (Section View) - {output_prefix}")
        cbar_grid = fig_grid.colorbar(pressure_img_grid, ax=ax_grid)
        cbar_grid.set_label('Pressure Value (Normalized)', fontsize=14)
        
        grid_output_file = f"images/{output_prefix}_section.png"
        fig_grid.savefig(grid_output_file, bbox_inches='tight')
        plt.close(fig_grid)
        print(f"Saved GridData plot: {grid_output_file}")
    
        return True

    except Exception as e:
        print(f"Error generating plots for {output_prefix}: {e}")
        return False

def upload_file_to_github(file_path, github_path):
    try:
        g = Github(os.getenv('GITHUB_TOKEN'))
        repo = g.get_repo(REPO_NAME)
        with open(file_path, 'rb') as file:
            file_content = file.read()
        try:
            existing_file = repo.get_contents(github_path)
            repo.update_file(
                path=github_path, 
                message=f"Update {file_path}", 
                content=file_content, 
                sha=existing_file.sha, 
                branch="main"
            )
            print(f"Updated existing file: {github_path}")
        except Exception:
            repo.create_file(
                path=github_path, 
                message=f"Add {file_path}", 
                content=file_content, 
                branch="main"
            )
            print(f"Created new file: {github_path}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

# Main execution: process each CSV file in the runData folder
def main(file_path, output_folder):
    data = load_data(file_path)
    if data is None:
        return
    # Convert the "Time" column to numeric
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    if data['Time'].isnull().any():
        print("Warning: Some time values could not be converted and will be dropped.")
    data = data.dropna(subset=['Time'])
    
    timestamps = data['Time'].values
    sensor_pressures = [data[f"Sensor_{i+1}"].values for i in range(22)]
    
    # Extract file timestamp from filename (supports .csv or .zip)
    m = re.search(r"Run_Data_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.(csv|zip)", file_path)
    if m:
        file_timestamp = m.group(1)
    else:
        file_timestamp = "unknown"
    
    # Compute full run pressure (average over all rows)
    full_data = np.mean(sensor_pressures, axis=1)
    
    # Define output prefix for symmetry plots
    sym_prefix = f"{file_timestamp}_symmetry"
    generate_plots(full_data, method='griddata', output_prefix=sym_prefix)
    
    # Divide the run into thirds using timestamps from the file
    run_duration = timestamps[-1] - timestamps[0]
    if run_duration <= 0:
        print("Invalid run duration")
        return
    third_duration = run_duration / 3
    thirds = [
        (timestamps[0], timestamps[0] + third_duration),
        (timestamps[0] + third_duration, timestamps[0] + 2 * third_duration),
        (timestamps[0] + 2 * third_duration, timestamps[-1])
    ]
    
    # Generate fatigue plots for each third of the run
    for i, (start_time, end_time) in enumerate(thirds):
        pressure_values = get_pressure_data_for_third(timestamps, sensor_pressures, start_time, end_time)
        print(f"Third {i+1} pressure values: {pressure_values}")
        fatigue_prefix = f"{file_timestamp}_fatigue{i+1}"
        generate_plots(pressure_values, method='rbf', output_prefix=fatigue_prefix)
    
    # Prepare to upload images with an upload timestamp appended
    upload_timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    images = [
        (f"{sym_prefix}_gradient.png", f"images/{sym_prefix}_gradient_{upload_timestamp}.png"),
        (f"{sym_prefix}_section.png", f"images/{sym_prefix}_section_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue1_gradient.png", f"images/{file_timestamp}_fatigue1_gradient_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue1_section.png", f"images/{file_timestamp}_fatigue1_section_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue2_gradient.png", f"images/{file_timestamp}_fatigue2_gradient_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue2_section.png", f"images/{file_timestamp}_fatigue2_section_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue3_gradient.png", f"images/{file_timestamp}_fatigue3_gradient_{upload_timestamp}.png"),
        (f"{file_timestamp}_fatigue3_section.png", f"images/{file_timestamp}_fatigue3_section_{upload_timestamp}.png")
    ]
    
    for local_path, github_path in images:
        if os.path.exists(local_path):
            upload_file_to_github(local_path, github_path)
        else:
            print(f"File not found: {local_path}")
            
def process_all_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv') or file_name.endswith('.zip'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path}")
            main(file_path, output_folder)

if __name__ == '__main__':
    process_all_files(input_folder, output_folder)
