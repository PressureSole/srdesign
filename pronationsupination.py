import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from github import Github
from github import GithubException

# GitHub repository and token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Set the GitHub Personal Access Token as an environment variable
REPO_NAME = "PressureSole/srdesign"

# Ensure GitHub connection
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    print(f"Connected to repository: {repo.full_name}")
except Exception as e:
    print(f"GitHub connection error: {e}")
    exit(1)

# Get the absolute path of the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output folders relative to the script's location
input_folder = os.path.join(script_dir, 'runData')
output_folder = os.path.join(script_dir, 'prosupvisual')

# Load the data from CSV or ZIP
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

# Sensor coordinates (already provided)
sensor_coords = np.array([
    [5, 3], [7.5, 3], [5, 7.5], [7.5, 7.5], [5, 12], [7.5, 12], [8, 15.5], [5.9, 16.4], [3.2, 17.7], [6.5, 18.9], [4.2, 20.4],
    [-5, 3], [-7.5, 3], [-5, 7.5], [-7.5, 7.5], [-5, 12], [-7.5, 12], [-8, 15.5], [-5.9, 16.4], [-3.2, 17.7], [-6.5, 18.9], [-4.2, 20.4]
])

# Foot outline coordinates (already provided)
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

# Calculate Center of Pressure (COP) using sensor data
def calculate_cop_from_sensors(sensor_data, sensor_coords, force_threshold=0.1):
    force = np.sum(sensor_data, axis=1)
    # Create arrays initialized with NaN
    cop_x = np.full(force.shape, np.nan)
    cop_y = np.full(force.shape, np.nan)
    # Compute COP only for rows with force above the threshold
    valid = force > force_threshold
    if np.any(valid):
        cop_x[valid] = np.sum(sensor_data[valid] * sensor_coords[:, 0], axis=1) / (force[valid] + 1e-6)
        cop_y[valid] = np.sum(sensor_data[valid] * sensor_coords[:, 1], axis=1) / (force[valid] + 1e-6)
    return cop_x, cop_y

# Plot COP trajectory on foot outline with timestamp-based brightness
def plot_cop_on_foot(l_cop_x, l_cop_y, r_cop_x, r_cop_y, left_time, right_time, label, output_folder):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot foot outlines
    ax.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], 'k-', lw=2)
    ax.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], 'k-', lw=2)

    # Vertical lines for foot centers
    ax.axvline(x=np.mean(smooth_foot_outline[:, 0]), color='gray', linestyle='--', alpha=0.8)
    ax.axvline(x=np.mean(left_foot_outline[:, 0]), color='gray', linestyle='--', alpha=0.8)

    # Sensor locations
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='black', s=100, marker='o', alpha=0.7)

    # Normalize left time for colormap; if constant, use ones
    if np.max(left_time) == np.min(left_time):
        norm_time_left = np.ones_like(left_time)
    else:
        norm_time_left = (left_time - np.min(left_time)) / (np.max(left_time) - np.min(left_time))
    
    # Normalize right time for colormap
    if np.max(right_time) == np.min(right_time):
        norm_time_right = np.ones_like(right_time)
    else:
        norm_time_right = (right_time - np.min(right_time)) / (np.max(right_time) - np.min(right_time))

    # Plot COP trajectories with colormap and no edge color, using the proper time arrays
    sc1 = ax.scatter(l_cop_x, l_cop_y, c=norm_time_left, cmap='YlOrRd', marker='o', s=67,
                     edgecolors='none', alpha=0.9, vmin=0, vmax=1)
    sc2 = ax.scatter(r_cop_x, r_cop_y, c=norm_time_right, cmap='YlOrRd', marker='o', s=67,
                     edgecolors='none', alpha=0.9, vmin=0, vmax=1)

    # Add colorbar with custom ticks and labels (using right-time colormap)
    cbar = fig.colorbar(sc2, ax=ax, pad=0.01)
    cbar.set_label('Run Progress (%)', fontsize=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout(pad=0)

    output_path = os.path.join(output_folder, f'{label}_cop_plot.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
# Main analysis
def main(file_path, output_folder):
    data = load_data(file_path)
    if data is None:
        return
    # Convert "Time" to numeric
    time_vals = pd.to_numeric(data['Time'], errors='coerce').values
    sensor_data = data.iloc[:, 1:23].values
    right_sensor_data = sensor_data[:, :11]
    left_sensor_data = sensor_data[:, 11:]
    
    # Calculate COP using a force threshold to filter out air-phase frames
    l_cop, _ = calculate_cop_from_sensors(left_sensor_data, sensor_coords[11:], force_threshold=0.1)
    r_cop, _ = calculate_cop_from_sensors(right_sensor_data, sensor_coords[:11], force_threshold=0.1)
    # We need both x and y values so:
    l_cop_x, l_cop_y = calculate_cop_from_sensors(left_sensor_data, sensor_coords[11:], force_threshold=0.1)
    r_cop_x, r_cop_y = calculate_cop_from_sensors(right_sensor_data, sensor_coords[:11], force_threshold=0.1)
    
    # Filter out invalid (NaN) COP values for left and right separately
    valid_left = ~np.isnan(l_cop_x) & ~np.isnan(l_cop_y)
    valid_right = ~np.isnan(r_cop_x) & ~np.isnan(r_cop_y)
    
    l_cop_x_plot = l_cop_x[valid_left]
    l_cop_y_plot = l_cop_y[valid_left]
    r_cop_x_plot = r_cop_x[valid_right]
    r_cop_y_plot = r_cop_y[valid_right]
    
    # Compute separate time arrays for left and right using the valid indices
    plot_time_left = time_vals[valid_left]
    plot_time_right = time_vals[valid_right]
    
    label = os.path.splitext(os.path.basename(file_path))[0]
    plot_cop_on_foot(l_cop_x_plot, l_cop_y_plot, r_cop_x_plot, r_cop_y_plot,
                     plot_time_left, plot_time_right, label, output_folder)

def upload_to_github(output_folder):
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
            github_path = f'prosupvisual/{file_name}'
            try:
                # Try to get the existing file.
                existing_file = repo.get_contents(github_path)
                try:
                    # Attempt to update using the SHA from the existing file.
                    repo.update_file(github_path, f"Update {file_name}", content, existing_file.sha)
                    print(f"Updated {file_name} on GitHub.")
                except GithubException as update_err:
                    # If conflict error, re-read the file to get the latest SHA and retry.
                    if update_err.status == 409:
                        existing_file = repo.get_contents(github_path)
                        repo.update_file(github_path, f"Update {file_name}", content, existing_file.sha)
                        print(f"Updated {file_name} on GitHub after conflict resolution.")
                    else:
                        raise update_err
            except GithubException as e:
                # If file is not found, create it.
                if e.status == 404:
                    try:
                        repo.create_file(github_path, f"Add {file_name}", content)
                        print(f"Uploaded {file_name} to GitHub.")
                    except GithubException as ce:
                        print(f"Failed to create file {file_name}: {ce}")
                else:
                    print(f"Failed to upload {file_name}: {e}")

# Process all files in the runData folder
def process_all_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv') or file_name.endswith('.zip'):
            file_path = os.path.join(input_folder, file_name)
            main(file_path, output_folder)
    upload_to_github(output_folder)

# Run the script on all files in runData
if __name__ == '__main__':
    process_all_files(input_folder, output_folder)
