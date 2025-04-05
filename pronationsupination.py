import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import splprep, splev
from github import Github

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

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Sensor coordinates (already provided by you)
sensor_coords = np.array([
    [5.9, 16.4], [5, 7.5], [7.5, 7.5], [5, 3], [7.5, 3], [5, 12], [7.5, 12], [3.2, 17.7], [8, 15.5], [6.5, 18.9], [4.2, 20.4],
    [-5.9, 16.4], [-5, 7.5], [-7.5, 7.5], [-5, 3], [-7.5, 3], [-5, 12], [-7.5, 12], [-3.2, 17.7], [-8, 15.5], [-6.5, 18.9], [-4.2, 20.4]
])

# Foot outline coordinates (already provided by you)
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
def calculate_cop_from_sensors(sensor_data, sensor_coords):
    force = np.sum(sensor_data, axis=1)
    cop_x = np.sum(sensor_data * sensor_coords[:, 0], axis=1) / (force + 1e-6)
    cop_y = np.sum(sensor_data * sensor_coords[:, 1], axis=1) / (force + 1e-6)
    return cop_x, cop_y

import matplotlib.pyplot as plt
import os
import numpy as np

# Plot COP trajectory on foot outline with timestamp-based brightness
def plot_cop_on_foot(l_cop_x, l_cop_y, r_cop_x, r_cop_y, time, label, output_folder):
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased figure size

    # Plot foot outlines
    ax.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], 'k-')
    ax.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], 'k-')

    # Vertical lines for foot centers
    ax.axvline(x=np.mean(smooth_foot_outline[:, 0]), color='gray', linestyle='--', alpha=0.8)
    ax.axvline(x=np.mean(left_foot_outline[:, 0]), color='gray', linestyle='--', alpha=0.8)

    # Sensor locations
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c='black', s=100, marker='o', alpha=0.7)

    # Normalize time for colormap
    norm_time = (time - np.min(time)) / (np.max(time) - np.min(time))

    # Plot COP trajectories with colormap
    sc1 = ax.scatter(l_cop_x, l_cop_y, c=norm_time, cmap='YlOrRd', marker='o', s=150, alpha=0.8)
    sc2 = ax.scatter(r_cop_x, r_cop_y, c=norm_time, cmap='YlOrRd', marker='o', s=150, alpha=0.8)


    # Add colorbar with larger font size
    cbar = fig.colorbar(sc2, ax=ax, pad=0.01)
    cbar.set_label('Run Progress (%)', fontsize=20)  # Bigger label
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=20)  # Bigger tick labels

    # Clean up plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout(pad=0)

    # Save plot
    output_path = os.path.join(output_folder, f'{label}_cop_plot.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


# Main analysis
def main(file_path, output_folder):
    data = load_data(file_path)
    time = data['Time'].values

    # Sensor data from CSV (Sensor_1 to Sensor_22)
    sensor_data = data.iloc[:, 1:23].values
    right_sensor_data = sensor_data[:, :11]
    left_sensor_data = sensor_data[:, 11:]

    # Calculate and plot COP from sensor data
    l_cop_x, l_cop_y = calculate_cop_from_sensors(left_sensor_data, sensor_coords[11:])
    r_cop_x, r_cop_y = calculate_cop_from_sensors(right_sensor_data, sensor_coords[:11])

    # Get the file name without extension for labeling
    label = os.path.splitext(os.path.basename(file_path))[0]
    plot_cop_on_foot(l_cop_x, l_cop_y, r_cop_x, r_cop_y, time, label, output_folder)

# Upload image to GitHub
def upload_to_github(output_folder):
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_folder, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Define the path in the GitHub repo
            github_path = f'prosupvisual/{file_name}'
            
            # Try to get the existing file to update it
            try:
                existing_file = repo.get_contents(github_path)
                repo.update_file(github_path, f"Update {file_name}", content, existing_file.sha)
                print(f"Updated {file_name} on GitHub.")
            except:
                repo.create_file(github_path, f"Add {file_name}", content)
                print(f"Uploaded {file_name} to GitHub.")

# Process all files in the runData folder
def process_all_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            main(file_path, output_folder)

    # Upload all generated plots to GitHub
    upload_to_github(output_folder)

# Run the script on all files in runData
if __name__ == '__main__':
    process_all_files(input_folder, output_folder)
