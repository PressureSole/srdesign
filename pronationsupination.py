import zipfile
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

# Sensor coordinates (already provided by you)
sensor_coords = np.array([
    [5, 3], [7.5, 3], [5, 7.5], [7.5, 7.5], [5, 12], [7.5, 12], [8, 15.5], [5.9, 16.4], [3.2, 17.7], [6.5, 18.9], [4.2, 20.4],
    [-5, 3], [-7.5, 3], [-5, 7.5], [-7.5, 7.5], [-5, 12], [-7.5, 12], [-8, 15.5], [-5.9, 16.4], [-3.2, 17.7], [-6.5, 18.9], [-4.2, 20.4]
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
        if file_name.endswith('.csv') or file_name.endswith('.zip'):
            file_path = os.path.join(input_folder, file_name)
            main(file_path, output_folder)

    # Upload all generated plots to GitHub
    upload_to_github(output_folder)

# Run the script on all files in runData
if __name__ == '__main__':
    process_all_files(input_folder, output_folder)
