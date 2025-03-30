import os
import time
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

# Ensure GitHub connection
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    print(f"Connected to repository: {repo.full_name}")
except Exception as e:
    print(f"GitHub connection error: {e}")

# Load the Excel file (make sure the file path is correct)
try:
    filename = "Copy of test_lab shortening.xlsx"  # Adjust this path based on the uploaded file
    data = pd.read_excel(filename)
except FileNotFoundError:
    print(f"Error: File {filename} not found. Please check the file path.")
    exit(1)

# Extract timestamps and relevant pressure values
timestamps = data["Time"].values
sensor_pressures = [data[f"Sensor_{i+1}"].values for i in range(22)]

# Sensor coordinates
sensor_coords = np.array([
    [5, 3], [7.5, 3], [5, 7.5], [7.5, 7.5], [5, 12], [7.5, 12],
    [8, 16.0], [5.9, 16.9], [6.5, 18.9], [3.2, 18.2], [4.2, 20.4],
    [-5, 3], [-7.5, 3], [-5, 7.5], [-7.5, 7.5], [-5, 12], [-7.5, 12],
    [-8, 16.0], [-5.9, 16.9], [-6.5, 18.9], [-3.2, 18.2], [-4.2, 20.4]
])

# Foot outline
foot_outline = np.array([
    [6, 0.4], [3.3, 2.2], [3.7, 7], [3, 10.8], [2.2, 15.7], [2.3, 19.8],
    [4, 23.6], [7, 22], [8.5, 19.2], [9.2, 15.2], [9, 11.5], [8.5, 6.6],
    [8.4, 2.3], [7.9, 0.9], [6, 0.4]
])

# Generate a smooth Bézier curve
tck, u = splprep(foot_outline.T, s=2)
u_fine = np.linspace(0, 1, 300)
smooth_foot_outline = np.array(splev(u_fine, tck)).T
left_foot_outline = smooth_foot_outline.copy()
left_foot_outline[:, 0] *= -1

# Grid setup
min_x, max_x = np.min(left_foot_outline[:, 0]), np.max(smooth_foot_outline[:, 0])
min_y, max_y = np.min(smooth_foot_outline[:, 1]), np.max(smooth_foot_outline[:, 1])
X, Y = np.meshgrid(np.linspace(min_x, max_x, 120), np.linspace(min_y, max_y, 300))

def create_foot_path(smooth_x, smooth_y):
    return Path(np.column_stack([smooth_x, smooth_y]))

r_foot_path = create_foot_path(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1])
l_foot_path = create_foot_path(left_foot_outline[:, 0], left_foot_outline[:, 1])

# Normalize function
def normalize(data):
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    return (data - min_val) / (max_val - min_val)

# Interpolate pressure data function
def interpolate_pressure_data(pressure_values, method):
    # Perform RBF Interpolation
    if method == 'rbf':
        rbf_interpolator = Rbf(sensor_coords[:, 0], sensor_coords[:, 1], pressure_values, function='linear')
        pressure_data = rbf_interpolator(X, Y)
    elif method == 'griddata':
        # GridData Interpolation
        grid_points = np.column_stack([X.flatten(), Y.flatten()])
        pressure_data = griddata(sensor_coords, pressure_values, grid_points, method='nearest').reshape(X.shape)
    
    # Mask points inside foot outlines
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    inside_right = r_foot_path.contains_points(grid_points).reshape(X.shape)
    inside_left = l_foot_path.contains_points(grid_points).reshape(X.shape)
    combined_pressure = np.where(inside_right | inside_left, pressure_data, np.nan)
    
    return combined_pressure

# Get pressure data for each third of the run
def get_pressure_data_for_third(start_time, end_time):
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    relevant_pressures = np.array(sensor_pressures)[:, mask]  # Filter based on timestamps
    avg_relevant_pressures = np.mean(relevant_pressures, axis=1)  # Average over sensors
    return avg_relevant_pressures

# Divide the run into thirds
run_duration = timestamps[-1] - timestamps[0]
third_duration = run_duration / 3

# Define the thirds of the run
thirds = [
    (timestamps[0], timestamps[0] + third_duration),  # First third
    (timestamps[0] + third_duration, timestamps[0] + 2 * third_duration),  # Second third
    (timestamps[0] + 2 * third_duration, timestamps[-1])  # Third third
]

# Generate plots function
def generate_plots(pressure_values, method, output_prefix):
    try:
        # Interpolate pressure data using RBF and GridData
        interpolated_pressure_rbf = interpolate_pressure_data(pressure_values, method='rbf')
        interpolated_pressure_grid = interpolate_pressure_data(pressure_values, method='griddata')
    
        # Normalize the pressure data
        normalized_pressure_rbf = normalize(interpolated_pressure_rbf)
        normalized_pressure_grid = normalize(interpolated_pressure_grid)
    
        # RBF plot (Gradient)
        fig_rbf, ax_rbf = plt.subplots(figsize=(8, 8))
        ax_grid.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False, labeltop=False, labelright=False)
        pressure_img_rbf = ax_rbf.imshow(normalized_pressure_rbf, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_rbf.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="black", lw=2, label="Right Foot")
        ax_rbf.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="black", lw=2, label="Left Foot")
        ax_rbf.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=50)
        ax_rbf.set_title(f"Pressure Mapping (Gradient View) - {output_prefix}")
        cbar_rbf = fig_rbf.colorbar(pressure_img_rbf, ax=ax_rbf)
        cbar_rbf.set_label('Pressure Value (Normalized)')
        
        # Save RBF plot
        rbf_output_file = f"{output_prefix}_gradient.png"
        fig_rbf.savefig(rbf_output_file, bbox_inches='tight')
        plt.close(fig_rbf)
        print(f"Saved RBF plot: {rbf_output_file}")

        # GridData plot (Section)
        fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
        ax_grid.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False, labeltop=False, labelright=False)
        pressure_img_grid = ax_grid.imshow(normalized_pressure_grid, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_grid.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="black", lw=2, label="Right Foot")
        ax_grid.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="black", lw=2, label="Left Foot")
        ax_grid.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=50)
        ax_grid.set_title(f"Pressure Mapping (Section View) - {output_prefix}")
        cbar_grid = fig_grid.colorbar(pressure_img_grid, ax=ax_grid)
        cbar_grid.set_label('Pressure Value (Normalized)')
        
        # Save GridData plot
        grid_output_file = f"{output_prefix}_section.png"
        fig_grid.savefig(grid_output_file, bbox_inches='tight')
        plt.close(fig_grid)
        print(f"Saved GridData plot: {grid_output_file}")

        return True

    except Exception as e:
        print(f"Error generating plots for {output_prefix}: {e}")
        return False

# GitHub upload function
def upload_file_to_github(file_path, github_path):
    try:
        # Authenticate
        g = Github(os.getenv('GITHUB_TOKEN'))
        repo = g.get_repo("PressureSole/srdesign")
        
        # Read the file
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        # Check if file already exists
        try:
            existing_file = repo.get_contents(github_path)
            # If file exists, update it
            repo.update_file(
                path=github_path, 
                message=f"Update {file_path}", 
                content=file_content, 
                sha=existing_file.sha, 
                branch="main"
            )
            print(f"Updated existing file: {github_path}")
        except:
            # If file doesn't exist, create it
            repo.create_file(
                path=github_path, 
                message=f"Add {file_path}", 
                content=file_content, 
                branch="main"
            )
            print(f"Created new file: {github_path}")
    
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

# Main execution
def main():
    try:
        # Full dataset plots
        full_data = np.mean(sensor_pressures, axis=1)

        # Symmetry plots
        generate_plots(full_data, method='griddata', output_prefix='symmetry')
    
        # Generate plots for each third of the dataset
        for i, (start_time, end_time) in enumerate(thirds):
            pressure_values = get_pressure_data_for_third(start_time, end_time)
            print(f"Third {i+1}: {pressure_values}")
        
            # Fatigue plots for each third
            generate_plots(pressure_values, method='rbf', output_prefix=f"fatigue{i+1}")

        # Timestamp for GitHub upload
        timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())

        # List of images to upload
        images = [
            ("symmetry_gradient.png", f"images/symmetry_gradient_{timestamp}.png"),
            ("symmetry_section.png", f"images/symmetry_section_{timestamp}.png"),
            ("fatigue1_gradient.png", f"images/fatigue1_gradient_{timestamp}.png"),
            ("fatigue1_section.png", f"images/fatigue1_section_{timestamp}.png"),
            ("fatigue2_gradient.png", f"images/fatigue2_gradient_{timestamp}.png"),
            ("fatigue2_section.png", f"images/fatigue2_section_{timestamp}.png"),
            ("fatigue3_gradient.png", f"images/fatigue3_gradient_{timestamp}.png"),
            ("fatigue3_section.png", f"images/fatigue3_section_{timestamp}.png")
        ]
    
        # Upload each image
        for local_path, github_path in images:
            if os.path.exists(local_path):
                upload_file_to_github(local_path, github_path)
            else:
                print(f"File not found: {local_path}")

    except Exception as e:
        print(f"An error occurred during script execution: {e}")

# Run the main function
if __name__ == "__main__":
    main()
