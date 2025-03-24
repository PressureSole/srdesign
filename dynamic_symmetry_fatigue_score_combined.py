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
GITHUB_TOKEN = os.getenv('EK_TOKEN')  # Set the GitHub Personal Access Token as an environment variable
REPO_NAME = "PressureSole/srdesign"
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

# Make sure the token has access to the PressureSole organization
g = Github(os.getenv("GITHUB_TOKEN"))
# Correctly reference the repo under the organization
repo = g.get_repo("PressureSole/srdesign")
print(repo.full_name)  # Add this to check if you're connecting to the correct repo

# Load the Excel file (uploaded from the website)
filename = "Copy of test_lab shortening.xlsx"  # Adjust this path based on the uploaded file

# Read the data from the Excel file
data = pd.read_excel(filename)

# Extract timestamps and relevant pressure values
timestamps = data["Time"].values
sensor_pressures = [data[f"Sensor_{i+1}"].values for i in range(22)]

# Sensor coordinates
sensor_coords = np.array([
    [5.8, 2.2], [7.0, 4.9], [4.8, 6.2], [6.4, 9.6], [4.5, 12.4], [2.8, 15.1],
    [6.8, 15.0], [4.9, 16.9], [6.5, 18.9], [2.2, 19.2], [4.6, 21.4],
    [-5.8, 2.2], [-7.0, 4.9], [-4.8, 6.2], [-6.4, 9.6], [-4.5, 12.4], [-2.8, 15.1],
    [-6.8, 15.0], [-4.9, 16.9], [-6.5, 18.9], [-2.2, 19.2], [-4.6, 21.4]
])

# Foot outline
foot_outline = np.array([
    [6, 0.4], [3.6, 2.2], [3.7, 7], [3, 10.8], [1.2, 15.7], [1.3, 19.8],
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

# Compute pressure mapping
avg_sensor_pressures = np.mean(sensor_pressures, axis=1)

# --- RBF Interpolation ---
rbf_interpolator = Rbf(sensor_coords[:, 0], sensor_coords[:, 1], avg_sensor_pressures, function='linear')
pressure_data_rbf = rbf_interpolator(X, Y)

# --- GridData Interpolation ---
points = sensor_coords
values = avg_sensor_pressures
grid_points = np.column_stack([X.flatten(), Y.flatten()])
pressure_data_grid = griddata(points, values, grid_points, method='nearest').reshape(X.shape)

# Mask points inside foot outlines
inside_right = r_foot_path.contains_points(grid_points).reshape(X.shape)
inside_left = l_foot_path.contains_points(grid_points).reshape(X.shape)

combined_pressure_rbf = np.where(inside_right | inside_left, pressure_data_rbf, np.nan)
combined_pressure_grid = np.where(inside_right | inside_left, pressure_data_grid, np.nan)

# Normalize pressure data for comparison
def normalize(data):
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    return (data - min_val) / (max_val - min_val)

# Function to interpolate pressure data using RBF and GridData
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

# Function to get pressure data for each third of the run
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

def generate_plots(pressure_values, method, output_prefix):
    # Interpolate pressure data using RBF and GridData
    interpolated_pressure_rbf = interpolate_pressure_data(pressure_values, method='rbf')
    interpolated_pressure_grid = interpolate_pressure_data(pressure_values, method='griddata')
    
    # Normalize the pressure data
    normalized_pressure_rbf = normalize(interpolated_pressure_rbf)
    normalized_pressure_grid = normalize(interpolated_pressure_grid)
    
    # Save RBF plot (Gradient)
    fig_rbf, ax_rbf = plt.subplots(figsize=(8, 8))
    pressure_img_rbf = ax_rbf.imshow(normalized_pressure_rbf, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
    ax_rbf.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
    ax_rbf.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
    ax_rbf.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
    ax_rbf.scatter(0, 0, color="green", s=150, marker="x", label="Origin")
    ax_rbf.set_title(f"Pressure Mapping (Gradient View) - {output_prefix}")
    ax_rbf.legend()
    cbar_rbf = fig_rbf.colorbar(pressure_img_rbf, ax=ax_rbf)
    cbar_rbf.set_label('Pressure Value (Normalized)')


    # Save GridData plot (Section)
    fig_grid, ax_grid = plt.subplots(figsize=(8, 8))
    pressure_img_grid = ax_grid.imshow(normalized_pressure_grid, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
    ax_grid.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
    ax_grid.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
    ax_grid.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
    ax_grid.scatter(0, 0, color="green", s=150, marker="x", label="Origin")
    ax_grid.set_title(f"Pressure Mapping (Section View) - {output_prefix}")
    ax_grid.legend()
    cbar_grid = fig_grid.colorbar(pressure_img_grid, ax=ax_grid)
    cbar_grid.set_label('Pressure Value (Normalized)')
    
    # Save the plot
    grid_output_file = f"{output_prefix}.png"
    print(f"Saving plot as {grid_output_file}")
    fig_grid.savefig(grid_output_file, bbox_inches='tight')
    plt.close(fig_grid)


# Full dataset plots
full_data = np.mean(sensor_pressures, axis=1)

# Symmetry plots
generate_plots(full_data, method='griddata', output_prefix='symmetry_section')
generate_plots(full_data, method='rbf', output_prefix='symmetry_gradient')

# Generate plots for each third of the dataset
for i, (start_time, end_time) in enumerate(thirds):
    pressure_values = get_pressure_data_for_third(start_time, end_time)
    print(f"Third {i+1}: {pressure_values}")
    # Fatigue plots
    generate_plots(pressure_values, method='rbf', output_prefix=f"fatigue{i+1}_gradient")
    generate_plots(pressure_values, method='griddata', output_prefix=f"fatigue{i+1}_section")

# GitHub upload
timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())

# For the full dataset images (RBF and GridData)
with open("symmetry_gradient.png", "rb") as f:
    image_data_rbf = f.read()
repo.create_file(f"images/symmetry_gradient_{timestamp}.png", "Upload symmetry gradient mapping", image_data_rbf, branch="main")

with open("symmetry_section.png", "rb") as f:
    image_data_grid = f.read()
repo.create_file(f"images/symmetry_section_{timestamp}.png", "Upload symmetry section mapping", image_data_grid, branch="main")

# For the first third images (RBF and GridData)
with open("fatigue1_gradient.png", "rb") as f:
    image_data_first_third_rbf = f.read()
repo.create_file(f"images/fatigue1_gradient_{timestamp}.png", "Upload first third gradient mapping", image_data_first_third_rbf, branch="main")

with open("fatigue1_section.png", "rb") as f:
    image_data_first_third_grid = f.read()
repo.create_file(f"images/fatigue1_section_{timestamp}.png", "Upload first third section mapping", image_data_first_third_grid, branch="main")

# For the second third images (RBF and GridData)
with open("fatigue2_gradient.png", "rb") as f:
    image_data_second_third_rbf = f.read()
repo.create_file(f"images/fatigue2_gradient_{timestamp}.png", "Upload second third gradient mapping", image_data_second_third_rbf, branch="main")

with open("fatigue2_section.png", "rb") as f:
    image_data_second_third_grid = f.read()
repo.create_file(f"images/fatigue2_section_{timestamp}.png", "Upload second third section mapping", image_data_second_third_grid, branch="main")

# For the third third images (RBF and GridData)
with open("fatigue3_gradient.png", "rb") as f:
    image_data_third_third_rbf = f.read()
repo.create_file(f"images/fatigue3_gradient_{timestamp}.png", "Upload third third gradient mapping", image_data_third_third_rbf, branch="main")

with open("fatigue3_section.png", "rb") as f:
    image_data_third_third_grid = f.read()
repo.create_file(f"images/fatigue3_section_{timestamp}.png", "Upload third third section mapping", image_data_third_third_grid, branch="main")

