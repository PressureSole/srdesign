import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import Rbf, splprep, splev
from github import Github
from io import BytesIO

# GitHub repository and token
GITHUB_TOKEN = os.getenv('EK_TOKEN')  # Set the GitHub Personal Access Token as an environment variable
REPO_NAME = "jakewang21/srdesign"
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

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

# Compute pressure mapping (method='multiquadric' for the first image)
avg_sensor_pressures = np.mean(sensor_pressures, axis=1)
rbf_multi = Rbf(sensor_coords[:, 0], sensor_coords[:, 1], avg_sensor_pressures, function='multiquadric')
pressure_data_grid_multi = rbf_multi(X, Y)

# Compute pressure mapping (method='nearest' for the second image)
rbf_nearest = Rbf(sensor_coords[:, 0], sensor_coords[:, 1], avg_sensor_pressures, function='nearest')
pressure_data_grid_nearest = rbf_nearest(X, Y)

# Mask points inside foot outlines
points = np.column_stack([X.flatten(), Y.flatten()])
inside_right = r_foot_path.contains_points(points).reshape(X.shape)
inside_left = l_foot_path.contains_points(points).reshape(X.shape)
combined_pressure_data_multi = np.where(inside_right | inside_left, pressure_data_grid_multi, np.nan)
combined_pressure_data_nearest = np.where(inside_right | inside_left, pressure_data_grid_nearest, np.nan)

# Normalize pressure data
min_pressure, max_pressure = np.nanmin(combined_pressure_data_multi), np.nanmax(combined_pressure_data_multi)
normalized_pressure_data_multi = (combined_pressure_data_multi - min_pressure) / (max_pressure - min_pressure)

min_pressure_nearest, max_pressure_nearest = np.nanmin(combined_pressure_data_nearest), np.nanmax(combined_pressure_data_nearest)
normalized_pressure_data_nearest = (combined_pressure_data_nearest - min_pressure_nearest) / (max_pressure_nearest - min_pressure_nearest)

# Plotting first image (multiquadric method)
fig, ax = plt.subplots(figsize=(12, 8))
pressure_img = ax.imshow(normalized_pressure_data_multi, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
ax.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
ax.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
ax.scatter(0, 0, color="green", s=150, marker="x", label="Origin")
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_title("Average Foot Pressure Mapping (Multiquadric)")
ax.legend()
cbar = fig.colorbar(pressure_img, ax=ax)
cbar.set_label('Pressure Value (Normalized from 0 to 1)', rotation=270, labelpad=20)

# Save the first image
output_file_multi = "dynamic_symmetry_score_visualization.png"
plt.savefig(output_file_multi, bbox_inches='tight')
plt.close()

# Plotting second image (nearest method)
fig, ax = plt.subplots(figsize=(12, 8))
pressure_img_nearest = ax.imshow(normalized_pressure_data_nearest, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
ax.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
ax.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
ax.scatter(0, 0, color="green", s=150, marker="x", label="Origin")
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_title("Average Foot Pressure Mapping (Nearest)")
ax.legend()
cbar = fig.colorbar(pressure_img_nearest, ax=ax)
cbar.set_label('Pressure Value (Normalized from 0 to 1)', rotation=270, labelpad=20)

# Save the second image
output_file_nearest = "dynamic_symmetry_score_visualization_nearest.png"
plt.savefig(output_file_nearest, bbox_inches='tight')
plt.close()

# GitHub upload
timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
new_output_file_multi = f"dynamic_symmetry_score_visualization_{timestamp}.png"
new_output_file_nearest = f"dynamic_symmetry_score_visualization_nearest_{timestamp}.png"

# Read the images as binary streams
with open(output_file_multi, "rb") as f:
    image_data_multi = f.read()

with open(output_file_nearest, "rb") as f:
    image_data_nearest = f.read()

# Upload both images to GitHub
repo.create_file(f"images/{new_output_file_multi}", "Upload multiquadric symmetry score plot", image_data_multi, branch="main")
repo.create_file(f"images/{new_output_file_nearest}", "Upload nearest symmetry score plot", image_data_nearest, branch="main")

print(f"Images uploaded to GitHub repository as {new_output_file_multi} and {new_output_file_nearest}!")
