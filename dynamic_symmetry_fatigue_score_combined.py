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

# Ensure GitHub connection
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    print(f"Connected to repository: {repo.full_name}")
except Exception as e:
    print(f"GitHub connection error: {e}")
    # Consider adding a way to continue without GitHub upload if connection fails

# Load the Excel file (make sure the file path is correct)
try:
    filename = "Copy of test_lab shortening.xlsx"  # Adjust this path based on the uploaded file
    data = pd.read_excel(filename)
except FileNotFoundError:
    print(f"Error: File {filename} not found. Please check the file path.")
    exit(1)

# Rest of the code remains the same as in the original script...

# Modify the generate_plots function to ensure file saving
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
        pressure_img_rbf = ax_rbf.imshow(normalized_pressure_rbf, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_rbf.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
        ax_rbf.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
        ax_rbf.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
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
        pressure_img_grid = ax_grid.imshow(normalized_pressure_grid, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap="YlOrRd", vmin=0, vmax=1)
        ax_grid.plot(smooth_foot_outline[:, 0], smooth_foot_outline[:, 1], color="red", lw=2, label="Right Foot")
        ax_grid.plot(left_foot_outline[:, 0], left_foot_outline[:, 1], color="blue", lw=2, label="Left Foot")
        ax_grid.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color="black", s=100)
        ax_grid.set_title(f"Pressure Mapping (Section View) - {output_prefix}")
        cbar_grid = fig_grid.colorbar(pressure_img_grid, ax=ax_grid)
        cbar_grid.set_label('Pressure Value (Normalized)')
        
        # Save GridData plot
        grid_output_file = f"{output_prefix}_section.png"
        fig_grid.savefig(grid_output_file, bbox_inches='tight')
        plt.close(fig_grid)
        print(f"Saved GridData plot: {grid_output_file}")

    except Exception as e:
        print(f"Error generating plots for {output_prefix}: {e}")

# Modify GitHub upload section to handle potential errors
def upload_image_to_github(file_path, github_path, commit_message):
    try:
        with open(file_path, "rb") as f:
            image_data = f.read()
        repo.create_file(github_path, commit_message, image_data, branch="main")
        print(f"Successfully uploaded {file_path} to GitHub")
    except Exception as e:
        print(f"Error uploading {file_path} to GitHub: {e}")

# Main execution with error handling
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

    # Upload images to GitHub (optional, can be commented out if GitHub upload is not needed)
    image_files = [
        ("symmetry_gradient.png", f"images/symmetry_gradient_{timestamp}.png", "Upload symmetry gradient mapping"),
        ("symmetry_section.png", f"images/symmetry_section_{timestamp}.png", "Upload symmetry section mapping"),
        ("fatigue1_gradient.png", f"images/fatigue1_gradient_{timestamp}.png", "Upload first third gradient mapping"),
        ("fatigue1_section.png", f"images/fatigue1_section_{timestamp}.png", "Upload first third section mapping"),
        ("fatigue2_gradient.png", f"images/fatigue2_gradient_{timestamp}.png", "Upload second third gradient mapping"),
        ("fatigue2_section.png", f"images/fatigue2_section_{timestamp}.png", "Upload second third section mapping"),
        ("fatigue3_gradient.png", f"images/fatigue3_gradient_{timestamp}.png", "Upload third third gradient mapping"),
        ("fatigue3_section.png", f"images/fatigue3_section_{timestamp}.png", "Upload third third section mapping")
    ]

    for local_path, github_path, commit_message in image_files:
        upload_image_to_github(local_path, github_path, commit_message)

except Exception as e:
    print(f"An error occurred during script execution: {e}")
