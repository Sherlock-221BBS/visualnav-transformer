import os
from tqdm import tqdm

# Base directory
base_path = "/home/scai/mtech/aib242295/scratch/navigation_models/datasets/mobile_videos/processed/"

# List of elements
elements = os.listdir(base_path)

# Iterate through the list and execute the command
for element in tqdm(elements, total = len(elements)):
    predictions_path = f"{base_path}{element}/predictions_8.npy"
    images_path = f"{base_path}{element}/"
    output_path = f"{base_path}{element}/vis_outputs"

    command = f"python visualize_waypoint_predictions.py --predictions {predictions_path} --images {images_path} --output {output_path}"

    print(f"Executing: {command}")  # Print for debugging
    os.system(command)  # Execute the command

