# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import argparse

# def visualize_waypoint_sequence(predictions_path, images_dir, output_dir=None, sample_count=None, start_index=3):
#     """
#     Visualize waypoints starting from the bottom middle of images.
    
#     Args:
#         predictions_path: Path to the .npy file containing predictions
#         images_dir: Directory containing input images
#         output_dir: Directory to save visualization images (optional)
#         sample_count: Number of samples to visualize (None for all)
#         start_index: Index of the first image that has a prediction (default: 3)
#     """
#     # Create output directory if specified
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     # Load predictions
#     predictions = np.load(predictions_path) * 30
#     print(f"Loaded predictions with shape: {predictions.shape}")
    
#     # Load image files
#     image_files = sorted(
#         [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')],
#         key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
#     )
    
#     # Ensure we have enough images considering the start_index
#     if len(image_files) <= start_index:
#         print(f"Error: Not enough images. Need at least {start_index+1} images but found {len(image_files)}")
#         return
    
#     # Adjust for the start_index offset
#     num_predictions = len(predictions)
#     if len(image_files) < start_index + num_predictions:
#         print(f"Warning: Not enough images for all predictions. Using only {len(image_files) - start_index} predictions.")
#         num_predictions = len(image_files) - start_index
#         predictions = predictions[:num_predictions]
    
#     # Limit samples if specified
#     if sample_count is not None:
#         sample_count = min(sample_count, num_predictions)
#         predictions = predictions[:sample_count]
    
#     print(f"Visualizing {len(predictions)} predictions starting from image index {start_index}...")

#     # For each image with predictions
#     for i in range(len(predictions)):
#         img_index = i + start_index
#         img_file = image_files[img_index]
#         img_path = os.path.join(images_dir, img_file)
#         img = Image.open(img_path)
#         img_array = np.array(img)
        
#         # Create figure
#         plt.figure(figsize=(10, 8))
#         plt.imshow(img_array)
        
#         # Get image dimensions
#         h, w = img_array.shape[:2]
#         start_x, start_y = w // 2, h - 20  # Start from bottom middle
        
#         # Get all 8 waypoints for this image
#         waypoints = predictions[i]  # Shape: (8, 2)
        
#         # Convert relative coordinates to absolute image coordinates
#         abs_waypoints = np.zeros_like(waypoints)
#         for j in range(8):
#             abs_waypoints[j, 0] = start_x - waypoints[j, 1]  # x-coordinate
#             abs_waypoints[j, 1] = start_y - waypoints[j, 0]  # y-coordinate
        
#         # Plot all waypoints
#         for j in range(8):
#             plt.scatter(
#                 abs_waypoints[j, 0], 
#                 abs_waypoints[j, 1], 
#                 color=plt.cm.viridis(j/8), 
#                 s=80, 
#                 marker='o',
#                 edgecolors='black',
#                 zorder=4
#             )
        
#         # Connect waypoints with a line
#         plt.plot(
#             abs_waypoints[:, 0], 
#             abs_waypoints[:, 1], 
#             '-', 
#             color='blue', 
#             linewidth=2,
#             zorder=2
#         )
        
#         plt.title(f"Image: {img_file} (Index: {img_index})\n8-Waypoint Prediction")
        
    
    
#         # Save or display
#         if output_dir:
#             out_path = os.path.join(output_dir, f"viz_{img_file.split('.')[0]}.png")
#             plt.savefig(out_path)
#             plt.close()
#             if i % 10 == 0:
#                 print(f"Saved {i+1}/{len(predictions)} visualizations")
#         else:
#             plt.show()
            
#     if output_dir:
#         print(f"All visualizations saved to {output_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize navigation waypoint sequences")
#     parser.add_argument("--predictions", required=True, help="Path to predictions.npy file")
#     parser.add_argument("--images", required=True, help="Directory containing input images")
#     parser.add_argument("--output", help="Directory to save visualization images (optional)")
#     parser.add_argument("--samples", type=int, help="Number of samples to visualize (default: all)")
#     parser.add_argument("--start-index", type=int, default=3, help="Index of the first image that has a prediction (default: 3)")
    
#     args = parser.parse_args()
    
#     visualize_waypoint_sequence(
#         args.predictions,
#         args.images,
#         args.output,
#         args.samples,
#         args.start_index
#     )


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

def visualize_multiple_waypoint_paths(predictions_path, images_dir, output_dir=None, sample_count=None, start_index=3):
    """
    Visualize multiple potential waypoint paths starting from the bottom middle of images.
    
    Args:
        predictions_path: Path to the .npy file containing predictions with multiple paths
        images_dir: Directory containing input images
        output_dir: Directory to save visualization images (optional)
        sample_count: Number of samples to visualize (None for all)
        start_index: Index of the first image that has a prediction (default: 3)
    """
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load predictions
    predictions = np.load(predictions_path) * 80
    print(f"Loaded predictions with shape: {predictions.shape}")
    
    # Determine the structure of the predictions
    if len(predictions.shape) == 4:  # (num_samples, num_paths, num_waypoints, 2)
        num_samples, num_paths, num_waypoints, _ = predictions.shape
    elif len(predictions.shape) == 3:  # (num_samples, num_paths*num_waypoints, 2)
        num_samples = predictions.shape[0]
        total_waypoints = predictions.shape[1]
        num_paths = 8  # Assuming 8 paths as requested
        num_waypoints = total_waypoints // num_paths
        
        # Reshape to (num_samples, num_paths, num_waypoints, 2)
        predictions = predictions.reshape(num_samples, num_paths, num_waypoints, 2)
    else:
        print(f"Unexpected predictions shape: {predictions.shape}")
        return
    
    print(f"Visualizing {num_paths} paths with {num_waypoints} waypoints each for {num_samples} samples")
    
    # Load image files
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
    )
    
    # Ensure we have enough images considering the start_index
    if len(image_files) <= start_index:
        print(f"Error: Not enough images. Need at least {start_index+1} images but found {len(image_files)}")
        return
    
    # Adjust for the start_index offset
    if len(image_files) < start_index + num_samples:
        print(f"Warning: Not enough images for all predictions. Using only {len(image_files) - start_index} predictions.")
        num_samples = len(image_files) - start_index
        predictions = predictions[:num_samples]
    
    # Limit samples if specified
    if sample_count is not None:
        sample_count = min(sample_count, num_samples)
        predictions = predictions[:sample_count]
        num_samples = sample_count
    
    # Create a colormap for the different paths
    colors = plt.cm.tab10(np.linspace(0, 1, num_paths))
    
    # For each image with predictions
    for i in range(num_samples):
        img_index = i + start_index
        img_file = image_files[img_index]
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        plt.imshow(img_array)
        
        # Get image dimensions
        h, w = img_array.shape[:2]
        start_x, start_y = w // 2, h - 20  # Start from bottom middle
        
        # Plot each path
        for path_idx in range(num_paths):
            # Get all waypoints for this path
            waypoints = predictions[i, path_idx]  # Shape: (8, 2)
            
            # Convert relative coordinates to absolute image coordinates
            abs_waypoints = np.zeros_like(waypoints)
            for j in range(num_waypoints):
                abs_waypoints[j, 0] = start_x - waypoints[j, 1]  # x-coordinate
                abs_waypoints[j, 1] = start_y - waypoints[j, 0]  # y-coordinate
            
            # Plot waypoints for this path
            for j in range(num_waypoints):
                plt.scatter(
                    abs_waypoints[j, 0], 
                    abs_waypoints[j, 1], 
                    color=colors[path_idx], 
                    s=80, 
                    marker='o',
                    edgecolors='black',
                    zorder=4,
                    alpha=0.7
                )
            
            # Connect waypoints with a line
            plt.plot(
                abs_waypoints[:, 0], 
                abs_waypoints[:, 1], 
                '-', 
                color=colors[path_idx], 
                linewidth=2,
                zorder=2,
                label=f"Path {path_idx+1}"
            )
        
        plt.title(f"Image: {img_file} (Index: {img_index})\nMultiple Path Predictions")
        plt.legend(loc='best')
        
        # Save or display
        if output_dir:
            out_path = os.path.join(output_dir, f"viz_{img_file.split('.')[0]}.png")
            plt.savefig(out_path)
            plt.close()
            if i % 10 == 0:
                print(f"Saved {i+1}/{num_samples} visualizations")
        else:
            plt.show()
            
    if output_dir:
        print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multiple navigation waypoint paths")
    parser.add_argument("--predictions", required=True, help="Path to predictions.npy file with multiple paths")
    parser.add_argument("--images", required=True, help="Directory containing input images")
    parser.add_argument("--output", help="Directory to save visualization images (optional)")
    parser.add_argument("--samples", type=int, help="Number of samples to visualize (default: all)")
    parser.add_argument("--start-index", type=int, default=3, help="Index of the first image that has a prediction (default: 3)")
    
    args = parser.parse_args()
    
    visualize_multiple_waypoint_paths(
        args.predictions,
        args.images,
        args.output,
        args.samples,
        args.start_index
    )




# waypoints_to_visualize = ['Dec-06-2022-bww8_00000000_0', 
#                             'Dec-06-2022-bww8_00000000_1', 
#                             'Dec-06-2022-bww8_00000001_0', 
#                             'Dec-06-2022-bww8_00000001_1', 
#                             'Dec-06-2022-bww8_00000001_2', 
#                             'Dec-06-2022-bww8_00000002_0', 
#                             'Dec-06-2022-bww8_00000002_1', 
#                             'Dec-06-2022-bww8_00000002_2', 
#                             'Dec-06-2022-bww8_00000002_3', 
#                             'Dec-06-2022-bww8_00000003_0']

# python visualize_waypoint_predictions.py --predictions /home/scai/mtech/aib242295/scratch/navigation_models/datasets/huron/huron_processed/Dec-06-2022-bww8_00000000_0/predictions_8.npy --images /home/scai/mtech/aib242295/scratch/navigation_models/datasets/huron/huron_processed/Dec-06-2022-bww8_00000000_0/ --output /home/scai/mtech/aib242295/scratch/navigation_models/datasets/huron/huron_processed/Dec-06-2022-bww8_00000000_0/vis_outputs