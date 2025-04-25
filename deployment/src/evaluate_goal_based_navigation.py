import os
import shutil
import numpy as np
import torch
from PIL import Image as PILImage
import yaml
from typing import List
from tqdm import tqdm

from utils import transform_images, load_model, to_numpy
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Paths
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_CONFIG_PATH = "../config/models.yaml"
ROBOT_CONFIG_PATH = "../config/robot.yaml"

# Load robot config
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
# Load model config
with open(MODEL_CONFIG_PATH, "r") as f:
    model_paths = yaml.safe_load(f)

# Global Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_queue = []
context_size = None


model_name = "nomad"  

# Load model params
model_config_path = model_paths[model_name]["config_path"]
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

ckpt_path = model_paths[model_name]["ckpt_path"]
if os.path.exists(ckpt_path):
    print(f"Loading model from {ckpt_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

# Load model
model = load_model(ckpt_path, model_params, device)
model = model.to(device)
model.eval()

num_diffusion_iters = model_params["num_diffusion_iters"]
noise_scheduler = DDPMScheduler(
    num_train_timesteps=model_params["num_diffusion_iters"],
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)




def load_topomap_images(dir_path: str) -> List[PILImage.Image]:
    """
    Load and return all topomap images from a directory in sorted order.
    """
    topomap_filenames = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".png") or f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    topomap = [
        PILImage.open(os.path.join(dir_path, f)) for f in topomap_filenames
    ]
    return topomap


def load_input_images(dir_path: str) -> List[str]:
    """
    Load and return input image file paths from a directory in sorted order.
    Ignore non-image files like traj_data.pkl.
    """
    filenames = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".png") or f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )
    file_paths = [os.path.join(dir_path, f) for f in filenames]
    return file_paths


def copy_images_to_topomap(input_dir: str, target_dir: str):
    """
    Copy images from input directory to the topomap directory.
    Ignore non-image files like traj_data.pkl.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filenames = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )

    for filename in filenames:
        src = os.path.join(input_dir, filename)
        dest = os.path.join(target_dir, filename)
        shutil.copy(src, dest)

    # print(f"Copied {len(filenames)} images to {target_dir}")


def update_context(obs_img):
    """
    Add image to context queue and maintain proper size.
    """
    global context_queue, context_size

    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def predict(model, image_paths: List[str], model_params: dict, num_samples: int):
    """
    Pass the images through the model and return the predicted waypoints.
    """
    global context_size

    context_size = model_params["context_size"]
    predictions = []

    for img_path in image_paths:
        obs_img = PILImage.open(img_path)
        update_context(obs_img)

        # Only process when context is ready
        if len(context_queue) == context_size + 1:
            obs_images = transform_images(
                context_queue, model_params["image_size"], center_crop=False
            )
            obs_images = obs_images.to(device)

            # Create fake goal and mask
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device)

            # Infer action
            with torch.no_grad():
                # Encoder vision features
                obs_cond = model(
                    'vision_encoder',
                    obs_img=obs_images,
                    goal_img=fake_goal,
                    input_goal_mask=mask
                )

                # Repeat obs_cond to match sample size
                # print(f"obs_cond shape: {obs_cond.shape}")
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(num_samples, 1, 1)

                # Initialize action from Gaussian noise
                naction = torch.randn(
                    (num_samples, model_params["len_traj_pred"], 2), device=device
                )

                # Initialize scheduler
                noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])

                for k in noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # Remove noise using diffusion step
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

                # Get action from model output
                # print(f"naction shape: {naction.shape}")
                naction = to_numpy(get_action(naction))
                # naction = naction[0]
                # print(f"naction shape: {naction.shape}")
                # chosen_waypoint = naction[2]
                chosen_waypoint = naction
                if model_params["normalize"]:
                    chosen_waypoint *= (MAX_V / RATE)
                predictions.append(chosen_waypoint)

    return predictions



def clear_topomap_images(dir_path: str):
    """
    Remove all files in the topomap directory.
    """
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def main():
    
    trajectories_folder = "/home/scai/mtech/aib242295/scratch/navigation_models/datasets/mobile_videos/processed"

    for trajectory_folder in tqdm(sorted(os.listdir(trajectories_folder)), total=len(os.listdir(trajectories_folder))):
        trajectory_folder_path = os.path.join(trajectories_folder, trajectory_folder)

        if not os.path.isdir(trajectory_folder_path):
            continue
        
        # Clear previous topomap images
        clear_topomap_images(TOPOMAP_IMAGES_DIR)
        
        # Copy images to topomap folder from the current trajectory folder
        copy_images_to_topomap(trajectory_folder_path, TOPOMAP_IMAGES_DIR)
        input_image_paths = load_input_images(TOPOMAP_IMAGES_DIR)

        if len(input_image_paths) == 0:
            print(f"No input images found in {trajectory_folder_path}")
            continue

        # Run predictions
        # print(f"Generating predictions for {trajectory_folder}...")
        predictions = predict(model, input_image_paths, model_params, num_samples=8)
        # print(f"predictions for each image shape: {predictions[0].shape}")
        # print(f"length of predictions: {len(predictions)}")
        print(f"next 8 waypoints in first image {predictions[0]}")

        # Ensure output directory exists
        output_file = os.path.join(trajectory_folder_path, "predictions_8.npy")
        os.makedirs(trajectory_folder_path, exist_ok=True)

        # Save predictions
        np.save(output_file, predictions)
        # print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()

