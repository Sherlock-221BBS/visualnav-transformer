import os
import cv2

def rotate_frame(frame, angle):
    # Rotate the frame by the given angle
    if angle == 180:
        # Rotate 180 degrees by flipping both vertically and horizontally
        return cv2.flip(frame, -1)
    else:
        # For other angles, use rotation matrix
        (h, w) = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        return rotated

def extract_frames_with_rotation_correction(input_folder, rotation_angle=180):
    # List all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        
        # Create a folder for image frames
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(input_folder, f"{video_name}_imageframes")
        os.makedirs(output_folder, exist_ok=True)

        # Read the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation correction if needed
            if rotation_angle != 0:
                frame = rotate_frame(frame, rotation_angle)

            # Save each frame as an image
            frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

    print("Image frames extracted and saved successfully with rotation correction.")

# Example usage
input_folder = "/home/scai/mtech/aib242295/scratch/navigation_models/datasets/mobile_videos/raw"
extract_frames_with_rotation_correction(input_folder, 0)  # Apply 180-degree rotation
