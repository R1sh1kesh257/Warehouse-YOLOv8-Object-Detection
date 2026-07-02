import os
import cv2
from ultralytics import YOLO

def main():
    # 1. Load your custom trained model weights
    model_path = os.path.join(os.getcwd(), "runs", "detect", "train-3", "weights", "best.pt")
    model = YOLO(model_path)
    
    # 2. Point to your downloaded video file
    video_name = "warehouse.mp4"  # <-- Make sure your video matches this name exactly!
    video_path = os.path.join(os.getcwd(), video_name)
    
    # Check if the video actually exists before running
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Could not find '{video_name}' in your project folder!")
        print("Please download a clip, drop it here, and rename it to 'warehouse.mp4'.\n")
        return

    print("\n=== PROCESSING VIDEO TRACKING ===")
    print("The AI is processing the video frame-by-frame...")
    
    # 3. Run prediction on the video and save the tracked output
    # show=False keeps it stable, save=True creates a brand new fully-tracked MP4 video!
    model.predict(source=video_path, show=False, save=True, conf=0.20)
    
    print("\nSUCCESS! The AI has finished tracking the video.")
    print("Look inside your 'runs/detect/' folder for a new 'predictX' folder to watch it!")
    print("=================================\n")

if __name__ == '__main__':
    main()