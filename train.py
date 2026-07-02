import os
import multiprocessing
from ultralytics import YOLO

if __name__ == '__main__':
    # Prevents Windows/OneDrive background freezes
    multiprocessing.freeze_support()
    
    current_folder = os.getcwd()
    
    # We matched this EXACTLY to your Get-ChildItem screenshot name!
    dataset_folder = "forklift-1.v5-raw-images-palletclassonly.yolov8"
    yaml_path = os.path.join(current_folder, dataset_folder, "data.yaml")
    
    print("\n=== STARTING TRAINING PROCESS ===")
    print(f"Loading data from: {yaml_path}")
    
    # Load the base YOLO model
    model = YOLO("yolov8n.pt")
    
    # Train the model 
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        device="cpu",
        workers=0  # Safe mode for OneDrive environments
    )