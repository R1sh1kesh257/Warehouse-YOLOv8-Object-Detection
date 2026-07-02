import os
from ultralytics import YOLO

def main():
    # 1. Load your custom trained model weights
    model_path = os.path.join(os.getcwd(), "runs", "detect", "train-3", "weights", "best.pt")
    model = YOLO(model_path)
    
    # 2. Grab the first test image automatically
    test_images_dir = os.path.join(
        os.getcwd(), 
        "forklift-1.v5-raw-images-palletclassonly.yolov8", 
        "test", 
        "images"
    )
    all_test_images = os.listdir(test_images_dir)
    image_name = all_test_images[0] 
    image_path = os.path.join(test_images_dir, image_name)
    
    print("\n=== PROCESSING IMAGE ===")
    
    # 3. Predict and save the visual result as a file (show=False ignores the buggy pop-up)
    model.predict(source=image_path, show=False, save=True, conf=0.20)
    
    print("\nSUCCESS! The AI has processed the image.")
    print("Look at the left sidebar in VS Code for a new folder called 'runs'!")
    print("=========================\n")

if __name__ == '__main__':
    main()