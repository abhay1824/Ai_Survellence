import cv2
import numpy as np
import os
from ultralytics import YOLO

def main():
    print("Testing Fight Classification Model on PLAIN BLACK image...")
    model_path = r"D:\AI SUrvellinace\runs\classify\fight_classifier\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    try:
        model = YOLO(model_path)
    except Exception as e:
        print("❌ Error loading model:", e)
        return

    # Create plain black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = r"D:\AI SUrvellinace\plain_test.jpg"
    cv2.imwrite(img_path, img)
            
    if os.path.exists(img_path):
        print(f"Running prediction on {img_path}...")
        result = model.predict(img_path)
        if result and len(result) > 0:
            probs = result[0].probs
            top1 = probs.top1
            class_name = model.names[top1]
            confidence = float(probs.top1conf)
            print("\n====================")
            print(f"Prediction Result: {class_name}")
            print(f"Confidence: {confidence:.4f}")
            print("====================")
        else:
            print("No prediction result output.")
        
if __name__ == "__main__":
    main()
