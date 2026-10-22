import cv2
import urllib.request
import os
from ultralytics import YOLO

def main():
    print("Testing Fight Classification Model Integration...")
    model_path = r"D:\AI SUrvellinace\weights\fight_yolo11n_cls.pt"
    if not os.path.exists(model_path):
        model_path = r"D:\AI SUrvellinace\runs\classify\fight_classifier\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", e)
        return

    # Download a sample fight image if it doesn't exist locally
    img_path = r"D:\AI SUrvellinace\test_fight.jpg"
    if not os.path.exists(img_path):
        print("Downloading sample fight image...")
        try:
            # Random fight snapshot from a generic URL (just for a quick test)
            url = "https://raw.githubusercontent.com/jaysr/Violence-Detection/master/dataset/violence/1.jpg"
            urllib.request.urlretrieve(url, img_path)
        except Exception as e:
            print("Failed to download image. Please manually place a 'test_fight.jpg' in the directory.", e)
            
    if os.path.exists(img_path):
        print(f"Running prediction on {img_path}...")
        result = model.predict(img_path)
        if result and len(result) > 0:
            probs = result[0].probs
            # It's a classification model, so result[0].probs holds the Top1 or confidence
            top1 = probs.top1
            class_name = model.names[top1]
            confidence = float(probs.top1conf)
            print("\n====================")
            print(f"Prediction Result: {class_name}")
            print(f"Confidence: {confidence:.4f}")
            print("Probabilities Object:", probs)
            print("====================")
        else:
            print("No prediction result output.")
    else:
        print("Image not found. Skip prediction.")
        
if __name__ == "__main__":
    main()
