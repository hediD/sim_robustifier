import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

def evaluate_model_comprehensive(model, image_paths, dataset_name, processor, device, tank_class_id):
    predictions, confidences, tank_predictions, top_5_predictions, failed_images = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(tqdm(image_paths, desc=f"Evaluating {dataset_name}")):
            try:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1).item()
                confidence = probs[0, pred].item()

                predictions.append(pred)
                confidences.append(confidence)
                if pred == tank_class_id:
                    tank_predictions.append(i)
            except:
                failed_images.append(path)
    # compute metrics...
    tank_acc = len(tank_predictions)/len(predictions) if predictions else 0
    return {"dataset_name": dataset_name, "tank_accuracy": tank_acc}
