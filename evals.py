# evals.py
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model_comprehensive(model, image_paths: List[str], dataset_name: str,
                                 processor, device: str, tank_class_id: int,
                                 show_detailed: bool = False) -> Dict:
    predictions, confidences, tank_predictions, top_5_predictions, failed = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_paths, desc=f"Evaluating {dataset_name}")):
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

                pred = torch.argmax(logits, dim=-1).item()
                conf = probs[0, pred].item()

                top5_p, top5_i = torch.topk(probs, 5)
                top5 = [(top5_i[0][j].item(), top5_p[0][j].item()) for j in range(5)]

                predictions.append(pred)
                confidences.append(conf)
                top_5_predictions.append(top5)
                if pred == tank_class_id:
                    tank_predictions.append(i)
            except Exception:
                failed.append(image_path)
                continue

    total = len(predictions)
    tank_correct = len(tank_predictions)
    tank_acc = tank_correct / total if total else 0.0

    top5_tank_count = sum(1 for x in top_5_predictions if any(cid == tank_class_id for cid, _ in x))
    top5_acc = top5_tank_count / total if total else 0.0

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    avg_tank_conf = float(np.mean([confidences[i] for i in range(len(predictions)) if predictions[i] == tank_class_id])) if predictions else 0.0

    print(f"\n{dataset_name.upper()} RESULTS:")
    print(f"  Tank accuracy:       {tank_acc:.4f} ({tank_acc*100:.2f}%)")
    print(f"  Top-5 tank accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"  Avg confidence:      {avg_conf:.4f}")
    print(f"  Avg tank confidence: {avg_tank_conf:.4f}")
    print(f"  Failed images:       {len(failed)}")

    return {
        "dataset_name": dataset_name,
        "total_images": total,
        "tank_accuracy": tank_acc,
        "top_5_tank_accuracy": top5_acc,
        "tank_predictions_count": tank_correct,
        "average_confidence": avg_conf,
        "average_tank_confidence": avg_tank_conf,
        "failed_images": len(failed),
        "predictions": predictions,
        "confidences": confidences,
        "top_5_predictions": top_5_predictions
    }

def quick_evaluate_accuracy(model, image_paths: List[str], dataset_name: str, processor, device: str,
                            tank_class_id: int, sample_size: int) -> float:
    model.eval()
    correct = 0
    total = 0
    subset = image_paths[:sample_size] if len(image_paths) > sample_size else image_paths
    with torch.no_grad():
        for p in subset:
            try:
                image = Image.open(p).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                if pred == tank_class_id:
                    correct += 1
                total += 1
            except Exception:
                continue
    return correct / total if total else 0.0

def evaluate_multiclass_accuracy(model, class_validation_sets: Dict[int, List[str]], dataset_name: str,
                                 imagenet_classes: List[str], processor, device: str) -> Dict[int, Dict]:
    results: Dict[int, Dict] = {}
    model.eval()

    print(f"Evaluating {dataset_name}:")
    with torch.no_grad():
        for cid, paths in class_validation_sets.items():
            if not paths:
                continue
            correct = 0
            total = 0
            confs: List[float] = []

            for p in paths:
                try:
                    image = Image.open(p).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    pred = torch.argmax(logits, dim=-1).item()
                    conf = probs[0, pred].item()

                    if pred == cid:
                        correct += 1
                    total += 1
                    confs.append(conf)
                except Exception:
                    continue

            acc = correct / total if total else 0.0
            avg_c = float(np.mean(confs)) if confs else 0.0
            cname = imagenet_classes[cid] if 0 <= cid < len(imagenet_classes) else f"class_{cid}"
            results[cid] = {"class_name": cname, "accuracy": acc, "confidence": avg_c, "total_samples": total}
            print(f"  {cname:40s}: {acc:.4f} ({acc*100:.2f}%) - {total} samples")
    return results

def evaluate_imagenet_overall_performance(model, val_dataset, dataset_name: str,
                                          processor, device: str, sample_size: int = 10000) -> Dict:
    import random
    model.eval()
    total_samples = len(val_dataset)
    actual_n = min(sample_size, total_samples)
    print(f"Evaluating {dataset_name} on {actual_n} ImageNet validation samples...")
    indices = random.sample(range(total_samples), actual_n)

    top1 = 0
    top5 = 0
    total = 0
    failed = 0

    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Evaluating {dataset_name}"):
            try:
                sample = val_dataset[idx]
                image = sample["image"]
                label = sample["label"]
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1).item()
                if pred == label:
                    top1 += 1

                probs = F.softmax(logits, dim=-1)
                _, top5_idx = torch.topk(probs, 5)
                top5_ids = [top5_idx[0][j].item() for j in range(5)]
                if label in top5_ids:
                    top5 += 1
                total += 1
            except Exception:
                failed += 1
                continue

    top1_acc = top1 / total if total else 0.0
    top5_acc = top5 / total if total else 0.0

    print(f"\n{dataset_name.upper()} IMAGENET OVERALL PERFORMANCE:")
    print(f"  Samples evaluated: {total}")
    print(f"  Failed samples:    {failed}")
    print(f"  Top-1 accuracy:    {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-5 accuracy:    {top5_acc:.4f} ({top5_acc*100:.2f}%)")

    return {"top1_accuracy": top1_acc, "top5_accuracy": top5_acc, "total_samples": total, "failed_samples": failed}
