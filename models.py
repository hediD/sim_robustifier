# models.py
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import ViTForImageClassification

from evals import quick_evaluate_accuracy

def create_model(model_name: str, device: str = "cuda"):
    model = ViTForImageClassification.from_pretrained(model_name)
    # Freeze backbone; train classifier head only
    for p in model.vit.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.to(device)
    return model

def train_model(
    model,
    dataloader,
    device: str,
    epochs: int,
    processor,  # not used here, kept for symmetry/extensibility
    tank_class_id: int,
    real_image_paths: List[str],
    sim_eval_paths: List[str],
    eval_sample_size: int,
    enable_eval: bool = True,
) -> torch.nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    total_batches = len(dataloader)
    eval_interval = max(1, total_batches // 3) if enable_eval else 10**9

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (pixel_values, labels) in enumerate(pbar):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

            if enable_eval and batch_idx > 0 and (batch_idx % eval_interval == 0):
                real_acc = quick_evaluate_accuracy(model, real_image_paths, "real", processor, device, tank_class_id, eval_sample_size)
                sim_acc  = quick_evaluate_accuracy(model, sim_eval_paths, "sim",  processor, device, tank_class_id, eval_sample_size)
                print(f"\n  [Epoch {epoch+1}, Batch {batch_idx}/{total_batches}] Real: {real_acc:.3f}, Sim: {sim_acc:.3f}")
                model.train()

            pbar.set_postfix({"Loss": f"{(epoch_loss/(batch_idx+1)):.4f}", "Acc": f"{(correct/max(1,total)):.4f}"})

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.4f}, Accuracy = {correct/max(1,total):.4f}")

    print("✅ Training completed!")
    return model
