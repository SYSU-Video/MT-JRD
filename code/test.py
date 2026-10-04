"""Evaluate AMT-JRD and report JRD error metrics for all three tasks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataSet_d
from model import swin_small_patch4_window7_224
from utils import test


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_jrd_metrics(predicted_jrds, ground_truth_jrds):
    """Calculate global MAE, local MAE, and population std of signed errors."""
    predictions = np.asarray(predicted_jrds, dtype=np.float64)
    targets = np.asarray(ground_truth_jrds, dtype=np.float64)
    errors = predictions - targets
    local_mask = (targets >= 27) & (targets <= 51)
    return {
        "E_A": float(np.mean(np.abs(errors))),
        "E_[27,51]": float(np.mean(np.abs(errors[local_mask]))),
        "sigma_e": float(np.std(errors, ddof=0)),
    }


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def flatten(batches):
    return [value for batch in batches for value in batch]


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    info_root = data_root / "infos"
    test_names = read_json(info_root / "test_names.json")
    labels = read_json(info_root / "three_JRD_info.json")
    image_paths = [data_root / "images" / "original" / f"{name}.png" for name in test_names]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = MyDataSet_d(
        images_paths=image_paths,
        images_names=test_names,
        JRD_info_dict=labels,
        object_attributes_path=info_root / "object_attributes.json",
        transform=transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    model = swin_small_patch4_window7_224(num_classes=64)
    load_checkpoint(model, args.checkpoint, device)
    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    if device.type == "cuda" and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    outputs = test(model=model, data_loader=data_loader, device=device)
    prediction_batches = outputs[9:12]
    target_batches = outputs[12:15]
    predictions = [flatten(values) for values in prediction_batches]
    targets = [flatten(values) for values in target_batches]

    # Internally, labels and outputs are ordered as KPD, OD, and IS.
    task_names = ("KPD", "OD", "IS")
    metrics = {
        task: calculate_jrd_metrics(pred, target)
        for task, pred, target in zip(task_names, predictions, targets)
    }
    metrics["Avg."] = {
        metric: float(np.mean([metrics[task][metric] for task in task_names]))
        for metric in ("E_A", "E_[27,51]", "sigma_e")
    }

    print(f"{'Task':<8}{'E_A':>12}{'E_[27,51]':>16}{'sigma_e':>14}")
    for task in (*task_names, "Avg."):
        result = metrics[task]
        print(
            f"{task:<8}{result['E_A']:>12.3f}"
            f"{result['E_[27,51]']:>16.3f}{result['sigma_e']:>14.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "GT_KPD", "Pred_KPD", "GT_OD", "Pred_OD", "GT_IS", "Pred_IS"])
        writer.writerows(zip(test_names, targets[0], predictions[0], targets[1], predictions[1], targets[2], predictions[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AMT-JRD")
    parser.add_argument("--data_root", type=str, default="./data/MT-JRD")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/amt_jrd/amt_jrd_best.pth")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    main(parser.parse_args())
