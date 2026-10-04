"""Train the AMT-JRD model on MT-JRD."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import MyDataSet_d
from model import swin_small_patch4_window7_224
from utils import evaluate, train_one_epoch


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data_root, split, transform):
    data_root = Path(data_root)
    info_root = data_root / "infos"
    names = read_json(info_root / f"{split}_names.json")
    labels = read_json(info_root / "three_JRD_info.json")
    image_paths = [data_root / "images" / "original" / f"{name}.png" for name in names]
    return MyDataSet_d(
        images_paths=image_paths,
        images_names=names,
        JRD_info_dict=labels,
        object_attributes_path=info_root / "object_attributes.json",
        transform=transform,
    )


def load_swin_pretrained(model, checkpoint_path, device):
    """Load matching layers from the ImageNet-22K Swin-S checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    message = model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} pretrained tensors from {checkpoint_path}")
    print(message)


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = build_dataset(args.data_root, "train", train_transform)
    val_dataset = build_dataset(args.data_root, "val", eval_transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
        "num_workers": args.num_workers,
        "collate_fn": MyDataSet_d.collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = swin_small_patch4_window7_224(num_classes=64)
    if args.weights:
        load_swin_pretrained(model, args.weights, device)

    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    if device.type == "cuda" and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    # All model parameters are optimized. No backbone layer is frozen.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            csv_filename=output_dir / "training_metrics.csv",
            tb_writer=writer,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, epoch)
        train_loss, train_mae = train_metrics[:2]
        val_loss, val_mae = val_metrics[:2]
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_MAE", val_mae, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.3f}, train MAE={train_mae:.3f}, "
            f"val loss={val_loss:.3f}, val MAE={val_mae:.3f}"
        )

        # Select the checkpoint using the equally averaged MAE of KPD, OD, and IS.
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "amt_jrd_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    writer.close()
    print(f"Best validation MAE: {best_mae:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMT-JRD")
    parser.add_argument("--data_root", type=str, default="./data/MT-JRD")
    parser.add_argument("--weights", type=str, default="", help="ImageNet-22K Swin-S checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/amt_jrd")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())
