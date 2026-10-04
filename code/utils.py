"""Training and evaluation utilities for AMT-JRD."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SoftCrossEntropy(nn.Module):
    def forward(self, logits, soft_labels):
        return -(soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def gaussian_soft_labels(labels, num_classes=64, sigma=3.0):
    classes = torch.arange(num_classes, device=labels.device, dtype=torch.float32)
    distributions = torch.exp(-0.5 * ((classes[None, :] - labels[:, None]) / sigma) ** 2)
    return distributions / distributions.sum(dim=1, keepdim=True)


def _batch_statistics(logits, labels):
    predictions = logits.argmax(dim=1)
    absolute_error = torch.abs(predictions - labels).sum().item()
    correct = (predictions == labels).sum().item()
    return predictions, absolute_error, correct


def _run_epoch(model, data_loader, device, epoch, optimizer=None, writer=None, csv_filename=None):
    is_training = optimizer is not None
    model.train(is_training)
    criterion = SoftCrossEntropy()
    total_loss = 0.0
    sample_count = 0
    task_errors = [0.0, 0.0, 0.0]
    task_correct = [0.0, 0.0, 0.0]
    progress = tqdm(data_loader, desc="train" if is_training else "val")

    for step, (images, _names, labels, attributes) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attributes = attributes.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(images, attributes)
            task_losses = [
                criterion(task_logits, gaussian_soft_labels(labels[:, index]))
                for index, task_logits in enumerate(logits)
            ]
            # KPD, OD, and IS contribute equally to the multi-task objective.
            loss = sum(task_losses) / 3.0
            if is_training:
                loss.backward()
                optimizer.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")
        total_loss += loss.item()
        sample_count += images.size(0)
        for index, task_logits in enumerate(logits):
            _, error, correct = _batch_statistics(task_logits, labels[:, index])
            task_errors[index] += error
            task_correct[index] += correct

        task_mae = [value / sample_count for value in task_errors]
        mean_mae = sum(task_mae) / 3.0
        progress.set_description(
            f"{'train' if is_training else 'val'} {epoch}: "
            f"loss={total_loss / (step + 1):.3f}, MAE={mean_mae:.3f}"
        )

        if is_training and writer is not None:
            global_step = epoch * len(data_loader) + step
            writer.add_scalar("train_loss", total_loss / (step + 1), global_step)
            writer.add_scalar("train_MAE", mean_mae, global_step)

        if is_training and csv_filename is not None:
            csv_path = Path(csv_filename)
            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(
                    f, fieldnames=["epoch", "step", "train_loss", "train_MAE"]
                )
                if not file_exists:
                    csv_writer.writeheader()
                csv_writer.writerow({
                    "epoch": epoch,
                    "step": step,
                    "train_loss": total_loss / (step + 1),
                    "train_MAE": mean_mae,
                })

    task_mae = [value / sample_count for value in task_errors]
    task_accuracy = [value / sample_count for value in task_correct]
    return (
        total_loss / len(data_loader),
        sum(task_mae) / 3.0,
        *task_mae,
        sum(task_accuracy) / 3.0,
        *task_accuracy,
    )


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler,
                    csv_filename, tb_writer):
    del scheduler  # The scheduler is stepped once per epoch in train.py.
    metrics = _run_epoch(
        model, data_loader, device, epoch, optimizer, tb_writer, csv_filename
    )
    loss, mean_mae, *remaining = metrics
    return loss, mean_mae, tb_writer, *remaining


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    return _run_epoch(model, data_loader, device, epoch)


@torch.no_grad()
def test(model, data_loader, device):
    model.eval()
    criterion = SoftCrossEntropy()
    total_loss = 0.0
    sample_count = 0
    task_errors = [0.0, 0.0, 0.0]
    task_correct = [0.0, 0.0, 0.0]
    all_predictions = [[], [], []]
    all_targets = [[], [], []]
    progress = tqdm(data_loader, desc="test")

    for step, (images, _names, labels, attributes) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attributes = attributes.to(device, non_blocking=True)
        logits = model(images, attributes)
        loss = sum(
            criterion(task_logits, gaussian_soft_labels(labels[:, index]))
            for index, task_logits in enumerate(logits)
        ) / 3.0
        total_loss += loss.item()
        sample_count += images.size(0)

        for index, task_logits in enumerate(logits):
            predictions, error, correct = _batch_statistics(task_logits, labels[:, index])
            task_errors[index] += error
            task_correct[index] += correct
            all_predictions[index].append(predictions.cpu().tolist())
            all_targets[index].append(labels[:, index].cpu().tolist())

        task_mae = [value / sample_count for value in task_errors]
        progress.set_description(
            f"test: loss={total_loss / (step + 1):.3f}, MAE={sum(task_mae) / 3.0:.3f}"
        )

    task_mae = [value / sample_count for value in task_errors]
    task_accuracy = [value / sample_count for value in task_correct]
    return (
        total_loss / len(data_loader),
        sum(task_mae) / 3.0,
        *task_mae,
        sum(task_accuracy) / 3.0,
        *task_accuracy,
        *all_predictions,
        *all_targets,
    )
