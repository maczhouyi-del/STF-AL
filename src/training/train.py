from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path
from typing import Dict

import torch

torch.set_num_threads(min(torch.get_num_threads(), 4))

from ..datasets import build_dataloaders
from ..losses import classification_loss, domain_adversarial_loss, frequency_domain_similarity_matrix
from ..models import build_model
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.config import load_config, save_config
from ..utils.logging import setup_logger
from ..utils.metrics import classification_metrics
from ..utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train STF-AL models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target-partition", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int):
    model.eval()
    y_true, y_pred, rows = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        pred = out["source_logits"].argmax(dim=-1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        for truth, prediction, subject_id in zip(y.cpu().tolist(), pred.cpu().tolist(), batch["subject_id"].tolist()):
            rows.append({"y_true": truth, "y_pred": prediction, "subject_id": int(subject_id)})
    return classification_metrics(y_true, y_pred, num_classes), rows


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
    if args.device is not None:
        config.setdefault("runtime", {})["device"] = args.device
    if args.output_dir is not None:
        config.setdefault("runtime", {})["output_dir"] = args.output_dir
    if args.target_partition is not None:
        config.setdefault("dataset", {})["target_partition"] = args.target_partition

    seed_everything(int(config.get("runtime", {}).get("seed", 42)))
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    save_config(config, output_dir / "config.yaml")

    logger.info("Using device: %s", device)
    source_loader, target_loader, source_val_loader, target_eval_loader, metadata = build_dataloaders(config)
    config["model"]["input_channels"] = int(metadata["input_channels"])
    config["model"]["num_classes"] = int(metadata["num_classes"])
    save_config(config, output_dir / "config.resolved.yaml")
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if metadata.get("normalization") is not None:
        with (output_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(metadata["normalization"], f, indent=2)
    logger.info("Dataset metadata: %s", metadata)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, optimizer=optimizer, map_location=device)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    num_classes = int(config["model"]["num_classes"])
    best_metric = -1.0
    history = []
    max_steps = config["training"].get("max_steps_per_epoch")
    max_steps = None if max_steps is None else int(max_steps)

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        target_iter = cycle(target_loader)
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        steps = 0
        for source_batch in source_loader:
            target_batch = next(target_iter)
            source_x = source_batch["x"].to(device)
            source_y = source_batch["y"].to(device)
            target_x = target_batch["x"].to(device)

            out = model(
                source_x,
                target_x=target_x,
                grl_lambda=float(config["training"].get("grl_lambda", 1.0)),
            )
            class_loss = classification_loss(out["source_logits"], source_y)
            domain_loss = torch.zeros((), device=device)
            if "source_domain_logits" in out:
                similarity = None
                if model.use_fdsm:
                    similarity = frequency_domain_similarity_matrix(
                        out.get("source_frequency_features"),
                        out.get("target_frequency_features"),
                    )
                domain_loss = domain_adversarial_loss(
                    out["source_domain_logits"],
                    out["target_domain_logits"],
                    similarity_matrix=similarity,
                )
            loss = class_loss + float(config["training"]["lambda_domain"]) * domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_class_loss += float(class_loss.item())
            total_domain_loss += float(domain_loss.item())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        source_val_metrics, _ = evaluate_model(model, source_val_loader, device, num_classes)
        epoch_record = {
            "epoch": epoch,
            "loss": total_loss / max(1, steps),
            "class_loss": total_class_loss / max(1, steps),
            "domain_loss": total_domain_loss / max(1, steps),
            **{f"source_val_{key}": value for key, value in source_val_metrics.items()},
        }
        history.append(epoch_record)
        logger.info("Epoch %d | %s", epoch, epoch_record)
        monitor_metric = config["training"].get("monitor_metric", "macro_f1")
        monitor_value = float(source_val_metrics[monitor_metric])
        if monitor_value > best_metric:
            best_metric = monitor_value
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, epoch, source_val_metrics, config, metadata=metadata)

    final_training_source_val_metrics, _ = evaluate_model(model, source_val_loader, device, num_classes)
    save_checkpoint(
        output_dir / "final_checkpoint.pt",
        model,
        optimizer,
        int(config["training"]["epochs"]),
        final_training_source_val_metrics,
        config,
        metadata=metadata,
    )

    best_checkpoint = load_checkpoint(output_dir / "best_checkpoint.pt", model, map_location=device)
    selected_source_val_metrics, _ = evaluate_model(model, source_val_loader, device, num_classes)
    final_target_metrics, prediction_rows = evaluate_model(model, target_eval_loader, device, num_classes)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "final_training_source_val": final_training_source_val_metrics,
                "selected_source_val": selected_source_val_metrics,
                "final_target": final_target_metrics,
                "checkpoint_selection": "source_val",
                "selected_checkpoint": "best_checkpoint.pt",
                "selected_epoch": int(best_checkpoint["epoch"]),
            },
            f,
            indent=2,
        )
    if config.get("runtime", {}).get("save_predictions", True):
        with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["y_true", "y_pred", "subject_id"])
            writer.writeheader()
            writer.writerows(prediction_rows)
    logger.info("Training finished. Selected source-val metrics: %s", selected_source_val_metrics)
    logger.info("Training finished. Final target metrics: %s", final_target_metrics)
    logger.info("Artifacts saved to: %s", output_dir)


if __name__ == "__main__":
    main()
