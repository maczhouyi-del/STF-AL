from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

torch.set_num_threads(min(torch.get_num_threads(), 4))

from ..datasets import build_eval_dataloader
from ..models import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config, save_config
from ..utils.logging import setup_logger
from ..utils.metrics import classification_metrics
from ..utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate STF-AL models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def resolve_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_eval(model, loader, device, num_classes):
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

    seed_everything(int(config.get("runtime", {}).get("seed", 42)))
    output_dir = Path(config["runtime"]["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, name="stfal_eval")
    save_config(config, output_dir / "config.yaml")
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))

    checkpoint_data = None
    normalization_stats = None
    if args.checkpoint:
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        normalization_stats = checkpoint_data.get("metadata", {}).get("normalization")

    eval_loader, metadata = build_eval_dataloader(config, normalization_stats=normalization_stats)
    config["model"]["input_channels"] = int(metadata["input_channels"])
    config["model"]["num_classes"] = int(metadata["num_classes"])
    model = build_model(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.warning("No checkpoint provided; evaluating current model parameters.")

    metrics, rows = run_eval(model, eval_loader, device, int(config["model"]["num_classes"]))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["y_true", "y_pred", "subject_id"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Artifacts saved to: %s", output_dir)


if __name__ == "__main__":
    main()
