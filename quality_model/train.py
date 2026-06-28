from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quality_model.dataset import build_dataset
from quality_model.model import QualityModelConfig, build_torch_model
from quality_model.registry import get_action_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny per-action quality scoring model.")
    parser.add_argument("--action-id", required=True, help="Action id to train.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_action_model_spec(args.action_id)
    dataset = build_dataset(spec.action_id, spec.config_path)
    if not dataset:
        raise SystemExit(f"No training samples found for action: {spec.action_id}")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, random_split
    except Exception as exc:  # pragma: no cover - depends on environment
        raise SystemExit(f"PyTorch is required for training: {exc}") from exc

    class RepDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return torch.from_numpy(row["input"]), torch.tensor([row["label"]], dtype=torch.float32)

    torch.manual_seed(7)
    dataset_obj = RepDataset(dataset)
    valid_size = max(1, int(round(len(dataset_obj) * 0.2))) if len(dataset_obj) > 1 else 1
    train_size = max(1, len(dataset_obj) - valid_size)
    train_set, valid_set = random_split(dataset_obj, [train_size, len(dataset_obj) - train_size] if len(dataset_obj) > 1 else [1, 0])
    train_loader = DataLoader(train_set, batch_size=max(1, args.batch_size), shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=max(1, args.batch_size), shuffle=False) if len(valid_set) > 0 else None

    model = build_torch_model(QualityModelConfig(input_channels=spec.input_channels, input_frames=spec.input_frames))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    best_valid = float("inf")
    history: list[dict[str, float]] = []
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            preds = model(batch_inputs.float())
            loss = criterion(preds, batch_targets.float())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        valid_loss = _evaluate_loss(model, valid_loader, criterion)
        history.append({"epoch": epoch, "train_loss": _mean_or_zero(train_losses), "valid_loss": valid_loss})
        if valid_loss <= best_valid:
            best_valid = valid_loss
            torch.save({"state_dict": model.state_dict(), "action_id": spec.action_id}, spec.torch_path)

    metrics = _evaluate_metrics(model, dataset)
    summary = {
        "action_id": spec.action_id,
        "sample_count": len(dataset),
        "history": history,
        "mae": metrics["mae"],
        "spearman": metrics["spearman"],
    }
    (spec.model_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate_loss(model, loader, criterion) -> float:
    if loader is None:
        return 0.0
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            preds = model(batch_inputs.float())
            loss = criterion(preds, batch_targets.float())
            losses.append(float(loss.detach().cpu().item()))
    return _mean_or_zero(losses)


def _evaluate_metrics(model, dataset_rows: list[dict[str, object]]) -> dict[str, float]:
    try:
        import torch
    except Exception:  # pragma: no cover
        return {"mae": 0.0, "spearman": 0.0}
    model.eval()
    preds: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for row in dataset_rows:
            tensor = torch.from_numpy(row["input"]).float()
            pred = float(model(tensor).detach().cpu().numpy().reshape(-1)[0])
            preds.append(pred)
            labels.append(float(row["label"]))
    if not preds:
        return {"mae": 0.0, "spearman": 0.0}
    mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(labels))))
    spearman = _spearman(preds, labels)
    return {"mae": mae, "spearman": spearman}


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_rank = np.argsort(np.argsort(np.asarray(left)))
    right_rank = np.argsort(np.argsort(np.asarray(right)))
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
