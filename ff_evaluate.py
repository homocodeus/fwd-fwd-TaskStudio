from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = [
    ROOT / "ff_multilayer.pt",
    ROOT / "ff_single_layer.pt",
]
DEFAULT_REPORT = ROOT / "ff_evaluation.json"


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-4)


def overlay_label(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, :10] = 0.0
    y[torch.arange(x.size(0), device=x.device), labels] = 1.0
    return y


class FFCheckpointModel:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.layers: list[dict[str, Any]] = []
        self.model_label = "Unknown FF checkpoint"
        self._load()

    def _load(self) -> None:
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("layers"), list):
            self.layers = [
                {
                    "weight": layer["weight"].to(self.device, dtype=torch.float32),
                    "bias": layer["bias"].to(self.device, dtype=torch.float32),
                }
                for layer in payload["layers"]
            ]
            self.model_label = f"{len(self.layers)}-layer FF checkpoint"
            return

        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict) or "weight" not in state or "bias" not in state:
            raise ValueError(f"Unsupported checkpoint format in {self.checkpoint_path}")

        self.layers = [
            {
                "weight": state["weight"].to(self.device, dtype=torch.float32),
                "bias": state["bias"].to(self.device, dtype=torch.float32),
            }
        ]
        self.model_label = "1-layer FF checkpoint"

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        scores = []
        for label in range(10):
            labels = torch.full((x.size(0),), label, device=x.device, dtype=torch.long)
            activations = overlay_label(x, labels)
            total_goodness = torch.zeros(x.size(0), device=x.device)
            for layer in self.layers:
                activations = torch.relu(normalize(activations) @ layer["weight"].t() + layer["bias"])
                total_goodness += activations.pow(2).mean(dim=1)
            scores.append(total_goodness)
        return torch.stack(scores, dim=1).argmax(dim=1)


@torch.no_grad()
def evaluate(model: FFCheckpointModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    correct = 0
    total = 0

    for images, labels in loader:
        x = images.flatten(1).to(device)
        y = labels.to(device)
        pred = model.predict(x)
        correct += (pred == y).sum().item()
        total += y.numel()

        for true_label, pred_label in zip(y.cpu(), pred.cpu()):
            confusion[int(true_label), int(pred_label)] += 1

    accuracy = correct / total if total else 0.0
    worst_pairs = []
    for true_digit in range(10):
        row = confusion[true_digit].clone()
        row[true_digit] = -1
        pred_digit = int(row.argmax().item())
        count = int(max(row[pred_digit].item(), 0))
        worst_pairs.append(
            {
                "true_digit": true_digit,
                "predicted_digit": pred_digit,
                "count": count,
            }
        )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion.tolist(),
        "worst_pairs": worst_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Forward-Forward checkpoint on MNIST")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = next((path for path in DEFAULT_CHECKPOINTS if path.exists()), DEFAULT_CHECKPOINTS[0])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FFCheckpointModel(checkpoint_path=checkpoint_path, device=device)
    metrics = evaluate(model, test_loader, device)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "model": model.model_label,
        "device": device_name,
        "evaluation_completed": True,
        **metrics,
    }

    args.report.write_text(json.dumps(report, indent=2))
    print(f"accuracy={metrics['accuracy']:.4%} correct={metrics['correct']} total={metrics['total']}")
    print(f"saved evaluation report to {args.report}")


if __name__ == "__main__":
    main()
