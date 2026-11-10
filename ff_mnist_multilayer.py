from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def overlay_label(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[:, :10] = 0.0
    x[torch.arange(x.size(0), device=x.device), labels] = 1.0
    return x


def sample_negative_labels(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    neg = torch.randint(0, num_classes - 1, labels.shape, device=labels.device)
    neg += (neg >= labels).long()
    return neg


class FFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 2.0,
        lr: float = 0.03,
        device: torch.device | str = "mps",
    ) -> None:
        super().__init__()
        scale = in_features ** -0.5
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * scale,
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=device),
            requires_grad=False,
        )
        self.threshold = threshold
        self.lr = lr

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.normalize(x) @ self.weight.t() + self.bias)

    @torch.no_grad()
    def train_batch(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        x_pos = self.normalize(x_pos)
        x_neg = self.normalize(x_neg)

        z_pos = x_pos @ self.weight.t() + self.bias
        z_neg = x_neg @ self.weight.t() + self.bias

        y_pos = torch.relu(z_pos)
        y_neg = torch.relu(z_neg)

        g_pos = y_pos.pow(2).mean(dim=1)
        g_neg = y_neg.pow(2).mean(dim=1)

        loss_pos = F.softplus(self.threshold - g_pos)
        loss_neg = F.softplus(g_neg - self.threshold)
        loss = loss_pos.mean() + loss_neg.mean()

        batch_size = x_pos.size(0)
        width = y_pos.size(1)

        dloss_dg_pos = -torch.sigmoid(self.threshold - g_pos) / batch_size
        dloss_dg_neg = torch.sigmoid(g_neg - self.threshold) / batch_size

        dz_pos = (2.0 / width) * dloss_dg_pos.unsqueeze(1) * y_pos
        dz_neg = (2.0 / width) * dloss_dg_neg.unsqueeze(1) * y_neg

        grad_w = dz_pos.t() @ x_pos + dz_neg.t() @ x_neg
        grad_b = dz_pos.sum(dim=0) + dz_neg.sum(dim=0)

        self.weight -= self.lr * grad_w
        self.bias -= self.lr * grad_b

        return y_pos.detach(), y_neg.detach(), loss.item()


class FFNet(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        threshold: float = 2.0,
        lr: float = 0.03,
        device: torch.device | str = "mps",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FFLayer(
                    in_features=in_features,
                    out_features=out_features,
                    threshold=threshold,
                    lr=lr,
                    device=device,
                )
                for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )

    @torch.no_grad()
    def train_batch(self, x: torch.Tensor, labels: torch.Tensor) -> list[float]:
        x_pos = overlay_label(x, labels)
        x_neg = overlay_label(x, sample_negative_labels(labels))

        losses = []
        for layer in self.layers:
            x_pos, x_neg, loss = layer.train_batch(x_pos, x_neg)
            losses.append(loss)
        return losses

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        scores = []
        for label in range(10):
            labels = torch.full((x.size(0),), label, device=x.device, dtype=torch.long)
            activations = overlay_label(x, labels)
            total_goodness = torch.zeros(x.size(0), device=x.device)

            for layer in self.layers:
                activations = layer.forward(activations)
                total_goodness += activations.pow(2).mean(dim=1)

            scores.append(total_goodness)

        return torch.stack(scores, dim=1).argmax(dim=1)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "type": "ff_multilayer",
            "layers": [
                {
                    "weight": layer.weight.detach().cpu(),
                    "bias": layer.bias.detach().cpu(),
                    "threshold": layer.threshold,
                }
                for layer in self.layers
            ],
        }


@torch.no_grad()
def evaluate(model: FFNet, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    for images, labels in loader:
        x = images.flatten(1).to(device)
        y = labels.to(device)
        pred = model.predict(x)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-layer Forward-Forward MNIST model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[500, 500])
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--checkpoint", type=Path, default=Path("ff_multilayer.pt"))
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Use Apple Silicon with a recent PyTorch build.")

    torch.manual_seed(0)
    device = torch.device("mps")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    layer_sizes = [28 * 28, *args.hidden_sizes]
    model = FFNet(
        layer_sizes=layer_sizes,
        threshold=args.threshold,
        lr=args.lr,
        device=device,
    )

    for epoch in range(args.epochs):
        running_layer_losses = [0.0 for _ in model.layers]
        for images, labels in train_loader:
            x = images.flatten(1).to(device)
            y = labels.to(device)
            layer_losses = model.train_batch(x, y)
            for index, loss in enumerate(layer_losses):
                running_layer_losses[index] += loss

        test_acc = evaluate(model, test_loader, device)
        mean_layer_losses = [loss / len(train_loader) for loss in running_layer_losses]
        layer_text = " ".join(
            f"layer{index + 1}_loss={loss:.4f}" for index, loss in enumerate(mean_layer_losses)
        )
        print(f"epoch={epoch + 1} {layer_text} test_acc={test_acc:.4%}")

    torch.save(model.checkpoint_payload(), args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
