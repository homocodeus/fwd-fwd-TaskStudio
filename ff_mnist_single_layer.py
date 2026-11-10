import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
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
        x = self.normalize(x)
        return torch.relu(x @ self.weight.t() + self.bias)

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).pow(2).mean(dim=1)

    @torch.no_grad()
    def train_batch(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> float:
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

        return loss.item()


@torch.no_grad()
def predict(layer: FFLayer, x: torch.Tensor) -> torch.Tensor:
    scores = []
    for label in range(10):
        labels = torch.full((x.size(0),), label, device=x.device, dtype=torch.long)
        scores.append(layer.goodness(overlay_label(x, labels)))
    return torch.stack(scores, dim=1).argmax(dim=1)


@torch.no_grad()
def evaluate(layer: FFLayer, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    for images, labels in loader:
        x = images.flatten(1).to(device)
        y = labels.to(device)
        pred = predict(layer, x)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Use Apple Silicon with a recent PyTorch build.")

    torch.manual_seed(0)
    device = torch.device("mps")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    layer = FFLayer(
        in_features=28 * 28,
        out_features=500,
        threshold=2.0,
        lr=0.03,
        device=device,
    )

    for epoch in range(5):
        running_loss = 0.0
        for images, labels in train_loader:
            x = images.flatten(1).to(device)
            y = labels.to(device)
            y_neg = sample_negative_labels(y)

            loss = layer.train_batch(
                overlay_label(x, y),
                overlay_label(x, y_neg),
            )
            running_loss += loss

        test_acc = evaluate(layer, test_loader, device)
        mean_loss = running_loss / len(train_loader)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f} test_acc={test_acc:.4%}")

    checkpoint_path = Path("ff_single_layer.pt")
    torch.save(
        {
            "weight": layer.weight.detach().cpu(),
            "bias": layer.bias.detach().cpu(),
            "threshold": layer.threshold,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
