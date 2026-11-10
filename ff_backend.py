from __future__ import annotations

import argparse
import json
import platform
import subprocess
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import torch
except Exception:  # pragma: no cover - backend remains usable without torch
    torch = None


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
DEFAULT_CHECKPOINTS = [
    ROOT / "ff_multilayer.pt",
    ROOT / "ff_single_layer.pt",
]
DEFAULT_EVALUATION_REPORT = ROOT / "ff_evaluation.json"
DEFAULT_COMPOSITION_REPORT = ROOT / "ff_composition.json"


def read_hw_model() -> str | None:
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            check=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None

    model = result.stdout.strip()
    return model or None


def normalize(x: "torch.Tensor") -> "torch.Tensor":
    return x / (x.norm(p=2, dim=1, keepdim=True) + 1e-4)


def overlay_label(x: "torch.Tensor", label: int) -> "torch.Tensor":
    y = x.clone()
    y[:, :10] = 0.0
    y[:, label] = 1.0
    return y


def select_torch_device() -> str:
    if torch is None:
        return "unavailable"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.hw_model = read_hw_model()
        self.device_name = select_torch_device()
        self.device = torch.device(self.device_name) if torch is not None and self.device_name != "unavailable" else None
        self.model_name = "Torch heuristic scorer"
        self.inference_mode = f"Heuristic backend on {self.device_name}"
        self.ff_layers: list[dict[str, Any]] = []
        self.evaluation_report = self._load_evaluation_report()
        self.composition_report = self._load_report(DEFAULT_COMPOSITION_REPORT)
        self._try_load_checkpoint()

    def _load_report(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None

        try:
            report = json.loads(path.read_text())
        except Exception:
            return None

        if not isinstance(report, dict):
            return None

        return report

    def _load_evaluation_report(self) -> dict[str, Any] | None:
        report = self._load_report(DEFAULT_EVALUATION_REPORT)
        if report is None:
            return None

        if self.checkpoint_path is None:
            return report

        checkpoint = report.get("checkpoint")
        checkpoint_name = report.get("checkpoint_name")
        resolved = str(self.checkpoint_path.resolve())
        if checkpoint == resolved or checkpoint_name == self.checkpoint_path.name:
            return report
        return None

    def _try_load_checkpoint(self) -> None:
        if torch is None or self.device is None or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("layers"), list):
            loaded_layers = self._load_multilayer(payload["layers"])
            if not loaded_layers:
                return

            self.ff_layers = loaded_layers
            self.model_name = f"{len(self.ff_layers)}-layer FF checkpoint"
            self.inference_mode = f"Forward-Forward goodness on {self.device_name}"
            return

        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict) or "weight" not in state or "bias" not in state:
            return

        weight = state["weight"]
        bias = state["bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            return

        self.ff_layers = [
            {
                "weight": weight.to(self.device, dtype=torch.float32),
                "bias": bias.to(self.device, dtype=torch.float32),
                "threshold": float(payload["threshold"]) if isinstance(payload, dict) and "threshold" in payload else 2.0,
            }
        ]
        self.model_name = "1-layer FF checkpoint"
        self.inference_mode = f"Forward-Forward goodness on {self.device_name}"

    def _load_multilayer(self, layers: list[Any]) -> list[dict[str, Any]]:
        loaded_layers = []
        for layer in layers:
            if not isinstance(layer, dict):
                return []
            weight = layer.get("weight")
            bias = layer.get("bias")
            if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
                return []

            loaded_layers.append(
                {
                    "weight": weight.to(self.device, dtype=torch.float32),
                    "bias": bias.to(self.device, dtype=torch.float32),
                    "threshold": float(layer.get("threshold", 2.0)),
                }
            )
        return loaded_layers

    def device_label(self) -> str:
        parts = []
        if self.hw_model:
            parts.append(self.hw_model)
        parts.append(platform.machine() or "unknown-arch")
        parts.append(self.device_name)
        return " / ".join(parts)

    def runtime_label(self) -> str:
        if torch is None:
            return "PyTorch unavailable"

        details = [f"PyTorch {torch.__version__}"]
        if hasattr(torch.backends, "mps"):
            details.append(f"mps_built={torch.backends.mps.is_built()}")
            details.append(f"mps_available={torch.backends.mps.is_available()}")
        details.append(f"selected={self.device_name}")
        return " / ".join(details)

    def info(self) -> dict[str, Any]:
        payload = {
            "device": self.device_label(),
            "runtime": self.runtime_label(),
            "model": self.model_name,
            "inference_mode": self.inference_mode,
            "host_os": platform.platform(),
            "hardware_model": self.hw_model,
            "machine": platform.machine(),
            "selected_device": self.device_name,
        }
        if self.evaluation_report:
            payload["evaluation"] = {
                "completed": bool(self.evaluation_report.get("evaluation_completed")),
                "accuracy": self.evaluation_report.get("accuracy"),
                "correct": self.evaluation_report.get("correct"),
                "total": self.evaluation_report.get("total"),
                "checkpoint_name": self.evaluation_report.get("checkpoint_name"),
            }
        else:
            payload["evaluation"] = {
                "completed": False,
            }
        if self.composition_report:
            payload["composition"] = {
                "completed": bool(self.composition_report.get("composition_completed")),
                "mode": self.composition_report.get("mode"),
                "checkpoint_name": self.composition_report.get("checkpoint_name"),
            }
        else:
            payload["composition"] = {
                "completed": False,
            }
        return payload

    def predict(self, pixels: list[float]) -> dict[str, Any]:
        if len(pixels) != 28 * 28:
            raise ValueError("Expected 784 pixels for a 28x28 input")

        if torch is None or self.device is None:
            scores = self._python_heuristic(pixels)
        else:
            x = torch.tensor(pixels, dtype=torch.float32, device=self.device).view(1, -1)
            if self.ff_layers:
                scores = self._ff_scores(x)
            else:
                scores = self._torch_heuristic(x)

        return {
            **self.info(),
            "scores": [float(score) for score in scores],
        }

    def _ff_scores(self, x: "torch.Tensor") -> list[float]:
        scores = []
        for label in range(10):
            activations = overlay_label(x, label)
            total_goodness = torch.zeros(x.size(0), device=self.device)

            for layer in self.ff_layers:
                activations = torch.relu(normalize(activations) @ layer["weight"].t() + layer["bias"])
                total_goodness += activations.pow(2).mean(dim=1)

            scores.append(total_goodness.item())
        return scores

    def _torch_heuristic(self, x: "torch.Tensor") -> list[float]:
        grid = x.view(28, 28)
        energy = grid.sum()

        rows = torch.arange(28, device=self.device, dtype=torch.float32).unsqueeze(1).expand(28, 28)
        cols = torch.arange(28, device=self.device, dtype=torch.float32).unsqueeze(0).expand(28, 28)
        dx = (cols - 13.5).abs() / 13.5
        dy = (rows - 13.5).abs() / 13.5
        center_bias = (grid * (1.0 - (dx + dy) * 0.25)).sum()

        scores = []
        for digit in range(10):
            phase = (digit + 1) * 0.61
            signal = torch.sin(energy * 0.037 + phase) + torch.cos(center_bias * 0.041 - phase)
            score = torch.clamp(0.8 + signal * 0.35 + digit * 0.015, min=0.01)
            scores.append(score.item())
        return scores

    def _python_heuristic(self, pixels: list[float]) -> list[float]:
        energy = sum(pixels)
        center_bias = 0.0
        for index, value in enumerate(pixels):
            row = index // 28
            col = index % 28
            dx = abs(col - 13.5) / 13.5
            dy = abs(row - 13.5) / 13.5
            center_bias += value * (1.0 - (dx + dy) * 0.25)

        import math

        scores = []
        for digit in range(10):
            phase = (digit + 1) * 0.61
            signal = math.sin(energy * 0.037 + phase) + math.cos(center_bias * 0.041 - phase)
            scores.append(max(0.01, 0.8 + signal * 0.35 + digit * 0.015))
        return scores


class FFRequestHandler(SimpleHTTPRequestHandler):
    predictor: Predictor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/info":
            self._send_json(HTTPStatus.OK, self.predictor.info())
            return

        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            pixels = payload["pixels"]
            if not isinstance(pixels, list):
                raise ValueError("pixels must be a list")
            result = self.predictor.predict([float(value) for value in pixels])
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, result)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-Forward frontend backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = next((path for path in DEFAULT_CHECKPOINTS if path.exists()), DEFAULT_CHECKPOINTS[0])

    predictor = Predictor(checkpoint_path=checkpoint_path)
    FFRequestHandler.predictor = predictor

    server = ThreadingHTTPServer((args.host, args.port), FFRequestHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {predictor.device_label()}")
    print(f"Runtime: {predictor.runtime_label()}")
    print(f"Model: {predictor.model_name}")
    if predictor.evaluation_report:
        accuracy = float(predictor.evaluation_report.get("accuracy", 0.0))
        print(f"Evaluation: accuracy={accuracy:.4%}")
    if predictor.composition_report:
        print("Composition: completed")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
