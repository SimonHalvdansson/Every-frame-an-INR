from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without dependencies
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
MEDIA_DIR = ROOT_DIR / "media"
CACHE_DIR = ROOT_DIR / ".cache" / "inr"
SOURCE_IMAGE = MEDIA_DIR / "image.png"
TARGET_SIZE = 256
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_FEATURE_DIM = 768
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 3
DEFAULT_FREQ_SCALE = 35.0
DEFAULT_LR = 8e-3
DEFAULT_LR_DECAY = 0.995
DEFAULT_MIN_LR = 1e-3
DEFAULT_PREVIEW_HOLD_SECONDS = 0.25

DISPLAY_NAME_OVERRIDES = {
    "image.png": "Harmonic screenshot",
    "google-deepmind-2ufXB-oAyG0-unsplash.jpg": "Code domes",
    "google-deepmind-37x7-sBhwf8-unsplash.jpg": "White flight paths",
    "google-deepmind-Fv39DqWqtHw-unsplash.jpg": "Glass compute rig",
    "google-deepmind-GVGnKgEomlw-unsplash.jpg": "Barcode blocks",
    "google-deepmind-hpIZ5T6SS-M-unsplash.jpg": "Folded color field",
    "google-deepmind-LcgLq78WZCQ-unsplash (1).jpg": "Blue circuitry model",
    "google-deepmind-mEawZ3YloK4-unsplash.jpg": "Pink particle strands",
    "google-deepmind-pLh_n9pnRhw-unsplash.jpg": "Orange organic folds",
    "google-deepmind-uakYuHuCnOw-unsplash.jpg": "Frosted architecture",
    "google-deepmind-UoJuws1wEzY-unsplash.jpg": "Black-white motion",
}


def _clamp_int(value: Any, default: int, minimum: int = 1, maximum: int = 10) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _finite_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _display_name(path: Path) -> str:
    if path.name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[path.name]
    stem = path.stem.replace("google-deepmind-", "").replace("-unsplash", "")
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"[-_]+", " ", stem).strip()
    return stem.title() or path.name


def list_media_images() -> list[dict[str, str]]:
    if not MEDIA_DIR.exists():
        return []
    images = [
        path
        for path in MEDIA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda item: (item.name != "image.png", item.name.lower()))
    return [
        {
            "name": path.name,
            "display_name": _display_name(path),
            "filename": path.name,
        }
        for path in images
    ]


if torch is not None:

    class FourierFeatures(nn.Module):
        def __init__(
            self,
            input_dim: int = 2,
            feature_dim: int = DEFAULT_FEATURE_DIM,
            freq_scale: float = DEFAULT_FREQ_SCALE,
        ) -> None:
            super().__init__()
            if feature_dim % 2 != 0:
                raise ValueError("feature_dim must be even")
            freqs = torch.randn(input_dim, feature_dim // 2) * freq_scale
            self.register_buffer("freqs", freqs)

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            phase = 2.0 * math.pi * (coords @ self.freqs)
            return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


    class FourierMLP(nn.Module):
        def __init__(
            self,
            feature_dim: int = DEFAULT_FEATURE_DIM,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            n_layers: int = DEFAULT_N_LAYERS,
            freq_scale: float = DEFAULT_FREQ_SCALE,
        ) -> None:
            super().__init__()
            if n_layers < 1:
                raise ValueError("n_layers must be at least 1")
            self.features = FourierFeatures(feature_dim=feature_dim, freq_scale=freq_scale)
            layers: list[nn.Module] = []
            dim = feature_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.SiLU())
                dim = hidden_dim
            layers.append(nn.Linear(dim, 3))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return self.net(self.features(coords))


class INRExperiment:
    def __init__(
        self,
        source_image: Path = SOURCE_IMAGE,
        cache_dir: Path = CACHE_DIR,
        target_size: int = TARGET_SIZE,
        lr: float = DEFAULT_LR,
        seed: int = 7,
    ) -> None:
        if TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(
                "PyTorch is not installed. Install the dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.source_image = Path(source_image)
        self.current_image_name = self.source_image.name
        self.cache_dir = Path(cache_dir)
        self.reference_path = self.cache_dir / "reference.png"
        self.latest_path = self.cache_dir / "latest.png"
        self.thumbs_dir = self.cache_dir / "thumbs"
        self.state_path = self.cache_dir / "state.json"
        self.target_size = int(target_size)
        self.lr = float(lr)
        self.current_lr = float(lr)
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.running = False
        self.training = False
        self.epoch = 0
        self.trained_epochs = 0
        self.loss: float | None = None
        self.psnr: float | None = None
        self.image_revision = 0
        self.error: str | None = None
        self.epochs_per_preview = 1
        self.loss_history: list[dict[str, float]] = []
        self.last_update_seconds: float | None = None
        self.latest_rgb_bytes: bytes = b""

        self.model: FourierMLP | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.coords: torch.Tensor | None = None
        self.pixels: torch.Tensor | None = None
        self.target_np: np.ndarray | None = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self, image_name: str | None = None) -> dict[str, Any]:
        self.stop()
        with self.training_lock:
            with self.lock:
                self.training = True
                self.error = None
                if image_name is not None:
                    self.source_image = self._resolve_media_image(image_name)
                    self.current_image_name = self.source_image.name
            try:
                self._initialize()
                self._evaluate_and_save()
            except Exception as exc:  # pragma: no cover - surfaced through state
                with self.lock:
                    self.error = str(exc)
                raise
            finally:
                with self.lock:
                    self.training = False
                    self.running = False
                self._write_state_file()
        return self.get_state()

    def select_image(self, image_name: str) -> dict[str, Any]:
        return self.reset(image_name=image_name)

    def start(self, epochs_per_preview: int) -> dict[str, Any]:
        interval = _clamp_int(epochs_per_preview, self.epochs_per_preview)
        with self.lock:
            self.epochs_per_preview = interval
            self.error = None
            if self.running:
                self._write_state_file()
                return self.get_state()
            if self.training:
                self.error = "Training is already in progress."
                self._write_state_file()
                return self.get_state()

            self.stop_event.clear()
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._continuous_worker,
                name="inr-trainer",
                daemon=True,
            )
            self.worker_thread.start()
            self._write_state_file()
            return self.get_state()

    def stop(self) -> dict[str, Any]:
        thread: threading.Thread | None
        with self.lock:
            self.stop_event.set()
            thread = self.worker_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=30.0)
        with self.lock:
            self.running = False
            if self.worker_thread is thread and (thread is None or not thread.is_alive()):
                self.worker_thread = None
            self._write_state_file()
            return self.get_state()

    def step(self, steps: int) -> dict[str, Any]:
        count = _clamp_int(steps, 1)
        with self.lock:
            if self.running:
                self.error = "Stop continuous training before stepping manually."
                self._write_state_file()
                return self.get_state()
            if self.training:
                self.error = "Training is already in progress."
                self._write_state_file()
                return self.get_state()
            self.error = None
        self._run_chunk(count)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "training": self.training,
                "epoch": self.epoch,
                "training_epoch": self.trained_epochs,
                "loss": _finite_number(self.loss),
                "psnr": _finite_number(self.psnr),
                "image_revision": self.image_revision,
                "device": str(self.device),
                "target_size": self.target_size,
                "error": self.error,
                "epochs_per_preview": self.epochs_per_preview,
                "epochs_per_update": self.epochs_per_preview,
                "current_image": self.current_image_name,
                "current_image_display_name": _display_name(self.source_image),
                "images": list_media_images(),
                "loss_history": list(self.loss_history),
                "last_update_seconds": _finite_number(self.last_update_seconds),
                "preview_hold_seconds": DEFAULT_PREVIEW_HOLD_SECONDS,
                "model": {
                    "type": "FourierMLP",
                    "feature_dim": DEFAULT_FEATURE_DIM,
                    "hidden_dim": DEFAULT_HIDDEN_DIM,
                    "n_layers": DEFAULT_N_LAYERS,
                    "freq_scale": DEFAULT_FREQ_SCALE,
                    "optimizer": "AdamW",
                    "base_lr": self.lr,
                    "current_lr": self.current_lr,
                    "lr_decay": DEFAULT_LR_DECAY,
                    "min_lr": DEFAULT_MIN_LR,
                },
            }

    def get_latest_raw(self) -> tuple[bytes, int, int, int]:
        with self.lock:
            return (
                bytes(self.latest_rgb_bytes),
                self.target_size,
                self.target_size,
                self.image_revision,
            )

    def get_thumbnail_path(self, image_name: str) -> Path:
        image_path = self._resolve_media_image(image_name)
        digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
        thumb_path = self.thumbs_dir / f"{digest}.jpg"
        if not thumb_path.exists() or thumb_path.stat().st_mtime < image_path.stat().st_mtime:
            thumb = ImageOps.fit(
                Image.open(image_path).convert("RGB"),
                (128, 128),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            thumb.save(thumb_path, quality=82, optimize=True)
        return thumb_path

    def _resolve_media_image(self, image_name: str) -> Path:
        candidate = Path(str(image_name)).name
        path = MEDIA_DIR / candidate
        if not path.exists() or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"Unknown media image: {image_name}")
        return path

    def _initialize(self) -> None:
        if not self.source_image.exists():
            raise FileNotFoundError(f"Missing source image: {self.source_image}")

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        target_image = Image.open(self.source_image).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size),
            Image.Resampling.BILINEAR,
        )
        target_image.save(self.reference_path)

        self.target_np = np.asarray(target_image, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(self.target_np).to(self.device)

        y_coords = torch.linspace(0.0, 1.0, steps=self.target_size, device=self.device)
        x_coords = torch.linspace(0.0, 1.0, steps=self.target_size, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        self.pixels = target_tensor.reshape(-1, 3)

        self.model = FourierMLP().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.current_lr = self.lr
        self.epoch = 0
        self.trained_epochs = 0
        self.loss = None
        self.psnr = None
        self.image_revision = 0
        self.loss_history = []
        self.last_update_seconds = None

    def _continuous_worker(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    interval = self.epochs_per_preview
                started = time.perf_counter()
                self._run_chunk(interval)
                elapsed = time.perf_counter() - started
                remaining = DEFAULT_PREVIEW_HOLD_SECONDS - elapsed
                if remaining > 0:
                    self.stop_event.wait(remaining)
        except Exception as exc:  # pragma: no cover - surfaced through state
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.running = False
                self.training = False
                if self.worker_thread is threading.current_thread():
                    self.worker_thread = None
                self._write_state_file()

    def _run_chunk(self, epochs: int) -> None:
        started = time.perf_counter()
        with self.training_lock:
            with self.lock:
                self.training = True
                self.error = None
                self._write_state_file()
            try:
                for _ in range(max(1, int(epochs))):
                    self._train_one_epoch()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                loss, psnr = self._evaluate_and_save()
                elapsed = time.perf_counter() - started
                with self.lock:
                    self.loss = loss
                    self.psnr = psnr
                    self.loss_history.append(
                        {
                            "epoch": float(self.epoch),
                            "loss": float(loss),
                            "psnr": float(psnr),
                        }
                    )
                    self.last_update_seconds = elapsed
            finally:
                with self.lock:
                    self.training = False
                    self._write_state_file()

    def _train_one_epoch(self) -> None:
        if self.model is None or self.optimizer is None or self.coords is None or self.pixels is None:
            raise RuntimeError("Experiment is not initialized.")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(self.coords)
        loss = F.mse_loss(prediction, self.pixels)
        loss.backward()
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            group["lr"] = max(DEFAULT_MIN_LR, float(group["lr"]) * DEFAULT_LR_DECAY)
        with self.lock:
            self.current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.trained_epochs += 1

    def _evaluate_and_save(self) -> tuple[float, float]:
        if self.model is None or self.coords is None or self.pixels is None:
            raise RuntimeError("Experiment is not initialized.")

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.coords)
            loss = F.mse_loss(prediction, self.pixels).item()
            image = prediction.reshape(self.target_size, self.target_size, 3)
        image_np = image.clamp(0.0, 1.0).detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        mse = max(float(loss), 1e-12)
        psnr = 10.0 * math.log10(1.0 / mse)
        output = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        self._save_rgb_image(output, self.latest_path)
        with self.lock:
            self.epoch = self.trained_epochs
            self.loss = float(loss)
            self.psnr = float(psnr)
            self.latest_rgb_bytes = output.tobytes()
            self.image_revision += 1
        return float(loss), float(psnr)

    def _save_rgb_image(self, image: np.ndarray, output_path: Path) -> None:
        Image.fromarray(image, mode="RGB").save(output_path)

    def _write_state_file(self) -> None:
        state = self.get_state()
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
        temp_path.replace(self.state_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive INR experiment server.")
    parser.add_argument("--host", default=os.environ.get("INR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INR_PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "PyTorch is not installed. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )

    from server import run_server

    args = parse_args()
    experiment = INRExperiment()
    run_server(experiment, host=args.host, port=args.port, root_dir=ROOT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
