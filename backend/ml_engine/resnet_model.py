"""
ml_engine/resnet_model.py
--------------------------
ResNet-18 tabanlı Caretta Caretta kimlik tanıma modeli.

Veri seti yapısı (klasör = sınıf = birey):
    data/turtles-data/data/images/
        t001/  img1.jpg  img2.jpg ...
        t002/  img1.jpg ...
        ...

Transfer Learning akışı:
    1. torchvision ResNet-18 (ImageNet ağırlıkları)
    2. Son fc katmanı → nn.Linear(512, num_classes)
    3. Yalnızca fc katmanı eğitilir (feature freezing)
    4. Epoch sonunda en iyi model kaydedilir (checkpoint)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from ml_engine.interface import ITurtleRecognizer, PredictionResult, TrainingResult

if TYPE_CHECKING:
    from agents.data_sources import DataRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
_IMG_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)   # ImageNet normalize
_STD  = (0.229, 0.224, 0.225)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ---------------------------------------------------------------------------
# Yardımcı: DataRecord listesinden ImageFolder uyumlu tmp dizini oluşturur
# ---------------------------------------------------------------------------

def _build_image_folder_from_records(
    records: list[DataRecord],
    images_root: Path,
) -> ImageFolder | None:
    """
    DataRecord listesindeki 'extra.turtle_id' veya 'title' bilgisinden
    sınıf etiketi çıkarır ve ImageFolder döndürür.

    Eğer kayıtlar yerel dosya yollarıysa (LocalDirectorySource) ve
    images_root altında alt-klasör yapısı varsa doğrudan ImageFolder kullanılır.
    """
    if images_root.exists() and any(images_root.iterdir()):
        logger.info("ImageFolder doğrudan kullanılıyor: %s", images_root)
        return ImageFolder(root=str(images_root), transform=_TRAIN_TRANSFORM)

    logger.warning("images_root boş veya yok; DataRecord'lardan klasör tahmini yapılıyor.")
    return None


# ---------------------------------------------------------------------------
# Ana Model Sınıfı
# ---------------------------------------------------------------------------

class ResNetTurtleModel(ITurtleRecognizer):
    """
    ResNet-18 tabanlı kaplumbağa kimlik tanıma modeli.

    SOLID:
      - LSP : ITurtleRecognizer arayüzünü eksiksiz uygular.
      - SRP : Yalnızca model yönetimi (train / predict / save / load).
      - OCP : Yeni backbone için sadece _build_backbone() override edilir.
    """

    def __init__(
        self,
        images_root: str | Path = "data/turtles-data/data/images",
        checkpoint_path: str | Path = "ml_engine/checkpoints/resnet_turtle.pth",
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 2e-3,
        device: str | None = None,
    ) -> None:
        self._images_root    = Path(images_root)
        self._checkpoint_path = Path(checkpoint_path)
        self._epochs         = epochs
        self._batch_size     = batch_size
        self._lr             = learning_rate
        self._device         = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Sınıf listesi eğitim sonrası doldurulur
        self._class_names: list[str] = []
        self._model: nn.Module | None = None

        # Kaydedilmiş checkpoint varsa yükle
        if self._checkpoint_path.exists():
            self._load_checkpoint()
        else:
            logger.info(
                "ResNetTurtleModel hazırlandı (checkpoint yok). "
                "Önce train() çağrılmalı. Device: %s", self._device
            )

    # ------------------------------------------------------------------
    # ITurtleRecognizer — train()
    # ------------------------------------------------------------------

    def train(self, data_records: list[DataRecord]) -> TrainingResult:
        """
        DataRecord listesini alır, ImageFolder dataset kurar, ResNet-18 fine-tune eder.

        Strateji:
            - Backbone dondurulur (requires_grad=False)
            - Yalnızca model.fc katmanı eğitilir
            - Her epoch sonunda validation loss hesaplanır
            - En iyi model checkpoint olarak kaydedilir
        """
        logger.info(
            "Eğitim başlıyor | %d kayıt | Device: %s | Epoch: %d",
            len(data_records), self._device, self._epochs,
        )

        # 1. Dataset
        dataset = _build_image_folder_from_records(data_records, self._images_root)
        if dataset is None or len(dataset) == 0:
            return TrainingResult(
                epochs_completed=0, final_loss=float("inf"),
                num_classes=0, success=False,
                error="Dataset boş veya oluşturulamadı.",
            )

        self._class_names = dataset.classes
        num_classes = len(self._class_names)
        logger.info("Sınıf sayısı: %d | Örnek sayısı: %d", num_classes, len(dataset))

        # Train / val split (%80 / %20)
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds, batch_size=self._batch_size, shuffle=True,
            num_workers=0, pin_memory=(self._device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self._batch_size, shuffle=False, num_workers=0,
        )

        # 2. Model
        model = self._build_backbone(num_classes)
        model.to(self._device)

        # 3. Optimizasyon — yalnızca fc katmanı
        optimizer = optim.Adam(model.fc.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs)
        criterion = nn.CrossEntropyLoss()

        # 4. Eğitim döngüsü
        best_val_loss = float("inf")
        final_loss = float("inf")

        for epoch in range(1, self._epochs + 1):
            # --- Train ---
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self._device), labels.to(self._device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / train_size

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self._device), labels.to(self._device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
            val_loss /= val_size
            final_loss = val_loss

            scheduler.step()
            logger.info(
                "Epoch %d/%d — train_loss: %.4f | val_loss: %.4f",
                epoch, self._epochs, train_loss, val_loss,
            )

            # En iyi checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model)
                logger.info("  ✓ Yeni en iyi model kaydedildi (val_loss=%.4f)", val_loss)

        self._model = model
        logger.info("Eğitim tamamlandı. En iyi val_loss: %.4f", best_val_loss)

        return TrainingResult(
            epochs_completed=self._epochs,
            final_loss=round(final_loss, 6),
            num_classes=num_classes,
            class_names=self._class_names,
            success=True,
        )

    # ------------------------------------------------------------------
    # ITurtleRecognizer — predict()
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> PredictionResult:
        """
        Tek görsel üzerinde kaplumbağa kimlik tahmini yapar.

        Args:
            image_path: Yerel dosya yolu veya HTTP URL.
        """
        if self._model is None:
            raise RuntimeError(
                "Model henüz yüklenmedi. Önce train() çağrın veya "
                "geçerli bir checkpoint sağlayın."
            )

        img = self._load_image(image_path)
        tensor = _INFER_TRANSFORM(img).unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        confidence = float(conf.item())
        class_idx  = int(idx.item())
        turtle_id  = (
            self._class_names[class_idx]
            if self._class_names and class_idx < len(self._class_names)
            else f"class_{class_idx}"
        )

        # %60 altı güven → yeni birey
        is_new = confidence < 0.60

        logger.info(
            "Tahmin: %s | Güven: %.2f%% | Yeni birey: %s",
            turtle_id, confidence * 100, is_new,
        )
        return PredictionResult(
            turtle_id=None if is_new else turtle_id,
            confidence=confidence,
            is_new_turtle=is_new,
            extra={"raw_class": turtle_id, "class_index": class_idx},
        )

    # ------------------------------------------------------------------
    # Yardımcı Metotlar (private)
    # ------------------------------------------------------------------

    def _build_backbone(self, num_classes: int) -> nn.Module:
        """ImageNet ağırlıklı ResNet-18 yükler, son katmanı değiştirir."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Backbone dondur
        for param in model.parameters():
            param.requires_grad = False

        # Yeni sınıflandırma başlığı (eğitilecek tek katman)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        logger.info("ResNet-18 backbone hazır. Çıkış: %d sınıf", num_classes)
        return model

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Model ağırlıklarını ve sınıf listesini kaydeder."""
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "class_names": self._class_names},
            self._checkpoint_path,
        )
        logger.debug("Checkpoint kaydedildi: %s", self._checkpoint_path)

    def _load_checkpoint(self) -> None:
        """Kaydedilmiş ağırlıkları yükler."""
        data = torch.load(self._checkpoint_path, map_location=self._device)
        self._class_names = data.get("class_names", [])
        num_classes = len(self._class_names)

        model = self._build_backbone(num_classes)
        model.load_state_dict(data["model_state"])
        model.to(self._device)
        self._model = model
        logger.info(
            "Checkpoint yüklendi: %s | Sınıf: %d",
            self._checkpoint_path, num_classes,
        )

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        """Yerel yol veya HTTP URL'den PIL Image döner."""
        if image_path.startswith("http://") or image_path.startswith("https://"):
            import io
            import urllib.request
            with urllib.request.urlopen(image_path) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(image_path).convert("RGB")
