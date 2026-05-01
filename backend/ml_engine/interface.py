"""ml_engine/interface.py — ITurtleRecognizer soyut arayüzü (DIP)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.data_sources import DataRecord


@dataclass
class PredictionResult:
    """predict() çağrısının standart çıktısı."""
    turtle_id: str | None        # tanınan birey ID'si; yeni bireyde None
    confidence: float            # 0.0 – 1.0
    is_new_turtle: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turtle_id": self.turtle_id,
            "confidence": self.confidence,
            "is_new_turtle": self.is_new_turtle,
            **self.extra,
        }


@dataclass
class TrainingResult:
    """train() çağrısının özeti."""
    epochs_completed: int
    final_loss: float
    num_classes: int
    class_names: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


class ITurtleRecognizer(ABC):
    """
    Tüm tanıma modellerinin uyması gereken sözleşme.

    SOLID:
      - ISP : Yalnızca train() ve predict() zorunlu tutulur.
      - DIP : TurtleService bu arayüze bağımlıdır, somut sınıfa değil.
      - OCP : EfficientNet, ViT vb. yeni modeller bu arayüzden türetilir.
    """

    @abstractmethod
    def train(self, data_records: list[DataRecord]) -> TrainingResult:
        """
        Eğitim verilerini alıp modeli fine-tune eder.

        Args:
            data_records: ResearcherAgent'ın onayladığı DataRecord listesi.
                          Her kaydın url alanı yerel dosya yolu veya HTTP URL'dir.
        """
        ...

    @abstractmethod
    def predict(self, image_path: str) -> PredictionResult:
        """
        Tek görsel üzerinde kimlik tahmini yapar.

        Args:
            image_path: Görselin yerel yolu veya URL'si.
        """
        ...

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        """
        Ham byte verisiyle tahmin yapar (geçici dosya üzerinden).
        Alt sınıflar override edebilir; varsayılan: tmp dosyaya yaz → predict().
        """
        import io
        import tempfile
        from pathlib import Path
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            return self.predict(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
