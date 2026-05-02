"""
api/endpoints.py
----------------
FastAPI Endpoint'leri — Caretta Caretta Tanıma API

SOLID prensipleri:
  - SRP : Her endpoint HTTP isteklerini parse ve cevaplamaktan sorumlu;
          iş mantığı TurtleService'te.
  - OCP : Yeni endpoint eklemek için mevcut olanlar değişmez.
  - DIP : Endpoints, TurtleService arayüzüne bağımlıdır.

Endpoint'ler:
  - POST /train       : Modeli yerel veri setiyle eğitir.
  - POST /predict     : Yüklenen fotoğrafı tahmin eder (ValidatorAgent → ResearcherAgent → Model).
  - GET  /health      : API sağlığını kontrol eder.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from services.turtle_service import TurtleService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic DTOs (Request / Response)
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """POST /train isteği."""
    data_dir: str | None = None
    max_results: int = 5000
    epochs: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "data_dir": "data/turtles-data/data/images",
                "max_results": 5000,
                "epochs": 3,
            }
        }


class TrainResponse(BaseModel):
    """POST /train yanıtı."""
    success: bool
    records_collected: int
    records_accepted: int
    epochs_trained: int | None = None
    final_loss: float | None = None
    error: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "records_collected": 1500,
                "records_accepted": 1200,
                "epochs_trained": 3,
                "final_loss": 0.1234,
                "error": "",
            }
        }


class PredictResponse(BaseModel):
    """POST /predict yanıtı."""
    success: bool
    request_id: str
    stage_reached: str  # "validation" | "research" | "prediction" | "complete"
    turtle_id: str | None = None
    confidence: float | None = None
    is_new_turtle: bool | None = None
    error: str = ""
    validation_passed: bool | None = None
    quality_passed: bool | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "request_id": "a1b2c3d4",
                "stage_reached": "complete",
                "turtle_id": "t042",
                "confidence": 0.87,
                "is_new_turtle": False,
                "error": "",
                "validation_passed": True,
                "quality_passed": True,
            }
        }


class HealthResponse(BaseModel):
    """GET /health yanıtı."""
    status: str
    message: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api", tags=["Caretta Track"])


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Sağlık Kontrolü",
    description="API'nin aktif ve çalışan olduğunu doğrular.",
)
async def health_check() -> HealthResponse:
    """
    GET /health

    Returns:
        HealthResponse: API durumu.
    """
    return HealthResponse(
        status="ok",
        message="Caretta Track API aktif ve çalışıyor.",
    )


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Model Eğitimi",
    description=(
        "Yerel veri setinden veri toplar (LocalDirectorySource), "
        "ResearcherAgent ile kalite filtresinden geçirir, "
        "ardından ResNetTurtleModel'i fine-tuning ile eğitir."
    ),
)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    POST /train

    Eğitim iş akışı:
      1. ResearcherAgent yerel veri dizini tarar
      2. Kalite metrikleri (çözünürlük, netlik) kontrol edilir
      3. ITurtleRecognizer.train() ile model eğitilir
      4. En iyi checkpoint kaydedilir

    Args:
        request: TrainRequest (data_dir, max_results)

    Returns:
        TrainResponse: Eğitim özeti

    Raises:
        HTTPException: Eğitim sırasında hata oluşursa.

    Examples:
        curl -X POST "http://localhost:8000/api/train" \\
             -H "Content-Type: application/json" \\
             -d '{"data_dir": "data/turtles-data/data/images", "max_results": 5000}'
    """
    logger.info("=== POST /train çağrısı === data_dir: %s | max_results: %d",
                request.data_dir, request.max_results)

    try:
        # TurtleService singleton'ı al
        service = _get_service()

        # Eğitim çalıştır
        result = service.train_system(
            data_dir=request.data_dir,
            max_results=request.max_results,
            epochs=request.epochs,
        )

        logger.info(
            "Eğitim sonucu: başarı=%s | toplanan=%d | kabul edilen=%d",
            result.success, result.records_collected, result.records_accepted,
        )

        if not result.success:
            logger.warning("Eğitim başarısız: %s", result.error)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Eğitim başarısız: {result.error}",
            )

        return TrainResponse(
            success=result.success,
            records_collected=result.records_collected,
            records_accepted=result.records_accepted,
            epochs_trained=(
                result.training_result.epochs_completed
                if result.training_result else None
            ),
            final_loss=(
                result.training_result.final_loss
                if result.training_result else None
            ),
            error=result.error,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("POST /train hata: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sunucu hatası: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Kaplumbağa Kimlik Tahmini",
    description=(
        "Yüklenen fotoğrafı ValidatorAgent (güvenlik), "
        "ResearcherAgent (kalite), ardından ResNetTurtleModel ile "
        "işler ve kaplumbağa kimliğini tahmin eder."
    ),
)
async def predict_turtle(file: UploadFile = File(...)) -> PredictResponse:
    """
    POST /predict

    Tahmin iş akışı:
      1. ValidatorAgent: Dosya güvenliği ve türü (image) kontrolü
      2. ResearcherAgent: Görsel kalitesi (çözünürlük, netlik)
      3. TurtleService._run_prediction(): ResNetTurtleModel tahmini
      4. Sonuç: turtle_id, confidence, is_new_turtle

    Args:
        file: Yüklenen fotoğraf dosyası (UploadFile).

    Returns:
        PredictResponse: Tahmin sonucu (success, turtle_id, confidence, vb.)

    Raises:
        HTTPException:
          - 400: Dosya doğrulama başarısız
          - 413: Dosya çok büyük
          - 500: Model hatası

    Examples:
        curl -X POST "http://localhost:8000/api/predict" \\
             -F "file=@/path/to/turtle_photo.jpg"
    """
    logger.info("=== POST /predict çağrısı === Dosya: %s | Boyut: %d byte",
                file.filename, file.size or 0)

    try:
        # Dosya boyut kontrolü (max 50MB)
        if file.size and file.size > 50 * 1024 * 1024:
            logger.warning("Dosya çok büyük: %d byte", file.size)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Dosya çok büyük (maks. 50MB).",
            )

        # Dosya içeriğini oku
        image_bytes = await file.read()
        if not image_bytes:
            logger.warning("Boş dosya yüklendi")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dosya boş.",
            )

        logger.info("Dosya başarıyla okundu: %d byte", len(image_bytes))

        # TurtleService singleton'ı al
        service = _get_service()

        # İşleme başla
        file_context = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(image_bytes),
        }

        result = service.process(
            image_bytes=image_bytes,
            file_context=file_context,
        )

        logger.info(
            "Tahmin sonucu: başarı=%s | aşama=%s | kaplumbağa=%s | güven=%.2f%%",
            result.success, result.stage_reached, result.prediction.get("turtle_id"),
            (result.prediction.get("confidence", 0) * 100) if result.success else 0,
        )

        # Response oluştur
        response_data: dict[str, Any] = {
            "success": result.success,
            "request_id": result.request_id,
            "stage_reached": result.stage_reached,
            "error": result.error_message,
        }

        if result.success:
            response_data.update({
                "turtle_id": result.prediction.get("turtle_id"),
                "confidence": result.prediction.get("confidence"),
                "is_new_turtle": result.prediction.get("is_new_turtle"),
            })

        if result.security_report:
            response_data["validation_passed"] = result.security_report.passed

        if result.research_result:
            response_data["quality_passed"] = result.research_result.passed

        return PredictResponse(**response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("POST /predict hata: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin sırasında sunucu hatası: {exc}",
        )


# ---------------------------------------------------------------------------
# Lazy Singleton — TurtleService
# ---------------------------------------------------------------------------

_service: TurtleService | None = None


def _get_service() -> TurtleService:
    """TurtleService singleton'ını al veya oluştur."""
    global _service
    if _service is not None:
        return _service

    from pathlib import Path

    from ml_engine.resnet_model import ResNetTurtleModel

    logger.info("TurtleService başlatılıyor...")

    images_root = Path(__file__).parent.parent / "data" / "turtles-data" / "data" / "images"
    checkpoint = Path(__file__).parent.parent / "ml_engine" / "checkpoints" / "resnet_turtle.pth"

    model = ResNetTurtleModel(
        images_root=str(images_root),
        checkpoint_path=str(checkpoint),
        epochs=10,
    )
    _service = TurtleService(ml_model=model, data_dir=str(images_root))

    logger.info("TurtleService başlatıldı. Endpoints hazır.")
    return _service
