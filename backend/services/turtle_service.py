"""
services/turtle_service.py
---------------------------
Koordinatör / Is Katmani (Service Layer)

Isleyis sirasi:
  1. ValidatorAgent  -> guvenlik & tur dogrulamasi
  2. ResearcherAgent -> gorsel kalite olcumu
  3. Her ikisi de onay verirse -> ML modeline ilet

SOLID prensipleri:
  - SRP : Yalnizca is akisi koordinasyonu yapar; isleri ajanlara birakir.
  - OCP : ML modeli interface uzerinden enjekte edilir; sinif degistirilmez.
  - LSP : MLModelPort uygulayan herhangi bir model seffaf sekilde calisir.
  - ISP : MLModelPort yalnizca predict() metodunu zorunlu kilar.
  - DIP : TurtleService, somut ML sinifina degil ITurtleRecognizer soyutlamasina bagimlidir.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.data_sources import LocalDirectorySource
from agents.researcher import QualityMetrics, ResearcherAgent, ResearchResult
from agents.validator import SecurityReport, ValidatorAgent, ValidationResult
from ml_engine.interface import ITurtleRecognizer, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geri Uyumluluk: eski MLModelPort yerine ITurtleRecognizer kullanilir
# ---------------------------------------------------------------------------
MLModelPort = ITurtleRecognizer


# ---------------------------------------------------------------------------
# Sonuc DTO
# ---------------------------------------------------------------------------

@dataclass
class TurtleServiceResult:
    """process() cagrisinin nihai ciktisi."""
    request_id: str
    timestamp: datetime
    success: bool
    stage_reached: str           # "validation" | "research" | "prediction" | "complete"
    security_report: SecurityReport | None = None
    research_result: ResearchResult | None = None
    prediction: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    @property
    def summary(self) -> str:
        if self.success:
            turtle_id  = self.prediction.get("turtle_id", "?")
            confidence = self.prediction.get("confidence", 0.0)
            return (
                f"[{self.request_id}] Basarili -- "
                f"Kaplumbaga ID: {turtle_id} | Guven: {confidence:.1%}"
            )
        return f"[{self.request_id}] Basarisiz ({self.stage_reached}): {self.error_message}"


@dataclass
class TrainingSystemResult:
    """train_system() cagrisinin ozet ciktisi."""
    success: bool
    records_collected: int
    records_accepted: int
    training_result: TrainingResult | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Koordinator Servis
# ---------------------------------------------------------------------------

class TurtleService:
    """
    API katmanindan gelen gorsel istegini alip ajanlari dogru sirada calistiran servis.

    Kullanim:
        from ml_engine.resnet_model import ResNetTurtleModel
        model   = ResNetTurtleModel()
        service = TurtleService(model)
        result  = service.process(image_bytes, file_context)
        # Egitim:
        tr = service.train_system()
    """

    def __init__(
        self,
        ml_model: ITurtleRecognizer,
        validator_agent: ValidatorAgent | None = None,
        researcher_agent: ResearcherAgent | None = None,
        data_dir: str = "data/turtles-data/data/images",
    ) -> None:
        self._model: ITurtleRecognizer = ml_model
        self._validator: ValidatorAgent = validator_agent or ValidatorAgent()
        self._researcher: ResearcherAgent = researcher_agent or ResearcherAgent()
        self._data_dir = data_dir

        logger.info(
            "TurtleService baslatildi. Model: %s | Validator: %s | Researcher: %s",
            type(self._model).__name__,
            type(self._validator).__name__,
            type(self._researcher).__name__,
        )

    # ------------------------------------------------------------------
    # 1. Tek Gorsel Isleme
    # ------------------------------------------------------------------

    def process(
        self,
        image_bytes: bytes,
        file_context: dict[str, Any] | None = None,
        past_metadata: dict[str, Any] | None = None,
    ) -> TurtleServiceResult:
        """Gorsel isleme boru hattini sirayla calistirir."""
        request_id = str(uuid.uuid4())[:8]
        timestamp  = datetime.now(tz=timezone.utc)   # utcnow() deprecated → timezone-aware
        ctx        = file_context or {}

        logger.info(
            "=== Yeni Istek [%s] === Dosya: %s | Boyut: %d byte",
            request_id, ctx.get("filename", "bilinmiyor"), len(image_bytes),
        )

        # 1. Guvenlik
        security_report = self._run_validation(image_bytes, ctx)
        if not security_report.passed:
            logger.warning("[%s] Guvenlik asamasinda reddedildi.", request_id)
            return TurtleServiceResult(
                request_id=request_id, timestamp=timestamp, success=False,
                stage_reached="validation", security_report=security_report,
                error_message=security_report.message,
            )

        # 2. Kalite
        research_result = self._run_research(image_bytes, past_metadata)
        if not research_result.passed:
            logger.warning("[%s] Kalite asamasinda reddedildi.", request_id)
            return TurtleServiceResult(
                request_id=request_id, timestamp=timestamp, success=False,
                stage_reached="research", security_report=security_report,
                research_result=research_result,
                error_message=research_result.message,
            )

        # 3. ML Tahmini
        prediction = self._run_prediction(image_bytes, request_id)
        if "error" in prediction:
            return TurtleServiceResult(
                request_id=request_id, timestamp=timestamp, success=False,
                stage_reached="prediction", security_report=security_report,
                research_result=research_result, prediction=prediction,
                error_message=prediction["error"],
            )

        logger.info("[%s] Islem basariyla tamamlandi. Tahmin: %s", request_id, prediction)
        return TurtleServiceResult(
            request_id=request_id, timestamp=timestamp, success=True,
            stage_reached="complete", security_report=security_report,
            research_result=research_result, prediction=prediction,
        )

    # ------------------------------------------------------------------
    # 2. Egitim Sistemi (YENİ)
    # ------------------------------------------------------------------

    def train_system(
        self,
        data_dir: str | None = None,
        max_results: int = 5000,
        epochs: int = 3,
    ) -> TrainingSystemResult:
        """
        Yerel veri setinden veri toplar, filtreler ve modeli egitir.

        Isleyis:
          1. LocalDirectorySource -> ham veri kayitlari
          2. ResearcherAgent.collect_training_data() -> kalite filtresi
          3. ITurtleRecognizer.train() -> fine-tuning

        Args:
            data_dir   : Gorsellerin bulundugu kok dizin.
                         None ise constructor'daki data_dir kullanilir.
            max_results: Kaynaktan cekilecek maksimum kayit sayisi.

        Returns:
            TrainingSystemResult
        """
        images_root = data_dir or self._data_dir
        logger.info("=== Egitim sistemi basliyor === Veri dizini: %s", images_root)

        # 1. Yerel kaynak
        try:
            source = LocalDirectorySource(directory_path=images_root)
            # ResearcherAgent'a bu kaynak uzerinden ara
            researcher = ResearcherAgent(sources=[source])
            collection_report = researcher.collect_training_data(
                queries=["local_dataset"],
                max_results_per_source=max_results,
            )
        except Exception as exc:
            logger.error("Veri toplama hatasi: %s", exc, exc_info=True)
            return TrainingSystemResult(
                success=False, records_collected=0, records_accepted=0,
                error=str(exc),
            )

        accepted = collection_report.accepted_records
        logger.info(
            "Veri toplama tamamlandi: %d/%d kayit kabul edildi.",
            len(accepted), collection_report.total_fetched,
        )

        if not accepted:
            return TrainingSystemResult(
                success=False,
                records_collected=collection_report.total_fetched,
                records_accepted=0,
                error="Kalite filtresini gecen kayit bulunamadi.",
            )

        # 2. Egitim
        try:
            training_result = self._model.train(accepted)
        except Exception as exc:
            logger.error("Model egitimi hatasi: %s", exc, exc_info=True)
            return TrainingSystemResult(
                success=False,
                records_collected=collection_report.total_fetched,
                records_accepted=len(accepted),
                error=str(exc),
            )

        logger.info(
            "Egitim tamamlandi. Sinif: %d | Son kayip: %.4f",
            training_result.num_classes, training_result.final_loss,
        )
        return TrainingSystemResult(
            success=training_result.success,
            records_collected=collection_report.total_fetched,
            records_accepted=len(accepted),
            training_result=training_result,
        )

    # ------------------------------------------------------------------
    # Yardimci Metotlar (private)
    # ------------------------------------------------------------------

    def _run_validation(self, image_bytes: bytes, context: dict[str, Any]) -> SecurityReport:
        try:
            logger.info("Adim 1/3: Guvenlik dogrulamasi basliyor...")
            report = self._validator.validate(image_bytes, context)
            logger.info("Guvenlik: %s", "GECTI" if report.passed else "BASARISIZ")
            return report
        except Exception as exc:
            logger.error("ValidatorAgent beklenmedik hata: %s", exc, exc_info=True)
            return SecurityReport(
                passed=False,
                results=[ValidationResult(
                    passed=False, validator_name="ValidatorAgent",
                    reason=f"Dogrulama sirasinda beklenmedik hata: {exc}",
                )],
            )

    def _run_research(
        self, image_bytes: bytes, past_metadata: dict[str, Any] | None,
    ) -> ResearchResult:
        try:
            logger.info("Adim 2/3: Kalite arastirmasi basliyor...")
            result = self._researcher.analyze(image_bytes, past_metadata or {})
            logger.info(
                "Kalite: %s | Sorunlar: %s",
                "GECTI" if result.passed else "BASARISIZ", result.issues,
            )
            return result
        except Exception as exc:
            logger.error("ResearcherAgent beklenmedik hata: %s", exc, exc_info=True)
            return ResearchResult(
                passed=False,
                quality_metrics=QualityMetrics(0, 0, 0.0, 0.0, 0.0, 0),
                issues=[f"Kalite analizi sirasinda beklenmedik hata: {exc}"],
            )

    def _run_prediction(self, image_bytes: bytes, request_id: str) -> dict[str, Any]:
        try:
            logger.info("Adim 3/3: ML tahmini basliyor... [%s]", request_id)
            result: PredictionResult = self._model.predict_bytes(image_bytes)
            logger.info("ML tahmini tamamlandi: %s", result)
            return result.to_dict()
        except Exception as exc:
            logger.error("[%s] ML modeli hatasi: %s", request_id, exc, exc_info=True)
            return {"error": f"ML modeli calistirilirken hata: {exc}"}
