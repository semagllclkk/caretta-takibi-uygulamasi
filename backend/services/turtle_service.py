"""
services/turtle_service.py
---------------------------
Koordinatör / İş Katmanı (Service Layer)

İşleyiş sırası:
  1. ValidatorAgent  → güvenlik & tür doğrulaması
  2. ResearcherAgent → görsel kalite ölçümü
  3. Her ikisi de onay verirse → ML modeline ilet

SOLID prensipleri:
  - SRP : Yalnızca iş akışı koordinasyonu yapar; işleri ajanlara bırakır.
  - OCP : ML modeli interface üzerinden enjekte edilir; sınıf değiştirilmez.
  - LSP : MLModelPort uygulayan herhangi bir model şeffaf şekilde çalışır.
  - ISP : MLModelPort yalnızca predict() metodunu zorunlu kılar.
  - DIP : TurtleService, somut ML sınıfına değil MLModelPort soyutlamasına bağımlıdır.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agents.researcher import ResearcherAgent, ResearchResult
from agents.validator import ValidatorAgent, SecurityReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ML Modeli Arayüzü  (DIP)
# ---------------------------------------------------------------------------

class MLModelPort(ABC):
    """
    Makine öğrenmesi modeli için soyut arayüz.
    ResNet, EfficientNet veya herhangi başka bir model bu arayüzü uygulayabilir.
    """

    @abstractmethod
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Args:
            image_bytes: Ham görsel verisi.

        Returns:
            dict: En azından aşağıdaki anahtarları içermelidir:
                  - "turtle_id"      : Tanımlanan kaplumbağanın ID'si (veya None)
                  - "confidence"     : 0.0 – 1.0 arası güven skoru
                  - "is_new_turtle"  : Daha önce görülmemiş birey mi?
        """
        ...


# ---------------------------------------------------------------------------
# Sonuç Veri Transferi Nesnesi
# ---------------------------------------------------------------------------

@dataclass
class TurtleServiceResult:
    """TurtleService.process() çağrısının nihai çıktısı."""
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
            turtle_id = self.prediction.get("turtle_id", "?")
            confidence = self.prediction.get("confidence", 0.0)
            return (
                f"[{self.request_id}] Başarılı — "
                f"Kaplumbağa ID: {turtle_id} | Güven: {confidence:.1%}"
            )
        return f"[{self.request_id}] Başarısız ({self.stage_reached}): {self.error_message}"


# ---------------------------------------------------------------------------
# Koordinatör Servis
# ---------------------------------------------------------------------------

class TurtleService:
    """
    API katmanından gelen görsel isteğini alıp ajanları doğru sırada
    çalıştıran iş mantığı servisi.

    Kullanım:
        model  = MyResNetModel()          # MLModelPort uygulayan sınıf
        service = TurtleService(model)
        result  = await service.process(image_bytes, file_context)
    """

    def __init__(
        self,
        ml_model: MLModelPort,
        validator_agent: ValidatorAgent | None = None,
        researcher_agent: ResearcherAgent | None = None,
    ) -> None:
        """
        Args:
            ml_model         : MLModelPort arayüzünü uygulayan model nesnesi.
            validator_agent  : Opsiyonel; verilmezse varsayılan ValidatorAgent kullanılır.
            researcher_agent : Opsiyonel; verilmezse varsayılan ResearcherAgent kullanılır.
        """
        self._model: MLModelPort = ml_model
        self._validator: ValidatorAgent = validator_agent or ValidatorAgent()
        self._researcher: ResearcherAgent = researcher_agent or ResearcherAgent()

        logger.info(
            "TurtleService başlatıldı. Model: %s | Validator: %s | Researcher: %s",
            type(self._model).__name__,
            type(self._validator).__name__,
            type(self._researcher).__name__,
        )

    # ------------------------------------------------------------------
    # Ana İş Akışı
    # ------------------------------------------------------------------

    def process(
        self,
        image_bytes: bytes,
        file_context: dict[str, Any] | None = None,
        past_metadata: dict[str, Any] | None = None,
    ) -> TurtleServiceResult:
        """
        Görsel işleme boru hattını sırayla çalıştırır.

        Args:
            image_bytes  : Yüklenen ham görsel verisi.
            file_context : Dosya adı, yükleyen kullanıcı ID'si, IP adresi, vs.
            past_metadata: Araştırmacı ajana iletilecek geçmiş kayıt metadata'sı.

        Returns:
            TurtleServiceResult: Tüm aşamaların birleşik sonucu.
        """
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow()
        ctx = file_context or {}

        logger.info(
            "═══ Yeni İstek [%s] ═══ Dosya: %s | Boyut: %d byte",
            request_id, ctx.get("filename", "bilinmiyor"), len(image_bytes),
        )

        # ── 1. Güvenlik Doğrulaması ──────────────────────────────────────
        security_report = self._run_validation(image_bytes, ctx)
        if not security_report.passed:
            logger.warning("[%s] Güvenlik aşamasında reddedildi.", request_id)
            return TurtleServiceResult(
                request_id=request_id,
                timestamp=timestamp,
                success=False,
                stage_reached="validation",
                security_report=security_report,
                error_message=security_report.message,
            )

        # ── 2. Kalite Araştırması ────────────────────────────────────────
        research_result = self._run_research(image_bytes, past_metadata)
        if not research_result.passed:
            logger.warning("[%s] Kalite aşamasında reddedildi.", request_id)
            return TurtleServiceResult(
                request_id=request_id,
                timestamp=timestamp,
                success=False,
                stage_reached="research",
                security_report=security_report,
                research_result=research_result,
                error_message=research_result.message,
            )

        # ── 3. ML Tahmini ────────────────────────────────────────────────
        prediction = self._run_prediction(image_bytes, request_id)
        if "error" in prediction:
            return TurtleServiceResult(
                request_id=request_id,
                timestamp=timestamp,
                success=False,
                stage_reached="prediction",
                security_report=security_report,
                research_result=research_result,
                prediction=prediction,
                error_message=prediction["error"],
            )

        logger.info(
            "[%s] İşlem başarıyla tamamlandı. Tahmin: %s",
            request_id, prediction,
        )
        return TurtleServiceResult(
            request_id=request_id,
            timestamp=timestamp,
            success=True,
            stage_reached="complete",
            security_report=security_report,
            research_result=research_result,
            prediction=prediction,
        )

    # ------------------------------------------------------------------
    # Yardımcı Metotlar (private)
    # ------------------------------------------------------------------

    def _run_validation(
        self, image_bytes: bytes, context: dict[str, Any]
    ) -> SecurityReport:
        """ValidatorAgent'ı çalıştırır; hataları yakalar ve loglar."""
        try:
            logger.info("Adım 1/3: Güvenlik doğrulaması başlıyor…")
            report = self._validator.validate(image_bytes, context)
            logger.info(
                "Güvenlik doğrulaması tamamlandı. Sonuç: %s",
                "GEÇTI" if report.passed else "BAŞARISIZ",
            )
            return report
        except Exception as exc:
            logger.error("ValidatorAgent beklenmedik hata: %s", exc, exc_info=True)
            # Güvenlik hatası → reddet
            from agents.validator import SecurityReport, ValidationResult
            return SecurityReport(
                passed=False,
                results=[
                    ValidationResult(
                        passed=False,
                        validator_name="ValidatorAgent",
                        reason=f"Doğrulama sırasında beklenmedik hata: {exc}",
                    )
                ],
            )

    def _run_research(
        self,
        image_bytes: bytes,
        past_metadata: dict[str, Any] | None,
    ) -> ResearchResult:
        """ResearcherAgent'ı çalıştırır; hataları yakalar ve loglar."""
        try:
            logger.info("Adım 2/3: Kalite araştırması başlıyor…")
            result = self._researcher.analyze(image_bytes, past_metadata or {})
            logger.info(
                "Kalite araştırması tamamlandı. Sonuç: %s | Sorunlar: %s",
                "GEÇTI" if result.passed else "BAŞARISIZ",
                result.issues,
            )
            return result
        except Exception as exc:
            logger.error("ResearcherAgent beklenmedik hata: %s", exc, exc_info=True)
            from agents.researcher import ResearchResult, QualityMetrics
            return ResearchResult(
                passed=False,
                quality_metrics=QualityMetrics(0, 0, 0.0, 0.0, 0.0, 0),
                issues=[f"Kalite analizi sırasında beklenmedik hata: {exc}"],
            )

    def _run_prediction(
        self, image_bytes: bytes, request_id: str
    ) -> dict[str, Any]:
        """ML modelini çalıştırır; hataları yakalar ve loglar."""
        try:
            logger.info("Adım 3/3: ML tahmini başlıyor… [%s]", request_id)
            prediction = self._model.predict(image_bytes)
            logger.info("ML tahmini tamamlandı: %s", prediction)
            return prediction
        except Exception as exc:
            logger.error(
                "[%s] ML modeli beklenmedik hata: %s", request_id, exc, exc_info=True
            )
            return {"error": f"ML modeli çalıştırılırken hata: {exc}"}
