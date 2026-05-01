"""
agents/data_sources.py
-----------------------
Veri Kaynağı Arayüzü ve Implementasyonları

SOLID – OCP:
  Yeni bir kaynak (Kaggle, iNaturalist, GBIF, vb.) eklemek için
  yalnızca bu dosyaya yeni bir IDataSource alt sınıfı eklenir;
  ResearcherAgent sınıfına dokunulmaz.

SOLID – DIP:
  ResearcherAgent somut kaynak sınıflarını değil,
  IDataSource arayüzünü bağımlılık olarak alır.

SOLID – ISP:
  IDataSource yalnızca search() ve source_name'i zorunlu kılar;
  kaynaklar ihtiyaç fazlası metot taşımak zorunda kalmaz.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Veri Kaydı DTO
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    """
    Herhangi bir kaynaktan dönen tek fotoğraf/metadata kaydı.
    Tüm IDataSource implementasyonları bu ortak formata dönüştürür.
    """
    url: str
    source: str                              # "bing_mock", "duckduckgo_mock", "kaggle_mock", …
    query: str                               # kullanılan arama terimi
    title: str = ""
    width: int = 0
    height: int = 0
    file_format: str = ""
    license: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolution_known(self) -> bool:
        return self.width > 0 and self.height > 0


# ---------------------------------------------------------------------------
# Soyut Arayüz  (IDataSource)
# ---------------------------------------------------------------------------

class IDataSource(ABC):
    """
    Tüm veri kaynaklarının uyması gereken sözleşme.

    Yeni kaynak eklemek:
        class KaggleDataSource(IDataSource):
            @property
            def source_name(self) -> str:
                return "kaggle"

            def search(self, query: str, max_results: int) -> list[DataRecord]:
                ...  # Kaggle API çağrısı
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Kaynağın benzersiz tanımlayıcısı."""
        ...

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[DataRecord]:
        """
        Verilen sorguya göre görsel URL'leri ve metadata'yı döner.

        Args:
            query      : Arama terimi (ör. "Caretta Caretta face").
            max_results: Döndürülecek maksimum kayıt sayısı.

        Returns:
            list[DataRecord]: Normalize edilmiş kayıt listesi.
        """
        ...


# ---------------------------------------------------------------------------
# Mock Implementasyonlar  (Gerçek API'lerin yerine geçer)
# ---------------------------------------------------------------------------

class MockBingImageSource(IDataSource):
    """
    Bing Image Search API'nin mock implementasyonu.

    Gerçek entegrasyon için:
      pip install azure-cognitiveservices-search-imagesearch
      client = ImageSearchClient(endpoint, CognitiveServicesCredentials(api_key))
      results = client.images.search(query=query, count=max_results)
    """

    _FAKE_DOMAINS = [
        "images.biologist.net",
        "cdn.marinelife.org",
        "static.iucnredlist.org",
        "photos.wwf.org",
        "media.nationalgeographic.com",
    ]

    _FAKE_FORMATS = ["jpg", "jpeg", "png"]

    @property
    def source_name(self) -> str:
        return "bing_mock"

    def search(self, query: str, max_results: int = 20) -> list[DataRecord]:
        logger.info("[BingMock] Arama: '%s' | max: %d", query, max_results)
        time.sleep(0.05)  # Sahte ağ gecikmesi

        records: list[DataRecord] = []
        for i in range(max_results):
            domain = random.choice(self._FAKE_DOMAINS)
            fmt = random.choice(self._FAKE_FORMATS)
            width = random.choice([320, 480, 640, 800, 1024, 1280, 1920])
            height = random.choice([240, 360, 480, 600, 768, 1080])
            records.append(DataRecord(
                url=f"https://{domain}/caretta/{query.replace(' ', '_')}_{i:04d}.{fmt}",
                source=self.source_name,
                query=query,
                title=f"Caretta Caretta – Bing Sonuç #{i + 1}",
                width=width,
                height=height,
                file_format=fmt,
                license="bing_standard",
                extra={"rank": i + 1, "engine": "bing"},
            ))

        logger.info("[BingMock] %d kayıt döndürüldü.", len(records))
        return records


class MockDuckDuckGoImageSource(IDataSource):
    """
    DuckDuckGo Image Search'ün mock implementasyonu.

    Gerçek entegrasyon için:
      pip install duckduckgo-search
      from duckduckgo_search import DDGS
      results = list(DDGS().images(query, max_results=max_results))
    """

    _FAKE_DOMAINS = [
        "upload.wikimedia.org",
        "flickr.com/caretta",
        "inaturalist-open-data.s3.amazonaws.com",
        "static.seaturtle.org",
    ]

    @property
    def source_name(self) -> str:
        return "duckduckgo_mock"

    def search(self, query: str, max_results: int = 20) -> list[DataRecord]:
        logger.info("[DDGMock] Arama: '%s' | max: %d", query, max_results)
        time.sleep(0.03)

        records: list[DataRecord] = []
        for i in range(max_results):
            domain = random.choice(self._FAKE_DOMAINS)
            width = random.choice([256, 512, 640, 800, 1024])
            height = random.choice([256, 384, 512, 600, 768])
            records.append(DataRecord(
                url=f"https://{domain}/sea_turtle/{i:05d}.jpg",
                source=self.source_name,
                query=query,
                title=f"Sea Turtle Head – DDG #{i + 1}",
                width=width,
                height=height,
                file_format="jpg",
                license="cc_by",
                extra={"engine": "duckduckgo", "safe_search": "on"},
            ))

        logger.info("[DDGMock] %d kayıt döndürüldü.", len(records))
        return records


class MockKaggleDatasetSource(IDataSource):
    """
    Kaggle Dataset API'nin mock implementasyonu.

    Gerçek entegrasyon için:
      pip install kaggle
      import kaggle
      kaggle.api.dataset_download_files("dataset/caretta-caretta", path="./data")
    """

    _DATASET_SLUG = "marine-bio/caretta-caretta-faces"

    @property
    def source_name(self) -> str:
        return "kaggle_mock"

    def search(self, query: str, max_results: int = 20) -> list[DataRecord]:
        logger.info("[KaggleMock] Dataset: '%s' | max: %d", self._DATASET_SLUG, max_results)
        time.sleep(0.08)

        records: list[DataRecord] = []
        subsets = ["train", "val", "test"]
        for i in range(max_results):
            subset = subsets[i % len(subsets)]
            records.append(DataRecord(
                url=(
                    f"https://storage.googleapis.com/kaggle-data-sets/"
                    f"{self._DATASET_SLUG}/{subset}/caretta_{i:05d}.jpg"
                ),
                source=self.source_name,
                query=query,
                title=f"Kaggle Caretta – {subset} set #{i}",
                width=1024,
                height=768,
                file_format="jpg",
                license="cc0",
                extra={
                    "dataset": self._DATASET_SLUG,
                    "subset": subset,
                    "label": "caretta_caretta",
                },
            ))

        logger.info("[KaggleMock] %d kayıt döndürüldü.", len(records))
        return records
