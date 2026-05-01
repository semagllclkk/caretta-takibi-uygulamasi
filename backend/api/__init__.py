"""
api/ — FastAPI Katmanı (Presentation Layer)

SOLID prensipleri:
  - SRP : Her endpoint yalnızca HTTP isteklerini işler, iş mantığı TurtleService'te.
  - OCP : Yeni endpoint eklemek için mevcut endpoints değişmez.
  - DIP : Endpoints, TurtleService (somut sınıf) değil, ITurtleRecognizer arayüzüyle çalışır.
"""
