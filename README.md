# 🐢 CarettaTrack: AI-Powered Sea Turtle Identification System

> "Nesli tükenmekte olan deniz kaplumbağalarını, yapay zeka ile bireysel düzeyde takip eden akıllı kimlik tanıma platformu."

## 📌 Proje Özeti
CarettaTrack, Caretta Caretta (deniz kaplumbağası) bireylerini yüzlerindeki benzersiz pul dizilimlerinden (scutes) -tıpkı bir parmak izi gibi- ayırt etmeyi sağlayan uçtan uca bir makine öğrenmesi ve web projesidir. Sistem, kullanıcıların yüklediği fotoğrafları derin öğrenme modelleriyle analiz ederek birey tahmini yapar ve veritabanına yeni bireyler kazandırır.

## 🚀 Öne Çıkan Mühendislik Çözümleri

* **SOLID ve Modüler Mimari:** Makine öğrenmesi modeli ile backend sıkı sıkıya bağlı değildir. `ITurtleRecognizer` soyut arayüzü (DIP) sayesinde, sistem ileride farklı yapay zeka modellerine (ViT, EfficientNet) kod değiştirilmeden entegre edilebilir.
* **Akıllı Doğrulama Katmanı (Validator):** Kullanıcıdan gelen görseller, ana yapay zeka modeline ulaşmadan önce bir güvenlik katmanından geçer. Geçersiz fotoğraflar sistem kaynaklarını tüketmeden reddedilir.
* **Şeffaf Kullanıcı Deneyimi (UX):** Başarılı veya başarısız tüm işlemler, güven skorları (% Confidence) ve detaylı, anlaşılır hata mesajları ("Bu görsel bir kaplumbağa içermiyor olabilir") ile kullanıcıya sunulur.

## 🛠️ Teknoloji Yığını
* **Makine Öğrenmesi:** PyTorch, Torchvision (ResNet-18)
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **Bulut Entegrasyonu:** Kaggle (Model eğitimi için T4 x2 GPU kullanılmıştır)

## 🧠 Model Eğitimi ve Altyapı
Sistem sıfırdan eğitilmek yerine, **ResNet-18** kullanılarak Transfer Learning (Öğrenme Aktarımı) yöntemiyle geliştirilmiştir:
* **Veri Seti:** 438 farklı bireyi içeren 8700+ fotoğraflık SeaTurtleID2022 veri seti.
* **Eğitim Stratejisi:** Aşırı öğrenmeyi (Overfitting) engellemek adına modelin temel özellikleri dondurulmuş (Feature Freezing), yalnızca son katman eğitilmiştir.
* **Aşırı Öğrenme Kalkanı:** Model eğitim sırasında `best_val_loss` algoritmasıyla sürekli denetlenmiş, 40 Epoch'luk eğitim boyunca yalnızca en başarılı (optimum) ağırlıklar `.pth` formatında kaydedilmiştir.

## 📸 Doğru Tanıma İçin İpuçları
Modelin en yüksek güven skoruyla (%90+) çalışabilmesi için yüklenen fotoğrafların:
1. Kaplumbağanın yüzünü sağ veya sol **yan profilden** net bir şekilde göstermesi,
2. Suyun bulanıklaştırmadığı, yüksek kaliteli kareler olması,
3. Sadece üst kabuğu değil, karakteristik kafa yapısını barındırması tavsiye edilir.

---
*Geliştirici:* Sema Gül Çelik
