# 📊 Log Analizi ve RAG Tabanlı Soru-Cevap Sistemi

Bu proje, büyük ölçekli sistem loglarını analiz ederek, kullanıcıların doğal dilde sorduğu sorulara yanıt veren bir **RAG (Retrieval-Augmented Generation)** sistemi geliştirmeyi amaçlar. Sistem, **FAISS** vektör veritabanı ve **GPT-2** dil modelini entegre ederek log verilerinden anlamlı içgörüler sunar.

## 🚀 Proje Hakkında
Geleneksel log analiz yöntemleri genellikle manuel sorgulara veya statik dashboard'lara dayanır. Bu projede ise Generative AI (Üretken Yapay Zeka) kullanılarak, log verileriyle "konuşulabilen" bir yapı kurulmuştur.

**Temel Özellikler:**
* **Doğal Dil İşleme:** Kullanıcı sorularını anlama ve bağlama uygun yanıt üretme.
* **Vektör Arama:** FAISS ile hızlı bilgi geri getirme (Retrieval).
* **Uçtan Uca Akış:** Ham veriden yanıt üretimine kadar tam entegre pipeline.

---

## 🛠️ Kullanılan Teknolojiler ve Mimari

Proje aşağıdaki temel bileşenler üzerine inşa edilmiştir:

| Bileşen | Teknoloji / Kütüphane | Açıklama |
| :--- | :--- | :--- |
| **Dil Modeli (LLM)** | GPT-2 (Fine-tuned) | Metin üretimi ve yanıt oluşturma. |
| **Vektör DB** | FAISS | Embedding vektörlerinin indekslenmesi ve hızlı aranması. |
| **Framework** | PyTorch & Hugging Face | Model eğitimi ve transformer mimarisi. |
| **Veri İşleme** | Pandas & NumPy | Log temizleme ve manipülasyon. |

### Model Mimarisi (RAG)
Sistem **Retrieval-Augmented Generation** mimarisini kullanır:
1.  **Retrieval (Geri Getirme):** Kullanıcı sorusu vektöre dönüştürülür ve FAISS veritabanında en alakalı log kayıtları bulunur.
2.  **Generation (Üretim):** Bulunan log kayıtları "bağlam" (context) olarak GPT-2 modeline verilir ve nihai yanıt üretilir.

---

## 📂 Veri Seti ve İşleme Süreci

### Veri Kaynağı
Proje başlangıcında **700.000 kayıt** içeren ham bir log dosyası kullanılmıştır.

### Ön İşleme ve Pilot Çalışma (PoC)
Büyük veri setinin işlenmesindeki donanım kısıtları ve optimizasyon ihtiyaçları nedeniyle, projenin bu aşaması bir **Proof of Concept (Kavram Kanıtı)** olarak tasarlanmıştır.

1.  **Temizleme:** Ham log verileri parse edildi, `NaN` değerler ve tutarsız kayıtlar temizlendi.
2.  **Örnekleme:** Analiz ve model eğitimi için 700.000 kayıt arasından stratejik olarak seçilen **200 adet yüksek nitelikli örnek** kullanıldı.
3.  **Vektörizasyon:** Metin verileri embedding katmanından geçirilerek vektör uzayına taşındı.

---

## 📈 Performans Değerlendirmesi

Modelin başarısı, test veri kümesi üzerinde yapılan karşılaştırmalarla ölçülmüştür.

* **Test Verisi:** 100 örneklem.
* **Metrik:** Cosine Similarity (Kosinüs Benzerliği).
* **Eğitim Parametreleri:** 3 Epoch, Batch Size: 4.

| Metrik | Skor | Not |
| :--- | :--- | :--- |
| **Ortalama Benzerlik** | **%90** | Modelin üretilen yanıtları ile referans yanıtlar arasındaki anlamsal yakınlık. |

*> **Not:** %90'lık başarı oranı, pilot veri setinin (200 örnek) sınırlı olmasından kaynaklı "overfitting" (aşırı öğrenme) etkisini içerebilir. Geniş veri setlerinde bu oran değişkenlik gösterebilir.*

---

## ⚠️ Karşılaşılan Zorluklar

1.  **Veri Hacmi ve İşlem Gücü:** 700.000 satırlık verinin vektörizasyonu ve eğitimi, mevcut donanım altyapısında darboğaz oluşturduğu için örneklem (sampling) yöntemine gidildi.
2.  **Bağlam Uzunluğu (Context Window):** GPT-2 modelinin token limitleri, çok uzun log satırlarının analizinde bilgi kaybına yol açabildi.
3.  **Veri Kalitesi:** CSV dosyasındaki kirlilik ve eksik veriler, ilk aşamada modelin halüsinasyon görmesine (hatalı bilgi üretimi) neden oldu.

---

## 🔮 Gelecek Çalışmalar ve Geliştirme Planı

Bu projeyi daha ileriye taşımak için hedeflenen adımlar şunlardır:

* **Modern LLM Entegrasyonu:** GPT-2 yerine daha güncel ve hafif modellerin (Örn: TinyLlama, DistilGPT veya Mistral) kullanılması.
* **Optimizasyon:** İşlem sürelerini düşürmek için **Model Quantization (4-bit/8-bit)** tekniklerinin ve GPU hızlandırmasının (CUDA) uygulanması.
* **Veri Genişletme:** Pilot çalışmanın başarısı üzerine, 700.000 verinin tamamının işlenebileceği dağıtık bir yapıya (Örn: Spark veya Ray) geçilmesi.
* **Gelişmiş Metrikler:** Performans ölçümüne ROUGE ve BLEU skorlarının eklenmesi.

---

## 💻 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için:

```bash
# Projeyi klonlayın
git clone [https://github.com/kullaniciadi/proje-adi.git](https://github.com/kullaniciadi/proje-adi.git)

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
python main.py
