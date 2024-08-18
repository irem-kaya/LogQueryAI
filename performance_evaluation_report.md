# Performans Değerlendirme Raporu

## 1. Proje Tanımı
Bu projede, log verilerini kullanarak bir soru-cevap sisteminin performansını değerlendirdik. Sistemin doğruluğunu ve performansını ölçmek amacıyla çeşitli metrikler kullanarak sistemin başarısını inceledik.

## 2. Veri İşleme
**Veri Kümesi:**
- Kullanılan veri kümesi, 700,000 kayıt içeren bir log dosyasından türetilmiştir. Ancak, işlem süresinin uzunluğu nedeniyle analiz için daha küçük bir veri alt kümesi kullanılmıştır.
- Küçük veri alt kümesi olarak 200 örnek seçilmiştir.

**Veri Temizleme:**
- Log verileri, ham formatından işlenebilir formata dönüştürülmüştür.
- Eksik değerler (NaN) ve tutarsızlıklar düzeltilmiş, veri kalitesi artırılmıştır.
- Soru-cevap çiftleri oluşturulmuş ve bu çiftler modelin değerlendirilmesinde kullanılmıştır.

## 3. Model Entegrasyonu
**RAG Modelinin Entegrasyonu:**
- **RAG Modeli:** Retrieval-Augmented Generation (RAG) modeli, bilgiyi geri getirme (retrieval) ve metin üretimi (generation) bileşenlerini birleştirir. Bu model, önceden eğitilmiş bir dil modeli ile birlikte bilgi tabanı sorgulama yeteneği sağlar.
- **Entegrasyon Süreci:** 
  - **Veri Hazırlığı:** 
    - **Vektör Veri Tabanı:** Log verilerinden elde edilen önemli bilgilerle bir vektör veri tabanı oluşturulmuştur. Bu işlem için `FAISS` kütüphanesi kullanılmıştır. 
    - **Vektörizasyon:** Log verileri, metin gömme (embedding) yöntemleri kullanılarak vektörlere dönüştürülmüştür. Bu vektörler, bilgilere hızlı erişim sağlamak amacıyla indekslenmiştir.
  - **RAG Modeli:** 
    - **Modül Yapılandırması:** PyTorch ve Hugging Face `transformers` kütüphaneleri kullanılarak, RAG modelinin hem bilgi geri getirme hem de metin üretim modülleri entegre edilmiştir.
    - **Bilgi Geri Getirme:** Model, kullanıcı sorularına en uygun bilgileri vektör veri tabanından getirir.
    - **Metin Üretimi:** Geri getirilen bilgileri kullanarak doğru ve anlamlı yanıtlar üretir.
  - **Performans ve Test:** Modelin performansı, doğruluk ve hız metrikleri kullanılarak değerlendirilmiştir.

**Model:**
- GPT-2 modeli, soru-cevap sistemine uygun şekilde yapılandırılmıştır.
- Eğitim süreci sırasında model, belirli bir hiperparametre seti ile optimize edilmiştir.

**Eğitim Parametreleri:**
- Eğitim süresi: 3 Epoch
- Batch boyutu: 4

## 4. Performans Değerlendirmesi
**Test Verisi:**
- 100 örnekten oluşan bir test veri alt kümesi kullanılmıştır.
- Gerçek ve tahmin edilen yanıtlar arasındaki benzerlik hesaplanmıştır.

**Benzerlik Hesaplaması:**
- `cosine_similarity` metodu kullanılarak ortalama benzerlik skoru hesaplanmıştır.
- Ortalama Benzerlik: **%90** (Not: Bu oran daha doğru bir hesaplama sonucu elde edilen yüzdedir, örneğin 0.90 x 100 = %90)

## 5. Karşılaşılan Zorluklar
- **Veri İşleme Süresi:** 700,000 kayıt içeren büyük veri setlerinin işlenmesi uzun sürdü. Bu nedenle, daha küçük bir veri kümesi ile çalışmak zorunda kaldık. Bu, modelin aşırı öğrenmesine neden olabilir.
- **Model Performansı:** Modelin yanıtlarının doğruluğu, özellikle uzun metinlerde sınırlı kalmıştır. Gömme vektörlerinin hesaplanması ve benzerlik ölçümü sırasında bazı teknik zorluklar yaşanmıştır.
- **CSV Dosyası Sorunları:** CSV dosyasındaki NaN değerleri doğruluk oranlarını olumsuz etkiledi.

## 6. Sonuçlar ve Öneriler
**Sonuçlar:**
- Sistem, belirli bir doğruluk seviyesine ulaşmış olsa da, benzerlik oranı daha yüksek olabilir. Bu, modelin eğitiminde kullanılan verilerin çeşitliliği ve kalitesine bağlıdır.

**Öneriler:**
- Eğitim veri setini artırmak ve modelin hiperparametrelerini yeniden yapılandırmak faydalı olabilir.
- Daha geniş bir veri kümesi üzerinde yeniden eğitim yapılarak modelin performansı artırılabilir.

## 7. Gelecek Çalışmalar
- Modelin daha uzun süreli eğitimleri ve daha büyük veri kümesi ile performans iyileştirmeleri hedeflenmelidir.
- Farklı metrikler ve performans testleri ile sistemin kapsamlı değerlendirmesi yapılmalıdır.
- **Paralel Yapı Kullanımı:** Bu projede paralel bir yapı kullandım, ancak daha büyük veri setlerinde bu yapı sınırlı kalabilir. Bunun yerine, C++ kütüphanelerinden faydalanılması modelin işlem sürelerini azaltabilir ve performansı artırabilir.
- **Gelişmiş Modeller:** GPT-4 veya daha yeni sürümler kullanıldığında, daha doğru ve kapsamlı yanıtlar elde edilebilir. 
