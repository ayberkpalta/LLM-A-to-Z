# ResearchMindAI - Makine Öğrenmesi ile Akademik Makale Sınıflandırma

ResearchMindAI, akademik makaleleri (ArXiv üzerinden çekilen) içeriklerine göre otomatik olarak analiz eden ve sınıflandıran, uçtan uca (end-to-end) tasarlanmış bir Çoklu Etiketli (Multi-Label) Doğal Dil İşleme (NLP) projesidir.

Bu proje, veri biliminin tüm aşamalarını (Veri çekme, doğrulama, ön işleme, veri bölme, modelleme, hyperparameter tuning ve canlı tahmin) adım adım ve profesyonel bir şekilde işlemektedir. Başlangıçta geleneksel TF-IDF ile yola çıkılmış, ardından gelişmiş **Sentence-Transformers (MPNet)** mimarisi ve **Linear SVM** ile state-of-the-art seviyeye çıkarılmıştır.

---

## 📂 Proje Yapısı

```
ResearchMindAI_project/
│
├── data/
│   ├── raw/                 # ArXiv'den çekilen işlenmemiş ham veriler (papers.json)
│   └── processed/           # Temizlenmiş, bölünmüş ve modele hazır veriler (train, val, test)
│       ├── embeddings/      # MPNet ile çıkarılan 768 boyutlu vektörler
│       └── labels/          # Binary matrix formatına çevrilmiş etiketler
│
├── models/                  # Eğitilmiş final makine öğrenmesi modelleri
│   ├── researchmindai_svm_final.joblib
│   └── model_metadata.npz
│
├── notebooks/               # Sürecin ilk adımlarının işlendiği Jupyter Notebook'lar
│
├── src/                     # Üretim ortamı (Production) için Python script'leri
│   ├── data_collection.py   # ArXiv API'den veri çeken script
│   ├── dataset_builder.py   # Veri birleştirme ve JSON oluşturma
│   ├── embedding_model.py   # MPNet ile metinleri embedding'e çevirme
│   ├── model_comparison.py  # Modeller arası (LR vs SVM) karşılaştırma (Benchmarking)
│   ├── svm_tuning.py        # SVM için Grid/Random search mantığıyla C parametresi optimizasyonu
│   ├── save_final_model.py  # En iyi modelin diske kaydedilmesi
│   └── predict.py           # Canlı metin girişi ile model tahmini yapan Inference scripti
│
├── try_code_apply/          # Deneme/Test scriptleri
└── README.md                # Proje dokümantasyonu
```

---

## 📓 Mimari ve Yapılan İşlemler

Aşağıda projenin baştan sona veri boru hattı (Data Pipeline) ve makine öğrenmesi süreçleri açıklanmıştır.

### 1. Veri Madenciliği ve Doğrulama (Data Ingestion & Validation)
Projenin ham verisi ArXiv API kullanılarak çekildi.
- **API Entegrasyonu:** Python `requests` ve XML parsing kullanılarak 4 ana Computer Science kategorisinden (`NLP`, `Computer Vision`, `Machine Learning`, `Robotics`) yüzlerce makale toplandı.
- **Data Validation:** Çekilen verinin kalitesini ölçmek için; eksik/boş (None/Null) alan kontrolü, aynı ID ve Başlığa sahip kopya (Duplicate) verilerin temizliği yapıldı.

### 2. Veri İşleme ve Bölme (Data Processing & Splitting)
Veriyi modele sokmadan önce doğru bir şekilde etiketlerini ayırmak ve modeli kör test edebilmek için eğitim (Train), doğrulama (Validation) ve Test setlerine bölme işlemi yapıldı.
- **One-Hot Encoding:** Etiketler (Labels), makinenin anlayabileceği Binary Matrix (0 ve 1'ler) formatına dönüştürüldü (Örn: `[1, 0, 1, 0]`).
- **Iterative Stratification:** Multi-label (Örn. bir makalenin hem NLP hem ML olması) dengesini bozmamak için gelişmiş `MultilabelStratifiedShuffleSplit` kullanılarak veri Train, Validation ve Test olarak ayrıldı.

### 3. Gelişmiş NLP: Word Embeddings (MPNet)
TF-IDF gibi kelime frekansına dayalı geleneksel yöntemlerin ötesine geçilerek semantik (anlamsal) analiz yapıldı.
- **Sentence-Transformers:** HuggingFace'in `all-mpnet-base-v2` modeli kullanılarak, her makalenin başlık ve özeti 768 boyutlu yoğun vektör uzayına (dense vectors) haritalandırıldı. Bu sayede kelimeler geçmese bile cümlenin anlamsal bağıntısı modele öğretildi.

### 4. Makine Öğrenmesi: Modelleme ve Karşılaştırma (Benchmarking)
Çıkarılan 768 boyutlu vektörler, farklı sınıflandırıcı algoritmalar ile test edildi.
- **Logistic Regression (Global Threshold):** Standart `0.5` eşik değeri yerine `0.45` kullanılarak optimize edildi.
- **Class-Specific Thresholds (Sınıfa Özel Eşik):** Özellikle yakalaması zor olan *Machine Learning* sınıfında (Eşik: `0.25`), diğer sınıflarda ise (`0.40 - 0.45`) gibi sınıfa özel eşik değerleri bulunarak Recall ve F1 skorları inanılmaz oranda artırıldı.
- **Linear SVM (Support Vector Machine):** Yüksek boyutlu verilerde hiper-düzlemlerle (hyperplanes) en iyi ayırıcı olan SVM, `OneVsRestClassifier` sarmalayıcısı ile uygulandı.

### 5. Hyperparameter Tuning ve Sonuçlar
SVM modelinin optimizasyonu için `C` (Regularization) parametresi `0.01`'den `10.0`'a kadar Validation seti üzerinde test edildi.
- **Optimum C Değeri:** `0.25` olarak bulundu.

**Final Modellerin Test Setindeki Karşılaştırması:**
| Model | Micro F1 | Macro F1 |
| --- | --- | --- |
| Logistic Regression — Global 0.45 | 0.832 | 0.829 |
| Logistic Regression — Class-specific | 0.824 | 0.829 |
| **Linear SVM (C=0.25)** | **0.842** | **0.838** |

*(Not: SVM modelinde Robotics sınıfı için F1-Score **0.96**'ya ulaşarak kusursuz bir başarı sergilemiştir.)*

### 6. Canlı Tahmin Motoru (Inference / Predict)
Eğitilen ve en iyi performansı gösteren SVM modeli (joblib formatında) diske kaydedildi.
- `predict.py` script'i yazılarak; dışarıdan verilen herhangi bir makale özetini önce MPNet ile embedding'e çeviren, ardından eğitilmiş SVM modeline sokarak sınıfını tahmin eden dinamik bir yapı kuruldu.

---

## 🛠 Kullanılan Teknolojiler
- **Veri Madenciliği:** `requests`, `xml.etree.ElementTree`
- **Doğal Dil İşleme (NLP):** `sentence-transformers` (MPNet), `nltk`
- **Veri İşleme ve Vektör Uzayı:** `numpy`, `json`
- **Makine Öğrenmesi (ML):** `scikit-learn`, `iterative-stratification`
- **Model Kayıt:** `joblib`

---

## 🚀 Sonraki Adımlar (Gelecek Planları)
1. **Web API (FastAPI):** Geliştirilen bu uçtan uca altyapının, FastAPI ile bir REST API'ye dönüştürülmesi.
2. **Kullanıcı Arayüzü (Streamlit / React):** API'ye bağlanan web tabanlı bir arayüz ile araştırmacıların (researcher) makalelerini kolayca sınıflayabileceği bir servis haline getirilmesi.
3. **Bulut (Cloud) Dağıtımı:** Dockerize edilip AWS/GCP üzerinde canlıya alınması.
