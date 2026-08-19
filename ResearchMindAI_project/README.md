# ResearchMindAI - Makine Öğrenmesi ile Akademik Makale Sınıflandırma

ResearchMindAI, akademik makaleleri (ArXiv üzerinden çekilen) içeriklerine göre otomatik olarak analiz eden ve sınıflandıran, uçtan uca (end-to-end) tasarlanmış bir Çoklu Etiketli (Multi-Label) Doğal Dil İşleme (NLP) projesidir.

Bu proje, veri biliminin tüm aşamalarını (Veri çekme, doğrulama, ön işleme, etiketleme, bölme, modelleme ve hata analizi) adım adım, öğretici ve detaylı bir şekilde işlemektedir.

---

## 📂 Proje Yapısı

```
ResearchMindAI_project/
│
├── data/
│   ├── raw/                 # ArXiv'den çekilen işlenmemiş ham veriler (papers.json)
│   └── processed/           # Temizlenmiş, bölünmüş ve modele hazır veriler (train, val, test)
│
├── notebooks/               # Tüm sürecin adım adım işlendiği Jupyter Notebook'lar
│   ├── 01_data_ingestion.ipynb
│   ├── 02_build_classification_dataset.ipynb
│   ├── 03_data_splitting.ipynb
│   └── 04_text_prep.ipynb
│
├── src/                     # Üretim ortamı için Python script'leri
├── try_code_apply/          # Deneme/Test scriptleri
└── .gitignore               # Versiyon kontrolünde gizlenecek dosyalar
```

---

## 📓 Notebook Detayları ve Yapılan İşlemler

Aşağıda projede yer alan tüm adımların detaylı bir analizi ve her adımda nelerin hedeflendiği anlatılmaktadır.

### 1. `01_data_ingestion.ipynb` (Veri Çekme ve Doğrulama)
Bu aşamanın amacı, projenin ham verisini elde etmek ve veri tutarlılığını test etmektir.
- **ArXiv API Entegrasyonu:** Python `requests` modülü kullanılarak `cs.CL` (NLP) kategorisinden 10 makale deneme amaçlı çekildi.
- **XML Parsing:** Gelen XML yanıtı `xml.etree.ElementTree` ile parse edildi ve içinden *ID, Başlık, Özet (Abstract), Yazarlar, Kategoriler ve URL'ler* çıkarılarak yapılandırılmış bir JSON formatına getirildi.
- **Data Validation (Veri Doğrulama):** Çekilen verinin kalitesini ölçmek için; eksik/boş (None/Null) alan kontrolü, aynı ID ve Başlığa sahip kopya verilerin kontrolü yapıldı.
- **NLTK ile Temel Metin Ön İşleme:** Metinler NLTK kullanılarak `word_tokenize` ile kelimelere bölündü, noktalama işaretleri temizlendi ve temel `stopwords`'ler atıldı.
- **TF-IDF Konsepti:** Sadece 10 makale üzerinde basit bir `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) kurularak, metinlerin sayılara (540 özellikli vektörlere) nasıl dönüştüğü gözlemlendi.

### 2. `02_build_classification_dataset.ipynb` (Geniş Çaplı Veriseti Oluşturma)
1. aşamadaki başarıdan sonra makine öğrenmesi modelinin eğitilebilmesi için asıl veriseti toplandı.
- **Kategori Çeşitlendirmesi:** 4 ana Computer Science kategorisinden (`NLP`, `Computer Vision`, `Machine Learning`, `Robotics`) 50'şer adet olmak üzere geniş çaplı veri çekildi.
- **Etiketleme (Labeling):** Çekilen makalelerin ait olduğu kategoriler `labels` array'i altına etiket olarak eklendi.
- **Veri Birleştirme:** Tüm çekilen makaleler tek bir JSON (`papers_merged.json`) dosyasında birleştirilerek kaydedildi.

### 3. `03_data_splitting.ipynb` (Veri Analizi ve Parçalama)
Veriyi modele sokmadan önce doğru bir şekilde etiketlerini ayırmak ve modeli kör test edebilmek için eğitim (Train), doğrulama (Validation) ve Test setlerine bölme işlemi yapıldı.
- **Kombinasyon Analizi:** Makalelerin sınıflara göre dağılımı ve çapraz kategori eşleşmeleri (`NLP + Machine Learning` gibi multi-label kombinasyonları) `Counter` ile analiz edildi.
- **One-Hot Encoding:** Etiketler string formatından makinenin anlayabileceği Binary Matrix (0 ve 1'ler) formatına dönüştürüldü (Örn: `[1, 0, 1, 0]`).
- **Stratified Split:** Verinin her parçada dengeli dağılması için `MultilabelStratifiedShuffleSplit` kullanılarak veri Train (%70), Validation (%15) ve Test (%15) olarak ayrıldı.

### 4. `04_text_prep.ipynb` (Gelişmiş Metin Ön İşleme, Modelleme ve Hata Analizi)
Bu notebook, projenin "Karar Verme" ve "Makine Öğrenmesi" merkezidir (74 hücre). Metinler tamamen pürüzsüzleştirilip model eğitilmiş ve ince ayarlar yapılmıştır.

**A. Veri Temizleme (Data Cleaning)**
- Modelin öğrenmesi için makale başlığı (`title`) ve makale özeti (`abstract`) birleştirilerek tek bir zengin `feature` haline getirildi.
- `Regex` (Regular Expressions) kullanılarak sayılar, özel semboller temizlendi ve "Simple Clean" adımı tamamlandı.

**B. Stopwords Karşılaştırması ve TF-IDF Optimizasyonu**
Sklearn kütüphanesinin `ENGLISH_STOP_WORDS` (318 kelime) listesi kullanılarak metindeki bağlaç ve dolgu kelimeleri ("the", "is", "at") çıkarıldı. 
- **Normal TF-IDF Vocabulary Size:** 3729 Kelime
- **Stopword Çıkartılmış TF-IDF Size:** 3522 Kelime
*Sonuç: Sadece Stopword filtresi uygulayarak modelin belleğinde gereksiz yer kaplayan ~200 kelime başarıyla elenerek model daha saf metinlere odaklandı.*

**C. Modelleme ve Hüsran**
- Veri, `OneVsRestClassifier` sarmalayıcısı ile `LogisticRegression` algoritmasına verildi.
- **İlk Sonuç:** Model varsayılan Threshold (Eşik) değeri olan `0.5` ile test edildiğinde tüm tahminler "0 (Negative)" çıktı. Yani model hiçbir makalenin hiçbir sınıfı temsil ettiğinden %50 oranında emin olamadı (`Micro F1: 0.0`). Bu durum Multi-label projelerinde sık karşılaşılan bir zorluktur.

**D. Threshold (Eşik) Analizi**
Modelin skorlarını optimize etmek için **Validation** seti üzerinden özel bir Threshold (Eşik Değeri) araması yapıldı:
- Eşik `0.20` -> Micro F1: 0.43
- Eşik `0.25` -> Micro F1: 0.53
- **Eşik `0.30` -> Micro F1: 0.667 (Optimum Nokta)**
- Eşik `0.35` -> Micro F1: 0.44

Eşik değeri `0.30`'a çekildiğinde modelin tahmin başarısı 0'dan %66'ya (F1 Score) sıçradı. Modelin "%30 ihtimal veriyorsan o sınıftır de" mantığı başarıyla çalıştı.

**E. Hata Analizi (Error Analysis)**
- Optimize edilmiş eşik değeri, hiç görülmemiş Test Seti üzerinde çalıştırıldı.
- Özellike *Machine Learning* sınıfında modelin neden yanıldığı, Gerçek Etiket (True Labels) vs Tahmin Edilen Etiket (Predicted Labels) olasılıkları manuel olarak ekrana basılarak derinlemesine analiz edildi.

---

## 🛠 Kullanılan Teknolojiler
- **Veri Toplama:** `requests`, `xml.etree.ElementTree` (ArXiv API)
- **Doğal Dil İşleme (NLP):** `nltk`, `re`, `sklearn.feature_extraction.text`
- **Veri İşleme ve Matematik:** `json`, `collections`, `numpy`
- **Makine Öğrenmesi (ML):** `scikit-learn`, `iterative-stratification`
- **Model:** `Logistic Regression` (One-vs-Rest stratejisi ile)

---

## 🚀 Sonraki Adımlar (Gelecek Planları)
1. **Derin Öğrenme ve Word Embeddings:** TF-IDF matrisleri yerine kelimelerin anlamsal uzayını yakalayan *Word2Vec*, *FastText* veya *GloVe* kullanılması.
2. **Transformers Mimarisi:** Makine Öğrenmesi tabanlı Logistic Regresyon modelinin yerini, kendi cihazımızda fine-tune edilmiş bir açık kaynaklı *BERT* veya *RoBERTa* modeline bırakması.
3. **Web API & Arayüz:** Geliştirilen modelin *FastAPI* ile canlıya alınıp *Streamlit* ile kullanıcı dostu bir arayüzde (ResearchMindAI as a Service) sunulması.
