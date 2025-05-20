# Klasifikasi Sentimen Review Film Menggunakan Model Deep Learning

# Anggota Kelompok 8

| Nama                             | NRP        | Kelas              |
| -------------------------------- | ---------- | ------------------ |
| Alif Nurrohman                   | 5025231057 | Machine Learning B |
| Alvin Zanua Putra                | 5025231064 | Machine Learning B |
| Christoforus Indra Bagus Pratama | 5025231124 | Machine Learning B |


## 📝 Deskripsi Proyek
Proyek ini bertujuan untuk mengklasifikasikan sentimen review film IMDB menggunakan pendekatan supervised learning dengan membandingkan dua model deep learning yang berbeda. Review film akan diklasifikasikan ke dalam sentimen positif atau negatif berdasarkan konten teksnya.

## 🎯 Tujuan
- Membangun sistem klasifikasi sentimen yang akurat untuk review film
- Membandingkan performa antara model LSTM dengan word embeddings dan model BERT fine-tuning
- Menganalisis faktor-faktor yang mempengaruhi akurasi prediksi sentimen

## 📊 Dataset
- **Sumber**: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Ukuran**: Subset 1000 review (500 positif, 500 negatif)
- **Distribusi Kelas**: Seimbang (50% positif, 50% negatif)
- **Fitur**: Teks review dan label sentimen (positif/negatif)

## 🔍 Metodologi

### Praproses Data
1. Pembersihan teks (menghapus HTML tags, tanda baca, dll.)
2. Normalisasi teks (lowercase, stemming/lemmatization)
3. Tokenisasi dan padding
4. Pembagian dataset (70% training, 15% validation, 15% testing)

### Model yang Diimplementasikan

#### Model 1: LSTM dengan Word Embeddings
```python
model_lstm = Sequential([
    Embedding(vocab_size, 100, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

#### Model 2: BERT Fine-tuning
```python
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
```

### Evaluasi
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Learning Curves
- Feature Importance Analysis

## 📆 Timeline Proyek
| Minggu | Kegiatan |
|--------|----------|
| 1 | Pengumpulan dan praproses data |
| 2 | Implementasi dan pelatihan model LSTM |
| 3 | Implementasi dan fine-tuning model BERT |
| 4 | Evaluasi model dan analisis perbandingan |
| 5 | Penyusunan presentasi dan finalisasi proyek |

## 👥 Pembagian Tugas
| Anggota | Tanggung Jawab |
|---------|----------------|
| Anggota 1 | Praproses data dan implementasi model LSTM |
| Anggota 2 | Implementasi model BERT dan fine-tuning |
| Anggota 3 | Evaluasi model, analisis hasil, dan visualisasi |

## 📊 Struktur Slide Presentasi
1. Pendahuluan
   - Latar belakang analisis sentimen
   - Tujuan dan ruang lingkup proyek
2. Definisi Masalah
   - Klasifikasi biner sentimen review film
3. Dataset
   - Karakteristik dataset IMDB
   - Proses praproses dan visualisasi data
4. Metodologi
   - Arsitektur model LSTM
   - Arsitektur model BERT
   - Strategi pelatihan dan hyperparameter tuning
5. Hasil dan Analisis
   - Perbandingan metrik performa
   - Visualisasi hasil
   - Analisis kasus kesalahan klasifikasi
6. Kesimpulan
   - Temuan utama dan implikasi
   - Potensi pengembangan di masa depan
7. Pembagian Tugas & Timeline

## 🔧 Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama
- **TensorFlow/Keras**: Framework deep learning
- **Hugging Face Transformers**: Implementasi model BERT
- **NLTK/SpaCy**: Natural Language Processing
- **Matplotlib/Seaborn**: Visualisasi data
- **Scikit-learn**: Evaluasi model

## 🌟 Keunikan Proyek
- Perbandingan model sekuensial klasik (LSTM) dengan model transformer modern (BERT)
- Analisis interpretabilitas model menggunakan visualisasi attention
- Eksplorasi pola linguistik dalam review film yang menentukan sentimen
- Aplikasi praktis untuk sistem rekomendasi film atau alat analisis feedback penonton

## 📈 Ekspektasi Hasil
- BERT diperkirakan memberikan performa lebih baik untuk klasifikasi sentimen
- LSTM mungkin lebih efisien dalam hal waktu training dan resource
- Analisis akan mengungkap pola linguistik yang menentukan sentimen dalam review film

## 📚 Referensi
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.
- Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. Proceedings of ACL.# FP-Machine-Learning---KEL8
