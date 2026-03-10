# 📈 Prediksi Penjualan Voucher Game Online  
## Menggunakan Metode XGBoost Berbasis Residual Prophet

## 🧠 Project Overview
Repository ini berisi implementasi lengkap dari skripsi S1 Informatika berjudul:

**“Prediksi Penjualan Voucher Game Online Menggunakan Metode XGBoost Berbasis Residual Prophet”**

Penelitian ini mengembangkan pendekatan **hybrid forecasting** dengan mengombinasikan:
- **Prophet** untuk menangkap tren dan pola musiman pada data time series
- **XGBoost** untuk memodelkan hubungan kompleks antar fitur berbasis residual Prophet

- Studi kasus dilakukan menggunakan **data transaksi riil penjualan voucher game online mobile legend.**

---

## 🎯 Research Objectives
- Menganalisis pola penjualan voucher game online berbasis data historis
- Menerapkan metode **XGBoost berbasis residual Prophet**
- Mengoptimalkan performa model melalui **hyperparameter tuning (Random Search)**
- Mengevaluasi performa model menggunakan **MAE, MSE, RMSE, dan MAPE**
- Mengimplementasikan model ke dalam **aplikasi prediksi berbasis web**

---

## 📦 Dataset Description
- **Sumber Data**: Data transaksi Internal PT XYZ
- **Tipe Data Awal**: Data transaksi (transaction-level)
- **Jumlah Data Awal**: ± **57.000 baris transaksi**
- **Periode Data**: Data historis penjualan voucher game online
- **Target Prediksi**: Nilai penjualan harian

### Atribut Utama Dataset Awal
- Tanggal transaksi
- Nama produk
- Harga modal & harga jual
- Jumlah pembelian
- Total pembayaran
- Metode pembayaran
- Kode promo

---

## 🔧 Dataset Preprocessing 

### 🔹 1. Data Awal (±57.000 Transaksi)
Dataset awal masih berbentuk **data transaksi mentah**, di mana:
- Satu baris merepresentasikan **satu transaksi**
- Dalam satu hari bisa terdapat **ratusan transaksi**
- Format ini **belum sesuai** untuk pemodelan time series

---

### 🔹 2. Konversi Format Data
- Kolom tanggal transaksi dikonversi dari `string` ke `datetime`
- Proses ini wajib agar data dapat diproses oleh **Prophet**
- Kesalahan format tanggal dapat menyebabkan kegagalan dalam analisis musiman

---

### 🔹 3. Agregasi Data Harian
Untuk memenuhi kebutuhan model time series, data transaksi diubah menjadi **data agregasi harian**.

**Proses agregasi:**
- Seluruh transaksi pada tanggal yang sama dijumlahkan
- Nilai target yang digunakan:
  - Total nilai penjualan per hari
  - Total jumlah transaksi per hari

📉 **Dampak agregasi:**
- Dari ±57.000 baris transaksi
- Menjadi **512 baris data time series harian**
- Setiap baris merepresentasikan **1 hari**

➡️ Agregasi ini:
- Mengurangi noise transaksi individual
- Memperjelas pola tren dan musiman
- Membuat data kompatibel dengan Prophet

---

### 🔹 4. Feature Engineering
Beberapa fitur tambahan dibangun untuk meningkatkan performa model:

#### a. Fitur Waktu
- Hari ke-n sejak tanggal awal
- Digunakan untuk menangkap tren temporal

#### b. Proporsi Metode Pembayaran
- Metode pembayaran (QRIS, e-wallet, VA, dll) awalnya bersifat kategorikal
- Diubah menjadi **fitur numerik berbasis proporsi harian**
- Contoh:
  - Persentase transaksi QRIS per hari
  - Persentase e-wallet per hari

➡️ Pendekatan ini:
- Mengurangi dimensi data
- Tetap mempertahankan informasi perilaku pelanggan

---

### 🔹 5. Normalisasi Data
- Digunakan **Min-Max Scaling**
- Semua fitur dinormalisasi ke rentang **[0, 1]**
- Bertujuan untuk:
  - Menghindari dominasi fitur berskala besar
  - Meningkatkan stabilitas model XGBoost

---

### 🔹 6. Pembagian Dataset
Dataset hasil preprocessing dibagi menjadi:
- **Data Train**
- **Data Validation**
- **Data Test**

Dua skema pembagian diuji:
- **80 : 10 : 10**
- **70 : 20 : 10**

Tujuan:
- Menguji generalisasi model
- Menghindari overfitting
- Mendukung proses hyperparameter tuning

---

## 🛠️ Tools & Technologies
- **Python**
- **Facebook Prophet**
- **XGBoost**
- **Scikit-learn**
- **Pandas & NumPy**
- **Random Search**
- **Flask**
- **HTML & CSS**

---

## 🔍 Modeling Approach

### 1️⃣ Prophet
- Menangkap tren jangka panjang dan musiman
- Menghasilkan komponen tren dan residual

### 2️⃣ Residual Prophet
- Residual Prophet digunakan sebagai **fitur tambahan**
- Membantu XGBoost mempelajari pola yang tidak tertangkap oleh model statistik

### 3️⃣ XGBoost
- Memodelkan hubungan kompleks antar fitur
- Menggunakan residual Prophet + fitur engineered
- Hyperparameter tuning menggunakan **Random Search**

---

## 📊 Model Evaluation
Evaluasi performa model menggunakan:
- **MAE**
- **RMSE**
- **MAPE**

### 📈 Hasil Terbaik
- **MAE**: **3.61%**
- **RMSE**: **7.97%**
- **MAPE**: **1.77%**

➡️ Hasil ini menunjukkan peningkatan akurasi yang signifikan dibandingkan pendekatan tunggal.

---

## 🌐 Deployment
Model diimplementasikan ke dalam **aplikasi prediksi berbasis web** menggunakan Flask:
- Input data penjualan
- Output hasil prediksi
- Visualisasi prediksi vs data aktual

---

## 🎓 Conclusion
Pendekatan **XGBoost berbasis residual Prophet**:
- Efektif dalam menangani data time series
- Mampu meningkatkan akurasi prediksi penjualan
- Berpotensi digunakan sebagai **decision support system** pada industri digital

---
