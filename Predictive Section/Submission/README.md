# Proyek Pertama Mobile Price Classification

Disusun Oleh : Bima Prastyaji

ini adalah proyek pertama predictive analytics submission dicoding. Proyek ini membangun model machine learning yang dapat memprediksi kategori harga jual suatu handphone.


## Domain Proyek

### Latar Belakang

Pada saat ini, pasar handphone sudah sangat berkembang.
Banyak produsen ponsel atau handphone yang menawarkan berbagai fitur dan spesifikasi yang beragam dengan harga yang berbeda. Dalam menentukan harga suatu handphone pastinya terdapat suatu faktor yang mempengaruhi harga jual handphone.

Oleh Karena itu, untuk mempermudah dalam menentukan harga suatu handphone maka dibuatlah penelitian menggunakan model machine learning yang dapat mengklasifikasikan handphone ke dalam beberapa kategori seperti 0(Low cost), 1(medium cost), 2(high cost) and 3(very high cost). Hasil prediksi ini nantinya dapat dijadikan standart dalam menentukan harga jual handphone pada suatu perusahaan.

Referensi : https://ijisrt.com/assets/upload/files/IJISRT22JAN380.pdf

## Business Understanding
### Problem Statements
- fitur apa saja yang mempengaruhi kategori handphone
- Bagaimana cara memproses data agar hasil prediksinya baik
- Apakah ada perbedaan signifikan pada spesifikasi antar handphone dengan kategori handphone

### Goals
- Mengetahui fitur yang paling berpengaruh terhadap kategori handphone
- Melakukan pembersihan data agar dapt dilatih oleh model
- Membuat model machine learning yang dapat mengklasifikasikan kategori handphone berdasarkan spesifikasi tertentu

### Solution Statement
- Menganalisis data dengan melakukan EDA, univariate analysis, multivariate analysis, dan visualisasi data. Dengan melakukan analisis kita dapat memahami data lebih dalam seperti mengetahui korelasi antar fitur dan mendeteksi outlier
- Melakukan normalisasi data menggunakan MinMaxScaler
- Menggunakan grid search untuk menentukan parameter model dan membuat model klasifikasi dengan algoritma K-Nearest Neighbour, Random Forest, dan AdaBoost.

## Data Understanding
Dataset yang digunakan pada proyek ini merupakan data spesifikasi handphone dari **Kaggle**. Dataset dapat di download pada tautan berikut [Resource Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data)

**Informasi Dataset antara lain** :
- Dataset memiliki format CSV
- Dataset memiliki 2 dataset yaitu train.csv dan test.csv
- Tidak ada missing value, namun terdapat error value pada dataset train
- Dataset train memiliki 19 fitur tipe int64 dan 2 fitur float64
- Dataset test memiliki 19 fitur tipe int64 dan 2 fitur float64

**Informasi fitur** :
1. **battery_power**: Total energi yang dapat disimpan baterai dalam satu waktu, diukur dalam mAh.
2. **blue**: Memiliki bluetooth atau tidak. (Categorical {0 : No, 1 ; Yes})
3. **clock_speed**: Kecepatan prosesor ponsel, diukur dalam GHz (gigahertz).
4. **dual_sim**: Support dual SIM card atau tidak. (Categorical {0 : No, 1 ; Yes})
5. **fc**: Mega pixel untuk kamera depan.
6. **four_g**: Support 4G atau tidak. (Categorical {0 : No, 1 ; Yes})
7. **int_memory**: Kapasotas memori internal, diukur dalam GB (gigabyte).
8. **m_dep**: Ketebalan handphone.
9. **mobile_wt**: Berat handphone.
10. **n_cores**: Jumlah core processor Handphone.
11. **pcv**: Mega pixel untuk kamera utama.
12. **px_height**: Tinggi resolusi layar ponsel, diukur dalam pixel.
13. **px_width**: Lebar resolusi layar ponsel, diukur dalam pixel.
14. **ram**: kapasitas RAM ponsel, diukur dalam GB (gigabyte).
15. **sc_h**: Tinggi layar ponsel, diukur dalam cm.
16. **sc_w**: Lebar layar ponsel, diukur dalam cm.
17. **talk_time**: lama waktu yang dapat dicapai baterai, dalam waktu 1 kali pengisian daya.
18. **three_g**: Support 3G atau tidak. (Categorical {0 : No, 1 ; Yes})
19. **touch_screen**: Memiliki layar sentuh atau tidak. (Categorical {0 : No, 1 ; Yes})
20. **wifi**: Support menggunakan Wifi atau tidak (Categorical {0 : No, 1 ; Yes})
21. **price_range**:  Target atau label data yang memiliki nilai 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)
22. **id**: Nomor unique data

### EDA
### Removing Outliers
### Univariate Analysis
### Multivariate Analysis

![correlation image](../Images/correlation_matrix.png)
## Data Preparation
## Modelling
## Evaluation

