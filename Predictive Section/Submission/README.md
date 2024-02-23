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
- Menganalisis data dengan melakukan univariate analysis, multivariate analysis, visualisasi data. Dengan melakukan analisis kita dapat memahami data lebih dalam seperti mengetahui korelasi antar fitur dan mendeteksi outlier
- Melakukan normalisasi data menggunakan MinMaxScaler
- Menggunakan grid search untuk menentukan parameter model dan membuat model klasifikasi dengan algoritma K-Nearest Neighbour, Random Forest, dan AdaBoost

[correlation image](Predictive%20Section/Images/correlation_matrix.png)
