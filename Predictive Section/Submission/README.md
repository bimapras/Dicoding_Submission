# Proyek Pertama Mobile Price Classification

Disusun Oleh : Bima Prastyaji

ini adalah proyek pertama predictive analytics submission dicoding. Proyek ini membangun model *multiclass classification* yang dapat memprediksi kategori harga jual suatu handphone.

## Domain Proyek

### Latar Belakang

![cover](https://s.kaskus.id/images/2022/01/12/10351509_20220112023258.jpg)

Gambar 1. cover

Pada saat ini, pasar handphone sudah sangat berkembang. Banyak produsen ponsel atau handphone yang menawarkan berbagai fitur dan spesifikasi yang beragam dengan harga yang berbeda. Dalam menentukan harga suatu handphone pastinya terdapat suatu faktor yang mempengaruhi harga jual handphone.

Oleh Karena itu, untuk mempermudah dalam menentukan harga suatu handphone maka dibuatlah penelitian menggunakan model *machine learning* yang dapat mengklasifikasikan handphone ke dalam beberapa kategori seperti 0(Low cost), 1(medium cost), 2(high cost) and 3(very high cost). Hasil prediksi ini nantinya dapat dijadikan standart dalam menentukan harga jual handphone pada suatu perusahaan.

## Business Understanding
### Problem Statements
- Fitur apa saja yang mempengaruhi kategori handphone
- Bagaimana cara memproses data agar hasil prediksinya baik
- Apakah ada perbedaan signifikan pada spesifikasi antar handphone dengan kategori handphone

### Goals
- Mengetahui fitur yang paling berpengaruh terhadap kategori handphone
- Melakukan pembersihan data agar dapat dilatih oleh model
- Membuat model *machine learning* yang dapat mengklasifikasikan kategori handphone berdasarkan spesifikasi tertentu

### Solution Statement
- Menganalisis data dengan melakukan *EDA*, *univariate analysis*, *multivariate analysis*, dan visualisasi data. Analisi dilakukan agar dapat memahami data lebih dalam seperti mengetahui korelasi antar fitur dan mendeteksi outlier
- Melakukan normalisasi data menggunakan MinMaxScaler
- Menerapkan grid search untuk menentukan parameter model dan membuat model klasifikasi dengan algoritma *K-Nearest Neighbour*, *Random Forest*, dan *AdaBoost*.

## Data Understanding
Dataset yang digunakan pada proyek ini merupakan data spesifikasi handphone dari **Kaggle**. Dataset dapat di download pada tautan berikut [Resource Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data)

**Informasi Dataset antara lain** :
- Dataset memiliki format CSV
- Terdapat 2 dataset yang digunakan yaitu train.csv dan test.csv
- Tidak ada missing value (Nan)
- Dataset train terdiri dari 2000 sample dengan 19 fitur tipe int64 dan 2 fitur float64
- Dataset test terdiri dari 1000 sample dengan 19 fitur tipe int64 dan 2 fitur float64

**Informasi fitur** :
- Categorical fitur :
  - **blue**: Memiliki bluetooth atau tidak
  - **four_g**: Support 4G atau tidak
  - **three_g**: Support 3G atau tidak
  - **touch_screen**: Memiliki layar sentuh atau tidak
  - **wifi**: Support menggunakan Wifi atau tidak
  - **dual_sim**: Support dual SIM card atau tidak.
  - **price_range**:  Target atau label data dengan value 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)

- Numerical fitur :
  - **battery_power**: Total energi yang dapat disimpan baterai dalam satu waktu, diukur dalam mAh.
  - **clock_speed**: Kecepatan prosesor ponsel, diukur dalam GHz (gigahertz).
  - **fc**: Megapixel untuk kamera depan.
  - **int_memory**: Kapasitas memori internal, diukur dalam GB (gigabyte).
  - **m_dep**: Ketebalan handphone.
  - **mobile_wt**: Berat handphone.
  - **n_cores**: Jumlah core processor handphone.
  - **pc**: Megapixel untuk kamera utama.
  - **px_height**: Tinggi resolusi layar ponsel, diukur dalam pixel.
  - **px_width**: Lebar resolusi layar ponsel, diukur dalam pixel.
  - **ram**: kapasitas RAM ponsel, diukur dalam GB (gigabyte).
  - **sc_h**: Tinggi layar ponsel, diukur dalam cm.
  - **sc_w**: Lebar layar ponsel, diukur dalam cm.
  - **talk_time**: lama waktu baterai ketika digunakan, dalam waktu 1 kali pengisian daya.
  - **id**: Nomor unique data

### EDA
Exploratory Data Analysis (EDA) adalah proses analisis awal data yang bertujuan untuk memahami karakteristik, struktur, dan komponen penting dari dataset sebelum melakukan analisis statistik atau pemodelan prediktif lebih lanjut. Tujuan dari *EDA* sendiri adalah untuk memahami data, mencari anomali seperti error values, mengidentifikasi pola atau tren dalam data, dan melihat hubungan antar fitur. 

Berikut tahapan - tahapan *EDA* yang dilakukan pada train.csv:
#### Error Value
Pada tahapan ini untuk mencari error value, fitur dibagi terlebih dahulu menjadi categorical dan numerical, kemudian gunakan fungsi *describe()* untuk menganalisis data pada fitur numerical.

| |battery_power|clock_speed|fc|int_memory|m_dep|mobile_wt|n_cores|pc|px_height|px_width|ram|sc_h|sc_w|talk_time|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|count|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|2000.000000|
|mean|1238.518500|1.522250|4.309500|32.046500|0.501750|140.249000|4.520500|9.916500|645.108000|1251.515500|2124.213000|12.306500|5.767000|	11.011000|
|std|439.418206|0.816004|4.341444|18.145715|0.288416|35.399655|2.287837|6.064315|443.780811|432.199447|1084.732044|4.213245|4.356398|5.463955|
|min|501.000000|0.500000|0.000000|2.000000|0.100000|80.000000|1.000000|0.000000|0.000000|500.000000|256.000000|5.000000|0.000000|2.000000|
|25%|851.750000|0.700000|1.000000|16.000000|0.200000|109.000000|3.000000|5.000000|282.750000|874.750000|1207.500000|9.000000|2.000000|6.000000|
|50%|1226.000000|1.500000|3.000000|32.000000|0.500000|141.000000|4.000000	|10.000000|564.000000|1247.000000|2146.500000|12.000000|5.000000|11.000000|
|75%|1615.250000|2.200000|7.000000|48.000000|0.800000|170.000000|7.000000|15.000000|947.250000|1633.000000|3064.500000|16.000000|9.000000|16.000000|
|max|1998.000000|3.000000|19.000000|64.000000|1.000000|200.000000|8.000000|20.000000|1960.000000|1998.000000|3998.000000|19.000000|18.000000|20.000000|

Tabel 1. deskripsi dataset

Dari tabel 1 terlihat terdapat beberapa fitur yang memiliki nilai minimum 0. Dimana nilai tersebut tidak wajar, sehingga dibutuhkan analisis yang lebih dalam lagi. Fitur yang memiliki nilai error antara lain fc, pc, px_height, sc_w. 

![missing_values](https://github.com/bimapras/Dicoding_Submission/assets/91962289/3c91e3e8-74ad-438c-b5c7-f15d4a45dd5f)

Gambar 2. jumlah error value

Berdasarkan gambar 2 dengan data train yang terdiri dari 2000 sampel dan jumlah error value pada fitur ‘pc’ sebanyak 474, dapat disimpulkan bahwa ini merupakan jumlah yang signifikan. Oleh karena itu, untuk mengatasinya, akan dilakukan penggantian dengan nilai rata-rata (mean).

![distribusi_target](https://github.com/bimapras/Dicoding_Submission/assets/91962289/cb1a1a16-e689-429f-a6fd-952e9f882b3f)

Gambar 3. distribusi target

Pada gambar 3 distribusi target menampilkan data sudah balance maka tidak perlu melakukan oversampling ataupun undersampling.

#### Removing Outliers
Outlier adalah nilai yang berbeda secara signifikan dari nilai-nilai lain dalam dataset. Menghilangkan outlier dapat membantu meningkatkan kualitas analisis dan model prediktif, untuk menghilangkan outlier teknik yang digunakan adalah visualisasi library [Seaborn](https://seaborn.pydata.org/generated/seaborn.boxplot.html) dan dilanjutkan dengan implementasi teknik *IQR* pada data train. Hasil dari implementasi teknik *IQR* dapat dilihat pada tabel 2.

*IQR* atau Jangkauan Interkuartil adalah ukuran yang digunakan dalam statistik untuk mengukur sebaran data dalam suatu himpunan. *IQR* mengukur jangkauan dari kuartil pertama *(Q1)* hingga kuartil ketiga *(Q3)* dalam data.

![outliers](https://github.com/bimapras/Dicoding_Submission/assets/91962289/06ac3404-53e0-42b0-a734-cd0fa0b38887)

Gambar 4. visualisasi outlier tiap fitur

|  Sebelum  |  Sesudah  |
|-----------|-----------|
|(2000, 21) | (1913, 21)|

Tabel 2. output data setelah melakukan *IQR*

#### Univariate Analysis
*Univariate Analysis* adalah menganalisis setiap fitur secara terpisah.

- Distribusi pada categorical fitur
![univariate_analysis](https://github.com/bimapras/Dicoding_Submission/assets/91962289/7f2d6cb3-d14e-421a-bf2d-ef664594320b)

Gambar 5. visualisasi distribusi data categorical fitur

Pada gambar 5 menunjukkan tiap-tiap fitur memiliki distribusi data yang seimbang atau hampir seimbang kecuali pada fitur 'three_g', sehingga dapat disimpulkan bahwa hanya sebagian kecil handphone yang     tidak support 3G.

- Distribusi pada numerical fitur
![histogram](https://github.com/bimapras/Dicoding_Submission/assets/91962289/926eaf54-b57a-4b24-ae30-936236c65510)

Gambar 6. visualisasi distribusi data pada numerical fitur

Berikut analisis dari gambar 6 :
- Sebagian besar handphone memilki daya baterai sekitar 1500 mAh.
- Kebanyakan handphone memiliki clock speed yang rendah.
- Banyak handphone memiliki kamera depan dengan megapiksel rendah.
- Distribusi kapasitas memori internal cukup merata.
- Distribusi kamera utama cukup merata, tetapi ada lonjakan pada beberapa megapiksel tertentu.
- Resolusi layar handphone bervariasi, mulai dari tinggi dan lebarnya.
- Distribusi ram cukup merata, walaupun terdapat lonjakan pada beberapa kapasitas ram.
- Lama waktu baterai ketika digunakan cukup bervariasi.
- Tinggi dan lebar handphone cukup beragam.

#### Multivariate Analysis
*Multivariate Analysis* menunjukkan hubungan antara dua atau lebih variabel pada data, salah satu cara untuk melihat hubungan antar fitur pada proyek ini adalah correlation matrix. Untuk lebih jelasnya dapat dilihat pada gambar 7.

![correlation_matrix](https://github.com/bimapras/Dicoding_Submission/assets/91962289/3b796d55-b7dc-4316-a362-5bb5c89a848e)
Gambar 7. matrix correlation

Dari correlation matrix diatas terdapat hubungan yang kuat pada fitur 'ram' dan 'price_range' dengan nilai korelasi sebesar 0,92.

![visual_correlation](https://github.com/bimapras/Dicoding_Submission/assets/91962289/7f0ba317-d1da-4a24-84ca-3885f9cd9289)

Gambar 8. korelasi fitur ram dan price_range

Pada gambar 8 menunjukkan grafik keatas yang berarti semakin tinggi kategori price_range maka hp tersebut memiliki kapasitas ram yang besar, dan juga sebaliknya. Hp yang memiliki kapasitas ram kecil maka akan masuk ke kategori price_range yang rendah.

## Data Preparation
Pada tahapan ini  melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Terdapat 2 tahapan yang dilakukan pada proyek ini yaitu *Split Data* dan *Normalization*.

- Split Data
  
  Dalam melakukan split data fitur harus dipisahkan menjadi x dan y, dimana y merupakan target atau label data. Untuk split data menggunakan [TrainTestSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari library [Sklearn](https://scikit-learn.org/stable/) dengan pembagian data train 90% dan data test 10%, dapat dilihat pada tabel 3.

  | Fitur |   Train    |   Test    |
  |-------|------------|-----------|
  |   x   | (1721, 20) | (192, 20) |
  |   y   | (1721) | (192) |

  Tabel 3. output *TrainTestSplit*
  
- Normalization
  
  Tujuan normalisasi adalah memastikan bahwa data memiliki distribusi yang baik, sehingga dapat meningkatkan kinerja model dan hasil prediksi. Salah satu teknik yang digunakan pada proyek ini adalah [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) dari library [Sklearn](https://scikit-learn.org/stable/). *MinMaxScaler* mengubah semua nilai fitur pada dataset dengan nilai minimum 0 dan nilai maksimum 1 pada tiap fitur.

  notes : lakukan normalisasi hanya pada data numerical
## Modelling
Tahapan ini merupakan proses pembuatan model *machine learning* dengan menerapkan algoritma tertentu pada data yang telah dipersiapkan sebelumnya. Model akan dilatih menggunakan data train untuk mempelajari pola dan hubungan antara fitur dengan target (label) yang akan diprediksi.
### Algoritma
Algoritma yang digunakan untuk membangun model *machine learning* pada proyek ini yaitu *K-Nearest Neighbour*, *Random Forest*, dan *AdaBoost*.

#### K-Nearest Neighbour
*K-Nearest Neighbour* bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dengan inputan x_train dan y_train untuk melatih model. 

  - Kelebihan:
    - Sederhana dan mudah diimplementasikan.
    - Tidak memerlukan asumsi tentang distribusi data.
    - Cocok untuk masalah klasifikasi dengan data berdimensi tinggi.
  - Kekurangan:
    - Sensitif terhadap skala data dan jumlah tetangga (nilai k).
    - Memerlukan perhitungan jarak untuk setiap prediksi.

Parameter yang digunakan yaitu :
- `n_neighbors`   = Jumlah k tetangga tedekat.
  
#### Random Forest
*RandomForest* merupakan salah satu model *machine learning* yang termasuk ke dalam kategori ensemble (group) learning. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Proyek ini menggunakan [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dengan inputan x_train dan y_train untuk melatih model.

  - Kelebihan :
    - Mengurangi overfitting dengan menggabungkan banyak pohon keputusan.
    - Stabil dan konsisten dalam performa.
    - Cocok untuk data berdimensi tinggi dan kategori yang tidak seimbang.
  - Kekurangan :
    - Sulit untuk diinterpretasi karena kompleksitas model.
    - Memerlukan lebih banyak sumber daya komputasi.

Parameter yang digunakan yaitu :
- `n_estimators`  = Jumlah pohon keputusan (estimator) yang akan digunakan dalam Random Forest.
- `max_depth`     = Kedalaman maksimum dari setiap pohon keputusan dalam Random Forest.
- `random_state`  = Menentukan seed untuk generator angka acak yang digunakan dalam algoritma.
  
#### AdaBoost
*Adaboost* bekerja dengan cara membangun banyak model klasifikasi lemah dan menggabungkan hasil prediksi dari setiap model untuk memprediksi label kelas. Proyek ini menggunakan [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dengan inputan x_train dan y_train untuk melatih model.

  - Kelebihan:
    - Meningkatkan performa model dengan menggabungkan beberapa weak learner.
    - Tidak memerlukan tuning parameter yang rumit.
    - Cocok untuk masalah klasifikasi dan regresi.
  - Kekurangan:
    - Rentan terhadap noise dan outlier.
    - Memerlukan lebih banyak iterasi untuk konvergensi.

Parameter yang digunakan yaitu :
- `learning_rate` = Menentukan seberapa besar bobot yang diberikan pada setiap model klasifikasi lemah.
- `n_estimators`  = Menentukan jumlah model klasifikasi lemah (estimator) yang akan dibangun
- `algorithm`     = Menentukan algoritma yang digunakan untuk membangun model klasifikasi lemah (SAMME atau SAMME.R)
- `random_state`  = Menentukan seed untuk generator angka acak yang digunakan dalam algoritma.
  
### GridSearch
Pada pembuatan model tentunya perlu mencoba beberapa parameter agar hasil dari prediksi maksimal, untuk mempermudahnya proyek ini menggunakan teknik [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) dengan nilai cv = 10. Hasil *GridSearch* dapat dilihat pada tabel 4.

  | model         | best_score |best_params                                                                           |
  |---------------|------------|--------------------------------------------------------------------------------------|
  | KNN           |  0.934642  |{'n_neighbors': 11}                                                                   |
  | AdaBoost      |  0.741260  |{'algorithm': 'SAMME', 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 11} |
  | Random Forest |  0.881329  |{'max_depth': None, 'n_estimators': 75, 'random_state': 55}                           |

Tabel 4. best_param algoritma

Dari hasil GridSearch algoritma yang memiliki score tertinggi yaitu *KNN*, mungkin dapat diasumsikan algoritma *KNN* dengan parameter tersebut sangat baik untuk sebuah model. Namun apakah hasil akurasinya bagus dengan nilai error yang kecil?.

## Evaluation
Metric yang digunakan pada proyek ini antara lain **Score** dan [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) (Mean Squared Error) untuk mengevaluasi model.

|Perhitungan Score|
|------------------|
|(TP + TN) / (TP + TN + FP + FN)|

Tabel 5. rumus score

- TP (True Positive) adalah jumlah data yang diprediksi benar sebagai kelas positif.
- TN (True Negative) adalah jumlah data yang diprediksi benar sebagai kelas negatif.
- FP (False Positive) adalah jumlah data yang diprediksi salah sebagai kelas positif.
- FN (False Negative) adalah jumlah data yang diprediksi salah sebagai kelas negatif.

| |KNN|RandomForest|Boosting|
|-----|---|------------|--------|
|Accuracy|0.411458|0.890625|0.75|

Tabel 6. akurasi algoritma

Tabel 6 menunjukkan bahwa pada algoritma KNN yang sebelumnya mempunyai skor tinggi namun nilai akurasi yang dihasilkan sangat rendah, sedangkan pada algoritma RandomForest dan Bossting menghasilkan nilai akurasi yang sangat baik dan tidak berbeda jauh dengan nilai skor GridSearch.

![mse_rumus](https://github.com/bimapras/Dicoding_Submission/assets/91962289/58e91608-f270-48fc-95dd-c0e70825a1ce)

Gambar 9. rumus mean squared error

Keterangan :
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi

Hasil MSE

![mse_algoritma](https://github.com/bimapras/Dicoding_Submission/assets/91962289/db8edbae-73a6-4384-bc4d-35be4fb63724)

Gambar 10. perbandingan nilai mse tiap algoritma

Pada gambar 10 terlihat model RandomForest memiliki nilai error yang sangat rendah baik pada data train ataupun data test, kemudian pada model AdaBoost nilai error yang dihasilkan juga cukup baik karena perbedaan nilai error pada data train dan test tidak jauh berbeda, sedangkan model KNN memiliki nilai error yang paling besar dibandingkan dengan algoritma AdaBoost atau RandomForest. Dari 3 model tersebut akan dipilih 1 model yang memiliki nilai error kecil dan nilai akurasi yang tinggi. Maka dari itu model yang akan dipilih pada proyek ini adalah RandomForest.

## Kesimpulan
Dapat disimpulkan bahwa algoritma dengan nilai score GridSearch besar belum tentu memiliki nilai akurasi dan nilai error yang optimal. Dapat dilihat hasil evaluasi menunjukkan model *RandomForest* (RF) memiliki kinerja yang sangat bagus daripada algoritma *KNN* atau *AdaBoost* dengan nilai akurasi sebesar 89% dan nilai error pada test sebesar 0.000109.

## Prediksi Dataset Test 
Memprediksi kategori harga jual pada dataset test.csv, berikut hasil prediksinya dalam bentuk csv [Hasil Prediksi](https://github.com/bimapras/Dicoding_Submission/blob/master/Predictive%20Section/Submission/prediction.csv)

## Referensi
[1] Varun Kiran A and Dr. Jebakumar R. (2022). "Prediction of Mobile Phone Price Class using Supervised Machine Learning Techniques",[International Journal of Innovative Science Research and Technology, Volume 7, Issue 1, pp. 248–251](https://ijisrt.com/assets/upload/files/IJISRT22JAN380.pdf)
