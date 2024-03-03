# Proyek Kedua Blog Recommendation System
Disusun oleh : Bima Prastyaji

Proyek *machine learning* ini membangun suatu sistem yang dapat merekomendasikan beberapa artikel blog pada pengguna.
## Project Overview
### Latar Belakang
<p align="center">
  <img src="https://wolacom.com/images/Article/News-201609/370043/tips-menambahkan-gambar-atau-foto-yang-menarik-pada-artikel-blog.jpg" />
  <p>Gambar 1. Cover</p>
</p>
Saat ini, perkembangan teknologi dan internet semakin maju, memungkinkan banyak pengguna untuk memiliki akses ke berbagai jenis konten, seperti artikel blog. Namun, jenis konten yang tersedia dalam artikel blog sangat bervariasi, sehingga pengguna mungkin kesulitan menemukan artikel yang sesuai dengan minat dan kesukaan mereka.

Dengan dibuatnya proyek *machine learning* ini, diharapkan dapat mempermudah pengguna dalam menemukan artikel blog yang sesuai dengan memanfaatkan data historis pengguna. Proyek ini bertujuan memberikan rekomendasi yang relevan dan sesuai dengan minat pengguna berdasarkan preferensi mereka.

## Bussiness Understanding
### Problem Statements
- Bagaimana cara mengolah data agar dapat digunakan pada model sistem rekomendasi?
- Bagaimana cara membuat model *machine learning* yang dapat memberikan rekomendasi artikel blog yang mungkin disukai pengguna?
  
### Goals
- Melakukan pengolahan data agar dapat digunakan pada model sistem rekomendasi.
- Membuat model machine learning untuk sistem rekomendasi yang dapat memberikan rekomendasi artikel blog pada pengguna.
  
### Solution Statements
Untuk menyelesaikan masalah ini terdapat beberapa cara yang digunakan antara lain :
- Menganalisis data dengan melakukan *Exploratory Data Analysis (EDA)* dan visualisasi data.
- Menggunakan metode pendekatan *Content Based Filtering*.
- Menggunakan metode pendekatan *Collaborative Filtering*.

## Data Understanding & Data Preprocessing
### Deskripsi Dataset
Dataset yang digunakan pada proyek ini merupakan data artikel blog dari medium yang dapat didownload pada link berikut [Resource](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data/data)
|Jenis|Keterangan|
|-|-|
|Owner|Yaksh Shah|
|Dataset|Author_Data.csv, Blog_Rating.csv, Medium_Blog_Data.csv|
|Usability|10.0|
|License|[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)|
|File|Zip|
|Size|3 MB|

Table 1. Informasi Dataset
  
- Dataset Blog_Rating.csv
  Terdapat 200140 data dengan 3 fitur :
  - `blog_id` : Id blog (9706 data unik)
  - `userId` : Id pengguna (5001 data unik)
  - `ratings` : rating yang diberikan oleh pembaca (nilai rating memiliki skala antra 0.5 hingga 5)

- Dataset Medium_Blog_Data.csv
  Terdapat 10467 data dan hanya 4 fitur yang digunakan pada dataset ini, antara lain :
  - `blog_id` : Id blog (10467 data unik)
  - `author_id` : Id penulis/pengarang (6824 data unik)
  - `blog_title` : Judul blog (10466 data unik)
  - `topic` : Jenis konten pada suatu blog (23 data unik)

- Dataset Author_Data.csv
  Terdapat 6868 dengan 2 fitur :
  - `author_id` : Id penulis/pengarang (6868 data unik)
  - `author_name` : nama sang penulis/pengarang (6867 data unik)
  
### EDA
*Exploratory Data Analysis* pada proyek dilakukan pada 3 dataset berikut *EDA* pada tiap dataset :
#### Rating
Analisis setiap fitur numerik dataset rating menggunakan fungsi *describe*

||	blog_id|	userId|	ratings|
|-|-|-|-|
|count|	200140.000000|	200140.000000|	200140.000000|
|mean|	5652.533621|	2545.710158|	3.117468|
|std|	2970.685946|	1446.195478|	1.768113|
|min|	1.000000|	10.000000|	0.500000|
|25%|	2906.000000|	1314.000000|	2.000000|
|50%|	5994.000000|	2552.000000|	3.500000|
|75%|	8510.000000|	3795.000000|	5.000000|
|max|	9755.000000|	5010.000000|	5.000000|

Table 2. Informasi Rating

Dapat dilihat pada table 2 pada fitur rating memiliki nilai minimum 0.5 dan nilai maksimum 5, berarti setiap artikel blog memiliki skala rating antara 0.5 hingga 5.

#### Blog
Melihat informasi fitur dataset blog
#### Author

## Data Preparation

## Modelling

## Evaluation
