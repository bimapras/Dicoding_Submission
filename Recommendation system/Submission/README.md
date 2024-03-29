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
- Membuat model *machine learning* untuk sistem rekomendasi yang dapat memberikan rekomendasi artikel blog pada pengguna.
  
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

  Terdapat 200140 data dengan 2 fitur tipe integer dan 1 fitur tipe float :
  - `blog_id` : Id dari blog
  - `userId` : Id pengguna 
  - `ratings` : Rating yang diberikan oleh pengguna 

- Dataset Medium_Blog_Data.csv
  
  Terdapat 10467 data dengan 2 fitur tipe integer dan 5 fitur tipe object :
  - `blog_id` : Id dari blog
  - `author_id` : Id penulis/pengarang
  - `blog_title` : Judul blog
  - `blog_content` : deskripsi singkat pada blog
  - `blog_link` : link atau url blog
  - `blog_img` : link cover gambar blog
  - `topic` : jenis konten pada suatu blog
  - `scrape_time` : waktu pengambilan data  

- Dataset Author_Data.csv

  Terdapat 6868 data dengan 1 fitur tipe integer dan 1 fitur tipe object :
  - `author_id` : Id penulis/pengarang 
  - `author_name` : nama sang penulis/pengarang 
  
### EDA
*Exploratory Data Analysis* pada proyek dilakukan pada 3 dataset berikut *EDA* pada tiap dataset :
#### Rating
Analisis setiap fitur numerik dataset Blog_Rating.csv menggunakan fungsi *describe*

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

![unik_rating](https://github.com/bimapras/Dicoding_Submission/assets/91962289/c3e90e50-e903-43b1-a3df-01eb5967469c)

Gambar 2. Data unik pada rating

Pada gambar 2 menunjukkan bahwa jumlah blog yang ada hanya 9706 dari 200140 data, dan banyaknya pengguna yang memberikan rating adalah 5001 dari 200140 data. Dapat dikatakan sebagian besar pengguna setidaknya telah membaca dan memberikan rating lebih dari 3 artikel blog.

![top10_user](https://github.com/bimapras/Dicoding_Submission/assets/91962289/15069829-ed7b-4abe-b58a-5c98168da1ab)

Gambar 3. Top 10 pengguna

Dari gambar 3 menunjukkan pengguna yang sering membaca dan memberikan rating adalah pengguna dengan *user* 3619, 3882, 4453, 4012, 4131, 1418, 622, 3742, 1855, dan 455.  

#### Blog
Pada dataset Medium_Blog_Data.csv fitur yang dibutuhkan hanya blog_id, author_id, blog_title, dan topic.

|blog_id|	author_id|	blog_title|	topic|
|-|-|-|-|
|1|	4|	Let’s Dominate The Launchpad Space Again|	ai|
|3|	4|	Let’s Dominate The Launchpad Space Again|	ai|
|4|	7|	Using ChatGPT for User Research|	ai|
|5|	8|	The Automated Stable-Diffusion Checkpoint Merger, autoMBW.|	ai|
|6|	9|	The Art of Lazy Creativity: My Experience Co-Writing a Monty Python Style Sketch with AI|	ai|

Table 3. Tampilan 5 data teratas pada blog

Terlihat pada table 3 ternyata author dengan author_id 4 memiliki 2 artikel blog dengan title yang sama namun berbeda blog_id.

![uniq_blog](https://github.com/bimapras/Dicoding_Submission/assets/91962289/58da5cc4-75ef-4a9d-ba75-f3819c9d8453)

Gambar 4. Data unik pada blog

Pada gambar 4 menunjukkan terdapat 10467 jumlah blog, 6824 author, 23 jenis konten, dan 10466 title. Perhatikan pada jumlah blog dan jumlah title, terdapat perbedaan yang kecil pada jumlahnya, namun hal tersebut tidak wajar, karena tiap blog pastinya memiliki judulnya masing-masing.

![distribusi_topic](https://github.com/bimapras/Dicoding_Submission/assets/91962289/57a95c4d-c716-4b4a-98f5-3072e9e7c564)

Gambar 5. Distribusi Topic

Dari gambar 5 terlihat bahwa grafik distribusi menunjukkan mayoritas blog yang ada merupakan artikel blog dengan topik ai, blockchain, cybersecurity, web-development, data-analysis, dan cloud-computing.
#### Author
![unik_author](https://github.com/bimapras/Dicoding_Submission/assets/91962289/1050a0aa-e1ee-48cc-b0d7-003a9360d28a)

Gambar 6. Data unik pada author

Pada gambar 6 terlihat jumlah author_id tidak seimbang dengan jumlah author_name, kemungkinan hal ini terjadi karena terdapat author yang memiliki lebih dari 1 author id.

![dup_name](https://github.com/bimapras/Dicoding_Submission/assets/91962289/97961f48-5fa5-4529-b205-97ec4b872341)

Gambar 7. Double blog_id

Dari gambar 7 dapat dilihat bahwa author dengan nama Kompetify.ai memili 2 id, maka dari itu perlu dilihat author tersebut memiliki author_id apa saja, author_id *Kompetify.ai* dapat dlihat pada tabel 4.

|author_id|author_name|
|-|-|
|84|Kompetify.ai|
|95|Kompetify.ai|

Table 4. author_id Kompetify.ai
### Preprocessing
Pada proyek ini terdapat beberapa teknik yang digunakan pada preprocessing, antara lain :
- **Create Helper Function**
  
  Membuat fungsi untuk menghapus tanda baca, emoji dan simbol menggunakan library [Regular Expression](https://docs.python.org/3/library/re.html)
  ```
  def remove_emoji_and_symbols(text):
    text_no_emoji = re.sub(r'[^\w\s\d]+', '', text)
    return text_no_emoji
  ```
  Fungsi ini dibuat untuk memudahkan dalam proses penghapusan tanda baca, emoji, dan simbol pada data.
  
- **Remove punctuation, emoji, dan symbol**
  
  Menghapus tanda baca, emoji, dan symbol pada data blog dan author. Proses ini dilakukan untuk mengurangi kompleksitas dan mempercepat proses pada saat data dimasukkan ke dalam model.
  
- **Merge author and blog**

  Melakukan penggabungan data author dan blog untuk dianalisis, seperti mencari *missing value* pada fitur tertentu dan melihat jumlah data.

  |Shape|
  |-|
  |(10511, 5)|

  Table 5. Dimensi data setelah digabungkan

  Dapat dilihat pada table 5 setelah digabungkan jumlah data yang dimiliki adalah 10511.
  
  ![missing_value](https://github.com/bimapras/Dicoding_Submission/assets/91962289/79aa7c7f-376c-49ac-b0c5-0ff0c83f8e1e)

  Gambar 8. Missing value
  
  Dapat dilihat pada gambar 8 terdapat 44 *missing value* pada fitur blog_id, blog_title, topic setelah digabungkan. Sehingga dapat dilakukan penghapusan pada data yang memiliki *missing value*.
- **Remove Missing Value**

  Menghapus data yang memiliki *missing value* menggunakan *dropna()*, proses ini dilakukan agar tidak terjadi bias pada data nantinya. Kemudian setelah *missing value* dihapus jumlah datanya dapat dilihat pada table 6.

  |Shape|
  |-|
  |(10467, 5)|

  Table 6. Dimensi data bersih
- **Create Dataframe blog_rate**

  Membuat Dataframe baru dengan menggabungkan data bersih dan rating. Penggabungan ini dilakukan untuk mendapatkan informasi tentang preferensi pengguna, dengan mendapatkan informasi ini model *machine learning* akan jauh lebih mudah merekomendasikan artikel blog yang relevan dengan minat pengguna.
  
![like_topic](https://github.com/bimapras/Dicoding_Submission/assets/91962289/f1be7415-8585-4774-a035-e2f4dada0559)
  Gambar 9. Distribusi Top Topic

  Dari hasil grafik pada gambar 9 dapat disimpulkan bahwa artikel blog dengan topic flutter, android, app-development, SoftwareDevelopment, dan web-development lebih banyak diminati oleh sebagian besar pengguna.
## Data Preparation
Dalam menyiapkan data agar dapat digunakan oleh model terdapat beberapa tahapan, yaitu :

- **Sorting Data**

  Melakukan pengurutan data berdasarkan blog_id agar mudah dalam melakukan penghapusan duplikat data.

- **Remove Duplicate Data**

  Melakukan penghapusan data duplikat agar tidak terjadi bias pada data.

- **Selection Feature & Convert Data to List**

  Memilih fitur yang akan digunakan pada model dan mengubah bentuk datanya menjadi list, hal ini dilakukan agar pemrosesan pada model lebih efisien.

## Modelling
Pada proyek ini proses modeling menggunakan algoritma *Neural Network* dan *Cosine Similarity*. *Neural Network* akan digunakan pada sistem rekomendasi dengan model *Collaborative Filtering*, sedangkan *Cosine Similarity* akan digunakan pada model *Content Based Filtering*.
### Content Based Filtering
Pada pembuatan model ini langkah pertama yang dilakukan adalah mengubah data pada fitur topic menjadi representasi vektor berdasarkan frekuensi kata dan bobotnya menggunakan [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), kemudian melakukan fit dan transformasi ke dalam bentuk matriks. Langkah yang kedua adalah menghitung kesamaan antara dua vektor dengan menggunakan [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) pada matriks TF-IDF, berikut rumus untuk menghitung kesamaan vektor :

![rumus](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:784efd3d2ba47d47153b050526150ba920210910171725.jpeg)

Gambar 10. Rumus Cosine Similarity

|Title|Building Your Developer Portal with Backstage a Comprehensive Tutorial|Is Offline reinforcement learning the future part2Machine Learning|CUDOS POWERING THE METAVERSE|Mobile Movement Dialect Rugged By Apple Solana Mobile Creates History Again METAVERTU Builds Luxury Web3 Phone|Socket Programming in Go Write a simple TCP clientserver|
|-|-|-|-|-|-|
|Building Your Developer Portal with Backstage a Comprehensive Tutorial|0.0|0.0|0.0|1.0|0.0|
|Is Offline reinforcement learning the future part2Machine Learning|0.0|0.0|0.0|0.0|0.0|
|CUDOS POWERING THE METAVERSE|0.0|0.0|0.0|0.0|0.0|
|Mobile Movement Dialect Rugged By Apple Solana Mobile Creates History Again METAVERTU Builds Luxury Web3 Phone|0.0|0.0|0.0|0.0|0.0|
|Socket Programming in Go Write a simple TCP clientserver|0.0|0.0|0.0|0.0|0.0|

Table 7. Hasil Cosine Similarity

*Cosine Similarity* menghitung kesamaan antar item dalam rentang 0 dan 1, dimana item yang memiliki nilai mendekati 1 memiliki kesamaan yang tinggi. Dari table 7 menunjukkan artikel blog dengan title *Mobile Movement Dialect Rugged By Apple Solana Mobile Creates History Again METAVERTU Builds Luxury Web3 Phone* memiliki kesamaan jenis konten dengan artikel blog dengan title *Why Web3 is the Next Big Thing*.

Berikut kelebihan dan kekurangan dari *Content Based Filtering* :

- Kelebihan
  - Tidak memerlukan data informasi pengguna.
  - Dapat memberikan item yang sesuai dengan preferensi pengguna

- Kekurangan
  - Memerlukan banyak data item.
  - Tidak dapat menentukan profil dari pengguna baru.
 
Hasil 10 rekomendasi artikel blog menggunakan *Content Based Filtering* dapat dilihat pada table 9, artikel blog yang digunakan untuk memberikan rekomendasi terdapat pada table 8.

|blog_id|title|topic|
|-|-|-|
|9145|sqlite from a web page|webdevelopment|

Tabel 8. Data uji coba

|title|topic|
|-|-|
|The Pros and Cons of Using OpenSource Software in Your Business|webdevelopment|
|Solution 5 Agnostics From hodling to thriving Web3 communities unlock the power of their tokens|webdevelopment|
|Mastering Reducers A Comprehensive Guide for Everyone|webdevelopment|
|Why I Wouldnt Hire Cool Programmers if I Launched a Startup|webdevelopment|
|React Application Gets Depressed and Finds Therapy in Nextjs and Jest|webdevelopment|
|10 RxJS operators which I use daily as an Angular developer|webdevelopment|
|Stop Using Docker|webdevelopment|
|Basics of Working with APIs|webdevelopment|
|Redux Simplified A Beginners Guide to the Core Concepts of Redux|webdevelopment|
|How im going to build OXINION Finance with ChatGPT|webdevelopment|

Tabel 9. Hasil rekomendasi content based filtering
### Collaborative Filtering
Pada model *Collaborative Filtering* informasi milik pengguna sangat dibutuhkan agar model dapat bekerja dengan baik, sehingga langkah pertama dalam pembuatan model ini adalah membuat dataframe profile yang berisikan informasi dari pengguna seperti userId, blog_id, dan ratings. Kemudian mengubah blog_id dan blog_id menjadi list, setelah bentuk datanya menjadi list lakukan mapping ke dataframe profile. Langkah kedua yaitu melakukan pembagian data menjadi train 80% dan validasi 20% menggunakan [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

Langkah yang terakhir adalah membuat model menggunakan *Neural Network*, dimana model ini menggunakan *layer embedding* dan *layer embedding* bias pada data userId dan blog_id. Hasil dari model ini merupakan perkalian *dot product* antara *embedding* userId dengan blog_id.

Berikut kelebihan dan kekurangan dari model *Collaborative Filtering* :

- Kelebihan
  - Dapat membuat rekomendasi tanpa harus selalu menggunakan data yang lengkap.
  - Unggul dari segi kecepatan dan *skalabilitas*.
    
- Kekurangan
  - Memerlukan data informasi pengguna
  - Memungkinkan hasil rekomendasi tidak sesuai dengan minat pengguna.
    
Berikut hasil rekomendasi untuk user 3036, dapat dilihat pada gambar 11.

![result collaborative](https://github.com/bimapras/Dicoding_Submission/assets/91962289/7be69af4-e83f-4d71-9761-c33ea13d428a)

Gambar 11. Hasil rekomendasi *collaborative filtering*
## Evaluation
Pada proyek ini menerapkan 2 metric evaluasi yaitu [RootMeanSquaredError](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) (RMSE) dan *Precision*

### Evaluasi Content Based Filtering
Pada evaluasi content based filtering menggunakan metrik precision content based filtering untuk menghitung precision model sistem yang telah dibuat sebelumnya. Rumus perhitungan precision dapat dilihat pada gambar 12.

![precision](https://miro.medium.com/v2/resize:fit:640/format:webp/1*MjZVU83RyTYp6gR8u4UZAQ.png)

Gambar 12. Rumus precision

Berikut analisis untuk menghitung precision hasil rekomendasi *Content Based Filtering* :
||title|topic|
|-|-|-|
|sample|sqlite from a web page|webdevelopment|
|Top-K|The Pros and Cons of Using OpenSource Software in Your Business|webdevelopment|
|Top-K|Solution 5 Agnostics From hodling to thriving Web3 communities unlock the power of their tokens|webdevelopment|
|Top-K|Mastering Reducers A Comprehensive Guide for Everyone|webdevelopment|
|Top-K|Why I Wouldnt Hire Cool Programmers if I Launched a Startup|webdevelopment|
|Top-K|React Application Gets Depressed and Finds Therapy in Nextjs and Jest|webdevelopment|
|Top-K|10 RxJS operators which I use daily as an Angular developer|webdevelopment|
|Top-K|Stop Using Docker|webdevelopment|
|Top-K|Basics of Working with APIs|webdevelopment|
|Top-K|Redux Simplified A Beginners Guide to the Core Concepts of Redux|webdevelopment|
|Top-K|How im going to build OXINION Finance with ChatGPT|webdevelopment|

Tabel 10. Analisis precision

Dari tabel 10 dapat dilihat bahwa sample artikel blog memiliki topic atau jenis konten yaitu web-development, dan hasil rekomendasi merupakan Top-K memiliki topic yang sama semua yaitu web-development. Artinya nilai precisionnya sebesar 100% (10/10).

### Evaluasi Collaborative Filtering
Pada *Collaborative Filtering* metric yang digunakan adalah *RMSE*, dimana semakin rendah nilai *root mean square error* yang dihasilkan semakin baik model tersebut. Berikut rumus perhitungan *RMSE* pada gambar 13.

![RMSE](https://community.qlik.com/legacyfs/online/128958_2016-06-23%2013_45_36-Root%20Mean%20Squared%20Error%20_%20Kaggle.png)

Gambar 13. Rumus RMSE

![rmse](https://github.com/bimapras/Dicoding_Submission/assets/91962289/deb8c4aa-ec00-4b19-8623-d838a74284be)

Gambar 14. Visualiasasi epoch rmse

Dapat dilihat pada gambar 14, data train mendapatkan nilai error sebesar 0.375 dan data validasi mendapatkan nilai error 0.399, nilai tersebut dihasilkan dengan melakukan training sebnayak 13 kali untuk menghindari overfitting. Walaupun nilai error yang didapat cukup besar, namun nilai tersebut sudah cukup bagus untuk sistem rekomendasi.

# Referensi

[1] Samir Brahim B. (2023). A Collaborative Filtering Recommendation Framework Utilizing Social Networks. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827023000488)

[2] Rianti, A., Majid, N. W. A., & Fauzi, A. (2022). Machine Learning Journal Article Recommendation System using Content Based Filtering. [*Jurnal Teknologi Informasi dan Ilmu Komputer, Volume 22, Number 1, January 2024: 1 – 10*](https://www.researchgate.net/publication/378695877_Machine_Learning_Journal_Article_Recommendation_System_using_Content_based_Filtering)

