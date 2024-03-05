#!/usr/bin/env python
# coding: utf-8

# # Blog Recommendation System
# 
# Nama : Bima Prastyaji

# # Import library

# In[107]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# # Load Dataset
# 
# Dataset dapat diunduh pada kaggle berikut linknya [resource](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data/data)

# In[108]:


# url dataset
rating = pd.read_csv('https://raw.githubusercontent.com/bimapras/Dicoding_Submission/master/Recommendation%20system/Dataset/blogdata/Blog%20Ratings.csv')
blog = pd.read_csv('https://raw.githubusercontent.com/bimapras/Dicoding_Submission/master/Recommendation%20system/Dataset/blogdata/Medium%20Blog%20Data.csv')
author = pd.read_csv('https://raw.githubusercontent.com/bimapras/Dicoding_Submission/master/Recommendation%20system/Dataset/blogdata/Author%20Data.csv')

print('Jumlah Blog :',len(blog['blog_id'].unique()))
print('Jumlah Author :', len(author['author_id'].unique()))
print('Jumlah Rating yang diberikan user :', len(rating['userId'].unique()))


# # EDA
# 
# Proyek ini menggunakan 3 dataset yaitu Blog_Ratings.csv, Medium_Blog_Data.csv, dan Author_Data.csv
# 
# **Deskripsi Dataset :**
# 
# - Dataset Blog_Ratings.csv :
#   - blog_id : Id blog
#   - userId : Id pengguna
#   - ratings : rating yang diberikan oleh pembaca
# 
# - Dataset Medium_Blog_data.csv :
#   - blog_id : Id blog
#   - author_id : Id penulis/pengarang
#   - blog_title : Judul blog
#   - blog_content : deskripsi singkat pada blog
#   - blog_link : link atau url blog
#   - blog_img : link cover gambar blog
#   - topic : jenis konten pada suatu blog
#   - scrape_time : wkatu pengambilan data
# 
# - Dataset Author_Data.csv :
#   - author_id : Id penulis/pengarang
#   - author_name : nama sang penulis/pengarang

# ### Rating

# In[109]:


# melihat data rating
rating.info()


# In[110]:


rating.describe()


# Dari output rating.describe(), diketahui nilai minimum rating adaah 0.5 dan nilai maksimum rating adalah 5. Berarti setiap blog memiliki skala rating antara 0.5 hingga 5.

# In[111]:


# melihat jumlah blog, jumlah pembaca yang memberikan rating, dan banyak data rating
print('Jumlah blog :', len(rating.blog_id.unique()))
print('Jumlah user :', len(rating.userId.unique()))
print('Banyak data rating :', len(rating))


# In[112]:


# menghitung tiap user telah memberikan penilaian
count = rating.userId.value_counts()
top_user = pd.DataFrame({'count_rating': count})

top_user[:10].plot(kind = 'barh')
plt.xlabel('Total Penilaian')
plt.ylabel('userId')
plt.title('Distribusi pemberian rating dari 10 user terbanyak')
plt.show()


# Dari grafik diatas menunjukkan bahwa user yang memberikan rating paling banyak adalah user 3619 dengan penilaian lebih dari 350 blog

# ### Blog

# In[113]:


# melihat data blog
blog.info()


# Pada dataset blog fitur yang digunakan hanya blog_id, author_id, blog_title, dan topic

# In[114]:


# mengambil fitur yang dibutuhkan
blog = blog[['blog_id', 'author_id', 'blog_title', 'topic']]
blog.head()


# In[115]:


# melihat data unique pada blog
print('Banyak blog :', len(blog.blog_id.unique()))
print('Banyak author :', len(blog.author_id.unique()))
print('Banyak jenis konten blog :', len(blog.topic.unique()))
print('Banyak title :', len(blog.blog_title.unique()))

# mencari data duplikat
print('Jumlah data duplikat :',blog.duplicated().sum())


# Dapat dilihat dari output diatas terdapat perbedaan pada jumlah blog yang ada dengan jumlah judul blog.

# In[116]:


# ditribusi topic blog
count = blog.topic.value_counts()
jenis_topic = pd.DataFrame({'count_topic': count})

plt.figure(figsize=(10, 10))
sns.barplot(y=jenis_topic.index, x='count_topic', data=jenis_topic, hue = jenis_topic.index)
plt.xlabel('Total Topic')
plt.ylabel('Topic')
plt.title('Distribusi Topic')
plt.show()


# Dari grafik distribusi menunjukkan mayoritas blog yang ada merupakan artikel blog dengan topic ai, blockchain, cybersecurity, web-development, data-analysis, dan cloud-computing.

# ### Author

# In[117]:


# melihat data author
author.info()


# In[118]:


# melihat data unique pada author
print('Jumlah author_id :', len(author.author_id.unique()))
print('Banyak nama author :', len(author.author_name.unique()))
print('Total dupllicate data :', author.duplicated().sum())


# Dari output diatas terlihat perbedaan antara jumlah author dengan nama author

# In[119]:


# mencari nama author yang memiliki lebih dari 1 author_id
author.author_name.value_counts()


# In[120]:


# melihat author_id yang memiliki lebih dari 1 nama author
author[author.author_name == 'Kompetify.ai']


# Dapat dilihat pada table diatas, author dengan nama Kompetify.ai memiliki author_id 84 dan 96. Maka dari itu perlu analasis lebih lanjut, apakah jumlah author_id tersebut akan berpengaruh atau tidak

# # Data Preprocessing

# ### Create Helper Function
# 

# In[121]:


# membuat function untuk menghilangkan tanda baca, symbol, emoji pada data
def remove_emoji_and_symbols(text):
  text_no_emoji = re.sub(r'[^\w\s\d]+', '', text)
  return text_no_emoji


# ### Remove punctuation, emoji, and symbol

# In[122]:


# menghapus tanda baca, symbol, emoji pada title
blog['blog_title'] = blog['blog_title'].apply(lambda x: remove_emoji_and_symbols(x))

# menghapus tanda baca, symbol, emoji pada topic
blog['topic'] = blog['topic'].apply(lambda x: remove_emoji_and_symbols(x))

# menghapus tanda baca, symbol, emoji pada author_name
author['author_name'] = author['author_name'].apply(lambda x: remove_emoji_and_symbols(x))


# ### Merge author and blog

# In[123]:


# menggabungkan dataset author dengan blog
author_new = pd.merge(author, blog, on = 'author_id', how = 'left')
author_new.shape


# In[124]:


# melihat missing value pada data
author_new.isnull().sum()


# In[125]:


# Melihat blog yang dimiliki author kompetify.ai
id1 = author_new[author_new.author_id == 84]
id2 = author_new[author_new.author_id == 96]

# menggabungkan dataframe id1 dan id2
dup_id = pd.concat([id1, id2])
dup_id


# Dapat dilihat author dengan nama Kompetifyai hanya memiliki 1 buah blog pada 2 id yang berbeda, maka id yang tidak memiliki blog akan dihapus.

# ### Remove Missing Value

# In[126]:


# melihat data nan
author_new.loc[(author_new[['blog_id', 'blog_title', 'topic']].isnull()).any(axis=1)].head()


# In[127]:


# menghapus nilai nan
author_clean = author_new.dropna(axis = 0)
author_clean.isnull().sum()


# In[128]:


# mengubah tipe data blog_id menjadi integer
author_clean['blog_id'] = author_clean['blog_id'].astype(int)
author_clean.shape


# In[129]:


# membuat dataframe untuk melihat jumlah author id
df_author = pd.DataFrame(columns=['Jumlah'], index=['Author', 'Author Clean'])
df_author.loc['Author', 'Jumlah'] = len(author.author_id.unique())
df_author.loc['Author Clean', 'Jumlah'] = len(author_clean.author_id.unique())
df_author


# Dari tabel diatas menunjukkan terdapat beberapa author yang tidak memiliki blog sama sekali sehingga author yang tidak memiliki blog akan dihapus.

# ### Create DataFrame blog_rate

# In[130]:


# membuat dataframe blog rating dengan menggabungkan rating dan author_clean
blog_rate = pd.merge(rating, author_clean, on = 'blog_id', how = 'left')
blog_rate


# In[131]:


# melihat topic yang disukai pembaca berdasarkan rating
top_topic = blog_rate.groupby('topic').sum().sort_values('ratings', ascending = 0)

plt.figure(figsize=(10, 10))
sns.barplot(y=top_topic.index, x='ratings', data=top_topic, hue = top_topic.index)
plt.xlabel('Total Rating')
plt.ylabel('Topic')
plt.title('Topic Yang Disukai User')
plt.show()


# Dari hasil distribusi user dapat disimpulkan bahwa blog dengan topic flutter, android, app-development, SoftwareDevelopment, dan web-development lebih banyak diminati oleh sebagian besar user.

# # Data Preparation

# In[132]:


# check missing values
blog_rate.isnull().sum()


# ## Sorting Data

# In[133]:


blog_rate = blog_rate.sort_values('blog_id', ascending = True)
blog_rate


# In[134]:


print('Jumlah Blog unique :',len(blog_rate.blog_id.unique()))
print('Jumlah Title unique :',len(blog_rate.blog_title.unique()))


# Terdapat perbedaan anatra jumlah blog yang ada dengan judul blog, dimana seharusnya tiap blog memiliki id dan judulnya masing-masing (unique)

# ## Remove Duplicate Data

# In[135]:


# membuat dataframe preparation yang berisi blog_rate dan membuang duplicate value
preparation = blog_rate.drop_duplicates('blog_title')
preparation


# ## Selection Feature & Convert Data to List

# In[136]:


# Mengonversi data series ‘blog_id’ menjadi dalam bentuk list
blog_id = preparation['blog_id'].tolist()

# Mengonversi data series ‘Name’ menjadi dalam bentuk list
blog_title = preparation['blog_title'].tolist()

# Mengonversi data series ‘Rcuisine’ menjadi dalam bentuk list
topic = preparation['topic'].tolist()

print(len(blog_id))
print(len(blog_title))
print(len(topic))


# In[137]:


# membuat dataframe final_preparation dan diisi dengan blog_id, blog_title, topic
final_preparation = pd.DataFrame({'blog_id': blog_id,
                                  'title': blog_title,
                                  'topic': topic})

final_preparation


# # Model Development
# 
# - Content Based Filtering
# - Collaborative Filtering

# ## Content Based Filtering

# In[138]:


data = final_preparation
data.head()


# ### TF-IDF

# In[139]:


tfidf = TfidfVectorizer()

# menghitung idf pada data topic
tfidf.fit(data['topic'])

# Mapping array dari fitur index integer ke fitur nama
tfidf.get_feature_names_out()


# In[140]:


# melakukan fit lalu tranformasiakan ke bentuk matrix
tfidf_matrix = tfidf.fit_transform(data['topic'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape


# In[141]:


# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()


# In[142]:


# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan topic
# Baris diisi dengan title

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=data.title
).sample(23, axis=1).sample(5, axis=0)


# ### Cosine Similarity

# In[143]:


cos_sim = cosine_similarity(tfidf_matrix)
cos_sim


# In[144]:


cos_sim_df = pd.DataFrame(cos_sim, index = data['title'], columns = data['title'])

# melihat similarity matrix pada setiap resto
cos_sim_df.sample(5, axis= 1).sample(5, axis = 0)


# ### Recommendation Function

# In[145]:


def blog_recommendation(title_blog, similarity_data = cos_sim_df, items = data[['title', 'topic']], k = 10):
  # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
  # Dataframe diubah menjadi numpy
  # Range(start, stop, step)

  index = similarity_data.loc[:,title_blog].to_numpy().argpartition(range(-1, -k, -1))

  # Mengambil data dengan similarity terbesar dari index yang ada
  closest = similarity_data.columns[index[-1:-(k+2):-1]]

  # Drop title_blog agar nama resto yang dicari tidak muncul dalam daftar rekomendasi
  closest = closest.drop(title_blog, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)


# ### Evaluasi Model Content Based Filtering

# In[146]:


data[data.title == 'sqlite from a web page']


# In[147]:


# mencoba mendapatkan rekomendai blog
blog_recommendation('sqlite from a web page')


# ## Collaborative Filtering

# In[148]:


df = blog_rate
df


# ### Membuat List Profile User

# In[149]:


# Mengubah userID menjadi list tanpa nilai yang sama
user_ids = df['userId'].unique().tolist()
print('list userId: ', user_ids)

# Melakukan encoding userID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userId : ', user_to_user_encoded)

# Melakukan proses encoding angka ke ke userID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userId: ', user_encoded_to_user)


# In[150]:


# Mengubah blog_id menjadi list tanpa nilai yang sama
blog_ids = df['blog_id'].unique().tolist()

# Melakukan proses encoding blog_id
blog_to_blog_encoded = {x: i for i, x in enumerate(blog_ids)}

# Melakukan proses encoding angka ke blog_id
blog_encoded_to_blog = {i: x for i, x in enumerate(blog_ids)}


# In[151]:


# Mapping userId ke dataframe user
df['user'] = df['userId'].map(user_to_user_encoded)

# Mapping blog_id ke dataframe blog
df['blog'] = df['blog_id'].map(blog_to_blog_encoded)


# In[152]:


# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah blog
num_blog = len(blog_encoded_to_blog)
print(num_blog)

# Nilai minimum rating
min_rating = min(df['ratings'])

# Nilai maksimal rating
max_rating = max(df['ratings'])

print('Number of User: {}, Number of blog: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_blog, min_rating, max_rating
))


# In[153]:


df = df.sample(frac = 1, random_state = 42)
df


# #### Split Data

# In[154]:


# membuat variabel x untuk mencocokkan data user dan blog
x = df[['user', 'blog']].values

# membuat variabel y untuk hasil ratings
y = df['ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

sample = int(0.8 * df.shape[0])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = sample, random_state = None)
print(x,y)


# ### Class ReccomenderNet

# In[155]:


class RecommenderNet(tf.keras.Model):
  # Insialisasi fungsi
  def __init__(self, num_users, num_blog, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_blog = num_blog
    self.embedding_size = embedding_size
    self.user_embedding = tf.keras.layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(num_users, 1) # layer embedding user bias
    self.blog_embedding = tf.keras.layers.Embedding( # layer embeddings blog
        num_blog,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.blog_bias = tf.keras.layers.Embedding(num_blog, 1) # layer embedding blog bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    blog_vector = self.blog_embedding(inputs[:, 1]) # memanggil layer embedding 3
    blog_bias = self.blog_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_blog = tf.tensordot(user_vector, blog_vector, 2)

    x = dot_user_blog + user_bias + blog_bias

    return tf.nn.sigmoid(x) # activation sigmoid


# In[156]:


# compile model
model = RecommenderNet(num_users, num_blog, 64) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# callbacks
callbacks = tf.keras.callbacks.EarlyStopping(
    min_delta=0.0001,
    monitor = 'val_loss',
    patience=5,
    restore_best_weights=True,
)


# ### Training

# In[157]:


history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 50,
    callbacks = [callbacks],
    validation_data = (x_val, y_val)
)


# ### Evaluasi Model Collaborative Filtering

# In[158]:


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Dari hasil visualisasi training diatas didapatkan nilai error akhir sebesar 0.375 dan nilai error sebesar 0.39 pada validasi. Nilai tersebut sudah cukup baik untuk sistem rekomendasi.

# ### Test Prediction

# In[159]:


blog_df = final_preparation
df = rating

# Mengambil sample user
user_id = df.userId.sample(1).iloc[0]
blog_read_by_user = df[df.userId == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
blog_not_read = blog_df[~blog_df['blog_id'].isin(blog_read_by_user.blog_id.values)]['blog_id']
blog_not_read = list(
  set(blog_not_read)
  .intersection(set(blog_to_blog_encoded.keys()))
)

blog_not_read = [[blog_to_blog_encoded.get(x)] for x in blog_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_blog_array = np.hstack(
  ([[user_encoder]] * len(blog_not_read), blog_not_read)
)


# In[160]:


recom_blog = model.predict(user_blog_array).flatten()

top_ratings_indices = recom_blog.argsort()[-10:][::-1]
recommended_blog_ids = [
    blog_to_blog_encoded.get(blog_not_read[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Blog with high ratings from user')
print('----' * 8)

top_blog_user = (
    blog_read_by_user.sort_values(
        by = 'ratings',
        ascending=False
    )
    .head(5)
    .blog_id.values
)

blog_df_rows = blog_df[blog_df['blog_id'].isin(top_blog_user)]
for row in blog_df_rows.itertuples():
    print(row.title, ':', row.topic)

print('----' * 8)
print('Top 10 Blog recommendation')
print('----' * 8)

recommended_blog = blog_df[blog_df['blog_id'].isin(recommended_blog_ids)]
for row in recommended_blog.itertuples():
    print(row.title, ':', row.topic)

