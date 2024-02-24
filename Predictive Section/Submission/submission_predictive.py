# -*- coding: utf-8 -*-
"""Submission Predictive.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C1Gwnb7Yahh7Aze15mWYsKzZe8W4dEtI

# Import Library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

"""# Load Dataset

Link Dataset : https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data
"""

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.shape)
print(test_data.shape)

"""# EDA"""

train_data.info()

# memisahkan categorical feature dan numerik feature
categorical = ['three_g','touch_screen', 'wifi', 'four_g', 'dual_sim', 'blue', 'price_range']

numerical = train_data.drop(columns = categorical, axis = 1).columns

train_data[numerical].describe()

"""## Error Value

Dari hasil describe terlihat terdapat beberapa fitur yang memiliki nilai minimun 0. Dimana seharusnya nilai tersebut bukan 0, sehingga kita harus mencari tahu lebih dalam lagi apakah fitur tersebut akan di isi dengan nilai mean, max, min, atau dihapus saja
"""

train_data.isnull().sum()

# Menghitung value error, seperti 0 atau value minus (lakukan pada data numerik saja)
for i in (train_data[numerical]):
  misValue = (train_data[i] <= 0).sum()
  print(f'Value error pada {i} :', misValue)

# menampilkan data yang memiliki error values
train_data.loc[(train_data[numerical]==0).any(axis=1)]

"""Dengan data train sebanyak 2000 dan jumlah missing values yang banyak maka kita dapat mengisi missing values tersebut menggunakan mean, max, ataupun min"""

train_data[numerical] = train_data[numerical].replace(0, train_data[numerical].mean())
train_data

# check kembali apakah masih ada missing values atau tidak
for i in (train_data[numerical]):
  misValue = (train_data[i] <= 0).sum()
  print(f'Value error pada {i} :', misValue)

# melihat distribusi data pada target
jenis = len(train_data['price_range'].value_counts())
count = train_data['price_range'].value_counts()
for i in range(jenis):
    print(f'Price Range {i} : ',count[i] )

fig , ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=count.index, y=count, palette='Set1', hue = count.index)
plt.title('Distribusi Kategori Price Range')
plt.legend(title = 'Price Range')
plt.xlabel('Kategori')
plt.ylabel('Count')
plt.show()

"""Distribusi pada label menampilkan data sudah balance maka tidak perlu melakukan oversampling ataupun undersampling

### Removing Outliers
- Detect Outliers
- IQR

## Outliers
"""

fig, axes = plt.subplots(7, 2, figsize=(14, 12))

for i, feature in enumerate(train_data[numerical]):
    r = i // 2
    c = i % 2
    sns.boxplot(data=train_data, x=feature, ax=axes[r, c])
    axes[r, c].set_title(feature)

plt.tight_layout()
plt.show()

"""## IQR

implementasi teknik IQR pada data numerik
"""

# menghapus outliers
Q1 = train_data.quantile(0.25)
Q3 = train_data.quantile(0.75)

IQR = Q3 - Q1

new_data = train_data[~((train_data[numerical]<(Q1 - 1.5*IQR))|(train_data[numerical]>(Q3 + 1.5*IQR))).any(axis=1)]

new_data.shape

"""## Univariate Analysis"""

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, feature in enumerate(categorical[:-1]):
  r = i // 3
  c = i % 3
  count = new_data[feature].value_counts()

  count.plot(kind='pie', ax=axes[r, c], autopct='%1.1f%%', startangle=90, cmap='Paired')
  axes[r,c].set_title(f'Distribusi {feature}')
  axes[r,c].legend(title=feature, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

"""Dari grafik diatas menunjukkan tiap tiap feature memiliki dsitribusi data yang seimbang atau hampir seimbang kecuali pada 'three_g', sehingga dapat disimpulkan bahwa hanya sebagian kecil handphone yang tidak support 3G"""

# histogram numerik feature
new_data[numerical].hist(bins=50, figsize=(20,15))
plt.show()

"""## Multivariate Analysis"""

plt.figure(figsize=(20, 12))
correlation_matrix = new_data.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix ", size=20)

# Viiualisasi korelasi ram dan price_range
plt.figure(figsize = (8,4))
plt.scatter(x = new_data['ram'], y = new_data['price_range'])
plt.title('Korelasi ram dan price_range')
plt.show()

"""Dari visualisasi diatas dapat disimpulkan bahwa semakin tinggi kategori price_range maka hp tersebut meiliki kapasitas ram yang besar, dan juga sebaliknya. Hp yang memiliki kapasitas ram kecil maka akan masuk ke kategori price_range yang rendah

# Data Preparation

- Split Data
- Normalization

## Split Data
"""

x = new_data.iloc[:, :-1]
y = new_data['price_range']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""## Normalization"""

scaler = MinMaxScaler()
x_train[numerical] = scaler.fit_transform(x_train[numerical])
x_test[numerical] = scaler.fit_transform(x_test[numerical])
x_train[numerical].describe().round(4)

"""# Create Model

- KNN
- RandomForest
- AdaBoost

## GridSearch
"""

KNN = KNeighborsClassifier()
RF = RandomForestClassifier()
Boost = AdaBoostClassifier()

def best_param(x, y):
  algorithms = {
        'knn': {
            'model': KNN,
            'params': {
                'n_neighbors': range(1,20),
            }
        },
        'boosting': {
            'model': Boost,
            'params': {
                'learning_rate' : [0.1, 0.05, 0.01, 0.05, 0.001],
                'n_estimators': [25, 50, 75, 100],
                'algorithm' : ['SAMME', 'SAMME.R'],
                'random_state': [11, 33, 55, 77]
            }
        },
        'random_forest': {
            'model': RF,
            'params': {
                'n_estimators': [25, 50, 75, 100],
                'max_depth' : [None, 10, 20],
                'random_state': [11, 33, 55, 77],
            }
        }

    }

  scores = []
  for algo_name, config in algorithms.items():
      gs =  GridSearchCV(config['model'], config['params'], cv=10, return_train_score=False)
      gs.fit(x,y)
      scores.append({
          'model': algo_name,
          'best_score': gs.best_score_,
          'best_params': gs.best_params_
      })

  return pd.DataFrame(scores,columns=['model','best_score','best_params'])

best_param(x,y)

"""# Evaluation"""

# implementasi best param
KNN = KNeighborsClassifier(n_neighbors = 11)
RF = RandomForestClassifier(max_depth = None, n_estimators= 75, random_state = 55)
Boost = AdaBoostClassifier(algorithm= 'SAMME', learning_rate = 0.1, n_estimators = 100, random_state = 11)

acc = pd.DataFrame(index = ['accuracy'], columns = ['KNN', 'RandomForest', 'Boosting'])

# Model KNN accuracy
KNN.fit(x_train, y_train)
acc.loc['accuracy', 'KNN'] = KNN.score(x_test,y_test)

# Model RandomForest accuracy
RF.fit(x_train, y_train)
acc.loc['accuracy', 'RandomForest'] = RF.score(x_test,y_test)

# Model AdaBoost accuracy
Boost.fit(x_train, y_train)
acc.loc['accuracy', 'Boosting'] = Boost.score(x_test,y_test)

acc

mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': KNN, 'RF': RF, 'Boosting': Boost}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3

mse

fig, ax = plt.subplots()
mse.sort_values(by = 'test', ascending = 0).plot(kind = 'barh', ax=ax, zorder=3)
ax.grid()

"""Dari hasil evaluasi model diatas dapat disimpulkan bahwa algoritma **Random Forest** memiliki tingkat accuracy yang tinggi dan nilai error yang kecil daripada algoritma KNN atau AdaBoost, maka dari itu kita dapat menggunakan model dengan algoritma **Random Forest** untuk prediksi pada data test

# Prediction pada Data Test
"""

# Normalisasi data
test_data.drop('id', inplace = True, axis = 1)
test_data[numerical] = scaler.transform(test_data[numerical])

# Prediksi
y_pred = RF.predict(test_data)
df_pred = pd.DataFrame(y_pred)

test_n_data = pd.read_csv('test.csv')
test_n_data['price_range'] = df_pred
test_n_data

# simpan hasil prediksi ke excel
test_n_data.to_csv('prediction.csv', index = False)