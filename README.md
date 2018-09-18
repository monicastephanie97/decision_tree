# Machine Learning - Decision Tree

_Wow! Ofc I know it! Everyone's been raving bout it recently! But, what is it actually?_  

_(Shout out to Kak Aji for giving a short lecture to me and Kak Rahmat!)_

Pada latihan pertama ini, data yang digunakan adalah data titanic dari Kaggle. saya akan membuat model untuk menentukan apakah seseorang bisa _survive_ atau tidak dari musibah titanic. Tentunya keputusan ini diambil berdasarkan data-data tersedia seperti nama, umur, jenis kelamin, dll. 

 Pengolahan data akan dilakukan dengan python. Disini akan saya tulis cara _step by step_ yang saya lakukan untuk menentukan model tersebut (tentunya ini masih model yang _basic_ banget).

 ## 1. _Always, always, clean your data FIRST_!
_Of course_, pertanyaan selanjutnya: gimana caranya tuh ngebersihin data? Ada beberapa tahap dalam membersihkan data. Yang akan saya pakai kali ini hanya:
1. Membuang data 'nirfaedah'
2. Mengisi nilai _missing values_

Kedua tahap ini akan dibahas dibawah.

### 1) Membuang data 'nirfaedah'
Pertama kali yang harus dilakukan yaitu menentukan apakah ada parameter-parameter yang tidak memengaruhi keputusan. Seperti contohnya dalam kasus titanic ini, nama seseorang dan nomor telepon tidak memengaruhi orang tersebut akan _survive_ atau tidak. Kedua data ini berarti masuk dalam kategori 'nirfaedah' saya. Cara menentukannya yaitu berdasarkan intuisi saya masing-masing dan logical reasoning.

```python
for x in ["Name","Survived"]: 
    dataset=dataset.drop([x], axis=1)
```

### 2) Mengisi nilai _missing values_
Pertama saya cari tahu dulu apakah data saya bersifat __numerikal__ atau __kategorikal__. Jika data saya merupakan data __numerikal__, biasanya nilai _missing values_ akan saya isi dengan rata-rata. Namun, harus diperhatikan juga bahwa rata-rata atau __mean__ sangat dipengaruhi oleh nilai _outlier_ atau pencilan. Sehingga, jika data saya memiliki _outliers_, saya akan menggantikannya dengan __median__. Sebaliknya, apabila data saya merupakan data __kategorikal__, saya akan mengganti nilai _missing values_ dengan modus atau __mode__.

Tapi ada 1 hal penting yang harus diperhatikan. Data yang _missing values_ nya diatas 30% lebih baik tidak dipakai :)

```python
#Filling Missing Numerical Values with Mean
data_titanic["Age"]=data_titanic["Age"].fillna(data_titanic["Age"].mean())

#Filling Missing Categorical Values with Mean
data_titanic["Embarked"]=data_titanic["Embarked"].fillna(data_titanic["Embarked"].mode()[0])
```

## 2. _Choose your result or label column_!
Pada data titanic, kolom "Survived" merupakah kolom hasil atau label saya, dengan angka 1 menandakan selamat dan 0 menandakan tidak selamat.

```python
#Determining Data Result named as Data_Label
data_result=data_titanic["Survived"]
```

## 3. _Transform your categorical data into numerical one_
saya akan mengubah kolom data kategorikal menjadi matriks biner seperti ini:
<center><a href="https://imgbb.com/"><img src="https://image.ibb.co/ekrxez/matriks_biner.jpg" alt="matriks_biner" border="0" width=400></a></center>

Matriks ini akan berupa tabel dengan banyak baris sejumlah banyak orang dan banyak kolom sejumlah banyak data unik pada kolom data kategorikal yang saya punya. Pada contoh diatas kolom data kategorikal adalah kolom __Gender__. Karena hanya ada 2 gender yaitu __Male__ dan __Female__ maka kolomnya hanya ada 2.

Kolom data kategorikal akan diubah menjadi matriks biner dengan method __pandas.get_dummies__ yang didapat dari library __pandas__.

```python
#Transforming Categorical Data into Numeric
categorical_label=["Pclass","Sex","Embarked"]

data_categorical=pd.DataFrame()
for x in categorical_label:
    data_dummy=pd.get_dummies(data_titanic[x],prefix=str(x))
    data_categorical=pd.concat([data_categorical,data_dummy],axis=1)
```

Setelah matriks-matriks biner digabungkan ke dataframe asli, jangan lupa untuk menghapus kolom data kategorikal sehingga isi dari dataframe seluruhnya merupakan angka.

```python
#Removing Categorical Columns
dataset=pd.concat([dataset,data_categorical],axis=1)
for x in categorical_label:
    dataset=dataset.drop([x],axis=1)
```

## 4. _Split your data into __data train__ as well as __data test___
__Data train__ berguna untuk melatih model, sedangkan __data test__ berguna untuk menguji model.

```python
#Dividing Dataset into Data Train and Data Test
X_train, X_test, y_train, y_test=train_test_split(dataset,data_result,test_size=0.20,random_state=14)

```

## 5. _This is the fun part! Finally, create your model_ ;)
Untuk membuat model, dapat digunakan beberapa metode seperti Random Forest, SVM, dll. Namun, pada kali ini saya akan memakai __Decision Tree__.

```python
#Use Decision Tree
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
data_predict=dt.predict(X_test)
```
## 6. _No, it's not over yet! __Test__ your model_ 
Model yang telah dibuat harus diuji nilai akurasinya. Nilai akurasi dikatan baik jika diatas 80%. 

```python
#Compute Accuracy Score
print(accuracy_score(data_predict,y_test))
```

Ada cara lain pula untuk menguji akurasi model. Kita akan membandingkan __y_train_predict__ dengan __y_train__, lalu __y_test_predict__ dengan __y_test__.

```python
y_train_predict=dt.predict(X_train)
y_test_predict=dt.predict(X_test)

#Compute Accuracy Score

acc1=accuracy_score(y_train_predict,y_train)
acc2=accuracy_score(y_test_predict,y_test)
```

Akurasi model dikatakan baik apabila nilai acc1 dan acc2 tidak berbeda jauh dan diatas 80%. Apabila nilai acc1 sangat tinggi (mendekati 1), namun nilai acc2 sangat rendah, maka kemungkinan besar model yang telah dibuat _overfitting_.

## Notes
Gambar dari _Decision Tree_ yang dipakai di model bisa dilihat. Pertama-tama, ekstrak filenya.

```python
#Export The Decision Tree
from sklearn import tree
tree.export_graphviz(dt,out_file="GambarDecisionTree",
                    feature_names=list(dataset.columns),
                    class_names=['0','1'],
                    filled=True)
```

Lalu, upload gambar tersebut ke http://webgraphviz.com/. 