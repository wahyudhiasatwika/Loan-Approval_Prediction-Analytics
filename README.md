# Predictive Analytics : Loan Approval Prediction - Wahyu Dhia Satwika

## Domain Proyek : Keuangan
Industri perbankan menghadapi tantangan besar dalam menentukan kelayakan pemohon untuk mendapatkan pinjaman. Semakin banyaknya permohonan pinjaman membuat lembaga keuangan perlu memiliki pendekatan yang efisien dan efektif untuk memutuskan persetujuan atau penolakan pinjaman, sambil mengurangi risiko kredit macet. Melalui penggunaan machine learning dan analisis data, lembaga keuangan dapat membuat keputusan lebih cerdas, mempercepat proses persetujuan pinjaman, dan meminimalkan risiko keuangan. Prediksi menggunakan teknik machine learning diperlukan yang bertujuan untuk menghindari permasalahan  kredit  macet  kedepannya (Sitepu & Manohar, 2022). 

## Business Understanding

### Problem Statements
- Bagaimana cara menentukan kelayakan persetujuan pinjaman secara otomatis untuk mengurangi waktu proses dan meminimalkan risiko kredit macet?
- Faktor apa saja yang paling berpengaruh dalam menentukan apakah pinjaman layak disetujui?

### Goals 
- Mengembangkan model prediksi yang mampu menentukan status persetujuan pinjaman dengan akurasi tinggi.
- Mengidentifikasi faktor-faktor yang memengaruhi kelayakan pinjaman sehingga dapat digunakan untuk mengoptimalkan proses persetujuan.

### Solution Statements 
- Membangun model machine learning untuk klasifikasi persetujuan pinjaman.
- Menganalisis variabel yang ada di dalam dataset untuk menemukan fitur-fitur yang paling berpengaruh dalam keputusan persetujuan pinjaman.

## Data Understanding
Dataset Loan Approval yang berasal dari kaggle merupakan sebuah dataset sintetis yang berdasarkan dari [(Sumber Utama)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). Dataset Loan Approval memiiliki 45000 records dengan 14 variabel, dengan rincian sebagai berikut:

**Informasi Dataset**
# Loan Approval Classification Dataset

| **Judul**       | Loan Approval Classification                                                        |                  
|-----------------|-------------------------------------------------------------------------------------|
| **Author**      | Ta-WEI LO                                                                           |
| **Source**      | [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) |
| **Visibility**  | Public                                                                              |
| **Usability**   | 10.00                                                                               |


**Metadata**
| Kolom                         | Deskripsi                                         | Tipe Data    |
|-------------------------------|---------------------------------------------------|--------------|
| person_age                    | Usia                                              | Float        |
| person_gender                 | Jenis kelamin orang                               | Kategorikal  |
| person_education              | Tingkat pendidikan tertinggi                      | Kategorikal  |
| person_income                 | Pendapatan tahunan                                | Float        |
| person_emp_exp                | Lama pengalaman kerja dalam tahun                 | Integer      |
| person_home_ownership         | Status kepemilikan rumah (misalnya, sewa, milik, hipotek) | Kategorikal  |
| loan_amnt                     | Jumlah pinjaman yang diminta                      | Float        |
| loan_intent                   | Tujuan dari pinjaman                              | Kategorikal  |
| loan_int_rate                 | Suku bunga pinjaman                               | Float        |
| loan_percent_income           | Jumlah pinjaman sebagai persentase dari pendapatan tahunan | Float |
| cb_person_cred_hist_length    | Lama riwayat kredit dalam tahun                   | Float        |
| credit_score                  | Skor kredit                                       | Integer      |
| previous_loan_defaults_on_file| Indikator default pinjaman sebelumnya             | Kategorikal  |
| loan_status (target variable) | Status persetujuan pinjaman: 1 = disetujui; 0 = ditolak | Integer  |

### Exploratory Data Analysis
Pada tahap ini dilakukan analisis untuk data yang ada di dalam dataset seperti dilihat

| Kolom                             | Jumlah Non-NUll    | TIpe Data     |
|-----------------------------------|--------------------|---------------|
| person_age                        | 45000              | float64       |
| person_gender                     | 45000              | object        |
| person_education                  | 45000              | object        |
| person_income                     | 45000              | float64       |
| person_emp_exp                    | 45000              | int64         |
| person_home_ownership             | 45000              | object        |
| loan_amnt                         | 45000              | float64       |
| loan_intent                       | 45000              | object        |
| loan_int_rate                     | 45000              | float64       |
| loan_percent_income               | 45000              | float64       |
| cb_person_cred_hist_length        | 45000              | float64       |
| credit_score                      | 45000              | int64         |
| previous_loan_defaults_on_file    | 45000              | object        |
| loan_status                       | 45000              | int64         |

Output di atas menunjukkan bahwa dataset memiliki 45000 data dan 14 kolom.
- Terdapat 6 tipe data float64
- Terdapat 3 tipe data int64
- Terdapat 5 tipe data object

| **Statistic** | **person_age** | **person_income** | **person_emp_exp** | **loan_amnt** | **loan_int_rate** | **loan_percent_income** | **cb_person_cred_hist_length** | **credit_score** | **loan_status** |
|---------------|----------------|-------------------|--------------------|----------------|-------------------|-------------------------|-------------------------------|------------------|-----------------|
| **Count**     | 45000.000000   | 45000.000000      | 45000.000000       | 45000.000000   | 45000.000000      | 45000.000000            | 45000.000000                  | 45000.000000     | 45000.000000    |
| **Mean**      | 27.764178      | 80319.050000      | 5.410333           | 9583.157556    | 11.006606         | 0.139725                | 5.867489                      | 632.608756       | 0.222222        |
| **Std**       | 6.045108       | 80422.500000      | 6.063532           | 6314.886691    | 2.978808          | 0.087212                | 3.879702                      | 50.435865        | 0.415744        |
| **Min**       | 20.000000      | 8000.000000       | 0.000000           | 500.000000     | 5.420000          | 0.000000                | 2.000000                      | 390.000000       | 0.000000        |
| **25%**       | 24.000000      | 47204.000000      | 1.000000           | 5000.000000    | 8.590000          | 0.070000                | 3.000000                      | 601.000000       | 0.000000        |
| **50%**       | 26.000000      | 67048.000000      | 4.000000           | 8000.000000    | 11.010000         | 0.120000                | 4.000000                      | 640.000000       | 0.000000        |
| **75%**       | 30.000000      | 95789.250000      | 8.000000           | 12237.250000   | 12.990000         | 0.190000                | 8.000000                      | 670.000000       | 0.000000        |
| **Max**       | 144.000000     | 7200766.000000    | 125.000000         | 35000.000000   | 20.000000         | 0.660000                | 30.000000                     | 850.000000       | 1.000000        |

Digunakan describe() untuk memberikan informasi statistik.

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum.
- 25% adalah kuartil pertama.
- 50% adalah kuartil kedua.
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

#### Checking Missing Value

| Kolom                            | Missing Values     |
|----------------------------------|--------------------|
| person_age                       | 0                  |
| person_gender                    | 0                  |
| person_education                 | 0                  |
| person_income                    | 0                  |
| person_emp_exp                   | 0                  |
| person_home_ownership            | 0                  |
| loan_amnt                        | 0                  |
| loan_intent                      | 0                  |
| loan_int_rate                    | 0                  |
| loan_percent_income              | 0                  |
| cb_person_cred_hist_length       | 0                  |
| credit_score                     | 0                  |
| previous_loan_defaults_on_file   | 0                  |
| loan_status                      | 0                  |

Dapat dilihat dari hasil di atas bahwa dataset tidak memiliki data duplikat dan missing value. Oleh karena itu, proses dapat dilanjutkan kepada visualisasi data

#### Boxplot Visualization

Untuk mempermudah visualisasi data, maka di feature dibagi menjadi categorical_feature dan numerical_feature.

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_age.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_person_income.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_person_emp_exp.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_loan_amnt.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_loan_int_rate.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_loan_percent_income.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_cb_person_cred_hist_length.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot1_credit_score.png?raw=true)

Untuk rata-rata distribusi box plot dapat dilihat sebagai berikut:
- person_age : rata-rata penggunaan loan_approval yaitu dari 23-28 tahun.
- person_income : rata-rata pengguna loan approval memiliki income sebesar 45000 - 85000.
- person_emp_exp : rata-rata pengalaman pekerjaan yang dimiliki oleh pengaju loan_approval adalah 1-7 tahun.
- loan_amnt : rata-rata pinjaman yang diminta yaitu 5000-13000
- loan_int_rate : rata-rata suku bunga pinjaman yang dimiliki berada pada angka 8.5-13
- loan_percent_income : rata-rata dari persentase pendapatan tahunan berada pada angka 0.07 - 0.18
- cb_person_cred_hist_length : rata-rata lama riwayat kredit yaitu 3 - 7 tahun
- credit_score : rata-rata skor kredit pengaju loan yaitu 600 - 660.

### EDA - Univariate Analysis

# Univariate Analysis - Numerical Feature
![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/univariate.png?raw=true)

- person_age : Distribusi dari umur memiliki angka tertinggi pada 23-24 tahun.
- person_income : Distribusi pendapatan pengaju memiliki angka terbanyak pada jangka 40000-70000
- loan_amnt : Distribusi permintaan loan memiliki angka tertinggi pada 500 dan 10000.
- loan_int_rate : Distribusi suku bunga pinjaman memiliki angka tertinggi pada angka 11
- credit_score : Distribusi skor kredit terdapat pada jangka angka 600-700.
- loan_percent_income: Distribusi jumlah pinjaman sebagai persentase dari pendapat tahunan memiliki angka tertinggi pada 0.05 - 0.15.

# Univariate Analysis - Multivariate analysis

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/univariate_categorical.png?raw=true)

Feature yang memiliki variasi terbanyak yaitu pada person_education, person_home_ownership, dan loan_intent.
- person_education : Angka tertinggi yaitu pada Bachelor dan disusul dengan High School dan Associate.
- person_home_ownership : Rata-rata dari pengaju loan yaitu bertempat tinggal rent dan mortgage.
- loan_intent : Angka tertinggi pada pengaju loan yaitu untuk education dan medical.

#### EDA - Multivariate Analysis


![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/multi_gender.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/multi_education.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/multi_loan_intent.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/multi_loan_defaults.png?raw=true)

- person_gender : Jenis kelamin tidak memiliki pengaruh besar dalam penerimaan pengajuan loan.
- person_education : Yang paling banyak diterima dalam pengajuan loan yaitu untuk Rent.
- loan_intent : Penerimaan loan paling banyak diterima untuk tujuan medical dan debtconsolidation
- previous_loan_default_on_file : Yang paling banyak diterima untuk pengajuan loan yaitu yang memiliki indikator peminjaman sebelumnya adalah "no".

#### Correlation Matrix

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/correlation.png?raw=true)

Visualisasi ini digunakan untuk mencari tahu feature apa saja yang memiliki korelasi paling besar. Feature Numerik yang memiliki korelasi paling besar yaitu loan_int_rate dengan skor sebesar 0.32 dan loan_percent_income sebesar 0.35.

### Data Preparation
Teknik yang digunakan:
- Filter Data : Melakukan filter terhada data yang tidak perlu
- Handling Outliers : Menghapus Outliers
- Label Encoding : Untuk feature categorical seperti umur, edukasi, kepimilikan tempat tinggal, tujuan loan, indikator default peminjaman sebelumnya akan diubah menjadi angka menggunakan LabelEncoder()
- Train-test split data : Dataset nantinya akan dibagi untuk feature menjadi variable X dan label menjadi variabel y. Untuk train dibagi menjadi 80% dan test 20%.

#### Filter Data
Dapat dilihat pada kolom max di kolom umur, terdapat kejanggalan dimana terdapat umur maksimal 144. 

[ 22.  21.  25.  23.  24.  26. 144. 123.  20.  32.  34.  29.  33.  28.
  35.  31.  27.  30.  36.  40.  50.  45.  37.  39.  44.  43.  41.  46.
  38.  47.  42.  48.  49.  58.  65.  51.  53.  66.  61.  54.  57.  59.
  62.  60.  55.  52.  64.  70.  78.  69.  56.  73.  63.  94.  80.  84.
  76.  67. 116. 109.]

Dikarenakan dataset memiliki data untuk umur pengguna dengan rentang 21 hingga 144 dimana umur 144 hampir tidak mungkin terjadi di dunia nyata. Oleh karena itu, dijadikan acuan untuk umur loan approval adalah 21 - 65 tahun dengan cara filter umur dan data menjadi 44961.

#### Handling Outliers
Outlier adalah nilai ekstrem dalam dataset yang dapat mengganggu analisis statistik dan kinerja model pembelajaran mesin. Dalam proyek ini, kami menggunakan metode **Interquartile Range (IQR)** untuk mendeteksi dan menghapus outlier dari fitur numerik.

##### Langkah-Langkah Metode IQR:
1. **Pilih Fitur Numerik**:  
   Kolom yang berisi data numerik dipilih karena lebih rentan terhadap keberadaan outlier.

2. **Hitung Kuartil dan IQR**:  
   - **Q1 (Kuartil 1)**: Nilai yang memisahkan 25% data terendah dari data lainnya.  
   - **Q3 (Kuartil 3)**: Nilai yang memisahkan 75% data terendah dari 25% data tertinggi.  
   - **IQR (Interquartile Range)**: Selisih antara Q3 dan Q1 (\( IQR = Q3 - Q1 \)).

3. **Tentukan Ambang Batas Outlier**:  
   - Batas bawah: \( Q1 - 1.5 \times IQR \)  
   - Batas atas: \( Q3 + 1.5 \times IQR \)  

4. **Filter Outlier**:  
   Baris data dengan nilai di luar batas bawah dan atas dianggap sebagai outlier dan dihapus dari dataset.

##### Alasan Menggunakan IQR?
- Metode ini tahan terhadap nilai ekstrem, berbeda dengan metode yang menggunakan rata-rata atau standar deviasi.  
- Memastikan data dalam rentang kuartil tidak terpengaruh, sehingga informasi yang penting tetap terjaga.  

Dapat dilihat setelah dilakukan filter untuk outliers menjadi 37549 data. Hasil Boxplot setelah difilter dapat dilihat di bawah ini.

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_age.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_person_income.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_person_emp_exp.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_loan_amnt.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_loan_int_rate.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_loan_percent_income.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_cb_person_cred_hist_length.png?raw=true)

![alt text](https://github.com/wahyudhiasatwika/Loan-Approval_Prediction-Analytics/blob/main/Gambar/boxplot_credit_score.png?raw=true)

#### Label Encoding
Untuk melakukan label encoding maka akan menggunakan function ```LabelEncoder()```. Pada tahap ini, untuk feature categorical seperti umur, edukasi, kepimilikan tempat tinggal, tujuan loan, indikator default peminjaman sebelumnya akan diubah menjadi angka menggunakan LabelEncoder()

#### Train-test Split
Pada tahap ini, dataset akan dibagi menjadi dua yaitu data train dan data test. Data train berguna untuk melatih model sedangkan data test berguna untuk menguji performa dari model. 
```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)```
Pada tahap ini, dibagi menjadi 80% data train dan 20% data test.

### Data Modelling
Digunakan 5 model yaitu Random Forest, Logistic Regression, Decision Tree, SVC, dan KNN untuk mencari tahu model yang terbaik.

##### Random Forest 
Random Forest adalah algoritma ensemble berbasis pohon keputusan. Algoritma ini membangun banyak pohon keputusan secara acak pada subset data, kemudian menggabungkan hasil prediksi dari masing-masing pohon dengan metode voting

Parameter:
- n_estimators=100: Jumlah pohon dalam hutan.
- criterion='gini': Kriteria untuk mengukur kualitas split (menggunakan Gini impurity).
- max_depth=None: Tidak ada batasan untuk kedalaman pohon.
- min_samples_split=2: Jumlah minimal sampel yang diperlukan untuk membagi node.
- min_samples_leaf=1: Jumlah minimal sampel per daun.
- bootstrap=True: Menggunakan pengambilan sampel ulang (bootstrap) untuk membuat setiap pohon.
- random_state=None: Tidak ada nilai seed tetap untuk reproduktivitas secara default.

##### Logistic Regression 
Logistic Regression adalah model statistik yang digunakan untuk klasifikasi biner. Model ini memperkirakan probabilitas kelas menggunakan fungsi sigmoid.

Parameter:
- penalty='l2': Regularisasi L2 digunakan secara default.
- dual=False: Tidak menggunakan formulasi dual (hanya berlaku untuk solver liblinear).
- C=1.0: Parameter regulasi (invers dari kekuatan regularisasi).
- solver='lbfgs': Metode optimisasi digunakan untuk fitting (cocok untuk dataset kecil hingga menengah).
- multi_class='auto': Memilih strategi multi-klasifikasi berdasarkan jumlah kelas (binary menggunakan ovr, multi-class menggunakan softmax).
- random_state=None: Tidak ada seed tetap untuk reproduksi.

##### Decision Tree 
Decision Tree adalah algoritma yang membagi data ke dalam kelompok berdasarkan fitur dengan aturan "if-then". Algoritma ini membuat keputusan dengan struktur pohon.
Parameter 

Parameter:
- criterion='gini': Kriteria untuk mengukur kualitas split.
- splitter='best': Memilih pemisahan terbaik (dibandingkan dengan metode acak).
- max_depth=None: Tidak ada batasan untuk kedalaman pohon.
- min_samples_split=2: Jumlah minimal sampel yang diperlukan untuk membagi node.
- min_samples_leaf=1: Jumlah minimal sampel per daun.
- random_state=None: Tidak ada seed tetap untuk reproduktivitas.

##### Support Vector Classifier 
Support Vector Classifier adalah algoritma berbasis Support Vector Machine (SVM) yang mencari hyperplane optimal untuk memisahkan kelas dalam dataset.

Parameter: 
- C=1.0: Parameter regulasi (trade-off antara kesalahan klasifikasi dan margin keputusan).
- kernel='rbf': Kernel radial basis function (RBF) digunakan.
- degree=3: Degree untuk kernel polynomial (tidak digunakan untuk kernel RBF).
- gamma='scale': Parameter kernel yang dihitung berdasarkan jumlah fitur.
- random_state=None: Tidak ada seed tetap untuk reproduksi.
- probability=False: Tidak mengaktifkan probabilitas estimasi.

##### K-Nearest Neighbors (KNN) 
 K-Nearest Neighbors adalah algoritma berbasis instance yang mengklasifikasikan data baru berdasarkan jarak (misalnya, Euclidean) ke 𝑘 tetangga terdekat.

Parameter: 
- n_neighbors=5: Jumlah tetangga terdekat yang digunakan.
- weights='uniform': Semua tetangga memiliki bobot yang sama.
- algorithm='auto': Memilih algoritma terbaik (di antara ball_tree, kd_tree, dan brute) berdasarkan data.
- leaf_size=30: Ukuran daun untuk BallTree atau KDTree.
- p=2: Nilai parameter jarak Minkowski (2 = Euclidean distance).


---

#### Kelebihan dan Kekurangan Model

##### 1. **Random Forest**
- **Kelebihan**:
  - Hasil akurasi tinggi pada data latih dan uji.
  - Dapat menangani data dengan dimensi tinggi dan korelasi antar fitur.
- **Kekurangan**:
  - Waktu komputasi lebih tinggi dibandingkan model sederhana seperti Logistic Regression.

##### 2. **Logistic Regression**
- **Kelebihan**:
  - Model yang cepat dan sederhana, cocok untuk dataset yang tidak terlalu kompleks.
  - Mudah diinterpretasikan (koefisien dapat menunjukkan pengaruh setiap fitur).
- **Kekurangan**:
  - Tidak mampu menangkap hubungan non-linear antara variabel.

##### 3. **Decision Tree**
- **Kelebihan**:
  - Mudah diinterpretasikan secara visual.
  - Kemampuan untuk menangani fitur numerik dan kategorikal tanpa preprocessing yang ekstensif.
- **Kekurangan**:
  - Rentan terhadap overfitting jika tidak dilakukan pruning.

##### 4. **Support Vector Classifier (SVC)**
- **Kelebihan**:
  - Cocok untuk data dengan dimensi tinggi.
  - Memberikan solusi optimal untuk margin keputusan.
- **Kekurangan**:
  - Performanya rendah dibandingkan model lain dalam eksperimen ini.
  - Tidak skala dengan baik pada dataset besar.

##### 5. **K-Nearest Neighbors (KNN)**
- **Kelebihan**:
  - Model non-parametrik, cocok untuk data dengan distribusi kompleks.
  - Implementasi sederhana.
- **Kekurangan**:
  - Waktu prediksi lambat untuk dataset besar.
  - Sensitif terhadap ukuran dataset dan pemilihan parameter `k`.

---

### Evaluation
# KFold Cross-Validation

| **Model**                 | **Train Accuracy** | **K-Fold Accuracy** |
|---------------------------|--------------------|---------------------|
| Random Forest             | 1.0000            | 0.9231             |
| Logistic Regression       | 0.8825            | 0.8815             |
| Decision Tree             | 1.0000            | 0.8916             |
| Support Vector Classifier | 0.8098            | 0.8085             |
| K-Nearest Neighbors       | 0.8793            | 0.8342             |

- Pada evaluasi ini digunakan **5 fold**, artinya dataset dibagi menjadi 5 subset. 
- Setiap subset digunakan bergantian sebagai data uji, sementara subset lainnya digunakan sebagai data latih.

##### Alasan Menggunakan K-Fold Cross Validation:
1. **Evaluasi Konsisten**: Membagi data ke dalam beberapa lipatan memberikan evaluasi model yang lebih stabil, karena setiap data digunakan sebagai data latih dan uji.
2. **Mengurangi Bias**: Dengan evaluasi pada berbagai subset data, potensi bias dari pembagian data secara acak dapat diminimalkan.
3. **Generalisasi Model**: Memberikan gambaran lebih baik tentang bagaimana model akan bekerja pada data baru.

#### Dampak Model terhadap Business Understanding

##### Apakah Model Menjawab Problem Statement?
- **Ya**, model yang dikembangkan mampu memberikan solusi untuk menentukan kelayakan persetujuan pinjaman dengan **akurasi tinggi**. Random Forest menjadi model terbaik dengan K-Fold Accuracy sebesar **92.31%**.

##### Apakah Model Berhasil Mencapai Goals?
- **Ya**, model dapat memprediksi persetujuan peminjaman dengan akurasi yang cukup tinggi. 

##### Apakah Solusi yang Direncanakan Berdampak?
- **Ya**, solusi ini berdampak positif karena mengurangi waktu evaluasi persetujuan pinjaman secara signifikan dan meminimalkan risiko kredit lambat karena dapat dilakukan secara otomatis

### Algoritma Terbaik

Dua Algoritma Terbaik yang didapatkan untuk dataset ini yaitu Random Forest dan Decision Tree

| **Model**       | **Train Accuracy** | **K-Fold Accuracy** |
|-----------------|--------------------|---------------------|
| Random Forest   | 1.0000             | 0.9231              |
| Decision Tree   | 1.0000             | 0.8916              |


####  Random Forest
Kelebihan :
- Random Forest lebih tahan terhadap overfitting
- Random Forest lebih tahan terhadap outlier dan noise dalam data
- Cocok untuk dataset besar dan kompleks
- Random Forest dapat memberikan estimasi pentingnya setiap fitur dalam membuat prediksi

Kekurangan :
- Pelatihan Random Forest memakan waktu dan memori lebih besar karena melibatkan banyak pohon.
- Dibandingkan Decision Tree tunggal, Random Forest lebih sulit diinterpretasikan karena sifatnya sebagai ensemble.

#### Decision Tree
Kelebihan:
- Decision Tree bekerja dengan baik tanpa memerlukan normalisasi atau standarisasi data.
- Decision Tree secara dapat menangani data kategori tanpa memerlukan encoding khusus.

Kekurangan:
- Decision Tree cenderung overfit jika pohon terlalu dalam.
- Untuk dataset besar atau kompleks, Decision Tree dapat menjadi lambat.

## Kesimpulan
Berdasarkan hasil di atas, dapat dikatakan bahwa model mampu memprediksi persetujuan peminjaman selaras dengan hasil akurasi menggunakan algoritma Decision Tree yaitu 89% dan Random Forest sebesar 92%. Selain itu, dapat dilihat feature yang paling berpengaruh pada visualisasi data Univariate Analysis dan Correlation Matrix dimana person_income (pendapatan) dan loan_percent_income (jumlah pinjaman sebagai persentase dari pendapatan tahunan) merupakan faktor yang memiliki peran penting dalam penentuan persetujuan peminjaman. Oleh karena itu, dapat dikatakan bahwa analisis ini dapat menjawab dari kedua problem statements.

## Referensi 
Implementasi Algoritma K-Nearest Neigbor Untuk Klasifikasi Pengajuan Kredit [_(Sumber Referensi)] (https://doi.org/10.55338/justikpen.v1i2.6)
