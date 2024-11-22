# Predictive Analytics : Loan Approval Prediction - Wahyu Dhia Satwika

## Domain Proyek : Keuangan
Industri perbankan menghadapi tantangan besar dalam menentukan kelayakan pemohon untuk mendapatkan pinjaman. Semakin banyaknya permohonan pinjaman membuat lembaga keuangan perlu memiliki pendekatan yang efisien dan efektif untuk memutuskan persetujuan atau penolakan pinjaman, sambil mengurangi risiko kredit macet. Melalui penggunaan machine learning dan analisis data, lembaga keuangan dapat membuat keputusan lebih cerdas, mempercepat proses persetujuan pinjaman, dan meminimalkan risiko keuangan. Prediksi menggunakan teknik machine learning diperlukan yang bertujuan untuk menghindari permasalahan  kredit  macet  kedepannya (Sitepu & Manohar, 2022). 

## Business Understanding

### Problem Statements
- Bagaimana cara menentukan kelayakan persetujuan pinjaman secara otomatis untuk mengurangi waktu proses dan meminimalkan risiko kredit macet?
- Faktor apa saja yang paling berpengaruh dalam menentukan apakah pinjaman layak disetujui?

### Goals 
- Mengembangkan model prediksi yang mampu menentukan status persetujuan pinjaman dengan akurasi tinggi.
- Mengidentifikasi faktor-faktor kunci yang memengaruhi kelayakan pinjaman sehingga dapat digunakan untuk mengoptimalkan proses persetujuan.

### Solution Statements 
- Membangun model machine learning untuk klasifikasi persetujuan pinjaman.
- Menganalisis variabel yang ada di dalam dataset untuk menemukan fitur-fitur yang paling berpengaruh dalam keputusan persetujuan pinjaman.

## Data Understanding
Dataset Loan Approval yang berasal dari kaggle merupakan sebuah dataset sintetis yang berdasarkan dari [_(Sumber Utama)_] (https://www.kaggle.com/datasets/laotse/credit-risk-dataset). Dataset Loan Approval memiiliki 45000 records dengan 14 variabel, dengan rincian sebagai berikut:

**Informasi Dataset**
|Judul|Loan Approval Classification|
|Author | Ta-WEI LO|
|Source||[Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
|Visibility| Public|
|Usability|10.00|

**Metadata**
| Kolom                         | Deskripsi                                         | Tipe         |
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


## Referensi 
Implementasi Algoritma K-Nearest Neigbor Untuk Klasifikasi Pengajuan Kredit [_(Sumber Referensi)] (https://doi.org/10.55338/justikpen.v1i2.6)