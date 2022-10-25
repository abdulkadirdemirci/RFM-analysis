################################
# iş problemi
################################
"""
Online ayakkabı mağazası olan FLO müşterilerini
segmentlere ayırıp bu segmentlere göre pazarlama
stratejileri belirlemek istiyor. Buna yönelik olarak
müşterilerin davranışları tanımlanacak ve bu
davranışlardaki öbeklenmelere göre gruplar oluşturulacak.
"""

################################
# veri seti hikayesi
################################
"""
Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

master_id          -Eşsiz müşteri numarası
order_channel      -Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
last_order_channel -En son alışverişin yapıldığı kanal
first_order_date   -Müşterinin yaptığı ilk alışveriş tarihi
last_order_date    -Müşterinin yaptığı son alışveriş tarihi
last_order_date_online       -Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_date_offline      -Müşterinin offline platformda yaptığı son alışveriş tarihi
order_num_total_ever_online  -Müşterinin online platformda yaptığı toplam alışveriş sayısı
order_num_total_ever_offline -Müşterinin offline'da yaptığı toplam alışveriş sayısı
customer_value_total_ever_offline  -Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online   -Müşterinin online alışverişlerinde ödediği toplam ücret
interested_in_categories_12        -Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

################################
# GÖREV1 : VERİYİ ANLAMA VE HAZIRLAMA
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################
# GÖREV1 - ADIM1 : flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
################################
df_ = pd.read_csv("C:/Users/Abdulkadir DEMİRCİ/Desktop/2022mvkpython/veri/FLO_RFM_Analizi/flo_data_20k.csv")
df = df_.copy()
################################
# GÖREV1 - ADIM2 : Veri setinde
# a. İlk 10 gözlem,
# b. Değişken isimleri,
# c. Betimsel istatistik,
# d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.
################################
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
df.head()
df.columns
df.describe().T
df.isnull().sum()
df.info()
################################
# GÖREV1 - ADIM3 : Omnichannel müşterilerin hem online'dan hemde offline
# platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz
################################
df.head(2)
df["all_over_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["all_over_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
################################
# GÖREV1 - ADIM4 :  Değişken tiplerini inceleyiniz.
# Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
################################
from datetime import date

df.info()
from datetime import datetime

df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
# -------------------------------------------------------------------------------------------------------------------
df['first_order_date'] = df['first_order_date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))
df['last_order_date'] = df['last_order_date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))
df['last_order_date_online'] = df['last_order_date_online'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))
df['last_order_date_offline'] = df['last_order_date_offline'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))
################################
# GÖREV1 - ADIM5 :  Alışveriş kanallarındaki müşteri sayısının,
# toplam alınan ürün sayısının ve
# toplam harcamaların dağılımına bakınız.
################################
df["order_channel"].nunique()
df["order_channel"].value_counts()
plt.pie(x=df["order_channel"].value_counts().values.tolist(),
        labels=df["order_channel"].value_counts().keys().tolist(),
        colors=["#ffdb00", "#ffa904", "#ee7b06", "#a12424"],
        autopct="%1.1f%%")
plt.show()

df.groupby("order_channel").agg({'all_over_order': ["sum"],
                                 'all_over_value': ["sum"],
                                 "order_channel": ["count"]}
                                )
################################
# GÖREV1 - ADIM6 :  En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
################################
df.sort_values(by="all_over_value", ascending=False).head(10)
################################
# GÖREV1 - ADIM7 :  En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
################################
df.sort_values(by="all_over_order", ascending=False).head(10)


################################
# GÖREV1 - ADIM8 : Veri ön hazırlık sürecini fonksiyonlaştırınız.
################################

def hazirlik(df, plot=False):
    print("*" * 50)
    print("genel bakış")
    print("*" * 50)
    df.head()
    df.shape
    df.columns
    df.describe().T
    df.isnull().sum()
    df.info()
    print("\n\n")
    print("*" * 50)
    print("birleştirilen değerler")
    print("*" * 50)
    df["all_over_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["all_over_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    print("birleştirmeler tamamlandı....")
    print("*" * 50)
    print("gereken degişkenleri date e çevirmek")
    print("*" * 50)
    from datetime import date
    df = df.astype({"first_order_date": "datetime64[ns]"})
    df = df.astype({"last_order_date": "datetime64[ns]"})
    df = df.astype({"last_order_date_online": "datetime64[ns]"})
    df = df.astype({"last_order_date_offline": "datetime64[ns]"})
    print(df.info())
    print("*" * 50)
    print("groupby altında incelemeler")
    print("*" * 50)
    print(df.groupby("order_channel").agg({'all_over_order': ["sum"],
                                           'all_over_value': ["sum"],
                                           "order_channel": ["count"]})
          )
    print("*" * 50)
    print("en fazla kazanc getiren ilk 10 müşteri")
    print("*" * 50)
    print(df.sort_values(by="all_over_value", ascending=False).head(10))
    print("*" * 50)
    print("en fazla alışveriş yapan ilk 10 müşteri")
    print("*" * 50)
    print(df.sort_values(by="all_over_order", ascending=False).head(10))
    if plot:
        plt.pie(x=df["order_channel"].value_counts().values.tolist(),
                labels=df["order_channel"].value_counts().keys().tolist(),
                colors=["#ffdb00", "#ffa904", "#ee7b06", "#a12424"],
                autopct="%1.1f%%")
    plt.show()


hazirlik(df, plot=True)

################################
# GÖREV2 : RFM METRİKLERİNİN HESAPLANMASI
################################
################################
# GÖREV2 - ADIM1: : Recency, Frequency ve Monetary tanımlarını yapınız.
################################
"""
Recency :
Yenilik olarak da tanımlanabilir. Birimi CLTV'nün aksine gündür.
Son alışverişin analiz tarihinden kaç gün önce yapıldığının bir ölçüsüdür.
[analiz_tarihi]-[son_alışveriş_tarihi] formülü ile bulunur.

Frequency :
Belirlenen bir zaman dilimin den başlayarak analiz tarihine kadar
müşterinin kaç kez alış veriş yaptığının bir göstergesidir.
Analizin periyoduna bağlı olarak aylık, 2 haftalık, 3 aylık gibi zaman dilimlerinde
müşteri frekansı değerlendirilebilir.

Monetary :
Müşterinin belirli zaman dilimi içerisinde yaptığı alışverişler 
neticesinde kuruma/şirkete kazandırdığı değerin parasal karşılığı.
Geçerli zaman dilimi içerisinde müşterinin aldığı ürünlerin fiyatları toplanarak elde edilir
"""
################################
# GÖREV2 - ADIM2-3- : Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
################################
df.shape
# veri setimiz 19945 gözlem ve 14 değişkenden oluşmaktadır.
# df["master_id"].nunique() 19945 den az çıkarsa çoklamalar vardır anlamına gelir
df["master_id"].nunique()
# çoklama yoktur her müşteri için sadece bir adet gözlem birimi vardır.
# veri setindeki en so alışveriş tarihini bul!
import datetime as dt
from datetime import datetime

df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df.info()
today = dt.datetime(2021, 6, 1)
rfm = pd.DataFrame({"recency": df["last_order_date"],
                    "frequency": df["all_over_order"],
                    "monetary": df["all_over_value"]}, columns=["recency", "frequency", "monetary"])
rfm["recency"] = pd.to_datetime(rfm["recency"])
rfm.info()
rfm["recency"] = rfm.apply(lambda row: (today - row["recency"]).days, axis=1)
# ------------------------------ same things but over df ----------------------------------------------------------------
df["recency"] = df.apply(lambda row: (today - row["last_order_date"]).days, axis=1)
df["frequency"] = df["all_over_order"]
df["monetary"] = df["all_over_value"]
df.head(2)
################################
# GÖREV3: RF Skorunun Hesaplanması
################################
################################
# GÖREV3 - ADIM1-2 : Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz
################################
rfm.head()
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
# ------------------------------ same things but over df ----------------------------------------------------------------
df["recency_score"] = pd.qcut(df["recency"], 5, labels=[5, 4, 3, 2, 1])
df["frequency_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df["monetary_score"] = pd.qcut(df["monetary"], 5, labels=[1, 2, 3, 4, 5])
df.head(2)
################################
# GÖREV3 - ADIM3 : recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
################################
rfm["RF_Score"] = rfm.apply(lambda row: "%s%s" % (row["recency_score"], row["frequency_score"]), axis=1)
# rfm["RF_Score"] = (rfm["recency_score"].astype("str")+rfm["frequency_score"].astype("str")
rfm.head()
rfm.info()
# ------------------------------ same things but over df ----------------------------------------------------------------
df["RF_Score"] = df.apply(lambda row: "%s%s" % (row["recency_score"], row["frequency_score"]), axis=1)
df.head(2)
################################
# GÖREV4 : RF Skorunun Segment Olarak Tanımlanması
################################
################################
# GÖREV4 - ADIM1 : Oluşturulan RF skorları için segment tanımlamaları yapınız.
################################
seg_map = {r"[1-2][5]": "cant_loose_them",
           r"[3-4][4-5]": "loyal_customers",
           r"[3][3]": "need_attention",
           r"[1-2][3-4]": "at_rist",
           r"[1-2][1-2]": "hibernating",
           r"[4-5][2-3]": "potential_royalist",
           r"[5][1]": "new_customers",
           r"[5][4-5]": "champions",
           r"[3][1-2]": "about_to_sleep",
           r"[4][1]": "promising"
           }
################################
# GÖREV4 - ADIM2 :  seg_map yardımı ile skorları segmentlere çeviriniz.
################################
rfm["Segment"] = rfm["RF_Score"]
rfm["Segment"] = rfm["Segment"].replace(seg_map, regex=True)
rfm.head(2)
# ------------------------------ same things but over df ----------------------------------------------------------------
df["Segment"] = df["RF_Score"]
df["Segment"] = df["Segment"].replace(seg_map, regex=True)
df.head(2)
################################
# GÖREV5 :  Aksiyon Zamanı !
################################
################################
# GÖREV5 - ADIM1 :  Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
################################
rfm.groupby("Segment").agg({"recency": ["mean", "count"],
                            "frequency": ["mean", "count"],
                            "monetary": ["mean", "count"]})
################################
# GÖREV5 - ADIM2-a :
# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
################################
df.loc[(df["interested_in_categories_12"].str.contains("KADIN")) & (
            (df["Segment"] == "loyal_customers") | (df["Segment"] == "champions"))]["master_id"]
################################
# GÖREV5 - ADIM2-b :
#  Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
################################
df.loc[((df["interested_in_categories_12"].str.contains("ERKEK")) |
        (df["interested_in_categories_12"].str.contains("COCUK"))) &
       ((df["Segment"] == "cant_loose_them") | (df["Segment"] == "about_to_sleep") | (df["Segment"] == "about_to_sleep")
        ]

df.loc[((df["interested_in_categories_12"].str.contains("ERKEK")) | (
    df["interested_in_categories_12"].str.contains("COCUK"))) & (
            (df["Segment"] == "cant_loose_them") | (df["Segment"] == "aboout_to_sleep") | (
                df["Segment"] == "new_customers"))]["master_id"]
