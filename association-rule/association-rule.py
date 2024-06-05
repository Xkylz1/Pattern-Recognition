# Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Membaca dataset dari file CSV (pastikan lokasi file CSV yang benar)
df = pd.read_csv('C:/Users/ACER/Documents/Pattern-Recognition/association-rule\dataset.csv')

# Menampilkan nama kolom untuk memeriksa kesalahan
print(df.columns)

# Menghapus leading/trailing spaces dari nama kolom
df.columns = df.columns.str.strip()

# Pastikan kolom 'Produk' ada di dalam DataFrame
if 'Produk' in df.columns:
    # Memisahkan kolom 'Produk' menjadi kolom item yang terpisah
    df_products = df['Produk'].str.get_dummies(',')

    # Menggabungkan DataFrame asli dengan DataFrame biner
    df = df.drop(columns=['ID Transaksi', 'Produk']).join(df_products)

    # Menampilkan DataFrame setelah transformasi
    print(df.head())

    # Menggunakan algoritma Apriori untuk mencari frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

    # Menghitung association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filter association rules based on desired conditions
    rice_eggs_rules = rules[rules['consequents'].astype(str).str.contains("Telur") & rules['consequents'].astype(str).str.contains("Beras")]

    # Menampilkan hasil aturan yang relevan
    print("\nAssociation Rules (Eggs and Rice):")
    print(rice_eggs_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

else:
    print("Kolom 'Produk' tidak ditemukan di dalam DataFrame.")
