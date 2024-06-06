import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Membaca dataset transaksi dari file CSV
df = pd.read_csv('C:/Users/ACER/Documents/Pattern-Recognition/association-rule/transactions.csv')

# Memisahkan kolom 'Produk' menjadi daftar produk
df['Produk'] = df['Produk'].apply(lambda x: x.split(','))

# Mengubah data transaksi ke dalam format yang sesuai untuk algoritma Apriori
te = TransactionEncoder()
te_ary = te.fit(df['Produk']).transform(df['Produk'])
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

# Menjalankan algoritma Apriori
frequent_itemsets = apriori(df_transformed, min_support=0.2, use_colnames=True)

# Menemukan aturan asosiasi
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Menampilkan aturan asosiasi
print("Aturan Asosiasi yang Ditemukan:")
print(rules)

# Menyaring aturan asosiasi berdasarkan confidence dan lift yang tinggi
filtered_rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] >= 1.2)]

print("\nAturan Asosiasi yang Disaring (confidence >= 0.6 dan lift >= 1.2):")
print(filtered_rules)

# Menyusun kesimpulan
if not filtered_rules.empty:
    print("\nKesimpulan:")
    for _, rule in filtered_rules.iterrows():
        antecedents = ', '.join(rule['antecedents'])
        consequents = ', '.join(rule['consequents'])
        support = rule['support']
        confidence = rule['confidence']
        lift = rule['lift']
        print(f"Jika seseorang membeli {antecedents}, mereka cenderung juga membeli {consequents} (Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}).")
else:
    print("\nTidak ada aturan asosiasi yang memenuhi kriteria penyaringan.")
