import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 150

# Özellikler: metrekare, oda_sayisi, kat, bina_yasi
X = np.column_stack([
    np.random.normal(120, 30, n_samples),  # metrekare
    np.random.randint(1, 6, n_samples),    # oda_sayisi
    np.random.randint(1, 10, n_samples),   # kat
    np.random.randint(0, 40, n_samples)    # bina_yasi
])

# Fiyat kategorileri: 0: Düşük, 1: Orta, 2: Yüksek
Y = np.where(X[:, 0] < 100, 0,
            np.where(X[:, 0] < 150, 1, 2))

# Veriyi böl
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print("Eğitim veri seti boyutu=", len(X_train))
print("Test veri seti boyutu=", len(X_test))

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_tahmin = knn_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
dt_tahmin = dt_model.predict(X_test)

# Hata matrisleri
kategoriler = ['Düşük', 'Orta', 'Yüksek']

# KNN hata matrisi
plt.figure(figsize=(10, 6))
knn_hata = confusion_matrix(Y_test, knn_tahmin)
sns.heatmap(pd.DataFrame(knn_hata, kategoriler, kategoriler), annot=True)
plt.title('KNN Hata Matrisi')
plt.show()

# Decision Tree hata matrisi
plt.figure(figsize=(10, 6))
dt_hata = confusion_matrix(Y_test, dt_tahmin)
sns.heatmap(pd.DataFrame(dt_hata, kategoriler, kategoriler), annot=True)
plt.title('Decision Tree Hata Matrisi')
plt.show()