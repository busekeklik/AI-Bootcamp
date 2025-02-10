import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(train_file, test_file):
    # Veri setlerini yükleme
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    # Eğitim ve test verilerini ayırma
    X_train = np.array(train)[:, :187]  # Özellikler
    y_train = np.array(train)[:, 187]   # Etiketler
    X_test = np.array(test)[:, :187]    # Test özellikleri
    y_test = np.array(test)[:, 187]     # Test etiketleri
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    print("\n=== Naive Bayes Sınıflandırma ===")
    # Naive Bayes modelini oluşturma ve eğitme
    nb_model = CategoricalNB()
    nb_model.fit(X_train, y_train)
    
    # Tahmin yapma
    y_pred = nb_model.predict(X_test)
    
    # Sonuçları değerlendirme
    plot_confusion_matrix(y_test, y_pred, "Naive Bayes Confusion Matrix")
    print(f"Naive Bayes Doğruluk: {accuracy_score(y_test, y_pred):.4f}")

def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    print("\n=== Decision Tree Sınıflandırma ===")
    # Decision Tree modelini oluşturma
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # Cross-validation değerlendirmesi
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=10)
    print(f"Cross-Validation Skorları: {cv_scores}")
    print(f"Ortalama CV Doğruluğu: {np.mean(cv_scores):.4f}")
    
    # Modeli tüm veri setiyle eğitme
    dt_model.fit(X_train, y_train)
    
    # Test verileri üzerinde tahmin
    y_pred = dt_model.predict(X_test)
    
    # Sonuçları değerlendirme
    plot_confusion_matrix(y_test, y_pred, "Decision Tree Confusion Matrix")
    print(f"Decision Tree Test Doğruluğu: {accuracy_score(y_test, y_pred):.4f}")

def plot_confusion_matrix(y_true, y_pred, title):
    # Karmaşıklık matrisini oluşturma
    cm = confusion_matrix(y_true, y_pred)
    
    # Görselleştirme için etiketler
    labels = ['No', 'S', 'V', 'F', 'Q']
    
    # DataFrame oluşturma
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.show()

def main():
    # Veri setlerini yükleme
    X_train, y_train, X_test, y_test = load_and_prepare_data(
        "mitbih_train.csv", 
        "mitbih_test.csv"
    )
    
    # Her iki modeli de eğitip değerlendirme
    train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test)
    train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()