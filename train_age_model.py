import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import warnings

# FitFailedWarning ve diğer olası uyarıları bastır
warnings.filterwarnings('ignore', category=UserWarning)

# Veriyi yükleyin
data = pd.read_csv('./csv_files/Age_sayısal_etiketlenmis_veri.csv')

# Çıkarılacak sütunlar
columns_to_drop = ['Energy', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff',
                   'Zero_Crossing_Rate', 'Chroma_STFT']

# Hedef değişken
target = 'Age'

# Özellik ve hedef değişkenleri ayırın
X = data.drop(columns=target)
y = data[target]

# Veri setini eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Başlangıç hiperparametre aralıkları
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_accuracy = 0
best_model = None
best_params = None
results = []

# Hiperparametre aralığını daraltarak iteratif olarak Grid Search
for i in range(5):  # 5 iterasyonla sınırlı
    print(f"Iteration {i+1}")
    # Random Forest modelini tanımla
    rf = RandomForestClassifier(random_state=42)
    # Grid Search ile en iyi hiperparametreleri bulma
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise')
    grid_search.fit(X_train, y_train)
    # En iyi modeli ve parametreleri al
    current_model = grid_search.best_estimator_
    current_params = grid_search.best_params_
    y_pred = current_model.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred)
    # Sonuçları kaydet
    results.append({
        'iteration': i + 1,
        'params': current_params,
        'accuracy': current_accuracy
    })
    print(f"Best params for iteration {i+1}: {current_params}")
    print(f"Accuracy for iteration {i+1}: {current_accuracy:.2f}")
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = current_model
        best_params = current_params
    # Hiperparametre aralıklarını daralt
    param_grid = {
        'n_estimators': [max(1, current_params['n_estimators'] - 100), current_params['n_estimators'], current_params['n_estimators'] + 100],
        'max_depth': [max(1, current_params['max_depth'] - 5), current_params['max_depth'], current_params['max_depth'] + 5],
        'min_samples_split': [max(2, current_params['min_samples_split'] - 1), current_params['min_samples_split'], current_params['min_samples_split'] + 1],
        'min_samples_leaf': [max(1, current_params['min_samples_leaf'] - 1), current_params['min_samples_leaf'], current_params['min_samples_leaf'] + 1]
    }
# Modeli ve doğruluğunu .pkl dosyasına kaydetme
model_filename = f'emotion_{best_accuracy:.2f}.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Best model saved as {model_filename} with accuracy: {best_accuracy:.2f}")
print(f"Best parameters: {best_params},{best_accuracy}")
# Sonuçları bir DataFrame'e dönüştür
results_df = pd.DataFrame(results)

# İterasyon başına doğruluk değerlerini görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(results_df['iteration'], results_df['accuracy'], marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy per Iteration')
plt.grid(True)
plt.show()

# Her iterasyondaki en iyi parametreleri ve doğruluk değerlerini yazdırma
for result in results:
    print(f"Iteration {result['iteration']}:")
    print(f"Params: {result['params']}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print("")

print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy:.2f}")