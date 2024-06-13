from emotion_extract_features import extract_features

# Duygu ve cinsiyet etiketleri
emotion_labels = {
    "0": "Neutral-Normal",
    "1": "Calm-Sakin",
    "2": "Happy-Mutlu",
    "3": "Sad-Üzgün",
    "4": "Angry-Kızgın",
    "5": "Fearful-Korkmuş",
    "6": "Disgust-Nefret",
    "7": "Surprised-Şaşkın"
}

gender_labels = {
    "0": "Female-Kadın",
    "1": "Male-Erkek"
}

# Yaş gruplarını etiketleri
age_labels = {
    "0": 'child(0-14)',
    "1": 'Young(15-24)',
    "2": 'Middle age(25-34)',
    "3": 'Old(35-65)'

}
import os
import joblib
import numpy as np

def predict(file_path, model_path):
    # Özellikleri çıkar
    features = extract_features(file_path)
    if features is None:
        return None, None, None
    # Özellikleri düz bir listeye dönüştür
    features_flattened = []
    for feature in features:
        if isinstance(feature, list):
            features_flattened.extend(feature)
        else:
            features_flattened.append(feature)
    features_flattened = np.array(features_flattened).reshape(1, -1)
    # Modelleri yükle
    emotion_model = joblib.load(os.path.join(model_path, 'models/emotion_0.72.pkl'))  # 0.71 doğruluk
    gender_model = joblib.load(os.path.join(model_path, 'gender_0.95.pkl'))  # 0.95 doğruluk
    age_model = joblib.load(os.path.join(model_path, 'Age_0.79.pkl'))#0.79 doğruluk
    # Tahminleri yap
    emotion_prediction = emotion_model.predict(features_flattened)[0]
    gender_prediction = gender_model.predict(features_flattened)[0]
    age_prediction = age_model.predict(features_flattened)[0]
    # Etiketleri al
    emotion_label = emotion_labels.get(str(emotion_prediction), "unknown")
    gender_label = gender_labels.get(str(gender_prediction), "unknown")
    age_label = age_labels.get(str(age_prediction), "unknown")
    # Tahmin edilen ham değerleri yazdırma
    print(f"Raw gender prediction: {gender_prediction}")
    print(f"Raw age prediction: {age_prediction}")

    return emotion_label, gender_label, age_label



'''
# Örnek kullanım
file_path = 'C:/Users/idris aktas/Desktop/BitirmeProjesi/agedata/1_24_f.flac'
model_path = 'models'
emotion, gender, age = predict(file_path, model_path)
if emotion is not None:
    print(f"Predicted Emotion: {emotion}, Gender: {gender}, Age: {age}")
else:
    print("Feature extraction failed.")
'''