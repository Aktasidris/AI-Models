import csv
import os
import numpy as np
import librosa

# CSV dosyasının adı ve yolu
csv_file = "csv_files/All_label_emotion20_veriseti.csv"

# Veri Setini Oluşturma
def create_dataset(directory_path):
    # CSV dosyasını oluştur ve özellikleri kaydet
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10", "MFCC11",
             "MFCC12", "MFCC13","MFCC14", "MFCC15", "MFCC16", "MFCC17", "MFCC18",
             "MFCC19", "MFCC20","Emotion", "Gender"])

        for actor_folder in range(1, 16):
            actor_folder_name = f"Actor_{actor_folder:02d}"
            actor_dir = os.path.join(directory_path, actor_folder_name)
            for root, dirs, files in os.walk(actor_dir):
                for file in files:
                    if file.endswith(".flac"):
                        file_path = os.path.join(root, file)
                        #mfccs, energy, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, chroma_stft = extract_features(file_path)
                        mfcc= extract_features(file_path)
                        # Etiketleri al
                        file_parts = file.split("-")
                        emotion = int(file_parts[2])-1
                        gender = 0 if int(file_parts[6].split(".")[0]) % 2 == 0 else 1

                        # emotion data özellikleri CSV dosyasına yaz
                        #writer.writerow(list(mfccs) + [energy, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, chroma_stft, emotion, gender])
                        writer.writerow(list(mfcc) + [emotion,gender])
import soundfile as sf

def extract_features(file_path, num_mfcc=20, n_fft=2048, hop_length=512):
    try:
        # FLAC yerine diğer formatları da desteklemek için librosa.load kullan
        signal, samplerate = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # MFCC'leri çıkar
    mfccs = librosa.feature.mfcc(y=signal, sr=samplerate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_avg = np.mean(mfccs.T, axis=0).tolist()  # Listeye dönüştür

    return mfccs_avg
create_dataset("emotiondata/flac_files")

'''
# Veri Ön İşleme ve Özellik Çıkarma
def extract_features(file_path, num_mfcc=20, n_fft=2048, hop_length=512):
    # Ses dosyasını yükle
    signal, sr = librosa.load(file_path, sr=None)
    # MFCC özelliklerini çıkar
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc)
    mfccs_avg = np.mean(mfccs, axis=1)
    # Enerji özelliğini çıkar
    energy = np.mean(librosa.feature.rms(y=signal))
    # Spektral özellikleri çıkar
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
    # Zaman özelliklerini çıkar
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    # Frekans özelliklerini çıkar
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr))
    return mfccs_avg, energy, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, chroma_stft
'''
# Kullanım
create_dataset("emotiondata/flac_files")

'''
    # Diğer özellikleri çıkar
    energy = np.sum(signal ** 2) / len(signal)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=samplerate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=samplerate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=samplerate))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(signal))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=signal, sr=samplerate))
   
    #return mfccs_avg, energy, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, chroma_stft
'''