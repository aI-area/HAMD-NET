"""Importing Libraries"""

#General
import pandas as pd
import numpy as np
import scipy
import scipy.io
import os
import zipfile
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import random
from scipy.stats import ttest_ind, f_oneway
import scipy.stats as stats
from scipy.signal import welch


#Deep Learning
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Reshape, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
###########
"""Mounting Google Drive"""
from google.colab import drive
drive.mount('/content/drive', force_remount = True)

zip_path = "/content/drive/MyDrive/Alzheimer and Frontotemporal/Alzheimers BIDS Validation.zip"
extract_to = "/tmp/Alzheimer and Frontotemporal/"

def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The file {zip_path} does not exist.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to {extract_to}")

unzip_file(zip_path, extract_to)

#######
extract_to = "/tmp/Alzheimer and Frontotemporal/"
participants_file = "/content/drive/MyDrive/Alzheimer and Frontotemporal/participants.tsv"

participants_data = pd.read_csv(participants_file, sep='\t')

fronto_data = []
fronto_labels = []
###Frontotemporal_Dementia
def get_label(group):
    if group == "F":
        return "Frontotemporal_Dementia"
    elif group == "C":
        return "Healthy"
    else:
        return None

for participant_id, group in zip(participants_data["participant_id"], participants_data["Group"]):
    label = get_label(group)
    if label:
        folder_path = os.path.join(extract_to, participant_id, "eeg")
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".set"):
                    file_path = os.path.join(folder_path, file_name)

                    try:
                        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
                        data = raw_data.get_data()

                        fronto_data.append(data)
                        fronto_labels.append(label)

                        print(f"Loaded: {file_path}, Label: {label}, Data Shape: {data.shape}")
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")

print("Total files loaded:", len(fronto_data))
print("Labels distribution:", pd.Series(fronto_labels).value_counts())
min_sample_length = min(data.shape[1] for data in fronto_data)
fronto_data = np.array([data[:, :min_sample_length] for data in fronto_data])
########
extract_to = "/tmp/Alzheimer and Frontotemporal/"
participants_file = "/content/drive/MyDrive/Alzheimer and Frontotemporal/participants.tsv"

participants_data = pd.read_csv(participants_file, sep='\t')

ad_data = []
ad_labels = []

def get_label(group):
    if group == "A":
        return "Alzheimer"
    elif group == "C":
        return "Healthy"
    else:
        return None

for participant_id, group in zip(participants_data["participant_id"], participants_data["Group"]):
    label = get_label(group)
    if label:
        folder_path = os.path.join(extract_to, participant_id, "eeg")
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".set"):
                    file_path = os.path.join(folder_path, file_name)

                    try:
                        raw_data = mne.io.read_raw_eeglab(file_path, preload=True)
                        data = raw_data.get_data()

                        ad_data.append(data)
                        ad_labels.append(label)

                        print(f"Loaded: {file_path}, Label: {label}, Data Shape: {data.shape}")
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")

print("Total files loaded:", len(ad_data))
print("Labels distribution:", pd.Series(ad_labels).value_counts())
#############
min_sample_length = min(data.shape[1] for data in ad_data)
ad_data = np.array([data[:, :min_sample_length] for data in ad_data])
###########


main_folder = "/content/drive/MyDrive/BIDA Validation/"

pd_data = []
pd_labels = []

def get_label(folder_name):
    if "sub-hc" in folder_name:
        return "Healthy"
    elif "sub-pd" in folder_name:
        return "PD"
    else:
        return None

for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    if os.path.isdir(subfolder_path) and subfolder.startswith("sub-"):
        label = get_label(subfolder)

        if label:
            for sub_subfolder in os.listdir(subfolder_path):
                eeg_folder = os.path.join(subfolder_path, sub_subfolder, "eeg")

                if os.path.isdir(eeg_folder):
                    for file_name in os.listdir(eeg_folder):
                        if file_name.endswith(".bdf"):
                            file_path = os.path.join(eeg_folder, file_name)

                            raw_data = mne.io.read_raw_bdf(file_path, preload=True)
                            data = raw_data.get_data()  # Shape (n_channels, n_samples)

                            pd_data.append(data)
                            pd_labels.append(label)

                            print(f"Loaded: {file_path}, Label: {label}, Data Shape: {data.shape}")
                            ####
min_sample_length = min(data.shape[1] for data in pd_data)
pd_data = np.array([data[:, :min_sample_length] for data in pd_data])
###
zip_path = "/content/drive/MyDrive/Schizophrenia.zip"
extract_to = "/tmp/Schizophrenia/"

def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The file {zip_path} does not exist.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to {extract_to}")

unzip_file(zip_path, extract_to)

def process_edf_files(folder_path):
    sch_data = []
    sch_labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".edf"):
            file_path = os.path.join(folder_path, file_name)

            # Read EDF file
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data()

            # Append the data
            sch_data.append(data)

            # Label: 0 for healthy (h), 1 for schizophrenia (s)
            if file_name.startswith("h"):
                sch_labels.append(0)
            elif file_name.startswith("s"):
                sch_labels.append(1)

    return sch_data, sch_labels

sch_data, sch_labels = process_edf_files(extract_to)
###
min_sample_length = min(data.shape[1] for data in sch_data)
sch_data = np.array([data[:, :min_sample_length] for data in sch_data])
# Step 1: Homogenize channels to 19
def homogenize_channels(data, target_channels=19):
    if data.shape[1] > target_channels:
        return data[:, :target_channels, :]  # Truncate extra channels
    elif data.shape[1] < target_channels:
        # Pad with zeros if fewer channels
        padding = target_channels - data.shape[1]
        return np.pad(data, ((0, 0), (0, padding), (0, 0)), mode='constant')
    else:
        return data

pd_data_homogenized = homogenize_channels(pd_data)

# Step 2: Homogenizing samples to the minimum samples across datasets
min_samples = min(ad_data.shape[2], fronto_data.shape[2], pd_data_homogenized.shape[2], sch_data.shape[2])

def homogenize_samples(data, target_samples):
    if data.shape[2] > target_samples:
        return data[:, :, :target_samples]  # Truncate extra samples
    elif data.shape[2] < target_samples:
        padding = target_samples - data.shape[2]
        return np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant')
    else:
        return data

ad_data_homogenized = homogenize_samples(ad_data, min_samples)
fronto_data_homogenized = homogenize_samples(fronto_data, min_samples)
pd_data_homogenized = homogenize_samples(pd_data_homogenized, min_samples)
sch_data_homogenized = homogenize_samples(sch_data, min_samples)

# Step 3: Combining all data and labels
all_data = np.concatenate([ad_data_homogenized, fronto_data_homogenized, pd_data_homogenized, sch_data_homogenized], axis=0)
all_labels = np.concatenate([ad_labels, fronto_labels, pd_labels, sch_labels], axis=0)

print("Combined Data Shape:", all_data.shape)
print("Combined Labels Shape:", all_labels.shape)
label_mapping = {
    '0': 'Healthy',
    'Healthy': 'Healthy',
    '1': 'Schizophrenia',
    'Alzheimer': 'Alzheimer',
    'Frontotemporal_Dementia': 'FrontotemporalDementia',
    'PD': 'Parkinsons'
}

all_labels = np.array([label_mapping[label] for label in all_labels])
from scipy.signal import stft, istft

def extract_and_combine_signals(data1, data2, fs):
    f1, t1, Zxx1 = stft(data1, fs=fs, nperseg=256)
    f2, t2, Zxx2 = stft(data2, fs=fs, nperseg=256)

    power1 = np.mean(np.abs(Zxx1), axis=1)
    power2 = np.mean(np.abs(Zxx2), axis=1)

    freq_range1 = (f1[np.where(power1 > 0.05 * np.max(power1))[0][0]],
                   f1[np.where(power1 > 0.05 * np.max(power1))[0][-1]])
    freq_range2 = (f2[np.where(power2 > 0.05 * np.max(power2))[0][0]],
                   f2[np.where(power2 > 0.05 * np.max(power2))[0][-1]])

    print(f"Disease 1 Frequency Range: {freq_range1}")
    print(f"Disease 2 Frequency Range: {freq_range2}")

    mask1 = (f1 >= freq_range1[0]) & (f1 <= freq_range1[1])
    mask2 = (f2 >= freq_range2[0]) & (f2 <= freq_range2[1])

    Zxx1_normalized = Zxx1 / np.max(np.abs(Zxx1)) if np.max(np.abs(Zxx1)) != 0 else Zxx1
    Zxx2_normalized = Zxx2 / np.max(np.abs(Zxx2)) if np.max(np.abs(Zxx2)) != 0 else Zxx2

    combined_Zxx = np.zeros_like(Zxx1)

    combined_Zxx[mask1, :] += Zxx1_normalized[mask1, :]

    combined_Zxx[mask2, :] += Zxx2_normalized[mask2, :]

    _, combined_signal = istft(combined_Zxx, fs=fs)
    return combined_signal
###
def synthesize_data_shape_preserved(all_data, all_labels, fs=256):
    synthesized_data = list(all_data)
    synthesized_labels = list(all_labels)

    unique_labels = [label for label in np.unique(all_labels) if label != "Healthy"]

    max_combinations = 3

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            idx1 = np.where(all_labels == label1)[0][:max_combinations]
            idx2 = np.where(all_labels == label2)[0][:max_combinations]

            for sample1 in idx1:
                for sample2 in idx2:
                    combined_sample = []
                    for ch in range(all_data.shape[1]):
                        combined_signal = extract_and_combine_signals(
                            all_data[sample1, ch],
                            all_data[sample2, ch],
                            fs
                        )
                        combined_signal = np.pad(combined_signal,
                                                 (0, max(0, all_data.shape[2] - len(combined_signal))),
                                                 mode='constant')[:all_data.shape[2]]
                        combined_sample.append(combined_signal)

                    synthesized_data.append(np.array(combined_sample))
                    synthesized_labels.append(f"{label1}_{label2}")

    return np.array(synthesized_data), np.array(synthesized_labels)

fs = 512
synthesized_data, synthesized_labels = synthesize_data_shape_preserved(all_data, all_labels, fs=fs)

combined_data = np.concatenate([all_data, synthesized_data], axis=0)
combined_labels = np.concatenate([all_labels, synthesized_labels], axis=0)

print("Combined Data Shape:", combined_data.shape)
print("Combined Labels Shape:", combined_labels.shape)
###
import matplotlib.pyplot as plt
from scipy.signal import stft

def plot_stft(signal, fs, title, ax):
    """
    Plots the STFT of a given signal on a provided axis with frequencies bounded between 0 and 45 Hz.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=256)
    f_limit_idx = np.where(f <= 45)
    ax.pcolormesh(t, f[f_limit_idx], np.abs(Zxx[f_limit_idx]), shading='gouraud', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

def visualize_combined_signals(all_data, all_labels, synthesized_data, synthesized_labels, fs):
    unique_combinations = np.unique([label for label in synthesized_labels if "_" in label])

    fig, axes = plt.subplots(len(unique_combinations), 3, figsize=(15, 5 * len(unique_combinations)))

    for idx, combination in enumerate(unique_combinations):
        try:
            disease1, disease2 = combination.split("_", 1)  # Split only at the first underscore

            print(disease1)
            print(disease2)

            idx1 = np.where(all_labels == disease1)[0][0]
            idx2 = np.where(all_labels == disease2)[0][0]

            combined_idx = np.where(synthesized_labels == combination)[0][0]

            plot_stft(all_data[idx1, 0], fs, f"{disease1} (Original)", axes[idx, 0])
            plot_stft(all_data[idx2, 0], fs, f"{disease2} (Original)", axes[idx, 1])
            plot_stft(synthesized_data[combined_idx, 0], fs, f"{combination} (Combined)", axes[idx, 2])
        except Exception as e:
            print(f"Error processing combination {combination}: {e}")

    plt.tight_layout()
    plt.show()

fs = 512
visualize_combined_signals(all_data, all_labels, synthesized_data, synthesized_labels, fs)
###
import numpy as np

# Mapping dictionary for synthesized_labels
synthesized_mapping = {
    'Alzheimer': 'AD',
    'Alzheimer_FrontotemporalDementia': 'AD + FTD',
    'Alzheimer_Parkinsons': 'AD + PD',
    'Alzheimer_Schizophrenia': 'AD + SCZ',
    'FrontotemporalDementia': 'FTD',
    'FrontotemporalDementia_Parkinsons': 'FTD + PD',
    'FrontotemporalDementia_Schizophrenia': 'FTD + SCZ',
    'Healthy': 'Healthy',
    'Parkinsons': 'PD',
    'Parkinsons_Schizophrenia': 'PD + SCZ',
    'Schizophrenia': 'SCZ'
}

# Updating synthesized_labels
synthesized_labels = np.array([synthesized_mapping[label] for label in synthesized_labels])

# Mapping dictionary for all_labels
all_labels_mapping = {
    'Alzheimer': 'AD',
    'FrontotemporalDementia': 'FTD',
    'Healthy': 'Healthy',
    'Parkinsons': 'PD',
    'Schizophrenia': 'SCZ'
}

# Updating all_labels
all_labels = np.array([all_labels_mapping[label] for label in all_labels])

# Printing the updated labels to verify
print("Updated synthesized_labels:", np.unique(synthesized_labels))
print("Updated all_labels:", np.unique(all_labels))
###
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

def plot_stft(signal, fs, title, ax):
    """
    Plots the STFT of a given signal on a provided axis with frequencies bounded
    dynamically based on signal content.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=256)

    magnitude_spectrum = np.abs(Zxx)
    max_freq_idx = np.where(magnitude_spectrum.mean(axis=1) > 0.01 * magnitude_spectrum.max())[0][-1]
    freq_limit = f[max_freq_idx] if max_freq_idx < len(f) else 45

    f_limit_idx = np.where(f <= freq_limit)
    ax.pcolormesh(t, f[f_limit_idx], magnitude_spectrum[f_limit_idx], shading='gouraud', cmap='viridis')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Frequency [Hz]", fontsize=16)
    ax.set_ylim(0, freq_limit)

    ax.tick_params(axis='both', which='major', labelsize=18)

def visualize_combined_signals(all_data, all_labels, synthesized_data, synthesized_labels, fs, save_path="STFT_Comparison.pdf"):
    """
    Visualizes STFT of individual and combined signals for different disease categories.
    Dynamically adjusts frequency limits and saves the output as a .pdf.
    """
    unique_combinations = np.unique([label for label in synthesized_labels if " + " in label])

    if len(unique_combinations) == 0:
        print("No valid multi-disease combinations found. Skipping visualization.")
        return

    fig, axes = plt.subplots(len(unique_combinations), 3, figsize=(15, 5 * len(unique_combinations)))

    for idx, combination in enumerate(unique_combinations):
        try:
            disease1, disease2 = combination.split(" + ", 1)  # Split using " + "

            idx1 = np.where(all_labels == disease1)[0][0]
            idx2 = np.where(all_labels == disease2)[0][0]
            combined_idx = np.where(synthesized_labels == combination)[0][0]

            plot_stft(all_data[idx1, 0], fs, f"{disease1} (Original)", axes[idx, 0])
            plot_stft(all_data[idx2, 0], fs, f"{disease2} (Original)", axes[idx, 1])
            plot_stft(synthesized_data[combined_idx, 0], fs, f"{combination} (Combined)", axes[idx, 2])
        except Exception as e:
            print(f"Error processing combination {combination}: {e}")

    plt.tight_layout()

    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    plt.show()
###
fs = 512
visualize_combined_signals(all_data, all_labels, synthesized_data, synthesized_labels, fs)
def rescale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = 2 * (data - min_val) / (max_val - min_val) - 1  # Scale to [-1, 1]
    return scaled_data

combined_data = rescale_data(combined_data)
###
from collections import Counter

label_counts = Counter(combined_labels)

for label, count in label_counts.items():
    print(f"{label}: {count}")
from imblearn.over_sampling import SMOTE

original_shape = combined_data.shape
flattened_data = combined_data.reshape(combined_data.shape[0], -1)

healthy_indices = np.where(combined_labels == "Healthy")[0]
other_indices = np.where(combined_labels != "Healthy")[0]

data_healthy = flattened_data[healthy_indices]
labels_healthy = combined_labels[healthy_indices]

data_other = flattened_data[other_indices]
labels_other = combined_labels[other_indices]

smote = SMOTE()
balanced_data_other, balanced_labels_other = smote.fit_resample(data_other, labels_other)

final_data = np.concatenate([data_healthy, balanced_data_other], axis=0)
final_labels = np.concatenate([labels_healthy, balanced_labels_other], axis=0)

final_data = final_data.reshape(-1, original_shape[1], original_shape[2])

print("Balanced Class Distribution:", Counter(final_labels))
print("Balanced Data Shape:", final_data.shape)

del combined_data, flattened_data, data_healthy, data_other
import pickle

save_path = '/content/drive/MyDrive/multiple_diseases.pkl'

with open(save_path, 'wb') as file:
    pickle.dump((final_data, final_labels), file)

print(f"Balanced data saved to: {save_path}")
