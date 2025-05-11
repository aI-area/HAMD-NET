

# Importing libraries
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

# Importing libraries
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

# Mounting Google Drive for data access
from google.colab import drive
drive.mount('/content/drive', force_remount = True)

# Loading single disease EEG data
import pickle
save_path = '/content/drive/MyDrive/single_diseases.pkl'

def load_balanced_data(file_path):
    with open(file_path, 'rb') as file:
        data, labels = pickle.load(file)
    return data, labels

single_data, single_labels = load_balanced_data(save_path)
print("Loaded Data Shape:", single_data.shape)
print("Loaded Labels Shape:", len(single_labels))

# Loading multiple disease EEG data
save_path = '/content/drive/MyDrive/multiple_diseases.pkl'
multi_data, multi_labels = load_balanced_data(save_path)
print("Loaded Data Shape:", multi_data.shape)
print("Loaded Labels Shape:", len(multi_labels))

# Combining and shuffling datasets
combined_data = np.concatenate((single_data, multi_data), axis=0)
combined_labels = np.concatenate((single_labels, multi_labels), axis=0)

indices = np.arange(combined_data.shape[0])
np.random.shuffle(indices)
combined_data = combined_data[indices]
combined_labels = combined_labels[indices]

del single_data, single_labels, multi_data, multi_labels

print("Combined Data Shape:", combined_data.shape)
print("Combined Labels Shape:", combined_labels.shape)

# ANOVA analysis
from scipy.stats import f_oneway
data = combined_data
labels = combined_labels

unique_diseases = np.unique(labels)
num_diseases = len(unique_diseases)

# Extracting statistical features
def extract_features(sample):
    mean_power = np.mean(np.abs(sample))
    variance = np.var(sample)
    skewness = np.mean((sample - mean_power) ** 3) / (np.std(sample) ** 3)

    return np.array([mean_power, variance, skewness])

feature_matrix = np.array([extract_features(sample) for sample in data])

# ANOVA matrix for disease comparison
columns = ['Global_MeanPower', 'Global_Variance', 'Global_Skewness']
df = pd.DataFrame(feature_matrix, columns=columns)
df['Disease'] = labels

anova_matrix = np.zeros((num_diseases, num_diseases))

for i in range(num_diseases):
    for j in range(i, num_diseases):
        disease_1 = unique_diseases[i]
        disease_2 = unique_diseases[j]

        group1 = df[df['Disease'] == disease_1][columns]
        group2 = df[df['Disease'] == disease_2][columns]

        f_stat, p_values = f_oneway(group1, group2)

        avg_p_value = np.mean(p_values)

        anova_matrix[i, j] = avg_p_value
        anova_matrix[j, i] = avg_p_value

anova_df = pd.DataFrame(anova_matrix, index=unique_diseases, columns=unique_diseases)
anova_df.to_csv('Anova_df.csv', index = False)

# Calculating phase and coherence measures

from scipy.signal import hilbert, coherence

# Defining brain regions and channels
region_map = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
    "Temporal": ["T3", "T4", "T5", "T6"],
    "Parietal": ["P3", "P4", "Pz"],
    "Occipital": ["O1", "O2"],
    "Central": ["C3", "C4", "Cz"]
}

channel_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T3", "T4", "T5", "T6",
                 "P3", "P4", "Pz", "O1", "O2", "C3", "C4", "Cz"]
channel_indices = {ch: idx for idx, ch in enumerate(channel_names)}

# Computing connectivity matrices between brain regions
def compute_regional_matrix(matrix, region_map, channel_indices):
    region_values = {region: [channel_indices[ch] for ch in channels if ch in channel_indices] for region, channels in region_map.items()}
    num_regions = len(region_map)
    regional_matrix = np.zeros((num_regions, num_regions))

    region_list = list(region_map.keys())

    for i, region1 in enumerate(region_list):
        for j, region2 in enumerate(region_list):
            indices1 = region_values[region1]
            indices2 = region_values[region2]
            region_values_matrix = matrix[np.ix_(indices1, indices2)]
            regional_matrix[i, j] = np.mean(region_values_matrix)

    return regional_matrix, region_list

# Calculating phase locking value
def compute_plv(data):
    n_channels, n_timepoints = data.shape
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase1 = np.angle(hilbert(data[i, :]))
            phase2 = np.angle(hilbert(data[j, :]))
            plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix

# Calculating coherence between channels
def compute_coherence(data, fs=256):
    n_channels, n_timepoints = data.shape
    coh_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            f, Cxy = coherence(data[i, :], data[j, :], fs=fs)
            coh_matrix[i, j] = np.mean(Cxy)
            coh_matrix[j, i] = coh_matrix[i, j]

    return coh_matrix

# Processing sample data and saving results
sample_data = data[:10]
plv_results = np.mean([compute_plv(sample) for sample in sample_data], axis=0)
coh_results = np.mean([compute_coherence(sample) for sample in sample_data], axis=0)

regional_plv_matrix, region_labels = compute_regional_matrix(plv_results, region_map, channel_indices)
regional_coh_matrix, _ = compute_regional_matrix(coh_results, region_map, channel_indices)

plv_df = pd.DataFrame(regional_plv_matrix, index=region_labels, columns=region_labels)
coh_df = pd.DataFrame(regional_coh_matrix, index=region_labels, columns=region_labels)

# Saving connectivity heatmaps
def save_heatmap(matrix, filename):
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xlabel("Brain Region")
    plt.ylabel("Brain Region")
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

save_heatmap(plv_df, "PLV_Brain_Regions")
save_heatmap(coh_df, "Coherence_Brain_Regions")

plv_pdf_path = "PLV_Brain_Regions.pdf"
coh_pdf_path = "Coherence_Brain_Regions.pdf"
print(f"PLV Matrix saved at: {plv_pdf_path}")
print(f"Coherence Matrix saved at: {coh_pdf_path}")


import numpy as np
from scipy.signal import resample

original_sampling_rate = 512
target_sampling_rate = 64
downsample_factor = original_sampling_rate // target_sampling_rate

frame_length_seconds = 3
frame_length_samples = frame_length_seconds * target_sampling_rate
overlap_percentage = 0.2
overlap_samples = int(frame_length_samples * overlap_percentage)

all_sample_frames = []

for sample in combined_data:
    downsampled_sample = resample(sample, sample.shape[1] // downsample_factor, axis=1)

    sample_frames = []
    for start in range(0, downsampled_sample.shape[1] - frame_length_samples + 1, frame_length_samples - overlap_samples):
        end = start + frame_length_samples
        sample_frames.append(downsampled_sample[:, start:end])

    all_sample_frames.append(np.array(sample_frames))

all_sample_frames = np.array(all_sample_frames)
print("Framed data shape:", all_sample_frames.shape)

del combined_data

# Applying data augmentation to EEG frames
from imgaug import augmenters as iaa

def DataAugmentation(Data, Labels):
  augmentation = iaa.Sequential([
    iaa.Flipud(p=0.05),
    iaa.Affine(rotate=(-10, 10)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.005)),
])

  augmented_features = []
  augmented_labels = []

  for i in range(len(Data)):
      feature = Data[i]
      label = Labels[i]

      augmented_feature = augmentation.augment_image(feature)

      augmented_features.append(augmented_feature)
      augmented_labels.append(label)

      for _ in range(10):
          augmented_feature = augmentation.augment_image(feature)
          augmented_features.append(augmented_feature)
          augmented_labels.append(label)

  augmented_features = np.array(augmented_features)
  augmented_labels = np.array(augmented_labels)

  print(augmented_features.shape)
  print(augmented_labels.shape)

  return augmented_features, augmented_labels

# Normalizing augmented data
all_sample_frames = all_sample_frames.astype(np.float32)
augmented_features, augmented_labels = DataAugmentation(all_sample_frames, combined_labels)

mean = np.mean(augmented_features, axis=(0, 2))
std = np.std(augmented_features, axis=(0, 2))
normalized_data = (augmented_features - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]

del all_sample_frames, augmented_features


import gc
gc.collect()

# Encoding disease labels
encoder = OneHotEncoder(sparse_output=False)
encoded_labels = encoder.fit_transform(augmented_labels.reshape(-1, 1))

label_mapping = {category: idx for idx, category in enumerate(encoder.categories_[0])}
print("Label Mapping:", label_mapping)

class_names = list(label_mapping.keys())

# Building hybrid transformer model
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Conv1D, Flatten, LSTM, Reshape, TimeDistributed, Attention
)
from tensorflow.keras.models import Model

# Adding positional encoding for transformer
def positional_encoding(seq_len, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    pos = np.arange(seq_len)[:, np.newaxis]
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads, dtype=tf.float32)

# Creating transformer encoder block
def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Adding attention mechanism
def attention_block(inputs):
    attention_output = Attention()([inputs, inputs])
    return attention_output

# Constructing hybrid CNN-LSTM-Transformer model
def build_hybrid_model(seq_len, n_channels, features_per_channel, d_model, num_heads, ff_dim, num_classes, dropout=0.1):
    inputs = Input(shape=(seq_len, n_channels, features_per_channel))

    x = tf.keras.layers.TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(inputs)
    x = tf.keras.layers.TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(Flatten())(x)

    x = Reshape((seq_len, -1))(x)
    x = LSTM(64, return_sequences=True)(x)

    x = attention_block(x)

    pos_encoding = positional_encoding(seq_len, x.shape[-1])
    x += pos_encoding

    for _ in range(2):
        x = transformer_encoder(x, d_model=x.shape[-1], num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    x = Flatten()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# Initializing model parameters
seq_len = normalized_data.shape[2]
n_channels = normalized_data.shape[1]
features_per_channel = normalized_data.shape[3]
num_classes = encoded_labels.shape[1]

d_model = 64
num_heads = 4
ff_dim = 128
dropout = 0.1

model = build_hybrid_model(seq_len, n_channels, features_per_channel, d_model, num_heads, ff_dim, num_classes, dropout)
learning_rate = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Analyzing attention patterns
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

samples, n_channels, seq_len, features_per_channel = normalized_data.shape

flattened_features = n_channels * features_per_channel

X_train, X_test, y_train, y_test = train_test_split(
    normalized_data, encoded_labels, test_size=0.2, random_state=42
)

X_train = X_train.reshape(X_train.shape[0], seq_len, n_channels, features_per_channel)
X_test = X_test.reshape(X_test.shape[0], seq_len, n_channels, features_per_channel)

attention_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=6).output)
attention_outputs = attention_layer_model.predict(X_test[:10])
attention_mean = np.mean(attention_outputs, axis=1)

mean_attention = np.mean(attention_mean, axis=0)



attention_map = np.mean(attention_outputs, axis=(0, 2))

plt.figure()

plt.plot(attention_map, linewidth=2)

plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Attention Strength", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.tight_layout()

plt.savefig("attention_over_time.pdf", format='pdf', bbox_inches='tight')

plt.show()

# Visualizing feature space with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

true_indices = np.argmax(y_test, axis=1)

original_labels = encoder.categories_[0]
true_class_names = [original_labels[i] for i in true_indices]

synthesized_mapping = {
    'Alzheimer': 'AD',
    'Alzheimer_FrontotemporalDementia': 'AD + FTD',
    'Alzheimer_Parkinsons': 'AD + PD',
    'Alzheimer_Schizophrenia': 'AD + SCZ',
    'FrontotemporalDementia': 'FTD',
    'FrontotemporalDementia_Parkinsons': 'FTD + PD',
    'FrontotemporalDementia_Schizophrenia': 'FTD + SCZ',
    'Frontotemporal_Dementia': 'FTD',
    'Healthy': 'Healthy',
    'Parkinsons': 'PD',
    'Parkinsons_Schizophrenia': 'PD + SCZ',
    'Schizophrenia': 'SCZ'
}
mapped_class_labels = [synthesized_mapping[label] for label in true_class_names]

feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(X_test)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_proj = tsne.fit_transform(features)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=tsne_proj[:, 0],
    y=tsne_proj[:, 1],
    hue=mapped_class_labels,
    palette='tab10',
    s=70,
    edgecolor='black'
)

plt.xlabel("t-SNE 1", fontsize=14)
plt.ylabel("t-SNE 2", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(
    title='Class Label',
    title_fontsize=13,
    fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fancybox=True,
    borderpad=0.8
)

plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()
# Tsne plot single diseases and comorbidy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


true_indices = np.argmax(y_test, axis=1)
original_labels = encoder.categories_[0]
true_class_names = [original_labels[i] for i in true_indices]


synthesized_mapping = {
    'Alzheimer': 'AD',
    'Alzheimer_FrontotemporalDementia': 'AD + FTD',
    'Alzheimer_Parkinsons': 'AD + PD',
    'Alzheimer_Schizophrenia': 'AD + SCZ',
    'FrontotemporalDementia': 'FTD',
    'FrontotemporalDementia_Parkinsons': 'FTD + PD',
    'FrontotemporalDementia_Schizophrenia': 'FTD + SCZ',
    'Frontotemporal_Dementia': 'FTD',
    'Healthy': 'Healthy',
    'Parkinsons': 'PD',
    'Parkinsons_Schizophrenia': 'PD + SCZ',
    'Schizophrenia': 'SCZ'
}
mapped_class_labels = [synthesized_mapping[label] for label in true_class_names]
label_array = np.array(mapped_class_labels)


single_disease_labels = ['AD', 'PD', 'SCZ', 'FTD', 'Healthy']
single_mask = np.isin(label_array, single_disease_labels)
combined_mask = ~single_mask

feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(X_test)

# t-SNE on all data
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_proj = tsne.fit_transform(features)


# Plot 1: Single Disease Only

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=tsne_proj[single_mask, 0],
    y=tsne_proj[single_mask, 1],
    hue=label_array[single_mask],
    palette='tab10',
    s=70,
    edgecolor='black'
)

plt.xlabel("t-SNE 1", fontsize=14)
plt.ylabel("t-SNE 2", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(
    title='Class Label',
    title_fontsize=13,
    fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fancybox=True,
    borderpad=0.8
)

plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_single_disease.pdf", format='pdf', bbox_inches='tight')
plt.show()


# Plot 2: Combined Disease Only

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=tsne_proj[combined_mask, 0],
    y=tsne_proj[combined_mask, 1],
    hue=label_array[combined_mask],
    palette='tab10',
    s=70,
    edgecolor='black'
)

plt.xlabel("t-SNE 1", fontsize=14)
plt.ylabel("t-SNE 2", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(
    title='Class Label',
    title_fontsize=13,
    fontsize=11,
    loc='upper left',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fancybox=True,
    borderpad=0.8
)

plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_combined_disease.pdf", format='pdf', bbox_inches='tight')
plt.show()
# Model Trainign
from sklearn.model_selection import train_test_split

samples, n_channels, seq_len, features_per_channel = normalized_data.shape

flattened_features = n_channels * features_per_channel

X_train, X_test, y_train, y_test = train_test_split(
    normalized_data, encoded_labels, test_size=0.2, random_state=42
)


X_train = X_train.reshape(X_train.shape[0], seq_len, n_channels, features_per_channel)
X_test = X_test.reshape(X_test.shape[0], seq_len, n_channels, features_per_channel)

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=8,
    validation_data=(X_test, y_test)
)
