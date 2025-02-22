# -*- coding: utf-8 -*-

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.preprocessing import normalize

#EVERYTHING BELOW THAT'S COMMENTED OUT IS ALL AUDIO PROCESSING / FEATURE DERIVATION
# !pip install datasets
# from datasets import load_dataset

# ds = load_dataset("JamesStratford/voice-of-birds")

# df = ds['train'].to_pandas()

# df = df.sample(n=200, random_state=42)  # random_state ensures reproducibility

# df.head()

# import librosa
# import librosa.display
# import io
# import pandas as pd

# features_list = []
# n = 0
# # Loop over each audio file in the dataset
# for audio_data in df['audio']:
#     # Convert byte data to a buffer for librosa to load
#     buffer = io.BytesIO(audio_data['bytes'])  # Assuming 'audio' column has 'bytes' subfield

#     # Load audio data
#     y, sr = librosa.load(buffer, sr=None)

#     # Calculate MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

#     # Calculate mean and standard deviation for each MFCC coefficient
#     mfcc_mean = np.mean(mfccs, axis=1)
#     mfcc_std = np.std(mfccs, axis=1)

#     # Combine mean and standard deviation into a single feature vector
#     mfcc_combined = np.hstack((mfcc_mean, mfcc_std))

#     # Calculate other features
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
#     zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
#     spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
#     harmonic_ratio = librosa.effects.harmonic(y=y)  # Harmonic component extraction
#     f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))  # Fundamental frequency

#     # Store features in a dictionary
#     features = {
#         "mfcc_combined": mfcc_combined,  # Contains all MFCC means and stds in one array
#         "spectral_centroid": spectral_centroid,
#         "spectral_bandwidth": spectral_bandwidth,
#         "zero_crossing_rate": zero_crossing_rate,
#         "spectral_rolloff": spectral_rolloff,
#         "harmonic_ratio": harmonic_ratio.mean(),  # Averaging harmonic component
#         "fundamental_frequency": np.mean(f0)  # Averaging over the audio duration
#     }

#     features_list.append(features)
#     print(n)
#     n += 1

# # Convert features list to a DataFrame
# features_df = pd.DataFrame(features_list)

# # View the first few rows of the extracted features
# features_df.head()

# features_df.to_pickle('/content/voice_of_birds_features.pkl')
# from google.colab import files

# # Download the file to your local machine
# files.download("/content/voice_of_birds_features.pkl")

# labels = []
# for bird in df['label']:

#   feature = {"bird_type": bird}
#   labels.append(feature)
# features_df['bird_type'] = pd.DataFrame(labels)

# features_df.head()

#MODEL TRAINING BEGINS HERE
features = pd.read_pickle('/content/voice_of_birds_features.pkl')
features = features[['spectral_centroid','spectral_bandwidth','zero_crossing_rate','spectral_rolloff','fundamental_frequency']]
features.head()
features = features.to_numpy()
nfeatures = sklearn.preprocessing.normalize(features, axis=0)

nfeatures = pd.DataFrame(nfeatures)

inertia = []
nk = range(1, 10)

for k in nk:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(nfeatures)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(nk, inertia, marker='o')
plt.xlabel('clusters')
plt.ylabel('inertia')
plt.title('e plot')
plt.grid(True)
plt.show()

# Create a KMeans model

kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10)

# Fit the model to the data
kmeans.fit(nfeatures)

# Get the cluster centers
centers = kmeans.cluster_centers_

# Get the labels for each data point (which cluster they belong to)
labels = kmeans.labels_

print(centers)
nfeatures

# distances = kmeans.transform(nfeatures)  # Distance of each point to all centroids
# assigned_distances = distances[np.arange(len(nfeatures)), kmeans.labels_]  # Distance to the assigned cluster

# # Calculate threshold for outliers
# threshold = np.mean(assigned_distances) + 2 * np.std(assigned_distances)  # Mean + 2 std deviations

# # Identify outliers
# outliers = np.where(assigned＿distances > threshold)[0]

# cleaned_features = nfeatures.drop(index=outliers)
# cleaned_features
# Calculate distances of points to their assigned cluster centroid
# Step 1: Small Cluster Detection
unique, counts = np.unique(kmeans.labels_, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
print("Cluster sizes:", cluster_sizes)

# Define a threshold for small cluster size (e.g., clusters with fewer than 2 points are flagged)
min_cluster_size = 2
small_clusters = [cluster for cluster, size in cluster_sizes.items() if size < min_cluster_size]

# Identify outliers from small clusters
small_cluster_outliers = np.where(np.isin(kmeans.labels_, small_clusters))[0]
print(f"Outlier indices from small clusters: {small_cluster_outliers}")

# Step 2: Distance-Based Outlier Detection
distances = kmeans.transform(nfeatures)  # Distances to all centroids
assigned_distances = distances[np.arange(len(nfeatures)), kmeans.labels_]  # Distance to assigned cluster

# Set threshold for outliers based on distances (e.g., Mean + 2*StdDev)
distance_threshold = np.mean(assigned_distances) + 2 * np.std(assigned_distances)
distance_outliers = np.where(assigned_distances > distance_threshold)[0]
print(f"Outlier indices from large distances: {distance_outliers}")

# Combine all outliers
all_outliers = np.unique(np.concatenate([small_cluster_outliers, distance_outliers]))
print(f"Final outlier indices: {all_outliers}")

# Step 3: Drop Outliers
cleaned_features = nfeatures.drop(index=all_outliers)
cleaned_features

kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10)

# Fit the model to the data
kmeans.fit(cleaned_features)

# Get the cluster centers
centers = kmeans.cluster_centers_

# Get the labels for each data point (which cluster they belong to)
labels = kmeans.labels_

print(centers)

fx = 0
fy = 3
fz = 1
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cleaned_features[fx], cleaned_features[fy], cleaned_features[fz], c=labels, cmap='viridis', s=75)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],kmeans.cluster_centers_[:, 2], s=300, c='green', marker='X')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Population')
ax.set_title('3D K-Means Clustering')


x_feature = 1  
y_feature = 0  

# Scatter plot
plt.figure(figsize=(8, 6))  # Optional: Set the figure size
plt.scatter(cleaned_features[x_feature], cleaned_features[y_feature], alpha=0.7, c='blue', edgecolor='k')  # Customize as needed
plt.title(f'Scatter Plot of {x_feature} vs {y_feature}', fontsize=14)
plt.xlabel(x_feature, fontsize=12)
plt.ylabel(y_feature, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Optional: Add grid
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
correlation_matrix = cleaned_features.iloc[:, 0:6].corr()  # Replace with the appropriate columns if needed

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)

# Customize the plot
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score

silhouette = silhouette_score(cleaned_features,kmeans.labels_)
print(silhouette)
