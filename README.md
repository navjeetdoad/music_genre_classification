# Music Genre Classification in Python

## Overview
This project aims to classify .wav files of music into 10 different genres using machine learning. The model learns to recognize patterns in audio features (ex. tempo, spectral contrast, etc.) and predicts the genre of a given track.

## Dataset
The dataset is taken from the GTZAN Music Genre Dataset on Kaggle, which includes Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae and Rock music data. The dataset includes the following:
* genres_original, a folder with 1000 .wav files, 100 of each of the 10 genres that are each 30 seconds long.
* images_original, a folder with 1000 .png files (mel-spectrograms), 100 of each of the 10 genres that visually represent each audio file.
* 2 .csv files which contain the numerical features of the audio files.
Note: The dataset can be found here: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Technologies / Libraries Used
* Python / Jupyter Notebook
* Librosa - Used for extracting audio features
* scikit-learn - Machine Learning models
* TensorFlow - Used for deep learning processes
* Matplotlib / Seaborn / Plotly

## Features
* Extracts audio features using Librosa
* Provides evaluation metrics and confusion matrix visualization
* Allows easy dataset customization
* Includes model saving and loading for deployment

## File Structure

```bash
music_genre_classification/
├── music_genre_classification.ipynb
├── data/
  ├── features_3_sec.csv
  ├── features_30_sec.csv
  ├── images_original/
    ├── rock/
    ├── reggae/
    ├── pop/
    ├── metal/
    ├── jazz/
    ├── hiphop/
    ├── disco/
    ├── country/
    ├── classical/
    └── blues/
  ├── genres_original/
    ├── rock/
    ├── reggae/
    ├── pop/
    ├── metal/
    ├── jazz/
    ├── hiphop/
    ├── disco/
    ├── country/
    ├── classical/
    └── blues/
```


