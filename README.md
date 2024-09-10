# Sentiment Analysis of Movie Reviews using NLP and Deep Learning

## Project Overview

This project focuses on sentiment analysis of movie reviews using a combination of Natural Language Processing (NLP) techniques and deep learning models. The goal is to classify movie reviews as **positive** or **negative** based on the text content. We utilized a dataset of 50,000 movie reviews, implementing various deep learning models to achieve accurate sentiment classification.

## Dataset

The dataset used in this project consists of 50,000 highly polar movie reviews. You can find the dataset at the following link:

[Stanford Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Models Used

### 1. LSTM (Long Short-Term Memory)
Used for capturing long-term dependencies in the text, which is crucial for understanding sentiment across lengthy reviews.

### 2. Bi-LSTM (Bidirectional LSTM)
Processes the sequence in both forward and backward directions to capture contextual information from both sides of a review.

### 3. CNN-LSTM (Convolutional Neural Network - LSTM)
Uses convolutional layers to extract features from the text, followed by an LSTM to capture temporal dependencies.

### 4. CNN (Convolutional Neural Network)
Extracts spatial features from the text, ideal for short-term dependencies and local patterns.

### 5. BERT (Bidirectional Encoder Representations from Transformers)
A pre-trained transformer model that excels at understanding the context in which words are used, making it a powerful model for sentiment classification.
