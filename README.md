# Sentiment Analyzer

## Overview
This project implements a machine learning-based sentiment analysis system using Python. It features a FastAPI web service that provides real-time sentiment analysis of text inputs using a Support Vector Machine (SVM) classifier with TF-IDF vectorization.

## Architecture

```text
                   Text Input
                        │
                        ▼
               ┌─────────────────┐
               │  FastAPI        │
               │  Endpoint       │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Text           │    ◄─── TF-IDF
               │  Preprocessing  │         Vectorization
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  SVM            │
               │  Classifier     │
               └────────┬────────┘
                        │
                        ▼
                   Sentiment
                   (Positive/
                   Negative)







## Components

### 1. Text Preprocessor (`src/preprocessor.py`)
- Handles text cleaning and normalization
- Implements TF-IDF vectorization
- Features:
  * Special character removal
  * Case normalization
  * Text vectorization using sklearn's TfidfVectorizer

### 2. Sentiment Model (`src/model.py`)
- Implements SVM classifier for sentiment analysis
- Provides:
  * Model training interface
  * Prediction capabilities
  * Model persistence (save/load)

### 3. API Service (`src/api.py`)
- FastAPI-based REST API
- Endpoints:
  * POST `/predict`: Analyzes sentiment of input text
- Features:
  * Error handling
  * Input validation
  * Asynchronous processing

## Technical Stack
- **Machine Learning**: scikit-learn, SVM
- **Text Processing**: TF-IDF Vectorization
- **Web Framework**: FastAPI
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Installation
```bash
# Clone repository
git clone git@github.com:easensoy/sentiment-analyzer.git
cd sentiment-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
