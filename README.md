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
