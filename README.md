# Document Q&A — Extractive Question Answering with Transformers

🔗 Live Demo: https://ai-document-transformers-muanpi.streamlit.app/


## Overview

This project implements an extractive question answering (QA) system using transformer-based models.
It allows users to paste a document and ask natural language questions, returning precise answers directly from the text along with a confidence score.

The goal of this system is to enable efficient information retrieval from long or complex documents without requiring full manual reading.


## Motivation

This project was developed to solve a real-world problem.
When working with official documents such as government notices, legal forms, or medical records, extracting a specific piece of information can be time-consuming.

This system provides a faster and more reliable way to locate exact answers within a document.


## How It Works

The application is built using Python, Hugging Face Transformers, and Streamlit.
	•	Model: deepset/roberta-base-squad2
	•	Architecture: Encoder-only transformer (RoBERTa)
	•	Task: Extractive Question Answering (SQuAD 2.0)

Unlike generative models, this system:
	•	Does not generate new text
	•	Extracts answers directly from the input document
	•	Returns the exact span where the answer appears

The model processes both the document (context) and the question together, predicting:
	•	Start position of the answer
	•	End position of the answer


## Why SQuAD 2.0

SQuAD 2.0 includes both answerable and unanswerable questions.

This allows the model to:
	•	Identify when the answer exists
	•	Recognize when the document does not contain the answer

This improves reliability compared to models trained only on answerable datasets.


## Features
	•	Extractive question answering from user-provided text
	•	Confidence score with visual indicators (high, medium, low)
	•	Highlighted answer span within the original document
	•	Session history of questions and answers
	•	Downloadable Q&A session
	•	Word count and input validation


## Project Structure
	•	app.py — Streamlit interface and user interaction
	•	qa_engine.py — Model loading and inference logic
	•	utils.py — Helper functions (text cleaning, word count, highlighting)
	•	requirements.txt — Dependencies for deployment
	•	runtime.txt — Python version configuration


## Design Decisions

Extractive vs Generative QA
Extractive QA was chosen to ensure that answers come directly from the source document, improving trust and verifiability.

Confidence Thresholds
A three-level system (high, medium, low) was used to provide clear interpretation of model confidence.

Model Caching
The model is loaded once using caching to reduce latency for subsequent queries.

Session History
Allows users to track and export their interactions for later reference.


## Key Observations
	•	Extractive QA provides verifiable answers directly from source text
	•	Confidence scores help identify uncertain predictions
	•	Model performance depends on input length and question clarity
	•	Initial model loading is slower, but subsequent queries are faster due to caching


## Limitations
	•	Input length is constrained by transformer token limits
	•	No domain-specific fine-tuning has been applied
	•	Performance may vary depending on document structure
	•	No quantitative evaluation metrics (e.g., Exact Match, F1) implemented
    •   No inference optimization (e.g., quantization, ONNX) applied

## How to Run Locally

```bash
git clone https://github.com/muanpi969-jpg/document-qa-transformers
cd document-qa-transformers
pip install -r requirements.txt
streamlit run app.py

## Tech Stack
	•	Python 3.10
	•	Hugging Face Transformers
	•	PyTorch (CPU)
	•	Streamlit
    •   Model: deepset/roberta-base-squad2 (SQuAD 2.0 fine-tuned)

## Deployment

The application is deployed on Streamlit Cloud and can be accessed via the live demo link above.

The deployment uses a CPU-only PyTorch configuration to ensure compatibility with cloud resource constraints.

## Notes

This project demonstrates the application of transformer-based NLP models in a real-world setting, focusing on reliability, interpretability, and practical usability rather than text generation.
