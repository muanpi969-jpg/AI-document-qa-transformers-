# Document Q&A — Extractive Question Answering with Transformers

#### Video Demo: <URL HERE>

#### Description:

This project is a web application that lets you paste any document and ask plain-English questions about it. Instead of reading through pages of text to find one specific fact, you type your question and the model finds the exact sentence or phrase inside your document that answers it. If the answer is not in the document, it tells you that too — with a confidence score — rather than making something up.

I built this as my CS50x final project because it solves a problem I have experienced directly. As a refugee in India, I regularly encounter official documents — government notices, legal letters, medical forms — written in formal English that is difficult to parse quickly. When you need to know one specific fact from a ten-page document, reading the whole thing word by word is not always possible. This tool changes that.

---

## How it works

The application is built in Python using Hugging Face Transformers and Streamlit. The model behind it is `deepset/roberta-base-squad2`, a RoBERTa model fine-tuned on the Stanford Question Answering Dataset version 2.0 (SQuAD 2.0).

This is deliberately different from my earlier AI text summarizer project, which used encoder-decoder models (BART and T5) to generate new sentences. This project uses a different architecture entirely. RoBERTa is an encoder-only transformer. It reads the full document and the question at the same time, then predicts two numbers: the position in the text where the answer starts, and the position where it ends. The answer it returns is always a span extracted directly from your document — it cannot produce words that were not already there.

The reason I chose SQuAD 2.0 specifically matters. SQuAD 1.0 only contains answerable questions, which means a model trained on it always tries to return something even when the document has no relevant information. SQuAD 2.0 includes questions that have no answer in the passage, which forces the model to learn when to say it does not know. That makes `roberta-base-squad2` much more honest and more useful for real documents.

---

## Project files

**app.py** — The main Streamlit application. This file handles everything the user sees: the document input area, the question field, the answer display, confidence indicators, a highlighted view showing exactly where in the document the answer came from, a session history of all questions asked, and a download button to save the full Q&A session as a text file. The confidence score is shown in three states: green for high confidence (above 70%), amber for moderate confidence (40–70%), and red for low confidence (below 40%), where the document likely does not contain the answer.

**qa_engine.py** — Model loading and inference. The model is loaded once using `@st.cache_resource` and stays in memory for the duration of the session, so subsequent questions run much faster than the first one. The `answer_question` function takes a context string and a question string, runs the pipeline, and returns the answer text, confidence score, and character positions of the answer within the original document.

**utils.py** — Helper functions used by both the UI and the engine. `clean_text` normalises whitespace. `word_count` counts words for the live counter displayed below the text input. `is_too_short` returns true if the document is under 50 words, which prevents the model from being run on inputs too small to be meaningful. `highlight_answer` takes the full context string and the start/end character indices returned by the model, then splits the text into three parts so the UI can render the answer highlighted in place inside the original document.

**requirements.txt** — Pinned dependency versions for reproducible deployment on Streamlit Cloud. PyTorch is pinned to the CPU-only wheel to avoid memory issues on the free tier. The `--extra-index-url` line points pip to PyTorch's own package index where the CPU build lives.

**runtime.txt** — Tells Streamlit Cloud to use Python 3.10, which is compatible with all pinned dependency versions.

---

## Design decisions

The main design decision in this project was choosing extractive QA over generative QA. I could have used a generative model like GPT-2 or a fine-tuned T5 to produce free-form answers. I chose not to, for one reason: when someone is trying to understand a legal or official document, they need to know that the answer came directly from the source text — not from a model that might have invented it. Extractive QA gives you that guarantee. The blue highlighted span in the document view makes this visible and verifiable.

The second design decision was the confidence threshold system. A single number is not enough information when the stakes of misreading a document are real. The three-level system — green, amber, red — gives the user an immediate signal about how much to trust the result, without requiring them to understand what a softmax probability score means.

The third decision was session history with a download option. A tool for understanding documents is only useful if you can keep a record of what you learned. The download button creates a plain text file of every question and answer from the current session.

---

## How to run locally

```bash
git clone https://github.com/muanpi969-jpg/document-qa
cd document-qa
pip install -r requirements.txt
streamlit run app.py
```

The model downloads automatically on first run from Hugging Face Hub (~500MB). Subsequent runs use the cached version.

---

## Tech stack

- Python 3.10
- Hugging Face Transformers
- PyTorch (CPU)
- Streamlit
- Model: deepset/roberta-base-squad2
