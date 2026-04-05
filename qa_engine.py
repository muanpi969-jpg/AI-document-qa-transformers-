import streamlit as st
from transformers import pipeline


# deepset/roberta-base-squad2 is a RoBERTa model fine-tuned on SQuAD 2.0.
# SQuAD 2.0 includes unanswerable questions, which means the model has
# learned to return a low confidence score when the answer is not in the
# document — rather than hallucinating one. This is important for a tool
# meant to handle real documents where the answer may genuinely be absent.
#
# Architecture difference from the summarizer:
# The AI text summarizer uses encoder-decoder models (BART, T5) that
# generate new text token by token. This model is encoder-only (RoBERTa).
# It reads the full document and question simultaneously, then predicts
# the start and end positions of the answer span within the source text.
# It cannot produce words that are not already in the document.
MODEL_NAME = "deepset/roberta-base-squad2"


@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model=MODEL_NAME)


def answer_question(context: str, question: str) -> dict:
    """
    Run extractive QA over the provided context.

    Returns a dict containing:
        answer  — the extracted text span
        score   — confidence float between 0 and 1
        start   — character index where the answer starts in context
        end     — character index where the answer ends in context
    """
    model = load_qa_model()
    result = model(question=question, context=context)
    return result
