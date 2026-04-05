import re


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def is_too_short(text: str, minimum: int = 50) -> bool:
    return word_count(text) < minimum


def highlight_answer(context: str, start: int, end: int):
    """
    Split the context into three parts around the answer span.
    Returns (text_before, answer_text, text_after) so the UI can
    render the answer highlighted inside the original document.
    """
    return context[:start], context[start:end], context[end:]
