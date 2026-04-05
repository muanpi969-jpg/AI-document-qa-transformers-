import streamlit as st
from qa_engine import answer_question
from utils import clean_text, word_count, is_too_short, highlight_answer


st.set_page_config(
    page_title="Document Q&A",
    page_icon="🔍",
    layout="centered",
)

st.title("Document Q&A")
st.caption(
    "Paste any document and ask questions about it in plain English. "
    "The model finds the exact answer inside your text — "
    "it does not generate or guess. If the answer is not in the document, "
    "it will tell you."
)

st.divider()

# --- Document input ---
context = st.text_area(
    "Paste your document here",
    height=280,
    placeholder="Paste any article, report, legal text, email, or document. Minimum 50 words.",
)

if context:
    count = word_count(context)
    st.caption(f"{count} words")

# --- Question input ---
question = st.text_input(
    "Ask a question about the document",
    placeholder="e.g. What is the deadline? Who is responsible? What does this mean?",
)

st.divider()

# --- Session history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Run inference ---
if st.button("Get Answer", type="primary", use_container_width=True):
    if not context.strip():
        st.warning("Please paste a document first.")
    elif is_too_short(context):
        st.warning("Document is too short — try at least 50 words.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading model and searching your document..."):
            try:
                cleaned = clean_text(context)
                result = answer_question(context=cleaned, question=question)

                answer = result["answer"]
                score = result["score"]
                start = result["start"]
                end = result["end"]

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "score": score,
                    "start": start,
                    "end": end,
                    "context": cleaned,
                })

            except Exception as e:
                st.error(f"Something went wrong during inference: {e}")
                st.stop()

# --- Display most recent result ---
if st.session_state.history:
    latest = st.session_state.history[-1]
    answer = latest["answer"]
    score = latest["score"]
    confidence_pct = round(score * 100, 1)

    st.subheader("Answer")
    st.write(answer)

    # Confidence indicator
    if score > 0.70:
        st.success(f"Confidence: {confidence_pct}%")
    elif score > 0.40:
        st.warning(
            f"Confidence: {confidence_pct}% — the model found a possible answer "
            "but is not certain. Check the highlighted location below."
        )
    else:
        st.error(
            f"Confidence: {confidence_pct}% — the document may not contain "
            "a clear answer to this question."
        )

    # Show where the answer appears in the document
    with st.expander("Show where this answer appears in the document"):
        before, found, after = highlight_answer(
            latest["context"], latest["start"], latest["end"]
        )
        # Render with the answer span bolded and colored
        st.markdown(f"{before}**:blue[{found}]**{after}")

    st.divider()

# --- Session history ---
if len(st.session_state.history) > 1:
    st.subheader("Previous questions this session")
    for item in reversed(st.session_state.history[:-1]):
        with st.expander(f"Q: {item['question']}"):
            st.write(f"**Answer:** {item['answer']}")
            st.caption(f"Confidence: {round(item['score'] * 100, 1)}%")

# --- Download session ---
if st.session_state.history:
    lines = []
    for item in st.session_state.history:
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer']}")
        lines.append(f"Confidence: {round(item['score'] * 100, 1)}%")
        lines.append("")
    session_text = "\n".join(lines)

    st.download_button(
        label="Download Q&A session as .txt",
        data=session_text,
        file_name="qa_session.txt",
        mime="text/plain",
    )

st.divider()
st.caption(
    "Model: deepset/roberta-base-squad2 · Fine-tuned on SQuAD 2.0 · "
    "Extractive QA — answers are spans found inside your document, not generated text. "
    "Model loads once per session and is cached."
)
