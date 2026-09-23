import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Config — filenames must match the files you place next to this script
# ---------------------------------------------------------------------------
MODEL_PATH = "lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAXLEN_PATH = "max_len.pickle"

st.set_page_config(page_title="LSTM Sentence Generator", page_icon="✍️", layout="centered")


# ---------------------------------------------------------------------------
# Load model + tokenizer + max_len once, then cache across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAXLEN_PATH, "rb") as f:
        max_len = pickle.load(f)

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


model, tokenizer, max_len, index_to_word = load_artifacts()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_next_word_probs(seed_text: str) -> np.ndarray:
    """Return the softmax probability vector for the next word."""
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len, padding="pre")

    try:
        preds = model.predict(token_list, verbose=0)[0]
    except Exception:
        # This model was saved with a FIXED batch size (e.g. 128) baked
        # into its input layer, so a single-row batch gets rejected.
        # Work around it by padding the batch up to that fixed size.
        fixed_batch = model.input_shape[0]
        if not fixed_batch:
            raise
        batch = np.zeros((fixed_batch, max_len), dtype=token_list.dtype)
        batch[0] = token_list[0]
        preds = model.predict(batch, verbose=0)[0]

    return preds


def generate_sentence(seed_text: str, num_words: int, temperature: float = 1.0) -> str:
    text = seed_text
    for _ in range(num_words):
        preds = predict_next_word_probs(text)

        if temperature and temperature != 1.0:
            preds = np.log(preds + 1e-9) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_index = int(np.random.choice(len(preds), p=preds))
        else:
            next_index = int(np.argmax(preds))

        next_word = index_to_word.get(next_index, "")
        if not next_word:
            break
        text += " " + next_word
    return text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("✍️ LSTM Next-Word Sentence Generator")
st.caption("Type a starting phrase — the model predicts the next word, then feeds its own output back in to build out a full sentence.")

with st.sidebar:
    st.header("Settings")
    num_words = st.slider("Words to generate", 1, 50, 10)
    use_sampling = st.checkbox("Add randomness (sampling)", value=False)
    temperature = st.slider(
        "Temperature", 0.2, 1.5, 1.0, 0.1,
        disabled=not use_sampling,
        help="Lower = safer/more repetitive. Higher = more random.",
    )
    st.markdown("---")
    st.caption(f"Vocabulary size: {model.output_shape[-1]}")
    st.caption(f"Padded sequence length: {max_len}")

seed_text = st.text_input("Start typing a sentence:", placeholder="e.g. once upon a time")

col1, col2 = st.columns(2)
generate_clicked = col1.button("Generate full sentence", type="primary", use_container_width=True)
predict_clicked = col2.button("Predict next word only", use_container_width=True)

if generate_clicked:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating..."):
            result = generate_sentence(seed_text, num_words, temperature if use_sampling else 1.0)
        st.success("Done!")
        st.markdown(f"**Generated text:**\n\n> {result}")

if predict_clicked:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        preds = predict_next_word_probs(seed_text)
        top_n = 5
        top_indices = preds.argsort()[-top_n:][::-1]
        st.markdown("**Top predicted next words:**")
        for idx in top_indices:
            word = index_to_word.get(int(idx), "<unk>")
            st.write(f"- **{word}** — {preds[idx] * 100:.2f}%")
