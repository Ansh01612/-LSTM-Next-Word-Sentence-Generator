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

st.html(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #68738a;
        --accent: #6259e8;
        --line: #e8eaf2;
        --surface: #ffffff;
    }
    html {
        color-scheme: light;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(115, 104, 255, .13), transparent 42%),
            #f7f8fc !important;
        color: var(--ink) !important;
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stMainBlockContainer"] {
        max-width: 820px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #fff !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h2 {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -.03em;
    }
    .eyebrow {
        color: var(--accent);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin: 0 0 .8rem;
    }
    .hero-title {
        font-family: 'Segoe UI', Arial, sans-serif !important;
        font-size: clamp(2.2rem, 6vw, 3.6rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.08;
        margin: 0;
        color: #172033 !important;
    }
    .hero-title span {
        color: var(--accent);
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 590px;
        margin: 1rem auto 0;
    }
    .hero {
        text-align: center;
        padding: 1.4rem 1rem 2rem;
        margin-bottom: 1rem;
    }
    .input-card, [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 14px 40px rgba(32, 42, 76, .06);
        padding: 1.35rem 1.5rem;
    }
    [data-testid="stTextInput"] label p {
        font-size: .92rem;
        font-weight: 700;
        color: var(--ink);
    }
    [data-testid="stTextInput"] input {
        min-height: 3.35rem;
        border-radius: 12px;
        border-color: #dfe2ed !important;
        background: #fbfbfe !important;
        color: var(--ink) !important;
        font-size: 1rem;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    .stButton > button {
        min-height: 3.05rem;
        border-radius: 11px;
        font-weight: 700;
        border-color: #e2e4ee !important;
        background: #fff !important;
        color: var(--ink) !important;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: var(--accent);
        box-shadow: 0 7px 18px rgba(98, 89, 232, .14);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #fff !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent);
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2.5rem;
        }
        .hero { padding: 1.6rem .2rem 1rem; }
        .hero-copy { font-size: .96rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 1rem; }
    }
    </style>
    """,
)


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
st.markdown(
    """
    <section class="hero">
      <p class="eyebrow">AI writing companion</p>
      <h1 class="hero-title">Your next thought,<br><span>starts here.</span></h1>
      <p class="hero-copy">Give the LSTM a few words. It will predict what comes next
      and help you turn a spark into a sentence.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

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

with st.container(border=True):
    seed_text = st.text_input(
        "What are you thinking?",
        placeholder="Try: once upon a time...",
        help="Enter a starting phrase for the model to continue.",
    )

col1, col2 = st.columns(2)
generate_clicked = col1.button("✨  Continue my thought", type="primary", use_container_width=True)
predict_clicked = col2.button("See next-word ideas", use_container_width=True)

if generate_clicked:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating..."):
            result = generate_sentence(seed_text, num_words, temperature if use_sampling else 1.0)
        st.success("Done!")
        st.markdown("#### Your completed thought")
        st.markdown(f"> {result}")

if predict_clicked:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        preds = predict_next_word_probs(seed_text)
        top_n = 5
        top_indices = preds.argsort()[-top_n:][::-1]
        st.markdown("#### Next-word ideas")
        for idx in top_indices:
            word = index_to_word.get(int(idx), "<unk>")
            st.write(f"- **{word}** — {preds[idx] * 100:.2f}%")
