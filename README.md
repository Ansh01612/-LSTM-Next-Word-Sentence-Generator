# LSTM Next-Word Sentence Generator — Streamlit App

## 1. Folder setup
Put these four files in the **same folder**:

```
next_word_app/
├── app.py
├── requirements.txt
├── lstm_model.h5        <- copy your uploaded model here
├── tokenizer.pickle      <- copy your uploaded tokenizer here
└── max_len.pickle        <- copy your uploaded max_len here
```

The filenames must match exactly (they're hardcoded at the top of `app.py`).

## 2. Run it in VS Code
1. Open this folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. It will open automatically at `http://localhost:8501`.

## Notes on your model
- Architecture: `Embedding(8900, 50) → LSTM(128) → Dense(8900, softmax)`.
- `max_len.pickle` = 745, the padded input length the model expects — the app pads every seed text to this length before predicting.
- Your model's input layer has a **fixed batch size of 128** baked in (not the usual flexible batch size). `app.py` handles this automatically by padding a single input up to a batch of 128 and reading row 0 of the result — you don't need to do anything, but if you retrain the model later, saving it with a flexible batch size (`Input(shape=(745,))` instead of `Input(batch_shape=(128, 745))`) will make things simpler and faster.
- Generation works by predicting one word at a time and appending it back into the seed text (standard approach for this kind of model) — that's what "Generate full sentence" does. "Predict next word only" just shows the top-5 candidates without extending the text.
- If TensorFlow can't load the `.h5` file (version mismatch), reinstalling with `tensorflow` version close to what you trained with, or re-saving the model as `.keras`, usually fixes it.
