# Next Thought — LSTM Sentence Generator

A browser-based writing companion powered by the repository's LSTM model. The
Vercel version runs inference on the visitor's device, so prompts are not sent
to a server.

## Deploy on Vercel

1. Push this repository to GitHub.
2. In Vercel, choose **Add New → Project** and import
   `Ansh01612/-LSTM-Next-Word-Sentence-Generator`.
3. Keep the project root at the repository root. Vercel detects Vite; the build
   command is `npm run build` and the output directory is `dist`.
4. Select **Deploy**. Future pushes to `main` trigger a new deployment.

The browser downloads the TensorFlow.js model from `public/model/` on first use.
The model weights and tokenizer are checked into the repository and total
about 6.8 MB.

## Run the Vercel version locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. For a production build, run:

```bash
npm run build
npm run preview
```

## Run the original Streamlit version

The original Python app remains available:

```bash
pip install -r requirements.txt
streamlit run app.py
```

It uses `lstm_model.h5`, `tokenizer.pickle`, and `max_len.pickle` from the
repository root.
