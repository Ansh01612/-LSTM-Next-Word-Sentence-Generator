import './style.css';

let tf;
const MAX_LEN = 745;
const BATCH_SIZE = 128;
const VOCAB_SIZE = 8900;
const MODEL_URL = '/model/model.json';
const TOKENIZER_URL = '/model/tokenizer.json';

const elements = {
  status: document.querySelector('#model-status'),
  seed: document.querySelector('#seed-text'),
  generate: document.querySelector('#generate-button'),
  predict: document.querySelector('#predict-button'),
  error: document.querySelector('#error-message'),
  result: document.querySelector('#result-card'),
  resultEyebrow: document.querySelector('#result-eyebrow'),
  resultText: document.querySelector('#result-text'),
  ideas: document.querySelector('#ideas-section'),
  ideasList: document.querySelector('#ideas-list'),
  wordCount: document.querySelector('#word-count'),
  wordCountValue: document.querySelector('#word-count-value'),
  sampling: document.querySelector('#sampling-toggle'),
  temperature: document.querySelector('#temperature'),
  temperatureValue: document.querySelector('#temperature-value'),
};

let model;
let tokenizer;
let indexToWord;

function setStatus(message, state = 'loading') {
  elements.status.dataset.state = state;
  elements.status.querySelector('.status-dot').textContent = state === 'ready' ? '✓' : '';
  elements.status.lastChild.textContent = ` ${message}`;
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
}

function clearResults() {
  elements.error.hidden = true;
  elements.result.hidden = true;
  elements.ideas.hidden = true;
}

function setBusy(isBusy) {
  elements.generate.disabled = isBusy || !model;
  elements.predict.disabled = isBusy || !model;
  elements.generate.classList.toggle('is-loading', isBusy);
  elements.generate.querySelector('span').textContent = isBusy ? '◌' : '✦';
}

function tokenize(text) {
  let normalized = tokenizer.lower ? text.toLowerCase() : text;
  for (const character of tokenizer.filters) {
    normalized = normalized.split(character).join(tokenizer.split);
  }
  return normalized
    .split(tokenizer.split)
    .filter(Boolean)
    .map((word) => tokenizer.wordIndex[word])
    .filter((index) => Number.isInteger(index) && index < tokenizer.numWords);
}

async function predictNextWordProbabilities(text) {
  const tokenIds = tokenize(text).slice(-MAX_LEN);
  const sequence = new Float32Array(BATCH_SIZE * MAX_LEN);
  const start = MAX_LEN - tokenIds.length;
  for (let i = 0; i < tokenIds.length; i += 1) {
    sequence[start + i] = tokenIds[i];
  }

  const input = tf.tensor2d(sequence, [BATCH_SIZE, MAX_LEN], 'float32');
  let output;
  try {
    output = model.predict(input);
    const values = await output.data();
    return Array.from(values.slice(0, VOCAB_SIZE));
  } finally {
    input.dispose();
    output?.dispose();
  }
}

function chooseSample(probabilities, temperature) {
  const logits = probabilities.map((probability) => Math.log(probability + 1e-9) / temperature);
  const maxLogit = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - maxLogit));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let target = Math.random() * total;
  for (let index = 0; index < weights.length; index += 1) {
    target -= weights[index];
    if (target <= 0) return index;
  }
  return weights.length - 1;
}

function topPredictions(probabilities, count = 5) {
  return probabilities
    .map((probability, index) => ({ index, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, count);
}

function renderIdeas(probabilities) {
  elements.ideasList.replaceChildren();
  for (const prediction of topPredictions(probabilities)) {
    const item = document.createElement('li');
    const word = document.createElement('strong');
    const percent = document.createElement('span');
    word.textContent = indexToWord[prediction.index] ?? '<unk>';
    percent.textContent = `${(prediction.probability * 100).toFixed(2)}%`;
    item.append(word, percent);
    elements.ideasList.append(item);
  }
  elements.ideas.hidden = false;
}

async function handleGenerate() {
  const seed = elements.seed.value.trim();
  if (!seed) {
    showError('Add a few starting words first, and I’ll help you continue.');
    elements.seed.focus();
    return;
  }

  clearResults();
  setBusy(true);
  elements.resultEyebrow.textContent = 'Your completed thought';
  elements.resultText.textContent = seed;
  elements.result.hidden = false;

  try {
    let text = seed;
    const wordCount = Number(elements.wordCount.value);
    const useSampling = elements.sampling.checked;
    const temperature = Number(elements.temperature.value);
    for (let i = 0; i < wordCount; i += 1) {
      elements.resultText.textContent = text;
      const probabilities = await predictNextWordProbabilities(text);
      const nextIndex = useSampling
        ? chooseSample(probabilities, temperature)
        : probabilities.indexOf(Math.max(...probabilities));
      const nextWord = indexToWord[nextIndex];
      if (!nextWord) break;
      text += ` ${nextWord}`;
    }
    elements.resultText.textContent = text;
  } catch (error) {
    elements.result.hidden = true;
    showError(`I couldn't generate a continuation: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function handlePredict() {
  const seed = elements.seed.value.trim();
  if (!seed) {
    showError('Add a few starting words first to see next-word ideas.');
    elements.seed.focus();
    return;
  }

  clearResults();
  setBusy(true);
  try {
    renderIdeas(await predictNextWordProbabilities(seed));
  } catch (error) {
    showError(`I couldn't get predictions: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function initialize() {
  try {
    const [tensorflow, tokenizerResponse] = await Promise.all([
      import('@tensorflow/tfjs'),
      fetch(TOKENIZER_URL),
    ]);
    if (!tokenizerResponse.ok) throw new Error('Could not load the tokenizer data.');
    tf = tensorflow;
    tokenizer = await tokenizerResponse.json();
    indexToWord = Object.fromEntries(
      Object.entries(tokenizer.wordIndex).map(([word, index]) => [index, word]),
    );

    try {
      await tf.setBackend('webgl');
    } catch {
      await tf.setBackend('cpu');
    }
    await tf.ready();
    model = await tf.loadLayersModel(MODEL_URL, {
      onProgress: (fraction) => {
        setStatus(`Loading model… ${Math.round(fraction * 100)}%`);
      },
    });
    elements.generate.disabled = false;
    elements.predict.disabled = false;
    setStatus('Ready — running on your device', 'ready');
  } catch (error) {
    setStatus('Could not load the model', 'error');
    showError(`The writing model could not be loaded: ${error.message}`);
  }
}

elements.generate.addEventListener('click', handleGenerate);
elements.predict.addEventListener('click', handlePredict);
elements.seed.addEventListener('input', clearResults);
elements.wordCount.addEventListener('input', () => {
  elements.wordCountValue.value = elements.wordCount.value;
  elements.wordCountValue.textContent = elements.wordCount.value;
});
elements.sampling.addEventListener('change', () => {
  elements.temperature.disabled = !elements.sampling.checked;
});
elements.temperature.addEventListener('input', () => {
  const value = Number(elements.temperature.value).toFixed(1);
  elements.temperatureValue.value = value;
  elements.temperatureValue.textContent = value;
});

initialize();
