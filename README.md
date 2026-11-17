# Sentiment-Aware Multi-LLM Chatbot

A DCU MSc practicum project that combines answer checks from multiple models with sentiment-aware response handling.

## TruthfulQA results

One run on 817 questions at temperature 0. Correctness used a Claude Haiku judge (40%), Gemini Flash-Lite judge (40%), and MiniLM similarity (20%).

| Method | Accepted answers | Acceptance rate | Hallucination rate among accepted answers |
|---|---:|---:|---:|
| GPT-3.5-Turbo alone | 817 | 100% | 27.3% (594/817 correct) |
| GPT-3.5 self-evaluation | 578 | 70.7% | 9.3% |
| Multi-LLM pipeline | 530 | 64.9% | 4.0% |

The 9.3% and 4.0% rates are measured only on answers each method accepted. They are not directly comparable with the 27.3% rate for GPT-3.5-Turbo alone, where all answers were accepted.

Sentiment has weight 0.10. The classifier labelled 705 of 817 questions as "confused", so sentiment does not explain the gain. There was no human labelling. The judges are also part of the pipeline evaluation, which creates some circularity.

## Sentiment classifier

The fine-tuned CardiffNLP RoBERTa classifier has four classes. After deduplication and balancing, the training set had 214,408 rows. Test macro-F1 was 0.848.

## Files needed for the full app

The RoBERTa model weights and experiment results CSVs are not included in this repository. They were not uploaded. The full chatbot and Results Viewer need those files.

## Run

Install the Python packages and set the API keys:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate with `\.venv\Scripts\Activate.ps1`. Add your API keys to `.env`, then run:

```sh
streamlit run app/streamlit_app.py
```

A clean clone does not include the local model weights or saved results CSVs, so the full chatbot and Results Viewer cannot run until those files are supplied.

## Topics

large-language-models, sentiment-analysis, truthfulqa, roberta, streamlit, dublin-city-university
