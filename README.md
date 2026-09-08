# Sentiment-Aware Multi-LLM Chatbot

A research project that combines sentiment-aware response adaptation with multi-LLM validation to improve the reliability and emotional alignment of chatbot responses.

## Overview

Large Language Models can produce factually incorrect or hallucinated responses, while conventional chatbots may fail to recognise the emotional state of the user.

This project addresses both challenges within a single framework by:

- Detecting the sentiment of the user's query
- Generating an initial response using an LLM
- Evaluating the response using multiple heterogeneous LLMs
- Measuring sentiment coherence between the user query and generated response
- Aggregating evaluation signals into a final decision
- Rewriting the response according to the user's detected sentiment
- Rejecting or flagging responses that are considered unreliable

## System Architecture


```text
User Query
    ↓
Sentiment Detection
    ↓
LLM Response Generation
    ↓
┌─────────────────────────────────────────────┐
│            Multi-LLM Validation             │
│                                             │
│  Claude → Factual Consistency               │
│  Gemini → Relevance + Hallucination Risk    │
│  RoBERTa → Sentiment Coherence              │
└─────────────────────────────────────────────┘
    ↓
Weighted Score Aggregation
    ↓
ACCEPT / FLAG / REJECT
    ↓
Sentiment-Aware Response Rewriting
    ↓
Final Response
```

## Main Components
### 1. Sentiment Detection

A fine-tuned RoBERTa-based sentiment classifier identifies the emotional state of the user's query. The framework uses four sentiment categories:

Positive
Neutral
Frustrated
Confused
### 2. Response Generation

GPT-3.5-Turbo generates the initial response to the user's question using a factual-response prompt.

### 3. Multi-LLM Validation

The generated response is independently evaluated using multiple signals:

Claude Haiku – evaluates factual consistency.
Gemini 2.5 Flash Lite – evaluates relevance and hallucination risk.
RoBERTa sentiment classifier – evaluates sentiment coherence between the user query and generated response.

Using heterogeneous evaluators reduces reliance on a single model's judgement.

### 4. Score Aggregation

The evaluation signals are combined using a weighted scoring mechanism:

Final Score =
0.50 × Claude Consistency
+ 0.40 × Gemini Combined Score
+ 0.10 × Sentiment Coherence

The resulting score determines one of three decisions:

ACCEPT – response is considered reliable.
FLAG – response has some uncertainty and requires caution.
REJECT – response is considered unreliable and is not directly presented to the user.
### 5. Sentiment-Aware Response Rewriting

The final response is adapted according to the detected sentiment of the user.

For example, a frustrated or confused user can receive a calmer and clearer response, while unreliable responses can result in an empathetic refusal rather than exposing the potentially incorrect answer.

## Streamlit Application

The project includes an interactive Streamlit application with three modes:

### Chatbot

Users can enter individual questions and receive a response that has been generated, evaluated and adapted according to their detected sentiment.

### Batch Test

The application can run the pipeline on a selected sample of TruthfulQA questions or the complete 817-question dataset.

### Results Viewer

Previously generated experimental results can be loaded and analysed without running the complete pipeline again.

## Evaluation

The framework was evaluated using the TruthfulQA benchmark, containing 817 questions across 38 categories.

Three experimental conditions were compared:

### Condition A – No Evaluation

Raw GPT-3.5-Turbo responses were accepted without validation.

### Condition B – Single-LLM Baseline

GPT-3.5-Turbo generated the response and also performed the self-evaluation.

### Condition C – Multi-LLM Pipeline

The proposed framework used:

GPT-3.5-Turbo for generation
Claude Haiku for factual consistency evaluation
Gemini 2.5 Flash Lite for relevance and hallucination-risk evaluation
A fine-tuned RoBERTa classifier for sentiment coherence

The experiments investigate whether multi-LLM validation can reduce hallucination-related errors compared with single-LLM approaches.

### Key Experimental Results

The full TruthfulQA evaluation produced the following results:

| Condition | Accuracy | Hallucination Rate |
|---|---:|---:|
| No Evaluation | 72.7% | 27.3% |
| Single-LLM Baseline | 90.7% | 9.3% |
| Multi-LLM Pipeline | 96.0% | 4.0% |

The multi-LLM pipeline achieved **96.0% accuracy** with a **4.0% hallucination rate** among accepted responses.

Compared with the no-evaluation condition, the multi-LLM pipeline improved accuracy by **23.3 percentage points**.

Compared with the single-LLM baseline, it improved accuracy by **5.4 percentage points**.

## Technologies
Python
PyTorch
Hugging Face Transformers
Sentence Transformers
OpenAI API
Anthropic API
Google Gemini API
Streamlit
Pandas
NumPy
Scikit-learn

## Project Structure

```text
sentiment-aware-multi-llm-chatbot/
│
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   └── pipeline.py
│
├── notebooks/
│   ├── Sentiment_Analysis.ipynb
│   └── pipeline.ipynb
│
├── data/
│   └── truthfulqa.csv
│
└── models/
    └── README.md
```
## Installation

Clone the repository:

git clone <YOUR-GITHUB-REPOSITORY-URL>
cd sentiment-aware-multi-llm-chatbot

Install the required Python packages:

pip install -r requirements.txt

## API Configuration

The application requires API keys for the external LLM providers.

Create a .env file in the project root based on .env.example:

OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key

Never commit your .env file or actual API keys to GitHub.

## Running the Application

From the project root directory, run:

streamlit run app/streamlit_app.py

## Research Objectives

This project investigates two main questions:

Can sentiment detection allow a chatbot to adapt its response tone according to the user's emotional state?
Can using multiple heterogeneous LLMs improve response reliability compared with relying on a single LLM?
Research Context

The project was developed as an MSc Computing (Artificial Intelligence) practicum project at Dublin City University.

The framework aims to combine two areas that are often treated separately: emotional awareness in conversational systems and factual validation of LLM-generated responses.

## Limitations
LLM-based evaluators may themselves make incorrect judgements.
API-based evaluation introduces additional latency and cost.
Sentiment classification may not correctly capture sarcasm, irony or complex emotional states.
TruthfulQA evaluates factual truthfulness but does not represent every type of real-world chatbot interaction.
The system is intended for research and demonstration rather than safety-critical use.
