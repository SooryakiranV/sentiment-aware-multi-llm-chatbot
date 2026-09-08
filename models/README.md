# Sentiment Model

The `best_model` directory contains the fine-tuned RoBERTa sentiment classification model used by the chatbot.

The trained model is not included in this repository because model files can be large.

To run the complete pipeline locally, place the trained model in this directory:

```text
models/
└── best_model/

The model should contain the files required by Hugging Face Transformers to load the fine-tuned classifier.

The sentiment model classifies user queries into four categories:

Positive
Neutral
Frustrated
Confused

The model was trained as part of the sentiment analysis component.