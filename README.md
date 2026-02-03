# BERT Sentiment Classification (Amazon Polarity)

This project fine-tunes a BERT-based classifier to predict review sentiment (positive vs negative) using the Amazon Polarity dataset.  
It includes a baseline model, architecture tweaks to the classification head, and hyperparameter experiments (layers, batch size, epochs, learning rate).

## Dataset
- Dataset: Amazon Polarity (via Hugging Face `datasets`)
- Task: Binary sentiment classification
- Subset used for faster training:
  - Train: [500] examples
  - Test: [5 or 200] examples

## What I built
### Tokenization
- BERT tokenizer: `bert-base-uncased`
- Inputs: `input_ids`, `attention_mask`, `token_type_ids`
- Padding/truncation applied to a fixed max length

### Models
- **Baseline / V2:** BERT encoder + 2-layer feed-forward classification head
- **V3:** BERT encoder + deeper (3-layer) classification head

## Experiments
I compared these configurations:
- V2 head, epochs=5, batch=32, lr=2e-5
- V2 head, epochs=5, batch=16, lr=2e-5
- **Learning rate test:** V2 head, epochs=5, batch=16, lr=1e-5
- **V3 architecture test:** deeper head, epochs=5, batch=16, lr=2e-5

## Results
- Logs: `outputs/bert_experiment_results.csv`
- Best run report: `outputs/best_run_report.txt`

**Best configuration (by F1):**  
[Paste your best run name here]

> Note: If your test subset is very small (ex: 5 samples), metrics can look inflated.  
> The log file documents the exact setup used.

## Files
- Notebook: `Week4_BERT_AmazonPolarity.ipynb`
- Outputs:
  - `outputs/bert_experiment_results.csv`
  - `outputs/best_run_report.txt`
  - `outputs/*config.json`
- Slides: `slides/Week4_BERT_AmazonPolarity.pptx`

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
