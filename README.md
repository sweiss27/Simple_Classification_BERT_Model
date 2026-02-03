# Simple Classification BERT Model (Amazon Polarity)

This project fine-tunes **BERT (bert-base-uncased)** for **binary sentiment classification** (positive vs negative) using the **Amazon Polarity** dataset.  
I built a baseline classifier head, then tested modified heads + hyperparameter combinations and compared results using **Accuracy, Precision, Recall, and F1-score**.

---

## Dataset
- **Amazon Polarity** (loaded via the Hugging Face `datasets` library)
- For faster experimentation, the notebook uses:
  - `train: 500 samples`
  - `test: 5 samples`

> Note: With a very small test set (n=5), metrics can swing heavily and may look “too perfect.”  
> That’s expected and doesn’t represent real-world generalization.

---

## Approach (high level)
1. Load Amazon Polarity train/test splits  
2. Tokenize with **BERT tokenizer** (input_ids, attention_mask, token_type_ids)  
3. Train a **Custom BERT classifier** (BERT encoder + classification head)  
4. Run experiments:
   - baseline vs modified head(s)
   - batch size / epochs / learning rate comparisons  
5. Evaluate with Accuracy, Precision, Recall, and F1

---

## Model Variants
### Baseline (V1)
- BERT pooled output → Dropout → Linear(2 classes)

### V2 (modified head)
- BERT pooled output → Dropout → Linear → activation → Dropout → Linear(2 classes)

### V3 (deeper head)
- BERT pooled output → deeper feed-forward stack → Linear(2 classes)

---

## Experiments Summary
| Experiment | Batch Size | LR | Epochs | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combo1: V2 head (2-layer) | 32 | 2e-5 | 5 | 0.80 | 1.00 | 0.75 | 0.857 |
| Combo2: V2 head (2-layer) | 16 | 2e-5 | 5 | 1.00 | 1.00 | 1.00 | 1.000 |
| LR Test: V2 head (2-layer) | 16 | 1e-5 | 5 | 0.80 | 1.00 | 0.75 | 0.857 |
| V3 Test: deeper head (3-layer) | 16 | 2e-5 | 5 | 0.80 | 0.80 | 1.00 | 0.889 |

**Best run (by F1):** Combo2: V2 head (2-layer), epochs=5, bs=16, lr=2e-5

---

## Outputs Saved (for submission)
Depending on what you saved in the notebook, you may have:
- `bert_experiment_results.csv` (experiment log)
- `best_bert_model_v2.pth` or `best_bert_model_v3.pth` (trained weights)
- `best_bert_model_v2_config.json` (best config metadata)

---

## How to Run
### 1) Install dependencies
```bash
pip install torch transformers datasets scikit-learn tqdm pandas
