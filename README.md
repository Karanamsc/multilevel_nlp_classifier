# Multi-Level NLP Ticket Classification System
An end-to-end NLP-based ticket classification system that automatically routes and categorizes customer support tickets across three hierarchical levels, combining classical machine learning with transformer-based models.

---

## Project Overview

Customer support organizations often struggle with manual ticket triage, leading to delays, inconsistent routing, and high operational cost. This project demonstrates a **production-oriented NLP solution** that balances **speed, accuracy, and reliability** through a hybrid modeling approach.

**Key capabilities:**

* Automated ticket routing (Queue level)
* Ticket type classification with confidence-based escalation
* Deep semantic topic classification using DistilBERT

---

## Classification Hierarchy

| Level | Target              | Model                        | Purpose                             |
| ----- | ------------------- | ---------------------------- | ----------------------------------- |
| L1    | Support Queue       | TF-IDF + Logistic Regression | Fast, high-precision routing        |
| L2    | Ticket Type         | Calibrated Linear SVM        | Confidence-aware classification     |
| L3    | Topic / Subcategory | DistilBERT                   | Fine-grained semantic understanding |

---

## Tech Stack

* **Languages:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn (TF-IDF, Logistic Regression, SVM, calibration)
  * Hugging Face Transformers (DistilBERT)
  * datasets, Trainer API
* **Environment:** Google Colab

---

## Dataset

* **Size:** 1,785 real-world support tickets
* **Features:**

  * Text: `subject`, `body`
  * Metadata: `queue`, `type`
  * Multi-level tags (`tag_1` – `tag_7`)

### Data Preparation

* Combined `subject + body` into a single text feature
* Filtered to English-only tickets
* Removed duplicates
* Consolidated rare labels (<30 samples) into an `Other` category
* Applied label encoding with consistent ID ↔ label mappings

---

## Modeling Approach

### Level 1 – Queue Classification

* **Model:** TF-IDF + Logistic Regression
* **Why:** Interpretable, fast inference, strong baseline
* **Accuracy:** **80%**

### Level 2 – Ticket Type Classification

* **Model:** Calibrated Linear SVM
* **Why:** Strong margins with calibrated probabilities
* **Accuracy:** **75%**

#### Confidence-Based Decisioning

Predictions below a probability threshold (0.75) are flagged as:

```text
Needs Review
```

This enables **human-in-the-loop workflows** for low-confidence cases.

---

### Level 3 – Topic Classification (Transformer)

* **Model:** DistilBERT (`distilbert-base-uncased`)
* **Why DistilBERT:**

  * Strong semantic performance
  * Lower latency than BERT
  * Production-friendly size

**Results:**

* Accuracy: **70%**
* Weighted F1: **0.69**
* Best performance on high-volume classes (Performance, Payment)

---

## Evaluation

* Stratified train/test splits
* Metrics:

  * Accuracy
  * Weighted F1-score (handles class imbalance)
  * Confusion matrices for error analysis

---

## Key Challenges & Solutions

### 1. Unseen Label Errors

**Issue:**

```text
ValueError: y contains previously unseen labels
```

**Solution:**

* Ensured label decoding only used encoders fitted on training labels
* Maintained consistent label mappings across pipelines

---

## Design Principles

* Hybrid ML + Transformer architecture for optimal tradeoffs
* Explicit handling of class imbalance
* Confidence-aware automation
* Evaluation aligned with real-world deployment constraints

---

## Impact

* Reduces manual ticket triage effort
* Improves routing accuracy and response time
* Scalable to new categories and domains
* Suitable for real-world support operations

---

## Future Enhancements

* Hierarchical loss across L1 → L3
* Active learning using human-reviewed tickets
* Explainability (attention visualization, SHAP)
* API deployment with FastAPI

---

## How to Run

1. Clone the repository
2. Open the notebook in Google Colab
3. Mount Google Drive
4. Install dependencies
5. Run cells sequentially

---

## Author

**Sneha Karanam**

Data Analytics Engineer

---

If you find this project interesting, feel free to star the repository or reach out!

