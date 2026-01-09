# 🧠 Naive Bayes Variants on Personality Traits  
*A clean, code-only machine learning project focused on probabilistic modeling, preprocessing discipline, and metric interpretation.*

---

## 📌 Project Overview

This project implements and compares **three variants of the Naive Bayes algorithm** — **Gaussian**, **Multinomial**, and **Bernoulli** — on a **personality traits dataset inspired by the Big Five model**.

The objective is **not** to chase state-of-the-art performance, but to:

- understand how different Naive Bayes assumptions interact with data  
- apply **correct preprocessing per model variant**  
- evaluate models responsibly  
- highlight the **limitations of accuracy**, especially on synthetic data  

The project is intentionally built **without notebooks**, using a **modular, production-style Python structure**.

---

## 🧩 Dataset Description

### Personality Traits (Features)

Each sample represents a person described by five continuous traits (scaled 0–10):

- **Extroversion**
- **Openness**
- **Conscientiousness**
- **Agreeableness**
- **Neuroticism**

These traits are generated using correlated Gaussian distributions to resemble real psychological survey data.

### Target Variable

- `0` → Introvert-leaning personality  
- `1` → Extrovert-leaning personality  

The label is derived from a **latent personality score with injected noise**, ensuring:
- no single feature directly encodes the label  
- probabilistic (not deterministic) class boundaries  

The dataset is **synthetic but statistically realistic**, created to avoid overused public datasets while preserving real-world behavior.

---

## 🏗️ Project Structure

naive_bayes_personality_traits/
├── data/
│   └── dataset.csv
│
├── src/
│   ├── data_generator.py
│   ├── preprocess.py
│   ├── gaussian_nb.py
│   ├── multinomial_nb.py
│   ├── bernoulli_nb.py
│   ├── evaluate.py
│   └── visualize.py
│
├── results/
│   ├── metrics.json
│   └── accuracy_comparison.png
│
├── main.py
├── requirements.txt
└── README.md

---

## ⚠️ Note on Perfect Accuracy with Synthetic Data

During experimentation, all three Naive Bayes variants achieved **near-perfect accuracy** on this dataset.

This behavior is **intentional and informative**, not a modeling bug.

Because the dataset is synthetically generated:

- training and test samples follow the same underlying data-generation process  
- feature distributions remain stable  
- class boundaries are statistically consistent  

Naive Bayes models, which estimate class-conditional probabilities, can exploit this consistency extremely well.

In real-world datasets, factors such as:

- measurement noise  
- concept drift  
- imperfect labels  
- missing values  

prevent such ideal performance.

**Key takeaway:**  
High accuracy does not necessarily imply a robust or realistic model.

This project intentionally preserves this behavior to demonstrate the **limitations of evaluation metrics on clean synthetic data**.

---
##🧠 Key Learning Outcomes

- Proper preprocessing is model-dependent, not optional

- Naive Bayes performs exceptionally well on stable probabilistic data

- Synthetic data can lead to misleadingly high evaluation metrics

- Accuracy alone is an insufficient measure of model quality

- Clean project structure matters as much as model choice

##🚀 Future Improvements

- Evaluation on real personality survey data

- Calibration curves and log-loss analysis

- Comparison with Logistic Regression

- Robustness testing under label noise and distribution shift

##📎 Final Note

- This project emphasizes machine learning reasoning, not just implementation.
- It focuses on understanding why models behave the way they do, and when their results should be questioned.
