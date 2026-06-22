# ============================================================
# FINAL IMPROVED EVALUATION PIPELINE
# HIGH + MEDIUM PRIORITY VERSION
# ============================================================

import pandas as pd
import numpy as np
import time
import re

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STYLE
# ============================================================

sns.set_style("whitegrid")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ground_truth.csv")

print("Loaded rows:", len(df))

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

# -----------------------------------
# SENTIMENT MODELS
# -----------------------------------

distilbert = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# -----------------------------------
# FLAN-T5 FOR ISSUE CLASSIFICATION
# -----------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

flan_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

# -----------------------------------
# EMBEDDING MODELS
# -----------------------------------

minilm = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

bge = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sentiment(label):

    label = str(label).lower()

    if "positive" in label:
        return "positive"

    if "negative" in label:
        return "negative"

    return "neutral"

# ============================================================
# FLAN ISSUE CLASSIFICATION
# ============================================================

VALID_ISSUES = [

    "design_issue",

    "performance_issue",

    "usability_issue",

    "durability_issue",

    "unexpected_behavior",

    "no_issue"
]

def clean_issue(output):

    output = output.lower()

    for issue in VALID_ISSUES:

        if issue in output:
            return issue

    return "no_issue"

def classify_issue_flan(text):

    prompt = f"""
Classify the issue into one of:

design_issue,
performance_issue,
usability_issue,
durability_issue,
unexpected_behavior,
no_issue

Text:
{text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = flan_model.generate(
        **inputs,
        max_new_tokens=20
    )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_issue(prediction)

# ============================================================
# PRODUCT EXTRACTION
# ============================================================

def extract_product(text):

    patterns = [

        r'product\s*id[:\s]*([a-z0-9]{8,12})',

        r'for\s+product\s+([a-z0-9]{8,12})',

        r'product\s+([a-z0-9]{8,12})'
    ]

    for p in patterns:

        match = re.search(
            p,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return None

# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(text):

    patterns = [

        r'(\d{2}\s\d{2},\s\d{4})',

        r'(\d{2}/\d{2}/\d{4})',

        r'(\d{4}-\d{2}-\d{2})',

        r'(\w+\s\d{1,2},\s\d{4})'
    ]

    for p in patterns:

        match = re.search(p, text)

        if match:
            return match.group(1)

    return None

# ============================================================
# PRECOMPUTE EMBEDDINGS
# ============================================================

all_texts = df["text"].astype(str).tolist()

all_issues = df["issue"].astype(str).tolist()

print("Generating MiniLM embeddings...")

minilm_text_emb = minilm.encode(
    all_texts
)

minilm_issue_emb = minilm.encode(
    all_issues
)

print("Generating BGE embeddings...")

bge_text_emb = bge.encode(
    all_texts
)

bge_issue_emb = bge.encode(
    all_issues
)

# ============================================================
# STORAGE
# ============================================================

results = []

true_sentiments = []

pred_sentiments_distil = []

pred_sentiments_roberta = []

true_issues = []

pred_issues = []

true_products = []

pred_products = []

true_dates = []

pred_dates = []

similarity_scores_minilm = []

similarity_scores_bge = []

# ============================================================
# MAIN LOOP
# ============================================================

for idx, row in df.iterrows():

    text = str(row["text"])

    # -----------------------------------
    # TRUE VALUES
    # -----------------------------------

    true_sentiment = normalize_sentiment(
        row["sentiment"]
    )

    true_issue = row["issue"]

    true_product = str(row["product"])

    true_date = str(row["date"])

    # ========================================================
    # DISTILBERT
    # ========================================================

    start = time.time()

    try:

        pred1 = distilbert(
            text[:512]
        )[0]["label"]

        pred1 = normalize_sentiment(
            pred1
        )

    except:

        pred1 = "neutral"

    t1 = time.time() - start

    # ========================================================
    # ROBERTA
    # ========================================================

    start = time.time()

    try:

        pred2 = roberta(
            text[:512]
        )[0]["label"]

        pred2 = normalize_sentiment(
            pred2
        )

    except:

        pred2 = "neutral"

    t2 = time.time() - start

    # ========================================================
    # FLAN ISSUE CLASSIFICATION
    # ========================================================

    start = time.time()

    pred_issue = classify_issue_flan(
        text
    )

    t3 = time.time() - start

    # ========================================================
    # PRODUCT + DATE EXTRACTION
    # ========================================================

    pred_product = extract_product(
        text
    )

    pred_date = extract_date(
        text
    )

    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    emb1 = [minilm_text_emb[idx]]

    emb2 = [minilm_issue_emb[idx]]

    sim1 = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    emb3 = [bge_text_emb[idx]]

    emb4 = [bge_issue_emb[idx]]

    sim2 = cosine_similarity(
        emb3,
        emb4
    )[0][0]

    similarity_scores_minilm.append(
        sim1
    )

    similarity_scores_bge.append(
        sim2
    )

    # ========================================================
    # STORE VALUES
    # ========================================================

    true_sentiments.append(
        true_sentiment
    )

    pred_sentiments_distil.append(
        pred1
    )

    pred_sentiments_roberta.append(
        pred2
    )

    true_issues.append(
        true_issue
    )

    pred_issues.append(
        pred_issue
    )

    true_products.append(
        true_product
    )

    pred_products.append(
        str(pred_product)
    )

    true_dates.append(
        true_date
    )

    pred_dates.append(
        str(pred_date)
    )

    # ========================================================
    # SAVE ROW
    # ========================================================

    results.append({

        "source": row["source"],

        "text_preview": text[:100],

        "true_sentiment": true_sentiment,

        "distilbert_sentiment": pred1,

        "roberta_sentiment": pred2,

        "true_issue": true_issue,

        "predicted_issue": pred_issue,

        "true_product": true_product,

        "predicted_product": pred_product,

        "true_date": true_date,

        "predicted_date": pred_date,

        "distilbert_time": round(t1, 4),

        "roberta_time": round(t2, 4),

        "issue_time": round(t3, 4),

        "minilm_similarity": round(sim1, 4),

        "bge_similarity": round(sim2, 4)
    })

# ============================================================
# SAVE OUTPUTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_output.csv",
    index=False
)

print("Saved model_outputs.csv")

# ============================================================
# METRICS
# ============================================================

metrics = []

# ========================================================
# DISTILBERT
# ========================================================

metrics.append({

    "model": "DistilBERT",

    "task": "Sentiment",

    "accuracy":
        accuracy_score(
            true_sentiments,
            pred_sentiments_distil
        ),

    "precision":
        precision_score(
            true_sentiments,
            pred_sentiments_distil,
            average="weighted",
            zero_division=0
        ),

    "recall":
        recall_score(
            true_sentiments,
            pred_sentiments_distil,
            average="weighted",
            zero_division=0
        ),

    "f1":
        f1_score(
            true_sentiments,
            pred_sentiments_distil,
            average="weighted",
            zero_division=0
        ),

    "avg_time":
        np.mean(
            results_df["distilbert_time"]
        )
})

# ========================================================
# ROBERTA
# ========================================================

metrics.append({

    "model": "RoBERTa",

    "task": "Sentiment",

    "accuracy":
        accuracy_score(
            true_sentiments,
            pred_sentiments_roberta
        ),

    "precision":
        precision_score(
            true_sentiments,
            pred_sentiments_roberta,
            average="weighted",
            zero_division=0
        ),

    "recall":
        recall_score(
            true_sentiments,
            pred_sentiments_roberta,
            average="weighted",
            zero_division=0
        ),

    "f1":
        f1_score(
            true_sentiments,
            pred_sentiments_roberta,
            average="weighted",
            zero_division=0
        ),

    "avg_time":
        np.mean(
            results_df["roberta_time"]
        )
})

# ========================================================
# FLAN-T5
# ========================================================

metrics.append({

    "model": "FLAN-T5",

    "task": "Issue Classification",

    "accuracy":
        accuracy_score(
            true_issues,
            pred_issues
        ),

    "precision":
        precision_score(
            true_issues,
            pred_issues,
            average="weighted",
            zero_division=0
        ),

    "recall":
        recall_score(
            true_issues,
            pred_issues,
            average="weighted",
            zero_division=0
        ),

    "f1":
        f1_score(
            true_issues,
            pred_issues,
            average="weighted",
            zero_division=0
        ),

    "avg_time":
        np.mean(
            results_df["issue_time"]
        )
})

# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    "metrics.csv",
    index=False
)

print("Saved metrics.csv")

# ============================================================
# EXTRACTION METRICS
# ============================================================

product_accuracy = accuracy_score(
    true_products,
    pred_products
)

date_accuracy = accuracy_score(
    true_dates,
    pred_dates
)

print("\nProduct Accuracy:",
      round(product_accuracy * 100, 2))

print("Date Accuracy:",
      round(date_accuracy * 100, 2))

# ============================================================
# GRAPH 1 — ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="model",
    y="accuracy",
    data=metrics_df,
    palette="viridis"
)

plt.title(
    "Model Accuracy Comparison",
    fontsize=15
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300
)

# ============================================================
# GRAPH 2 — F1 SCORE
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="model",
    y="f1",
    data=metrics_df,
    palette="magma"
)

plt.title(
    "Model F1-Score Comparison",
    fontsize=15
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "f1_comparison.png",
    dpi=300
)

# ============================================================
# GRAPH 3 — INFERENCE TIME
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="model",
    y="avg_time",
    data=metrics_df,
    palette="coolwarm"
)

plt.title(
    "Inference Time Comparison",
    fontsize=15
)

plt.ylabel("Seconds")

plt.tight_layout()

plt.savefig(
    "inference_time.png",
    dpi=300
)

# ============================================================
# GRAPH 4 — EFFICIENCY TRADEOFF
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="avg_time",
    y="accuracy",
    hue="model",
    s=250,
    data=metrics_df
)

for i, row in metrics_df.iterrows():

    plt.text(
        row["avg_time"] + 0.003,
        row["accuracy"],
        row["model"]
    )

plt.title(
    "Efficiency vs Accuracy",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "efficiency_tradeoff.png",
    dpi=300
)

# ============================================================
# GRAPH 5 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_sentiments,
    pred_sentiments_roberta
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "RoBERTa Confusion Matrix",
    fontsize=15
)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

# ============================================================
# GRAPH 6 — SIMILARITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    similarity_scores_minilm,
    color="blue",
    label="MiniLM",
    kde=True
)

sns.histplot(
    similarity_scores_bge,
    color="orange",
    label="BGE-small",
    kde=True
)

plt.legend()

plt.title(
    "Semantic Similarity Distribution",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "similarity_distribution.png",
    dpi=300
)

# ============================================================
# GRAPH 7 — PRECISION / RECALL
# ============================================================

metrics_melted = metrics_df.melt(

    id_vars=["model"],

    value_vars=[
        "precision",
        "recall"
    ],

    var_name="Metric",

    value_name="Score"
)

plt.figure(figsize=(8, 5))

sns.barplot(
    x="model",
    y="Score",
    hue="Metric",
    data=metrics_melted
)

plt.title(
    "Precision vs Recall",
    fontsize=15
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "precision_recall.png",
    dpi=300
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n===================================")
print("FINAL EVALUATION COMPLETE")
print("===================================")

print("""
Generated Files:

1. model_outputs.csv
2. metrics.csv
3. accuracy_comparison.png
4. f1_comparison.png
5. inference_time.png
6. efficiency_tradeoff.png
7. confusion_matrix.png
8. similarity_distribution.png
9. precision_recall.png
""")