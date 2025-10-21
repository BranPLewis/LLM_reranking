#!/usr/bin/env python3

"""
https://aistudio.google.com/ "Get API key"
https://aistudio.google.com/apikey
Put key in .env file with format:
GEMINI_API_KEY="the key"
Use OpenAIServerModel due to API compatibility
[Select model](https://ai.google.dev/gemini-api/docs/models)
"""

"""
Evaluate retrieval quality on the RAG sample dataset.

Computes Precision@K, Recall@K, and nDCG@K
for each query and averages the results.

Assumes the dataset columns:
  query_id, query_text, candidate_id, candidate_text,
  baseline_rank, baseline_score, gold_label
"""
from smolagents import OpenAIServerModel
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np
import dotenv, os, time
g_dotenv_loaded = False
def getenv(variable: str) -> str:
    global g_dotenv_loaded
    if not g_dotenv_loaded:
        g_dotenv_loaded = True
        dotenv.load_dotenv()
    value = os.getenv(variable)
    return value

api_key = getenv("GEMINI_API_KEY")

if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

#############################################
# Model connection
#############################################
model_id="gemini-2.5-flash"
model = OpenAIServerModel(
    model_id=model_id,
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)

#############################################
# 1. Load data
#############################################
df = pd.read_csv("rag_sample_queries_candidates.csv")
query_text = df.query_text
candidate_text = df.candidate_text
q_c = list(zip(query_text, candidate_text))
llm_scores = []
query_times = []
overall_time_1 = time.time()
for query, candidate_text in q_c:
    time_1 = time.time()
    try:
        message = [
            {
                "role": "user",
                "content": f"""Query: {query}\nCandidate passage: {candidate_text}\n
                Give me a rating for the relevance of the candidate passage to the query
                on a scale from 1 (not relevant) to 5 (highly relevant). Respond only with
                a single number.""",
            }
        ]
        answer = model.generate(messages=message)
    except: 
        print("Error querying LLM")
    llm_scores.append(answer.content)
    time_2 = time.time()
    query_times.append(time_2 - time_1)
overall_time_2 = time.time()
# Time reporting
print("Time overall: ", overall_time_2 - overall_time_1)
print("Average time per query: ", np.mean(query_times))
# Add LLM scores and ranks to the dataframe
df['llm_score'] = [float(score) for score in llm_scores]
df['llm_rank'] = df.groupby('query_id')['llm_score'].rank(
    method='first', 
    ascending=False
).astype(int)
# Save reranked results to a CSV
df.sort_values(["query_id", "llm_score"], ascending=[True,False])
results = pd.DataFrame(df[['query_id','candidate_id','llm_score','llm_rank']])
results.to_csv("results.csv", index=False)

#############################################
# 2. Metric helpers
#############################################
def precision_at_k(labels, k):
    """labels: list/array of 0/1 relevance sorted by baseline rank"""
    topk = labels[:k]
    return np.sum(topk) / len(topk)

def recall_at_k(labels, k):
    """Recall = retrieved relevant / total relevant"""
    total_relevant = np.sum(labels)
    if total_relevant == 0:
        return np.nan  # undefined
    topk = labels[:k]
    return np.sum(topk) / total_relevant

def ndcg_at_k(labels, k):
    """Compute nDCG@k with binary relevance (0/1)."""
    labels = np.array(labels)
    k = min(k, len(labels))
    gains = (2 ** labels[:k] - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)

    # Ideal DCG: sorted by true relevance
    ideal = np.sort(labels)[::-1]
    ideal_gains = (2 ** ideal[:k] - 1)
    idcg = np.sum(ideal_gains * discounts)
    return 0.0 if idcg == 0 else dcg / idcg

#############################################
# 3. Compute metrics per query
#############################################
results = []
K = 3

for qid, group in df.groupby("query_id"):
    baseline_sorted = group.sort_values("baseline_rank", ascending=False)
    labels_baseline = baseline_sorted.gold_label.tolist()

    llm_sorted = group.sort_values("llm_score", ascending=False)
    labels_llm = llm_sorted.gold_label.tolist()

    query_result = {
        "query_id": qid,
        f"P@{K}_baseline": precision_at_k(labels_baseline, K),
        f"R@{K}_baseline": recall_at_k(labels_baseline, K),
        f"nDCG@{K}_baseline": ndcg_at_k(labels_baseline, K),

        f"P@{K}_llm": precision_at_k(labels_llm, K),
        f"R@{K}_llm": recall_at_k(labels_llm, K),
        f"nDCG@{K}_llm": ndcg_at_k(labels_llm, K)
    }   
    results.append(query_result)    
metrics = pd.DataFrame(results)

#############################################
# 4. Display per-query and average metrics
#############################################
average_metrics = metrics.drop(columns="query_id").mean()
print("\nAverage metrics:")
print(average_metrics)
