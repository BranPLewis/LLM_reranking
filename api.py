#!/usr/bin/env python3

"""
https://aistudio.google.com/ "Get API key"
https://aistudio.google.com/apikey
Put key in .env file with format:
GEMINI_API_KEY="the key"
Use OpenAIServerModel due to API compatibility
[Select model](https://ai.google.dev/gemini-api/docs/models)
"""

#############################################
# Environment loading
#############################################
from rerank import df
import dotenv
import os
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
from smolagents import OpenAIServerModel

#model_id="gemini-2.0-flash"
#model_id="gemini-2.0-flash-lite"
model_id="gemini-2.5-flash"
model = OpenAIServerModel(
    model_id=model_id,
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)
query_text = df.query_text
candidate_text = df.candidate_text
q_c = list(zip(query_text, candidate_text))
llm_scores = []
for query, candidate_text in q_c:
    message = [
        {
            "role": "user",
            "content": f"""Query: {query}\nCandidate passage: {candidate_text}\n
            Give me a rating for the relevance of the candidate passage to the query
            on a scale from 0 (not relevant) to 100 (highly relevant). Respond only with
            a single number.""",
        }
    ]
    answer = model.generate(messages=message)
    llm_scores.append(answer.content)
df['llm_score'] = [float(score) for score in llm_scores]
df.to_csv('llm_ranked_results.csv', index=False)
print("Saved results to llm_ranked_results.csv")