import pandas as pd
from datasets import Dataset
import  json
from gdoc.gdoc.retrieval import searching
from gdoc.gdoc.models import dense_model_
# RAGAS
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_accuracy
)

# embedd_model = "nomic-ai/nomic-embed-text-v2-moe"
# judge_model="claude"
# eval_results = []

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_accuracy
]
# def prepare_eval_data(eval_res : list):
#     ragas_data = {
#         "question" = r["question"] for r in eval_res,
#         "context" = r["context"] for r in eval_res,
#         "ground_truth" = r["ground_truth"] for r in eval_res,
#         "answer" = r["answer"] for r in eval_res,
#     }
#     ragas_dataset = Dataset.from_dict(ragas_data)
#     return ragas_dataset

# def run_ragas(ragas_data):
#     llm = gemini
#     results = evaluate(
#         dataset = ragas_data,
#         metrics = metrics,
#         llm = llm,
#         embedding = dense_model_()
#     )
#     res_df = results.to_pandas()
#     res_df.to_csv("/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/ragas_data.csv",index=False)
#     return "Success"


import frappe
@frappe.whitelist(allow_guest=True)
def test():
    f = open("/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/ragas_data.json", "r") #use with open
    data = json.load(f)
    eval_res = []
    for i,item in enumerate(data):
        query = item["question"]
        result = searching(query)
        print(result)
        # result["ground_truth"] =  item["ground_truth"]
        # print(result)
    #     eval_res.append(result)
    # ragas_data = prepare_eval_data(eval_res)
    # run_ragas(ragas_data)
    f.close()





#     # searching(query)
