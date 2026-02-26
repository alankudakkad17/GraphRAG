import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.qa import build_hybrid_qa_chain
from app.graph import get_graph

# 1. Setup the evaluation data
# In a real scenario, these would be complex medical questions
eval_questions = [
    "What medications are prescribed for Patient X?",
    "Does the treatment for Disease Y have any side effects mentioned?",
    "Which doctor treated the patient with symptoms of hypertension?"
]

# You would ideally have 'ground_truth' answers for these
ground_truths = [
    ["Patient X is prescribed Lisinopril and Atorvastatin."],
    ["Treatment for Disease Y includes rest and Ibuprofen; no specific side effects noted."],
    ["Dr. Smith treated the patient with hypertension."]
]

def run_evaluation():
    graph = get_graph()
    hybrid_qa = build_hybrid_qa_chain(graph)
    
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": ground_truths
    }

    # 2. Run the pipeline to collect results
    for query in eval_questions:
        # We need to capture the raw response AND the context used
        # Note: You may need to modify your hybrid_qa to return context for Ragas
        response, context_list = hybrid_qa.get_detailed_response(query)
        
        data_samples["question"].append(query)
        data_samples["answer"].append(response)
        data_samples["contexts"].append(context_list)

    # 3. Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data_samples)

    # 4. Perform Evaluation
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return score.to_pandas()

if __name__ == "__main__":
    results = run_evaluation()
    print(results)
