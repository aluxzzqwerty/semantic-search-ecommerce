from src.preprocessing import load_data, prepare_dataframe
from src.embeddings import EmbeddingModel
from src.search import SemanticSearch
from src.evaluation import precision_at_k, recall_at_k, mean_reciprocal_rank

TOP_K = 5

def main():

    df = load_data("data/products.csv")
    df = prepare_dataframe(df)

    # Build embeddings
    model = EmbeddingModel()
    product_embeddings = model.encode_texts(df["product_title"].tolist())

    search_engine = SemanticSearch(product_embeddings)

    # Evaluation queries
    eval_queries = [
        {"query": "portable fan", "relevant_ids": [1, 3, 9, 16, 44]},
        {"query": "oral irrigator", "relevant_ids": [4, 112, 350, 527, 565]},
        {"query": "eyelash curler", "relevant_ids": [2, 11, 37, 47, 84]}
    ]

    all_relevant = []
    all_predictions = []

    print("\n🔎 Running evaluation...\n")

    for item in eval_queries:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        query_embedding = model.encode_query(query)
        distances, indices = search_engine.search(query_embedding, top_k=TOP_K)

        predicted_ids = indices.tolist()

        p = precision_at_k(relevant_ids, predicted_ids, TOP_K)
        r = recall_at_k(relevant_ids, predicted_ids, TOP_K)

        print(f"Query: {query}")
        print("Predicted IDs:", predicted_ids)
        print("Precision@5:", round(p, 3))
        print("Recall@5:", round(r, 3))
        print("-" * 40)

        all_relevant.append(set(relevant_ids))
        all_predictions.append(predicted_ids)

    # 4️⃣ Final MRR
    mrr = mean_reciprocal_rank(all_relevant, all_predictions)

    print("\n========================")
    print("FINAL METRICS")
    print("========================")
    print("MRR:", round(mrr, 3))


if __name__ == "__main__":
    main()