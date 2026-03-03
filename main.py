from src.preprocessing import load_data, prepare_dataframe
from src.embeddings import EmbeddingModel
from src.search import SemanticSearch

df = load_data("data/products.csv")
df = prepare_dataframe(df)

model = EmbeddingModel()
product_embeddings = model.encode_texts(df["product_title"].tolist())

search_engine = SemanticSearch(product_embeddings)

query = "comfortable running shoes"
query_embedding = model.encode_query(query)

distances, indices = search_engine.search(query_embedding, top_k=5)

print(df.iloc[indices][["product_title", "category_name"]])
