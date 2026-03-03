import streamlit as st

from src.preprocessing import load_data, prepare_dataframe
from src.embeddings import EmbeddingModel
from src.search import SemanticSearch

st.set_page_config(page_title="Semantic Search Demo", layout="wide")

st.title("🛍️ E-commerce Semantic Search")

@st.cache_resource
def load_model_and_index():
    df = load_data("data/products.csv")
    df = prepare_dataframe(df)

    model = EmbeddingModel()
    product_embeddings = model.encode_texts(df["product_title"].tolist())

    search_engine = SemanticSearch(product_embeddings)

    return df, model, search_engine

df, model, search_engine = load_model_and_index()

# Search bar
query = st.text_input("Search for products:")
search_clicked = st.button("Search")

def safe_image(url):
    if isinstance(url, str) and url.strip():
        return url
    return "https://via.placeholder.com/150"

# Function to display cards
def display_cards(df_display):
    cards_per_row = 4
    for i in range(0, len(df_display), cards_per_row):
        cols = st.columns(cards_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(df_display):
                break
            row = df_display.iloc[i + j]
            with col:
                st.image(safe_image(row.get("img_source_url")), width=200)
                st.markdown(f"**{row['product_title']}**")
                st.caption(row.get("category_name", ""))

# Search results
if search_clicked and query.strip() != "":
    query_embedding = model.encode_query(query)
    distances, indices = search_engine.search(query_embedding, top_k=20)

    results = df.iloc[indices][["product_title", "category_name", "img_source_url"]]

    st.subheader("🔍 Search Results")
    display_cards(results)

# Default product listing
st.subheader("📦 All Products")

if "display_count" not in st.session_state:
    st.session_state.display_count = 100

display_df = df.iloc[:st.session_state.display_count]
display_cards(display_df)

if st.session_state.display_count < len(df):
    if st.button("Load More"):
        st.session_state.display_count += 100
