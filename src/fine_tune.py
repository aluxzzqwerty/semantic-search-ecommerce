import random
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def generate_query_from_title(title):
    title = clean_text(title)
    words = title.split()

    if len(words) <= 3:
        return title

    # randomly select 2-4 words
    k = random.randint(2, min(4, len(words)))
    selected_words = random.sample(words, k)

    return " ".join(selected_words)


def create_training_examples(df, num_samples=20000):
    examples = []

    df = df.sample(min(num_samples, len(df)))

    for _, row in df.iterrows():
        title = str(row["product_title"])

        if len(title.split()) < 3:
            continue

        query = generate_query_from_title(title)

        examples.append(InputExample(texts=[query, title]))

    return examples


def train_model(data_path="data/products.csv"):
    df = pd.read_csv(data_path)

    print("Generating training examples...")
    train_examples = create_training_examples(df)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=32
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Training started...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        show_progress_bar=True
    )

    model.save("fine_tuned_model")
    print("Model saved to fine_tuned_model")


if __name__ == "__main__":
    train_model()