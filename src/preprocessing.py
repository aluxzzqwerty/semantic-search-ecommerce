import re
import pandas as pd

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    df["img_source_url"] = df["img_source_url"].fillna("")
    df["product_title"] = df["product_title"].fillna("")
    df["product_title"] = df["product_title"].apply(clean_text)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
