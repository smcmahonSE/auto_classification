"""
Legacy entrypoint kept for convenience.

This script orchestrates:
1) build_training_data.py
2) train_model.py
"""
import subprocess
import sys


def main():
    cmds = [
        [sys.executable, "build_training_data.py"],
        [sys.executable, "train_model.py"],
    ]
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print("Done. ONNX model available at artifacts/model/product_classifier.onnx")


if __name__ == "__main__":
    main()
"""
Legacy entrypoint kept for convenience.

This script now orchestrates:
1) build_training_data.py
2) train_model.py

For full control, run those scripts directly.
"""
import subprocess
import sys


def main():
    cmds = [
        [sys.executable, "build_training_data.py"],
        [sys.executable, "train_model.py"],
    ]
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print("Done. ONNX model available at artifacts/model/product_classifier.onnx")


if __name__ == "__main__":
    main()
"""
Train product category classifier locally:
- Load training data from Snowflake
- Embed text with AWS Titan (Bedrock)
- Train LightGBM and evaluate
"""
import json
import os
import pickle

import boto3
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from snowflake.snowpark import Session
from lightgbm import LGBMClassifier

# Optional: progress bar for embedding step
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ----------------------------
# Snowflake connection & data load
# ----------------------------
def get_snowflake_session(connection_params=None):
    """Create Snowflake session. Uses env or the provided connection_params."""
    if connection_params is None:
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT", "NTWRVFU-UEC95409"),
            "user": os.environ.get("SNOWFLAKE_USER", "STEPHANIE.MCMAHON@SCIENCEEXCHANGE.COM"),
            "authenticator": "externalbrowser",
            # Optional: set if your role needs a specific warehouse/role to see the table
            # "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
            # "role": os.environ.get("SNOWFLAKE_ROLE"),
        }
        # Drop None values so optional env vars don't break the connection
        connection_params = {k: v for k, v in connection_params.items() if v is not None}
    return Session.builder.configs(connection_params).create()


# Fully qualified table: DATABASE.SCHEMA.TABLE. Override with env SNOWFLAKE_PRODUCTS_TABLE if needed.
PRODUCTS_TABLE = os.environ.get(
    "SNOWFLAKE_PRODUCTS_TABLE",
    "SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_L3_SUB",
)


def load_product_data(session, table=None):
    """Load product training data from Snowflake (categories with >= 100 samples)."""
    table = table or PRODUCTS_TABLE
    query = f"""
    SELECT *
    FROM {table}
    WHERE parent_3_category IN (
        SELECT parent_3_category
        FROM {table}
        GROUP BY parent_3_category
        HAVING COUNT(*) >= 100
    )
    """
    df_snowflake = session.sql(query)
    try:
        # Fast path when connector pandas extras are installed
        return df_snowflake.to_pandas()
    except Exception as exc:
        # Fallback path if Snowflake connector cannot use pandas fetch helpers
        # (commonly missing pandas/pyarrow optional connector dependencies)
        err = str(exc)
        if "Optional dependency: 'pandas' is not installed" in err:
            rows = df_snowflake.collect()
            return pd.DataFrame([row.as_dict() for row in rows])
        raise


# ----------------------------
# AWS Bedrock Titan embeddings
# ----------------------------
def invoke_titan_embed(client, text, model_id="amazon.titan-embed-text-v1"):
    """Get embedding for a single text via Amazon Titan Embeddings."""
    text = str(text).strip()
    if not text:
        text = " "
    payload = {"inputText": text}
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )
    result = json.loads(response["body"].read().decode("utf-8"))
    return result.get("embedding", [])


def get_bedrock_client(profile_name=None, region="us-east-1"):
    """Create Bedrock runtime client. Uses profile_name if provided."""
    if profile_name:
        boto3.setup_default_session(profile_name=profile_name)
    return boto3.client(service_name="bedrock-runtime", region_name=region)


# ----------------------------
# Text concatenation
# ----------------------------
class TextConcatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            "Name: "
            + X["PRODUCT_NAME"].fillna("Unknown")
            + ", Description: "
            + X["DESCRIPTION"].fillna("No description provided")
            + ", Pricing Status: "
            + X["PRICING_STATUS_C"].fillna("Unknown")
            + ", List Price: "
            + X["LIST_PRICE_C"].astype(str).fillna("Not available")
        ).tolist()


# ----------------------------
# Titan embedding transformer
# ----------------------------
class TitanEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that embeds text via AWS Titan (Bedrock)."""

    def __init__(
        self,
        bedrock_client=None,
        model_id="amazon.titan-embed-text-v1",
        profile_name=None,
        region="us-east-1",
        show_progress=True,
    ):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
        self.profile_name = profile_name
        self.region = region
        self.show_progress = show_progress

    def fit(self, X, y=None):
        if self.bedrock_client is None:
            self.bedrock_client_ = get_bedrock_client(
                profile_name=self.profile_name, region=self.region
            )
        else:
            self.bedrock_client_ = self.bedrock_client
        return self

    def transform(self, X):
        client = self.bedrock_client_
        model_id = self.model_id
        it = tqdm(X, desc="Titan embed") if (self.show_progress and HAS_TQDM) else X
        embeddings = []
        for text in it:
            emb = invoke_titan_embed(client, text, model_id=model_id)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


# ----------------------------
# Main
# ----------------------------
def main():
    # AWS profile for Bedrock (e.g. staging.admin). If token expired: aws-sso-util login
    boto3.setup_default_session(profile_name="staging.admin")
    bedrock_client = get_bedrock_client(profile_name=None, region="us-east-1")

    print("Connecting to Snowflake (browser auth may open)...")
    session = get_snowflake_session()
    df = load_product_data(session)
    print(f"Loaded {len(df)} rows from Snowflake.")

    # Prepare targets
    X = df.copy()
    y = X["PARENT_3_CATEGORY"]
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.1, stratify=y_enc, random_state=42
    )
    num_classes = len(label_encoder.classes_)

    # Pipeline: text concat -> Titan embed -> LightGBM
    pipeline_lgbm = Pipeline(
        [
            ("text", TextConcatTransformer()),
            ("embed", TitanEmbeddingTransformer(bedrock_client=bedrock_client)),
            (
                "lgbm",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=num_classes,
                    learning_rate=0.05,
                    num_leaves=64,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_estimators=1000,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training pipeline (text -> Titan embed -> LightGBM)...")
    pipeline_lgbm.fit(X_train, y_train)

    y_pred = pipeline_lgbm.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )

    # Save pipeline and label encoder for deployment
    with open("product_classifier_pipeline.pkl", "wb") as f:
        pickle.dump({"pipeline": pipeline_lgbm, "label_encoder": label_encoder}, f)
    print("Saved pipeline and label encoder to product_classifier_pipeline.pkl")


if __name__ == "__main__":
    main()
