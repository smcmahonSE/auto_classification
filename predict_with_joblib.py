import argparse

import joblib
import numpy as np
import pandas as pd

from product_classifier_utils import (
    build_product_text,
    get_bedrock_client,
    invoke_titan_embed,
)


class JoblibProductCategoryPredictor:
    """Inference wrapper: dataframe -> text concat -> Titan embedding -> joblib model output."""

    def __init__(
        self,
        model_path: str = "artifacts/model/product_classifier.joblib",
        aws_profile: str = "staging.admin",
        aws_region: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v1",
    ):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.pca = bundle.get("pca")
        self.label_encoder = bundle["label_encoder"]
        self.bedrock_client = get_bedrock_client(profile_name=aws_profile, region=aws_region)
        self.model_id = model_id

    def embed_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        texts = build_product_text(df).tolist()
        embeddings = [
            invoke_titan_embed(self.bedrock_client, text=t, model_id=self.model_id) for t in texts
        ]
        X = np.asarray(embeddings, dtype=np.float32)
        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        return X

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.embed_dataframe(df)
        pred_idx = self.model.predict(X).astype(int)
        labels = self.label_encoder.inverse_transform(pred_idx)
        result = pd.DataFrame({"predicted_category": labels})

        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(X))
            if probs.ndim == 2:
                result["prediction_confidence"] = probs.max(axis=1)

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run joblib inference on a CSV of product rows.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="artifacts/model/predictions.csv")
    parser.add_argument("--model-path", default="artifacts/model/product_classifier.joblib")
    parser.add_argument("--aws-profile", default="staging.admin")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model-id", default="amazon.titan-embed-text-v1")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    predictor = JoblibProductCategoryPredictor(
        model_path=args.model_path,
        aws_profile=args.aws_profile,
        aws_region=args.aws_region,
        model_id=args.model_id,
    )
    pred_df = predictor.predict(df)
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
