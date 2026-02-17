import argparse
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from product_classifier_utils import (
    build_product_text,
    get_bedrock_client,
    invoke_titan_embed,
)


class OnnxProductCategoryPredictor:
    """Inference wrapper: dataframe -> text concat -> Titan embedding -> ONNX model output."""

    def __init__(
        self,
        onnx_path: str = "artifacts/model/product_classifier.onnx",
        classes_path: str = "artifacts/model/classes.npy",
        pca_path: str = "artifacts/model/pca.pkl",
        aws_profile: str = "staging.admin",
        aws_region: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v1",
    ):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.classes = np.load(classes_path, allow_pickle=True)
        self.bedrock_client = get_bedrock_client(profile_name=aws_profile, region=aws_region)
        self.model_id = model_id
        self.pca = None
        if pca_path and Path(pca_path).exists():
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
            print(f"Loaded PCA transformer from {pca_path}")

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
        outputs = self.session.run(None, {self.input_name: X})

        if len(outputs) >= 1 and np.issubdtype(np.asarray(outputs[0]).dtype, np.integer):
            pred_idx = np.asarray(outputs[0]).astype(int)
            probs = np.asarray(outputs[1]) if len(outputs) > 1 else None
        else:
            probs = np.asarray(outputs[-1])
            pred_idx = probs.argmax(axis=1)

        labels = self.classes[pred_idx]
        result = pd.DataFrame({"predicted_category": labels})

        if probs is not None and probs.ndim == 2:
            result["prediction_confidence"] = probs.max(axis=1)

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference on a CSV of product rows.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="artifacts/model/predictions.csv")
    parser.add_argument("--onnx-path", default="artifacts/model/product_classifier.onnx")
    parser.add_argument("--classes-path", default="artifacts/model/classes.npy")
    parser.add_argument(
        "--pca-path",
        default="artifacts/model/pca.pkl",
        help="Optional PCA transformer. If file exists, it is applied before ONNX inference.",
    )
    parser.add_argument("--aws-profile", default="staging.admin")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model-id", default="amazon.titan-embed-text-v1")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    predictor = OnnxProductCategoryPredictor(
        onnx_path=args.onnx_path,
        classes_path=args.classes_path,
        pca_path=args.pca_path,
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
