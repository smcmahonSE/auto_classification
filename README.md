# Product Classifier (Modular Local Workflow)

This project is split into modular steps so you do not need to regenerate Titan embeddings every time:

1. Build feature artifact from Snowflake data (`build_training_data.py`)
2. Train LightGBM from cached embeddings and export ONNX (`train_model.py`)
3. Run inference wrapper (concat + Titan + ONNX model) (`predict_with_onnx.py`)

## Setup (venv)

```bash
cd /Users/stephanie.mcmahon/smcmahon_repo/smcmahon_notebooks
source /Users/stephanie.mcmahon/smcmahon_repo/.venv/bin/activate
python -m pip install -r requirements.txt
```

## Environment

- AWS profile defaults to `staging.admin` (override with CLI args).
- Snowflake defaults can be overridden with env vars:
  - `SNOWFLAKE_ACCOUNT`
  - `SNOWFLAKE_USER`
  - `SNOWFLAKE_WAREHOUSE` (optional)
  - `SNOWFLAKE_ROLE` (optional)
  - `SNOWFLAKE_PRODUCTS_TABLE` (DATABASE.SCHEMA.TABLE; default is `SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_L3_STG`)

If SSO token expired, run:

```bash
aws-sso-util login
```

## Step 1: Build Training Features (cached embeddings)

```bash
python build_training_data.py
```

Useful flags:

- Small test run:
  ```bash
  python build_training_data.py --row-limit 500 --sample-per-category 25
  ```
- Stratified sample by category (recommended for staged testing):
  ```bash
  python build_training_data.py --stratified-sample-size 5000 --random-state 42
  python build_training_data.py --stratified-sample-size 100000 --random-state 42
  python build_training_data.py --stratified-sample-size 500000 --random-state 42
  ```
- Faster embedding runs (parallel + checkpointing):
  ```bash
  python build_training_data.py --stratified-sample-size 500000 --max-workers 16 --checkpoint-every 500
  ```
- Custom table:
  ```bash
  python build_training_data.py --table SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_L3_STG
  # later swap to prod
  python build_training_data.py --table SNOWFLAKE_LEARNING_DB.SMCMAHON_PRODUCTS.PRODUCTS_L3_SUB
  ```

Outputs:

- `artifacts/training_data/features_titan.npz` (embeddings + labels + ids)
- `artifacts/training_data/features_metadata.json`
- `artifacts/cache/embedding_cache.pkl` (hash -> embedding cache)

For large runs, start with `--max-workers 8` or `16`, then tune up/down based on throttling and stability.

## Step 2: Train Model and Export ONNX

```bash
python train_model.py
```

Optional PCA experiment:

```bash
python train_model.py --experiment-name lgbm_no_pca
python train_model.py --experiment-name lgbm_pca_512 --pca-components 512
python train_model.py --experiment-name lgbm_pca_256 --pca-components 256
```

Outputs:

- `artifacts/model/lightgbm_model.pkl`
- `artifacts/model/product_classifier.onnx`
- `artifacts/model/classes.npy`
- `artifacts/model/label_encoder.pkl`
- `artifacts/model/pca.pkl` (only if `--pca-components` is set)
- `artifacts/model/metrics.json`

`metrics.json` includes:

- performance metrics (`accuracy`, `balanced_accuracy`, macro/weighted precision-recall-F1)
- per-class precision/recall/F1/support
- artifact size report in MB and a rough runtime memory recommendation

Each run also appends a summary line to:

- `artifacts/model/metrics_history.jsonl`

You can compare runs automatically:

```bash
python compare_model_runs.py
python compare_model_runs.py --sort-by onnx_model_mb --ascending
```

Comparison output CSV:

- `artifacts/model/experiment_comparison.csv`

## Step 3: Inference using ONNX

Prepare a CSV containing:

- `PRODUCT_NAME`
- `DESCRIPTION`
- `PRICING_STATUS_C`
- `LIST_PRICE_C`

Then run:

```bash
python predict_with_onnx.py --input-csv path/to/input.csv --output-csv artifacts/model/predictions.csv
```

If `artifacts/model/pca.pkl` exists, inference applies PCA automatically before ONNX scoring.

## Notes on ONNX

ONNX contains the classifier only (LightGBM on embedding vectors).  
Titan embedding calls remain in Python wrapper code, which is expected for Bedrock-based embeddings.

## One-command convenience

`train_product_classifier_local.py` is kept as a convenience entrypoint that runs:

1. `build_training_data.py`
2. `train_model.py`
