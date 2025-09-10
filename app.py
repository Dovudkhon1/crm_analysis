# app_old.py
import numpy as np
import pandas as pd
import io
import json
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from fastapi.responses import StreamingResponse
import xgboost as xgb
from catboost import CatBoostClassifier
# ==============================
# Load trained models
# ==============================
clf_stage1 = CatBoostClassifier()
clf_stage1.load_model("clf_stage1.cbm")

clf_stage2 = CatBoostClassifier()
clf_stage2.load_model("clf_stage2.cbm")

reg_stage3 = xgb.XGBRegressor()
reg_stage3.load_model("reg_stage3.json")

rl_model = xgb.XGBClassifier()
rl_model.load_model("rl_model.json")

app = FastAPI(
    title="RA & RL Prediction API",
    description="Upload a JSON file without RA/RL columns. Returns P2, P21, RA, RL as JSON",
    version="1.0.0"
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a JSON file without RA and RL columns.
    Returns P2, P21, RA, RL as JSON.
    """
    contents = await file.read()
    try:
        df = pd.read_json(BytesIO(contents))
    except ValueError:
        # fallback: in case JSON is jsonl
        df = pd.read_json(BytesIO(contents), lines=True)

    # --- Ensure required columns ---
    df = df[["P2", "P4", "P8", "P9", "P19", "P21", "P22", "P23"]].copy()
    df["P19"] = pd.factorize(df["P19"])[0]  # encode categorical

    # Save IDs
    df_out = df[["P2", "P21"]].copy()

    feature_cols = [c for c in df.columns if c not in ['P2', 'P21']]

    # Aggregate features per company
    agg_df = df.groupby("P21")[feature_cols].agg(["sum", "mean", lambda x: x.std(ddof=0), "min", "max"])
    agg_df = agg_df.rename(columns={"<lambda_0>": "std"})
    agg_df["count"] = df.groupby(["P21"]).size()
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]

    df_merged = df.merge(agg_df, on="P21", how="left")

    # --- Step 3: Row-level ratios (value / sum) ---
    for col in feature_cols:
        sum_col = f"{col}_sum"
        ratio_col = f"{col}_ratio"
        df_merged[ratio_col] = (df_merged[col] / df_merged[sum_col])

    # --- Step 4: Ranking inside each P21 ---
    for col in feature_cols:
        rank_col = f"{col}_rank"
        df_merged[rank_col] = df_merged.groupby("P21")[col].rank(method="dense", ascending=True)

    df_merged.fillna(1.00)

    # Feature matrix for RA prediction
    X = df_merged.drop(columns=["P21", "P2"])

    # --------------------------
    # Stage 1: 0 vs nonzero
    # --------------------------
    stage1_preds = clf_stage1.predict(X)

    final_ra_preds = []
    stage2_idx, reg_idx = 0, 0

    # Stage 2 predictions for nonzero
    stage2_mask = stage1_preds != 0
    stage2_preds = clf_stage2.predict(X[stage2_mask])

    # Stage 3 regression for >1
    stage3_mask = stage2_preds != 0
    stage3_inputs = X[stage2_mask][stage3_mask]
    stage3_preds = reg_stage3.predict(stage3_inputs)
    stage3_preds = np.rint(np.expm1(stage3_preds)).astype(int)

    # Recombine all stages
    for pred in stage1_preds:
        if pred == 0:
            final_ra_preds.append(0)
        else:
            stage2_pred = stage2_preds[stage2_idx]
            if stage2_pred == 0:
                final_ra_preds.append(1)
            else:
                final_ra_preds.append(int(stage3_preds[reg_idx]))
                reg_idx += 1
            stage2_idx += 1

    df_out["RA"] = final_ra_preds

    # --------------------------
    # RL prediction
    # --------------------------
    # Add RA back into features for RL model
    df_for_rl = df.copy()
    df_for_rl["RA"] = df_out["RA"]

    # Define feature columns (exclude IDs)
    feature_cols = [c for c in df_for_rl.columns if c not in ["P2", "P21"]]

    # Aggregate features per company
    agg_df = df_for_rl.groupby("P21")[feature_cols].agg(["mean", "std", "min", "max"])
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    agg_df["count"] = df.groupby(["P21"]).size()
    agg_df = agg_df.fillna(0)

    # Predict RL per company (P21)
    rl_preds = rl_model.predict(agg_df)
    rl_preds = rl_preds + 1

    # Map RL back to each row using P21
    p21_to_rl = dict(zip(agg_df.index, rl_preds))
    df_out["RL"] = df_out["P21"].map(p21_to_rl)

    # Return JSON
    json_data = df_out.to_dict(orient="records")
    json_str = json.dumps(json_data, indent=4)
    # --------------------------
    # Send as downloadable file
    return StreamingResponse(
        io.BytesIO(json_str.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=predictions.json"}
    )
