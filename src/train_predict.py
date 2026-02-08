import os
import argparse
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def to_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    for sep in ["-", "/"]:
        if sep in s:
            left = s.split(sep)[0]
            if left.isdigit():
                return int(left)
    digits = "".join(c for c in s if c.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else np.nan


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--year_col", default="Season")
    parser.add_argument("--target_col", default="Attendance")
    parser.add_argument("--predict_year", type=int, default=2025)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    df[args.year_col] = df[args.year_col].apply(to_year)

    train_df = df[df[args.year_col].between(2021, 2024)]
    pred_df = df[df[args.year_col] == args.predict_year]

    train_df["_y"] = to_num(train_df[args.target_col])
    train_df = train_df[train_df["_y"].notna()]

    feature_cols = [c for c in df.columns if c not in [args.target_col, "_y"]]

    X_train = train_df[feature_cols]
    y_train = train_df["_y"]
    X_pred = pred_df[feature_cols]

    num_cols = X_train.select_dtypes(include="number").columns
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    # Validation (2024)
    tr = train_df[train_df[args.year_col] <= 2023]
    va = train_df[train_df[args.year_col] == 2024]

    if not tr.empty and not va.empty:
        pipe.fit(tr[feature_cols], tr["_y"])
        preds = pipe.predict(va[feature_cols])
        mae = mean_absolute_error(va["_y"], preds)
        rmse = mean_squared_error(va["_y"], preds, squared=False)
        print(f"2024 Validation MAE: {mae:,.0f}")
        print(f"2024 Validation RMSE: {rmse:,.0f}")

    # Final train + predict
    pipe.fit(X_train, y_train)
    pred = np.maximum(pipe.predict(X_pred), 0)

    out = pred_df.copy()
    out["Pred_Attendance_2025_Model"] = np.round(pred).astype(int)

    out.to_csv(f"{args.out_dir}/attendance_predictions_2025.csv", index=False)
    dump(pipe, f"{args.out_dir}/model.joblib")


if __name__ == "__main__":
    main()
