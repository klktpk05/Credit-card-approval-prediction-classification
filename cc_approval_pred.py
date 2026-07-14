"""
Credit Card Approval Prediction - ML Training Pipeline
======================================================
Trains 4 classifiers and saves the best model locally.
Models: Logistic Regression, Random Forest, XGBoost, Decision Tree
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Skipping XGBoost model.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

print("=" * 60)
print("  Credit Card Approval Prediction - Training Pipeline")
print("=" * 60)

print("\n[1/5] Loading datasets...")

train_original = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
test_original  = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

print(f"  Train shape : {train_original.shape}")
print(f"  Test shape  : {test_original.shape}")

full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()
test_copy  = test_original.copy()


def value_cnt_norm_cal(df, feature):
    """Calculate the count and percentage of each value in a feature."""
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ["Count", "Frequency (%)"]
    return ftr_value_cnt_concat


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING PIPELINE TRANSFORMERS
# ─────────────────────────────────────────────────────────────────────────────

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1  = df[self.feat_with_outliers].quantile(0.25)
            Q3  = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            is_outlier = ((df[self.feat_with_outliers] < (Q1 - 3 * IQR))
                          | (df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)
            if "ID" in df.columns:
                is_outlier = is_outlier & (df["ID"] != 0)
            df = df[~is_outlier]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=["Has a mobile phone", "Children count",
                                         "Job title", "Account age"]):
        self.feature_to_drop = feature_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=["Employment length", "Age"]):
        self.feat_with_days = feat_with_days

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if set(self.feat_with_days).issubset(X.columns):
            X[["Employment length", "Age"]] = np.abs(X[["Employment length", "Age"]])
            return X
        else:
            print("One or more features are not in the dataframe")
            return X


class RetireeHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if "Employment length" in df.columns:
            df_ret_idx = df["Employment length"][df["Employment length"] == 365243].index
            df.loc[df_ret_idx, "Employment length"] = 0
            return df
        else:
            print("Employment length is not in the dataframe")
            return df


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=["Income", "Age"]):
        self.feat_with_skewness = feat_with_skewness

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feat_with_skewness).issubset(df.columns):
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=["Has a work phone", "Has a phone", "Has an email"]):
        self.feat_with_num_enc = feat_with_num_enc

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feat_with_num_enc).issubset(df.columns):
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1: "Y", 0: "N"})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=["Gender", "Marital status", "Dwelling",
                                        "Employment status", "Has a car", "Has a property",
                                        "Has a work phone", "Has a phone", "Has an email"]):
        self.one_hot_enc_ft = one_hot_enc_ft

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.one_hot_enc_ft).issubset(df.columns):
            def one_hot_enc(df, one_hot_enc_ft):
                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                enc.fit(df[one_hot_enc_ft])
                feat_names = enc.get_feature_names_out(one_hot_enc_ft)
                df_enc = pd.DataFrame(
                    enc.transform(df[self.one_hot_enc_ft]),
                    columns=feat_names,
                    index=df.index
                )
                return df_enc

            def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
                return df_concat

            one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
            full_df_one_hot_enc = concat_with_rest(df, one_hot_enc_df, self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df


class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=["Education level"]):
        self.ordinal_enc_ft = ordinal_enc_ft

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if "Education level" in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df


class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=["Age", "Income", "Employment length"]):
        self.min_max_scaler_ft = min_max_scaler_ft

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.min_max_scaler_ft).issubset(df.columns):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if "Is high risk" in df.columns:
            df["Is high risk"] = pd.to_numeric(df["Is high risk"])
            return df
        else:
            print("Is high risk is not in the dataframe")
            return df


class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if "Is high risk" in df.columns:
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])
            X_y_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return X_y_bal
        else:
            print("Is high risk is not in the dataframe")
            return df


def full_pipeline(df):
    """Full preprocessing pipeline from raw data to model-ready data."""
    pipeline = Pipeline([
        ("outlier_remover",        OutlierRemover()),
        ("feature_dropper",        DropFeatures()),
        ("time_conversion_handler", TimeConversionHandler()),
        ("retiree_handler",        RetireeHandler()),
        ("skewness_handler",       SkewnessHandler()),
        ("binning_num_to_yn",      BinningNumToYN()),
        ("one_hot_with_feat_names", OneHotWithFeatNames()),
        ("ordinal_feat_names",     OrdinalFeatNames()),
        ("min_max_with_feat_names", MinMaxWithFeatNames()),
        ("change_to_num_target",   ChangeToNumTarget()),
        ("oversample_smote",       OversampleSMOTE()),
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Running preprocessing pipeline...")
train_prep = full_pipeline(train_copy)
print(f"  Preprocessed training shape: {train_prep.shape}")

# Separate features and target
X = train_prep.drop(columns=["ID", "Is high risk"])
y = train_prep["Is high risk"].astype(int)

# Train/val split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")
print(f"  Class distribution (train): {dict(pd.Series(y_train).value_counts())}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN FOUR CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Training classifiers...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1
    )
else:
    # Fallback: Gradient Boosting (sklearn)
    models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)

results = []
trained_models = {}

for name, model in models.items():
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        roc    = round(roc_auc_score(y_val, y_prob), 4)
    except Exception:
        roc = "N/A"

    acc    = round(accuracy_score(y_val, y_pred), 4)
    recall = round(recall_score(y_val, y_pred, zero_division=0), 4)
    f1     = round(f1_score(y_val, y_pred, zero_division=0), 4)

    results.append({
        "Model":    name,
        "Accuracy": acc,
        "Recall":   recall,
        "F1 Score": f1,
        "ROC-AUC":  roc,
    })
    trained_models[name] = model
    print(f"Done! Acc={acc:.2%} | Recall={recall:.2%} | F1={f1:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPARE AND SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Model comparison:")
results_df = pd.DataFrame(results).set_index("Model")
print("\n" + results_df.to_string())

# Choose best model by Recall (minimise credit default risk)
numeric_results = [(r["Model"], r["Recall"]) for r in results if isinstance(r["Recall"], float)]
best_name  = max(numeric_results, key=lambda x: x[1])[0]
best_model = trained_models[best_name]

print(f"\n  [BEST] Best model selected: {best_name} (Recall = {results_df.loc[best_name, 'Recall']:.2%})")

# Save best model
model_path = os.path.join(BASE_DIR, "best_model.pkl")
joblib.dump(best_model, model_path)

# Save model metadata for Flask app
metadata = {
    "best_model_name": best_name,
    "metrics": results_df.loc[best_name].to_dict(),
    "all_results": results_df.to_dict(),
    "feature_names": list(X.columns),
}
import json
metadata_path = os.path.join(BASE_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[5/5] Saved:")
print(f"  -> Model    : {model_path}")
print(f"  -> Metadata : {metadata_path}")
print("\n" + "=" * 60)
print("  Training complete. Run python app.py to start the web app.")
print("=" * 60 + "\n")
