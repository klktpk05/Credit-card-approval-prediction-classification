"""
Credit Card Approval Prediction — Flask Web Application
=======================================================
Serves the trained model via a REST API and premium web UI.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (needed to build the preprocessing pipeline)
# ─────────────────────────────────────────────────────────────────────────────

print("[INFO] Loading dataset for pipeline reference...")
train_original = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
test_original  = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)

from sklearn.model_selection import train_test_split

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_df, _ = data_split(full_data, 0.2)
train_copy  = train_df.copy()


def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    return pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CLASSES (must mirror cc_approval_pred.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            is_outlier = ((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) |
                          (df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)
            if "ID" in df.columns:
                is_outlier = is_outlier & (df["ID"] != 0)
            df = df[~is_outlier]
            return df
        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=["Has a mobile phone", "Children count", "Job title", "Account age"]):
        self.feature_to_drop = feature_to_drop
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
        return df

class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=["Employment length", "Age"]):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        if set(self.feat_with_days).issubset(X.columns):
            X[["Employment length", "Age"]] = np.abs(X[["Employment length", "Age"]])
        return X

class RetireeHandler(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None): return self
    def transform(self, df):
        if "Employment length" in df.columns:
            idx = df["Employment length"][df["Employment length"] == 365243].index
            df.loc[idx, "Employment length"] = 0
        return df

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=["Income", "Age"]):
        self.feat_with_skewness = feat_with_skewness
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.feat_with_skewness).issubset(df.columns):
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
        return df

class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=["Has a work phone", "Has a phone", "Has an email"]):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.feat_with_num_enc).issubset(df.columns):
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1: "Y", 0: "N"})
        return df

class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=["Gender", "Marital status", "Dwelling",
                                        "Employment status", "Has a car", "Has a property",
                                        "Has a work phone", "Has a phone", "Has an email"]):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.one_hot_enc_ft).issubset(df.columns):
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            enc.fit(df[self.one_hot_enc_ft])
            feat_names = enc.get_feature_names_out(self.one_hot_enc_ft)
            df_enc = pd.DataFrame(enc.transform(df[self.one_hot_enc_ft]),
                                   columns=feat_names, index=df.index)
            rest = [c for c in df.columns if c not in self.one_hot_enc_ft]
            return pd.concat([df_enc, df[rest]], axis=1)
        return df

class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=["Education level"]):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self, df, y=None): return self
    def transform(self, df):
        if "Education level" in df.columns:
            enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = enc.fit_transform(df[self.ordinal_enc_ft])
        return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=["Age", "Income", "Employment length"]):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self, df, y=None): return self
    def transform(self, df):
        if set(self.min_max_scaler_ft).issubset(df.columns):
            scaler = MinMaxScaler()
            df[self.min_max_scaler_ft] = scaler.fit_transform(df[self.min_max_scaler_ft])
        return df

class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None): return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            df["Is high risk"] = pd.to_numeric(df["Is high risk"])
        return df

class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None): return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])
            return pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
        return df


def full_pipeline(df):
    pipeline = Pipeline([
        ("outlier_remover",         OutlierRemover()),
        ("feature_dropper",         DropFeatures()),
        ("time_conversion_handler", TimeConversionHandler()),
        ("retiree_handler",         RetireeHandler()),
        ("skewness_handler",        SkewnessHandler()),
        ("binning_num_to_yn",       BinningNumToYN()),
        ("one_hot_with_feat_names", OneHotWithFeatNames()),
        ("ordinal_feat_names",      OrdinalFeatNames()),
        ("min_max_with_feat_names", MinMaxWithFeatNames()),
        ("change_to_num_target",    ChangeToNumTarget()),
        ("oversample_smote",        OversampleSMOTE()),
    ])
    return pipeline.fit_transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Load model
MODEL_PATH    = os.path.join(BASE_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

model    = None
metadata = {}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
else:
    print("[WARNING] best_model.pkl not found. Run cc_approval_pred.py first.")

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        metadata = json.load(f)


# Lookup maps (must match the mapping from full_data)
marital_status_values = list(full_data["Marital status"].value_counts().index)
marital_status_keys   = ["Married", "Single/not married", "Civil marriage", "Separated", "Widowed"]
marital_dict          = dict(zip(marital_status_keys, marital_status_values))

dwelling_type_values = list(full_data["Dwelling"].value_counts().index)
dwelling_type_keys   = ["House / apartment", "Live with parents", "Municipal apartment",
                         "Rented apartment", "Office apartment", "Co-op apartment"]
dwelling_dict        = dict(zip(dwelling_type_keys, dwelling_type_values))

employment_status_values = list(full_data["Employment status"].value_counts().index)
employment_status_keys   = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
employment_dict          = dict(zip(employment_status_keys, employment_status_values))

edu_level_values = list(full_data["Education level"].value_counts().index)
edu_level_keys   = ["Secondary school", "Higher education", "Incomplete higher",
                     "Lower secondary", "Academic degree"]
edu_dict         = dict(zip(edu_level_keys, edu_level_values))


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=metadata.get("best_model_name", "ML Model"),
        metrics=metadata.get("metrics", {}),
        all_results=metadata.get("all_results", {}),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please run cc_approval_pred.py first."}), 503

    try:
        data = request.get_json()

        gender           = data.get("gender", "M")
        age              = float(data.get("age", 30))
        marital_status   = marital_dict.get(data.get("marital_status", "Married"),
                                             marital_status_values[0])
        fam_member_count = float(data.get("family_members", 2))
        dwelling         = dwelling_dict.get(data.get("dwelling", "House / apartment"),
                                              dwelling_type_values[0])
        income           = float(data.get("income", 50000))
        emp_status       = employment_dict.get(data.get("employment_status", "Working"),
                                                employment_status_values[0])
        emp_length       = float(data.get("employment_length", 5))
        edu_level        = edu_dict.get(data.get("education_level", "Higher education"),
                                         edu_level_values[0])
        has_car          = data.get("has_car", "Y")
        has_property     = data.get("has_property", "Y")
        work_phone       = 1 if data.get("work_phone", "N") == "Y" else 0
        phone            = 1 if data.get("phone", "N") == "Y" else 0
        email            = 1 if data.get("email", "N") == "Y" else 0

        # Convert to pipeline-compatible format (negative days)
        age_days        = -(age * 365.25)
        emp_length_days = -(emp_length * 365.25)

        profile = [
            0,                  # ID
            gender[:1],         # Gender
            has_car[:1],        # Has a car
            has_property[:1],   # Has a property
            0,                  # Children count (dropped)
            income,             # Income
            emp_status,         # Employment status
            edu_level,          # Education level
            marital_status,     # Marital status
            dwelling,           # Dwelling
            age_days,           # Age (in days, negative)
            emp_length_days,    # Employment length (in days, negative)
            1,                  # Has a mobile phone (dropped)
            work_phone,         # Has a work phone
            phone,              # Has a phone
            email,              # Has an email
            "to_be_dropped",    # Job title (dropped)
            fam_member_count,   # Family member count
            0.00,               # Account age (dropped)
            0,                  # Target placeholder
        ]

        profile_df = pd.DataFrame([profile], columns=train_copy.columns)
        combined   = pd.concat([train_copy, profile_df], ignore_index=True)
        combined_prep = full_pipeline(combined)

        row = combined_prep[combined_prep["ID"] == 0]
        if row.empty:
            return jsonify({"error": "Could not process input. Applicant may be an outlier."}), 400

        row = row.drop(columns=["ID", "Is high risk"])

        prediction = int(model.predict(row)[0])
        try:
            probability = float(model.predict_proba(row)[0][1])
        except Exception:
            probability = float(prediction)

        result = {
            "approved": prediction == 0,
            "prediction": "Approved" if prediction == 0 else "Rejected",
            "confidence": round((1 - probability if prediction == 0 else probability) * 100, 1),
            "model_used": metadata.get("best_model_name", "ML Model"),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": metadata.get("best_model_name", "N/A"),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
