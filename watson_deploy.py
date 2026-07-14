"""
IBM Watson Machine Learning Deployment Pipeline
================================================
Deploy the trained best model to IBM Watson ML for cloud-based inference.

Prerequisites:
  pip install ibm-watson-machine-learning

Setup:
  1. Create an IBM Cloud account at https://cloud.ibm.com
  2. Create a Watson Machine Learning service instance
  3. Create a project in IBM Watson Studio
  4. Copy your API key and service URL from the WML service credentials
  5. Set your SPACE_ID from Watson Studio deployment space
  6. Fill in the credentials below and run this script
"""

import os
import json
import joblib

# ─── IBM Watson ML Credentials (fill these in) ─────────────────────────────
WML_CREDENTIALS = {
    "apikey": os.environ.get("IBM_API_KEY", "YOUR_IBM_API_KEY_HERE"),
    "url":    os.environ.get("IBM_URL",     "https://us-south.ml.cloud.ibm.com"),
}
SPACE_ID   = os.environ.get("IBM_SPACE_ID",   "YOUR_DEPLOYMENT_SPACE_ID")
MODEL_NAME = "CreditCard-Approval-BestModel"
DEPLOY_NAME = "CreditCard-Approval-Endpoint"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "best_model.pkl")
META_PATH   = os.path.join(BASE_DIR, "model_metadata.json")


def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def deploy_to_watson():
    print("=" * 60)
    print("  IBM Watson ML — Deployment Pipeline")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] best_model.pkl not found.")
        print("  → Run: python cc_approval_pred.py first.")
        return

    metadata   = load_metadata()
    model_name = metadata.get("best_model_name", "Unknown")
    print(f"\n[INFO] Best model: {model_name}")

    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("\n[ERROR] ibm-watson-machine-learning not installed.")
        print("  → Run: pip install ibm-watson-machine-learning")
        return

    if WML_CREDENTIALS["apikey"] == "YOUR_IBM_API_KEY_HERE":
        print("\n[ERROR] Please fill in your IBM credentials in watson_deploy.py")
        print("  Required: IBM_API_KEY, IBM_URL, IBM_SPACE_ID")
        print("\n  You can also set environment variables:")
        print("    set IBM_API_KEY=your_key")
        print("    set IBM_URL=https://us-south.ml.cloud.ibm.com")
        print("    set IBM_SPACE_ID=your_space_id")
        return

    print("\n[1/4] Connecting to Watson ML...")
    client = APIClient(WML_CREDENTIALS)
    client.set.default_space(SPACE_ID)
    print("  ✅ Connected.")

    print("\n[2/4] Storing model...")
    model = joblib.load(MODEL_PATH)

    # Detect model framework
    model_type = type(model).__name__
    framework_map = {
        "LogisticRegression":          ("scikit-learn_1.3", "python_3.10"),
        "RandomForestClassifier":      ("scikit-learn_1.3", "python_3.10"),
        "GradientBoostingClassifier":  ("scikit-learn_1.3", "python_3.10"),
        "DecisionTreeClassifier":      ("scikit-learn_1.3", "python_3.10"),
        "XGBClassifier":               ("scikit-learn_1.3", "python_3.10"),
    }
    sw_spec, runtime = framework_map.get(model_type, ("scikit-learn_1.3", "python_3.10"))

    sw_spec_uid = client.software_specifications.get_uid_by_name(sw_spec)

    model_props = {
        client.repository.ModelMetaNames.NAME:              MODEL_NAME,
        client.repository.ModelMetaNames.TYPE:              f"scikit-learn_1.3",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
        client.repository.ModelMetaNames.DESCRIPTION:       f"Credit card approval prediction — {model_name}",
    }

    stored_model = client.repository.store_model(
        model=MODEL_PATH,
        meta_props=model_props,
    )

    model_uid = client.repository.get_model_id(stored_model)
    print(f"  ✅ Model stored with UID: {model_uid}")

    print("\n[3/4] Creating deployment...")
    deploy_props = {
        client.deployments.ConfigurationMetaNames.NAME: DEPLOY_NAME,
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }

    deployment = client.deployments.create(model_uid, meta_props=deploy_props)
    deployment_uid = client.deployments.get_uid(deployment)
    scoring_url    = client.deployments.get_scoring_href(deployment)

    print(f"  ✅ Deployment UID: {deployment_uid}")
    print(f"  ✅ Scoring URL:    {scoring_url}")

    # Save deployment info
    deploy_info = {
        "model_name":      MODEL_NAME,
        "model_uid":       model_uid,
        "deployment_uid":  deployment_uid,
        "scoring_url":     scoring_url,
        "best_model_type": model_name,
    }
    deploy_path = os.path.join(BASE_DIR, "watson_deployment_info.json")
    with open(deploy_path, "w") as f:
        json.dump(deploy_info, f, indent=2)

    print(f"\n[4/4] Deployment info saved to: {deploy_path}")

    print("\n" + "=" * 60)
    print("  ✅ Deployment Complete!")
    print(f"  Scoring Endpoint: {scoring_url}")
    print("\n  To make predictions via Watson API, use:")
    print("""
    import requests, json

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_TOKEN"
    }
    payload = {
        "input_data": [{
            "fields": [...],  # feature names
            "values": [[...]] # feature values
        }]
    }
    r = requests.post(scoring_url, headers=headers, json=payload)
    print(r.json())
    """)
    print("=" * 60)


if __name__ == "__main__":
    deploy_to_watson()
