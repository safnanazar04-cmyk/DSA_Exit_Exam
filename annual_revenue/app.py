import os
import secrets
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("REVENUE_CSV", BASE_DIR / "data" / "G3.csv"))
CATEGORICAL_FEATURES = [
    "company_type",
    "enterprise_group_jurisdiction",
    "is_standalone_entity",
]
NUMERIC_FEATURES = [
    "employees",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
COMPANY_TYPES = ["PUBLIC", "PRIVATE", "STATE_OWNED", "NONPROFIT", "OTHER", "UNKNOWN"]


def load_training_data(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {csv_path}. "
            "Place G3.csv in the data folder or set REVENUE_CSV."
        )

    data = pd.read_csv(csv_path)
    required = FEATURES + ["revenue", "revenue_currency"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data = data.loc[
        data["revenue_currency"].fillna("").str.upper().eq("USD")
        & data["revenue"].gt(0)
    ].copy()
    data["company_type"] = (
        data["company_type"].fillna("").astype(str).str.strip().str.upper()
    )
    data.loc[~data["company_type"].isin(COMPANY_TYPES), "company_type"] = "UNKNOWN"
    data["enterprise_group_jurisdiction"] = (
        data["enterprise_group_jurisdiction"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )
    data["is_standalone_entity"] = data["is_standalone_entity"].map(
        {True: "Yes", False: "No", "True": "Yes", "False": "No"}
    )

    if data.empty:
        raise ValueError("The CSV contains no positive revenue records in USD.")

    return data


def train_model(data):
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(data[FEATURES], data["revenue"])
    return model


training_data = load_training_data(DATA_PATH)
model = train_model(training_data)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))


@app.get("/")
def home():
    return render_template("index.html", training_rows=len(training_data))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    jurisdictions = sorted(
        value
        for value in training_data["enterprise_group_jurisdiction"].unique()
        if value and value != "UNKNOWN"
    )

    if request.method == "GET":
        return render_template(
            "predict.html",
            company_types=COMPANY_TYPES,
            jurisdictions=jurisdictions,
            errors=[],
            form_values={},
        )

    errors = []
    values = {}
    company_type = request.form.get("company_type", "").strip().upper()
    jurisdiction = request.form.get("enterprise_group_jurisdiction", "").strip()
    standalone = request.form.get("is_standalone_entity", "").strip()

    if company_type not in COMPANY_TYPES:
        errors.append("Choose a valid company type.")
    if jurisdiction not in jurisdictions and jurisdiction != "UNKNOWN":
        errors.append("Choose a jurisdiction from the list.")
    if standalone not in {"Yes", "No"}:
        errors.append("Choose whether the enterprise is standalone.")

    values["company_type"] = company_type
    values["enterprise_group_jurisdiction"] = jurisdiction
    values["is_standalone_entity"] = standalone

    for feature in NUMERIC_FEATURES:
        raw_value = request.form.get(feature, "").strip()
        if not raw_value:
            values[feature] = None
            continue
        try:
            number = float(raw_value)
            if number < 0:
                raise ValueError
            if feature != "employees" and not number.is_integer():
                raise ValueError
            values[feature] = number
        except ValueError:
            errors.append(f"Enter a valid non-negative value for {feature.replace('_', ' ')}.")
            values[feature] = None

    if errors:
        return (
            render_template(
                "predict.html",
                company_types=COMPANY_TYPES,
                jurisdictions=jurisdictions,
                errors=errors,
                form_values=request.form,
            ),
            400,
        )

    prediction = float(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    session["prediction"] = max(0, prediction)
    session["inputs"] = {
        **values,
        "employees": values["employees"],
        "legal_units_count": values["legal_units_count"],
        "direct_subsidiaries_count": values["direct_subsidiaries_count"],
        "max_hierarchy_depth": values["max_hierarchy_depth"],
    }
    return redirect(url_for("result"))


@app.get("/result")
def result():
    if "prediction" not in session:
        return redirect(url_for("predict"))
    return render_template(
        "result.html",
        prediction=session["prediction"],
        inputs=session.get("inputs", {}),
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
