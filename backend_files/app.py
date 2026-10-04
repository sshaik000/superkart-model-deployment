
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load serialized model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

# Exact features expected by the trained model
MODEL_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area_Log",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# --------------------------------------------------
# Single prediction
# --------------------------------------------------

@app.route("/v1/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    # Standardize reg -> Regular
    sugar = data.get("Product_Sugar_Content")

    if str(sugar).strip().lower() == "reg":
        sugar = "Regular"

    # Check required input fields
    required_inputs = [
        "Product_Weight",
        "Product_Sugar_Content",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
        "Product_Id_char",
        "Store_Age_Years",
        "Product_Type_Category"
    ]

    missing = [
        column
        for column in required_inputs
        if column not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing input fields",
            "columns": missing
        }), 400

    # Create model input
    row = {
        "Product_Weight": data["Product_Weight"],
        "Product_Sugar_Content": sugar,

        # Feature engineering used during model training
        "Product_Allocated_Area_Log": np.log1p(
            float(data["Product_Allocated_Area"])
        ),

        "Product_MRP": data["Product_MRP"],
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        "Product_Id_char": data["Product_Id_char"],
        "Store_Age_Years": data["Store_Age_Years"],
        "Product_Type_Category": data["Product_Type_Category"]
    }

    input_df = pd.DataFrame(
        [row],
        columns=MODEL_COLUMNS
    )

    prediction = float(
        model.predict(input_df)[0]
    )

    return jsonify({
        "predicted_sales": prediction
    })


# --------------------------------------------------
# Batch prediction
# --------------------------------------------------

@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():

    if "file" not in request.files:

        return jsonify({
            "error": "CSV file is required"
        }), 400

    file = request.files["file"]

    try:
        batch_df = pd.read_csv(file)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

    # Input columns supplied by the batch CSV
    required_inputs = [
        "Product_Weight",
        "Product_Sugar_Content",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
        "Product_Id_char",
        "Store_Age_Years",
        "Product_Type_Category"
    ]

    missing = [
        column
        for column in required_inputs
        if column not in batch_df.columns
    ]

    if missing:

        return jsonify({
            "error": "Missing columns",
            "columns": missing
        }), 400

    batch_df = batch_df[
        required_inputs
    ].copy()

    # Standardize sugar values
    batch_df[
        "Product_Sugar_Content"
    ] = batch_df[
        "Product_Sugar_Content"
    ].replace({
        "reg": "Regular"
    })

    # Recreate the feature engineering used during training
    batch_df[
        "Product_Allocated_Area_Log"
    ] = np.log1p(
        batch_df["Product_Allocated_Area"]
    )

    # Remove original feature
    batch_df.drop(
        columns=["Product_Allocated_Area"],
        inplace=True
    )

    # Match model feature order
    batch_df = batch_df[
        MODEL_COLUMNS
    ]

    predictions = model.predict(
        batch_df
    )

    return jsonify({
        "predictions": [
            float(value)
            for value in predictions
        ]
    })


# --------------------------------------------------
# Start Flask
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
