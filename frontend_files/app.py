
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="📈"
)

# Flask backend URL inside Docker network
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://superkart-backend:7860"
)

st.title("SuperKart Sales Forecast")

st.write(
    "Predict product-store sales using the "
    "SuperKart machine learning model."
)


# ==================================================
# SINGLE PREDICTION
# ==================================================

st.header("Single Prediction")

col1, col2 = st.columns(2)

with col1:

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.66
    )

    sugar = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.027
    )

    mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=117.08
    )

    store_size = st.selectbox(
        "Store Size",
        [
            "Small",
            "Medium",
            "High"
        ]
    )


with col2:

    city_type = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

store_type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Departmental Store",
        "Food Mart"
    ]
)

    product_prefix = st.selectbox(
        "Product ID Prefix",
        [
            "FD",
            "NC",
            "DR"
        ]
    )

    store_age = st.number_input(
        "Store Age (Years)",
        min_value=0,
        value=15
    )

    product_category = st.selectbox(
        "Product Type Category",
        [
            "Perishables",
            "Non Perishables"
        ]
    )


if st.button("Predict Sales"):

    payload = {

        "Product_Weight": product_weight,

        "Product_Sugar_Content": sugar,

        "Product_Allocated_Area": allocated_area,

        "Product_MRP": mrp,

        "Store_Size": store_size,

        "Store_Location_City_Type": city_type,

        "Store_Type": store_type,

        "Product_Id_char": product_prefix,

        "Store_Age_Years": store_age,

        "Product_Type_Category": product_category
    }

    try:

        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        st.success(
            f"Predicted Sales: "
            f"{result['predicted_sales']:.2f}"
        )

    except Exception as e:

        st.error(
            f"Backend request failed: {e}"
        )


# ==================================================
# BATCH PREDICTION
# ==================================================

st.header("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(
        uploaded_file
    )

    st.dataframe(
        batch_df.head()
    )

    if st.button("Run Batch Prediction"):

        try:

            response = requests.post(

                f"{BACKEND_URL}/v1/predictbatch",

                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                },

                timeout=60
            )

            response.raise_for_status()

            predictions = response.json()[
                "predictions"
            ]

            result_df = batch_df.copy()

            result_df[
                "Predicted_Sales"
            ] = predictions

            st.dataframe(
                result_df
            )

            st.download_button(

                "Download Predictions",

                result_df.to_csv(
                    index=False
                ).encode("utf-8"),

                "superkart_predictions.csv",

                "text/csv"
            )

        except Exception as e:

            st.error(
                f"Batch prediction failed: {e}"
            )
