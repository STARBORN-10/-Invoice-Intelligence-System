import streamlit as st
from inference.predict_invoice_flag import predict_invoice_flag
from inference.predict_freight import predict_freight_cost

st.set_page_config(
    page_title="Vendor Invoice Intelligence System",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar: Model Selection
# ---------------------------------------------------------
st.sidebar.title("🔍 Model Selection")
st.sidebar.markdown("**Choose Prediction Module**")

module = st.sidebar.radio(
    label="",
    options=["Freight Cost Prediction", "Invoice Manual Approval Flag"]
)

st.sidebar.divider()

st.sidebar.markdown("**Business Impact**")
st.sidebar.markdown("""
- 📈 Improved cost forecasting
- 🧾 Reduced invoice fraud & anomalies
- ⚙️ Faster finance operations
""")

# ---------------------------------------------------------
# Main panel: Freight Cost Prediction
# ---------------------------------------------------------
if module == "Freight Cost Prediction":

    st.markdown("- Reduce financial leakage and manual workload")
    st.divider()

    st.title("🚚 Freight Cost Prediction")

    st.markdown("""
**Objective:**
Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
to support budgeting, forecasting, and vendor negotiations.
""")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "📦 Quantity",
                min_value=0,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "💵 Invoice Dollars",
                min_value=0.0,
                value=18500.0
            )

        submit = st.form_submit_button("🔮 Predict Freight Cost")

    if submit:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)
        predicted_freight = prediction["Predicted_Freight"][0]

        st.divider()
        st.success(f"📦 Predicted Freight Cost: **{predicted_freight}**")
        st.write(prediction)

# ---------------------------------------------------------
# Main panel: Invoice Manual Approval Flag
# ---------------------------------------------------------
else:

    st.markdown("- Reduce financial leakage and manual workload")
    st.divider()

    st.title("🧾 Invoice Manual Approval Flag")

    st.markdown("""
**Objective:**
Predict whether a vendor invoice should be flagged for **manual approval**
using a trained **Random Forest Classifier**.
""")

    with st.form("invoice_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight",
                min_value=0.0,
                value=5.0
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=1000.0
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=50
            )

        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=995.0
            )

        submit = st.form_submit_button("🧠 Evaluate Invoice Risk")

    if submit:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        prediction = predict_invoice_flag(input_data)

        is_flagged = bool(prediction["Predicted_Flag"][0])

        st.divider()

        if is_flagged:
            st.error("🚨 Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("✅ Invoice is **SAFE for Auto Approval**")

        st.write(prediction)