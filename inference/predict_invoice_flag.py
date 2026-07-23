import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load fitted scaler used during training.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    # Scale features exactly as done during training
    scaled_values = scaler.transform(input_df[FEATURES])
    scaled_df = pd.DataFrame(scaled_values, columns=FEATURES)

    input_df["Predicted_Flag"] = model.predict(scaled_df)

    return input_df


if __name__ == "__main__":

    sample_invoice = {
        "invoice_quantity": [120],
        "invoice_dollars": [4500],
        "Freight": [180],
        "total_item_quantity": [118],
        "total_item_dollars": [4495]
    }

    result = predict_invoice_flag(sample_invoice)

    print(result)