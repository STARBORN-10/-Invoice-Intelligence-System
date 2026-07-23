import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    db_path = "data/inventory.db"

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    print("Linear Regression")
    print(evaluate_model(lr_model, X_test, y_test))

    print("\nDecision Tree")
    print(evaluate_model(dt_model, X_test, y_test))

    print("\nRandom Forest")
    print(evaluate_model(rf_model, X_test, y_test))

    # Save models
    joblib.dump(lr_model, model_dir / "linear_regression.pkl")
    joblib.dump(dt_model, model_dir / "decision_tree.pkl")
    joblib.dump(rf_model, model_dir / "random_forest.pkl")

    print("\nModels saved successfully!")


if __name__ == "__main__":
    main()