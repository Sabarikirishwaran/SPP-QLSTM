# inferencing.py
import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import joblib
from QLSTM import QShallowRegressionLSTM, SequenceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def normalize_data(df, normalization_params=None):
    """
    Same normalization function as in training.py
    """
    if normalization_params is None:
        normalization_params = {}
        for c in df.columns:
            mean = df[c].mean()
            stdev = df[c].std()
            df[c] = (df[c] - mean) / stdev
            normalization_params[c] = (mean, stdev)
        return df, normalization_params
    else:
        for c in df.columns:
            mean, stdev = normalization_params[c]
            df[c] = (df[c] - mean) / stdev
        return df

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output.cpu(), y_star.cpu()), 0)
    return output

def main():
    # Paths
    file_path = 'SPP-QLSTM/data/river_level.csv'
    normalization_params_path = "SPP-QLSTM/results/normalization_params.pkl"
    model_checkpoint_path = "results/model_qubits_4.pt"
    os.makedirs("SPP-QLSTM/results", exist_ok=True)

    # Load normalization parameters
    normalization_params = joblib.load(normalization_params_path)

    # -----------------------------------------
    # Reconstruct df_test just like in training
    # -----------------------------------------
    df = pd.read_csv(file_path, skiprows=2)
    # Clean the dataset just like training.py
    df = df[df['date'].apply(lambda x: not isinstance(x, str) or 'terms of use' not in x)]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)

    columns = ["wlvalue"]
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    data = df.filter(columns)
    dataset = data.values

    # Same 80-20 split
    size = int(len(df) * 0.8)
    df_train = dataset[:size].copy()
    df_test = dataset[size:].copy()

    # Create DataFrames
    df_train = pd.DataFrame(df_train, columns=columns)
    df_test = pd.DataFrame(df_test, columns=columns)

    # Normalize using the saved parameters
    df_test = normalize_data(df_test, normalization_params)

    # Prepare test dataset with the same parameters as training
    features = df_test.columns
    target = "wlvalue"
    sequence_length = 3
    batch_size = 1

    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=sequence_length
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters as in training.py
    hidden_size = 16
    n_qubits = 4
    n_qlayers = 1

    # Load the trained model
    model = QShallowRegressionLSTM(
        num_sensors=len(features),
        hidden_units=hidden_size,
        n_qubits=n_qubits,
        n_qlayers=n_qlayers,
    ).to(device)

    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Perform prediction
    ystar_col = "Model Forecast"
    df_test[ystar_col] = predict(test_loader, model).numpy()

    # Compute RMSE
    test_rmse = math.sqrt(mean_squared_error(df_test[target], df_test[ystar_col]))
    print(f"Test RMSE: {test_rmse:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df_test[target].values, label="Actual", color="green")
    plt.plot(df_test[ystar_col].values, label="Predicted", color="red")
    plt.xlabel("Samples")
    plt.ylabel("Normalized Water Level")
    plt.title("Test Actual vs Predicted")
    plt.legend()
    plt.grid()
    plt.savefig("SPP-QLSTM/results/test_actual_vs_predicted.png")
    plt.show()

    # Save inference results
    df_test.to_csv("SPP-QLSTM/results/inference_predictions.csv", index=False)

    print("Inference completed. Results saved as 'SPP-QLSTM/results/test_actual_vs_predicted.png' and 'SPP-QLSTM/results/inference_predictions.csv'.")

if __name__ == "__main__":
    main()