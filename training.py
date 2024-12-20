import os
import pandas as pd
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
from QLSTM import SequenceDataset, QShallowRegressionLSTM

#changes 4<=qubits <10, learning rate = 0.001, test-train split 80-20 - kept seq_length as 3
def normalize_data(df, normalization_params=None):
    """
    Normalize the data using mean and standard deviation.
    If normalization_params is None, calculate and return them.
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

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    return avg_loss

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output

def main():
        
    # Load dataset
    file_path = 'SPP-QLSTM/data/river_level.csv'
    df = pd.read_csv(file_path, skiprows=2)
    df = df[df['date'].apply(lambda x: not isinstance(x, str) or 'terms of use' not in x)]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)


    columns = ["wlvalue"]
    # Ensure columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    
    data = df.filter(columns)
    dataset = data.values

    # Splitting the data into train and test
    size = int(len(df) * 0.8)
    df_train = dataset[:size].copy()
    df_test = dataset[size:].copy()

    # If dataset is empty after filtering, raise an error
    if df_train.shape[1] == 0:
        raise ValueError(
            "Filtered training dataset is empty. Ensure the 'columns' list matches the CSV file headers."
        )

    if df_test.shape[1] == 0:
        raise ValueError(
            "Filtered testing dataset is empty. Ensure the 'columns' list matches the CSV file headers."
        )

    # Create DataFrames
    df_train = pd.DataFrame(df_train, columns=columns)
    df_test = pd.DataFrame(df_test, columns=columns)

    # Normalize data
    df_train, normalization_params = normalize_data(df_train)
    df_test = normalize_data(df_test, normalization_params)

    # Save normalization parameters
    joblib.dump(normalization_params, "results/normalization_params.pkl")

    # Prepare datasets
    features = df_train.columns
    target = "wlvalue"
    batch_size = 1
    sequence_length = 3

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=sequence_length
    )
    joblib.dump(train_dataset, "SPP-QLSTM/results/train_dataset.pkl")
    joblib.dump(test_dataset, "SPP-QLSTM/results/test_dataset.pkl")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001
    num_hidden_units = 16

    results = ""

    os.makedirs("results", exist_ok=True)

    # Loop over different qubit counts
    for qubit in range(4, 5):
        quantumModel = QShallowRegressionLSTM(
            num_sensors=len(features),
            hidden_units=num_hidden_units,
            n_qubits=qubit,
            n_qlayers=1,
        )
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(quantumModel.parameters(), lr=learning_rate)

        quantum_loss_train = []
        quantum_loss_test = []

        num_epochs = 50

        start = time.time()

        for ix_epoch in range(num_epochs):
            train_loss = train_model(
                train_loader, quantumModel, loss_function, optimizer=optimizer
            )
            test_loss = test_model(test_loader, quantumModel, loss_function)
            quantum_loss_train.append(train_loss)
            quantum_loss_test.append(test_loss)

        end = time.time()

        # Save loss data
        with open(f"results/quantum_loss_qubits_{qubit}.csv", "w") as f:
            f.write("train_loss,test_loss\n")
            for i in range(num_epochs):
                f.write(f"{quantum_loss_train[i]},{quantum_loss_test[i]}\n")

        # Save model
        torch.save(quantumModel.state_dict(), f"results/model_qubits_{qubit}.pt")

        quantumModel.eval()
        with torch.no_grad():
            train_eval_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )
            test_eval_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            ystar_col = "Model Forecast"
            df_train[ystar_col] = predict(train_eval_loader, quantumModel).numpy()
            df_test[ystar_col] = predict(test_eval_loader, quantumModel).numpy()

            train_rmse = math.sqrt(
                mean_squared_error(df_train["wlvalue"], df_train["Model Forecast"])
            )
            test_rmse = math.sqrt(
                mean_squared_error(df_test["wlvalue"], df_test["Model Forecast"])
            )

            # Save train/test results
            df_train.to_csv(f"results/train_predictions_qubits_{qubit}.csv", index=False)
            df_test.to_csv(f"results/test_predictions_qubits_{qubit}.csv", index=False)

            # Plot losses
            plt.figure(figsize=(10, 6))
            plt.plot(quantum_loss_train, label="Training Loss")
            plt.plot(quantum_loss_test, label="Testing Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training and Testing Loss (Qubits={qubit})")
            plt.legend()
            plt.grid()
            plt.savefig(f"results/loss_plot_qubits_{qubit}.png")
            plt.close()

            # Results summary
            results += f"Qubits: {qubit}\n"
            results += f"Train time: {end - start:.2f}s\n"
            results += f"Train RMSE: {train_rmse:.4f}\n"
            results += f"Test RMSE: {test_rmse:.4f}\n\n"

    # Save results summary
    with open("results/summary.txt", "w") as summary_file:
        summary_file.write(results)

if __name__ == "__main__":
    main()