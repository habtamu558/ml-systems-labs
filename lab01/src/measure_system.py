import random
import os
import time
import statistics

import numpy as np
import pandas as pd
import torch
import joblib

from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------
# Load and split dataset
# ---------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)


# ---------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------
def create_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )


# ---------------------------------------------------------
# Training function
# ---------------------------------------------------------
def train_model(model_factory):
    model = model_factory()
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# Measure training time
# Warm-up: 1 run
# Timed runs: 5
# Report median
# ---------------------------------------------------------
def measure_training(model_factory):
    # Warm-up
    train_model(model_factory)

    times = []

    for _ in range(5):
        start = time.perf_counter()
        train_model(model_factory)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times)


# ---------------------------------------------------------
# Measure peak memory during training
# ---------------------------------------------------------
def measure_training_memory(model_factory):
    memory_values = memory_usage(
        (train_model, (model_factory,)),
        interval=0.01,
        timeout=None,
    )

    return max(memory_values) - min(memory_values)


# ---------------------------------------------------------
# Measure single-sample inference latency
# Warm-up: 1 run
# Timed runs: 100
# Report median
# ---------------------------------------------------------
def measure_inference(model):
    sample = X_test[:1]

    # Warm-up
    model.predict(sample)

    times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times)


# ---------------------------------------------------------
# Measure peak memory during inference
# ---------------------------------------------------------
def measure_inference_memory(model):
    sample = X_test[:1]

    def predict_once():
        model.predict(sample)

    memory_values = memory_usage(
        predict_once,
        interval=0.01,
        timeout=None,
    )

    return max(memory_values) - min(memory_values)


# ---------------------------------------------------------
# Measure model size
# ---------------------------------------------------------
def measure_model_size(model, filename):
    joblib.dump(model, filename)

    size_bytes = os.path.getsize(filename)
    size_kb = size_bytes / 1024

    return size_bytes, size_kb


# ---------------------------------------------------------
# Main measurement process
# ---------------------------------------------------------
def main():

    models = {
        "LogisticRegression": create_logistic_regression,
        "RandomForestClassifier": create_random_forest,
    }

    results = []

    for model_name, model_factory in models.items():

        print(f"\nMeasuring {model_name}...")

        # Train final model for inference and model-size measurements
        model = train_model(model_factory)

        # Training time
        training_time = measure_training(model_factory)

        # Training memory
        training_memory = measure_training_memory(model_factory)

        # Inference latency
        inference_latency = measure_inference(model)

        # Inference memory
        inference_memory = measure_inference_memory(model)

        # Model size
        filename = f"lab01/results/{model_name}.joblib"
        size_bytes, size_kb = measure_model_size(model, filename)

        results.append(
            {
                "model": model_name,
                "training_time_median_seconds": training_time,
                "inference_latency_median_seconds": inference_latency,
                "inference_latency_median_ms": inference_latency * 1000,
                "model_size_bytes": size_bytes,
                "model_size_kb": size_kb,
                "training_peak_memory_mb": training_memory,
                "inference_peak_memory_mb": inference_memory,
            }
        )

        print(f"Training time median: {training_time:.6f} seconds")
        print(
            f"Inference latency median: "
            f"{inference_latency * 1000:.6f} ms"
        )
        print(
            f"Model size: {size_bytes} bytes "
            f"({size_kb:.2f} KB)"
        )
        print(
            f"Training peak memory: "
            f"{training_memory:.2f} MB"
        )
        print(
            f"Inference peak memory: "
            f"{inference_memory:.2f} MB"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "lab01/results/system_measurements.csv",
        index=False,
    )

    print("\nResults saved to lab01/results/system_measurements.csv")


# ---------------------------------------------------------
# Windows-safe entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()

