import pandas as pd
import os

def load_f1_data(raw_path="data/raw"):
    # Load files
    results = pd.read_csv(os.path.join(raw_path, "results.csv"))
    races = pd.read_csv(os.path.join(raw_path, "races.csv"))
    drivers = pd.read_csv(os.path.join(raw_path, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(raw_path, "constructors.csv"))
    circuits = pd.read_csv(os.path.join(raw_path, "circuits.csv"))

    # Merge
    df = (
        results.merge(races[["raceId", "year", "circuitId"]], on="raceId")
              .merge(drivers[["driverId", "driverRef"]], on="driverId")
              .merge(constructors[["constructorId", "name"]], on="constructorId")
              .merge(circuits[["circuitId", "circuitRef"]], on="circuitId")
    )

    # Filters
    df = df[(df["year"] >= 2000) & (df["year"] <= 2024)]
    df = df[df["grid"] > 0]

    # Target: top 10?
    df["top10"] = (df["positionOrder"] <= 10).astype(int)

    return df
