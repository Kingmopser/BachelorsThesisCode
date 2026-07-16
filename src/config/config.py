import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "task_metadata_tabarena51.csv"
MLFLOW_DB = ROOT / "data" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ROOT / "mlruns"
MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

ROBUST_SEEDS = [24,35,123]
GLOBAL_SEED = 12
IQR_TO_STD = 1.3489795

DATASETS = [
     "wine_quality",
     "healthcare_insurance_expenses", 
     "Another-Dataset-on-used-Fiat-500", 
     "miami_housing"
]

BDE_GRID = {
    "hidden_layers": ["[16,16]", "[32,32]", "[16,16,16,16]", "[32,32,32]"],
    "var_start_end": ["(0.5,0.1)", "(0.05,0.01)", "(0.005,0.001)", "(0.0005,0.0001)"],
    "warmup_steps_n_samples": ["(1000,200)", "(2500,500)", "(5000,1000)", "(10000,5000)"],
    "epochs": ["400"]
}
BDE_FIAT500_ABLATION_GRID = {
    "hidden_layers": ["[4,4]","[1]","[4]","[8]"],
    "var_start_end": ["(0.5,0.1)"],
    "warmup_steps_n_samples": ["(2500,500)"],
    "epochs" : ["25","50","75"]}

BDE_MIAMIH_ABLATION_GRID = {
    "hidden_layers": ["[16,16,16,16,16,16]"],
    "var_start_end": ["(0.5,0.1)","(0.05,0.01)","(0.005,0.001)" ],
    "warmup_steps_n_samples": ["(50000,1000)"],
    "epochs" : ["1200"]
    }

BDE_EPOCHS_800_OVERRIDE = {"epochs" : 800}
BDE_EPOCHS_1200_OVERRIDE = {"epochs" : 1200}

BDE_MIAMIH_OVERRIDE = {
    "warmup_steps": 10000,
    "n_samples": 2000,
    "n_members": 40,
}

BDE_ACTIVE_OVERRIDE = {}


XGBOOSTLSS_CONFIG = {
    "Another-Dataset-on-used-Fiat-500": {
        "distribution": "Gaussian",
        "stabilization": "None",
        "response_fn": "exp",
        "loss_fn": "nll",
        "num_boost_round": 50,
        "params": {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        },
    },
    "healthcare_insurance_expenses": {
        "distribution": "Gaussian",
        "stabilization": "None",
        "response_fn": "exp",
        "loss_fn": "nll",
        "num_boost_round": 50,
        "params": {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        },
    },
    "miami_housing": {
        "distribution": "Gaussian",
        "stabilization": "None",
        "response_fn": "exp",
        "loss_fn": "nll",
        "num_boost_round": 200,
        "params": {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        },
    },
    "wine_quality": {
        "distribution": "Gaussian",
        "stabilization": "MAD",
        "response_fn": "softplus",
        "loss_fn": "nll",
        "num_boost_round": 150,
        "params": {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        },
    },
}


