import mlflow 
import mlflow.data
import os
from pathlib import Path
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from bde import BdeRegressor
from bde.loss import GaussianNLL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "data" / "mlflow.db"
MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("test123")
mlflow.autolog()

with mlflow.start_run():
        
    data = fetch_openml(name="airfoil_self_noise", as_frame='auto') # requires pandas
    dataml = mlflow.data.from_pandas(data.data,source = data.url, name = data.details.get("name","test"))
    mlflow.log_input(dataset=dataml ,context="training")
    X = data.data.values
    y = data.target.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    Xmu, Xstd = jnp.mean(X_train, 0), jnp.std(X_train, 0) + 1e-8
    Ymu, Ystd = jnp.mean(y_train, 0), jnp.std(y_train, 0) + 1e-8

    Xtr = (X_train - Xmu) / Xstd
    Xte = (X_test - Xmu) / Xstd
    ytr = (y_train - Ymu) / Ystd
    yte = (y_test - Ymu) / Ystd

    # Build the regressor
    regressor = BdeRegressor(
        hidden_layers=[16, 16],
        n_members=8,
        seed=0,
        loss=GaussianNLL(),
        epochs=50,
        validation_split=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_steps=50,
        n_samples=2000,
        n_thinning=2,
        patience=10,
        desired_energy_var_start=0.05,
        desired_energy_var_end=0.01
    )

    # Fit the regressor
    mlflow.log_params(regressor.get_params(deep=True))
    regressor.fit(x=Xtr, y=ytr.ravel())
    mlflow.sklearn.log_model(sk_model=regressor,name = f"FirstModel with{regressor.get_params(deep=True).get('epochs','')} ")
    # Get results from regressor
    means, sigmas = regressor.predict(Xte, mean_and_std=True)
    mean, intervals = regressor.predict(Xte, credible_intervals=[0.1, 0.9])
    train_score = regressor.score(Xtr,ytr)
    test_score=  regressor.score(Xte,yte)
    raw = regressor.predict(Xte, raw=True) # (ensemble members, n_samples/n_thinning, n_test_data, (mu,sigma))
    print(f"train score: {train_score} and test score : {test_score}")
    mlflow.log_metric("train_score", regressor.score(Xtr, ytr))
    mlflow.log_metric("test_score", regressor.score(Xte, yte))
  
