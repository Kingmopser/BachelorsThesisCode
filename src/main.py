from dotenv import load_dotenv
load_dotenv()
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from bde import BdeRegressor
import mlflow 
import os
from PreProcessing.PreProcessing import LoadData,PreProcessing
from config.metrics import GaussianNll, WinklerScore
from config.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_DB,
    IQR_TO_STD,
    BDE_GRID,
    GLOBAL_SEED,
    ROBUST_SEEDS,
    BDE_ACTIVE_OVERRIDE,
    XGBOOSTLSS_CONFIG
)
import optuna
from functools import partial
from mlflow.tracking import MlflowClient
import ast
import xgboost as xgb
import traceback
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
from tabicl import TabICLRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


def objective(trial,X_train,X_test,y_train,y_test):
            
            with mlflow.start_run(nested=True,run_name=f"trial_{trial.number}") as child_run:
                
                
                bde_hidden_layers = ast.literal_eval(
                    trial.suggest_categorical("hidden_layers", BDE_GRID["hidden_layers"])
                )
                bde_desired_energy_var_start, bde_desired_energy_var_end = ast.literal_eval(
                    trial.suggest_categorical("var_start_end", BDE_GRID["var_start_end"])
                )
                bde_warmup_steps, bde_n_samples = ast.literal_eval(
                    trial.suggest_categorical(
                        "warmup_steps_n_samples", BDE_GRID["warmup_steps_n_samples"]
                    )
                )                                                                                         
                bde_epochs = ast.literal_eval(
                    trial.suggest_categorical("epochs", BDE_GRID["epochs"])
                )
                validation_split = 0.15 
                patience = 10
                
                params = {
                    "hidden_layers": bde_hidden_layers,
                    "desired_energy_var_start": bde_desired_energy_var_start,
                    "desired_energy_var_end": bde_desired_energy_var_end,
                    "warmup_steps":bde_warmup_steps,
                    "n_samples":bde_n_samples,
                    "seed": GLOBAL_SEED,
                    "epochs": bde_epochs,
                    "validation_split": validation_split,
                    "patience":patience
                }
                
                mlflow.log_params(params=params)
                
                regressor = BdeRegressor(**params)
                regressor.fit(x=X_train,y=y_train.ravel()) # fitted on train. 
                
                '''
                performance on inner Test data 
                #main metrics: RMSE, Winkler Score, Coverage, NLL 
                '''
                y_pred = regressor.predict(X_test)
                #rmse
                rmse = root_mean_squared_error(y_true=y_test.ravel(),y_pred=y_pred)
                #winklerscore, coverage
                _, intervals = regressor.predict(X_test, credible_intervals=[0.05, 0.95]) # get 90% credible interval
                meanWinkler, coverage = WinklerScore(y_test.ravel(),pi_lower=intervals[0],pi_upper=intervals[1], returnCoverage=True)
                #nll
                mu,sigma = regressor.predict(X_test,mean_and_std=True)
                nll = GaussianNll(y_test.ravel(),mu=mu,sigma=sigma)
                
                mlflow.log_metrics({
                    "RMSE": float(rmse),
                    "Mean_Winkler_Score": float(meanWinkler),
                    "Winkler_Coverage": float(coverage),
                    "Negative_log_likelihood": float(nll)
                })
                mlflow.sklearn.log_model(regressor,name=f"model_trial_{trial.number}")
                trial.set_user_attr("run_id",child_run.info.run_id)
                
                return float(meanWinkler)
                #basically, after we found best config, lets say 50 runs, 

def extractParams(paramdict):
    
    params = {
                    "hidden_layers": ast.literal_eval(paramdict.get("hidden_layers")),
                    "desired_energy_var_start": float(paramdict.get("desired_energy_var_start")),
                    "desired_energy_var_end": float(paramdict.get("desired_energy_var_end")),
                    "warmup_steps":int(paramdict.get("warmup_steps")),
                    "n_samples":int(paramdict.get( "n_samples")),
                    "seed": int(paramdict.get("seed")),
                    "epochs":int(paramdict.get("epochs")),
                    "validation_split": float(paramdict.get("validation_split")),
                    "patience": None if paramdict.get("patience") in (None, "", "none", "None") else int(paramdict.get("patience"))
                }
    return params


def RobustTest(X_train_HO, X_test_HO, y_train_HO, y_test_HO, seeds, run_name, datasetname,global_seed = 12):
    '''
    RobustTest would be a seperate run. So that we can log the performance of the best model. 
    Ideally in Optuna we would have per dataset 1 HPO Study Experiment, and 1 Stresstest 
    '''
    
    
    '''
    This whole block is just to extract the params of the bde HPO from the dataset
    '''
    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB}")
    experiment = client.get_experiment_by_name(f"Experiment_Dataset_{datasetname}")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"attributes.run_name = '{run_name}'",
                                order_by=["attributes.start_time DESC"],
                                max_results=1,
        )
   
    parent_run_id = runs[0].info.run_id
    best_child_run_id = client.get_run(parent_run_id).data.params["best_child_run_id"]
    best_child_params = extractParams(client.get_run(best_child_run_id).data.params) # save best params from BDE HPO
    
    #current models
    models = {
       "BDE":BDEPredictor,
        "XGboostLSS":XGBoostLSSPredictor,
        "TabICL":TabICLPredictor,
        "LG":LGPredictor,
        "RF":RFPredictor
    }
    
    with mlflow.start_run(run_name=f"Robustnesstest_Data_{datasetname}") as run:
        for model,model_fn in models.items():
            with mlflow.start_run(nested=True,run_name=f"Model{model}_robust") as model_run:  
                try:
                    SeedRun(model_fn,seeds,X_train_HO, X_test_HO, y_train_HO, y_test_HO,best_child_params,datasetname)
                except Exception:
                    mlflow.set_tag("run_failed", True)
                    mlflow.log_text(traceback.format_exc(), "error.txt")
                    print(f"Model {model} failed")
                    continue
                    
                
        
                    
def BDEPredictor(X_train, y_train, X_test_HO,best_child_params,**kwargs):
    
    params = dict(best_child_params).copy()  # copy
    params.update(BDE_ACTIVE_OVERRIDE) #can be varied for each preferred ablation  
   
    def _mlflow_safe(v):
        # MLflow's UI can display `None` as empty; stringify for clarity.
        return "none" if v is None else v

    mlflow.log_params(
        {
            "epochs": _mlflow_safe(params.get("epochs")),
            "patience": _mlflow_safe(params.get("patience")),
            "validation_split": _mlflow_safe(params.get("validation_split")),
            "warmup_steps": params.get("warmup_steps"),
            "n_samples": params.get("n_samples"),       
            "seed": params.get("seed"),
            "n_members": params.get("n_members")
        }
    )
    print(params)
    
   
    regressor = BdeRegressor(**params)
    regressor.fit(X_train,y_train.ravel())
    
    try:
        if getattr(regressor, "_bde", None) is not None and getattr(regressor._bde, "history", None):
            model0 = regressor._bde.history.get("Model0", {})
            epoch_history = model0.get("epoch", [])

            mlflow.log_param("bde_history_length_model0", int(len(epoch_history)))
            if "stop_epoch" in model0:
                mlflow.log_param("bde_stop_epoch_model0", int(model0.get("stop_epoch")))

            trainloss = model0.get("trainloss")
            valloss = model0.get("valloss")

            snapshot_index = 400
            # Explicit checkpoints for comparing runs like epochs=800 vs epochs=1200.
            checkpoint_epochs = [100, 200, 400, 600, 800, 1200]

            def _log_loss_history(loss_name, loss_values):
                if loss_values is None or len(loss_values) == 0:
                    return

                mlflow.log_metric(f"bde_{loss_name}_final_value_model0", float(loss_values[-1]))
                mlflow.log_param(
                    f"bde_{loss_name}_final_history_index_model0",
                    int(len(loss_values) - 1),
                )

                if epoch_history and len(epoch_history) == len(loss_values):
                    mlflow.log_param(
                        f"bde_{loss_name}_final_epoch_value_model0",
                        int(epoch_history[-1]),
                    )

                if len(loss_values) > snapshot_index:
                    mlflow.log_metric(
                        f"bde_{loss_name}_history_idx_{snapshot_index}_value_model0",
                        float(loss_values[snapshot_index]),
                    )
                    mlflow.log_param(
                        f"bde_{loss_name}_history_idx_{snapshot_index}_model0",
                        snapshot_index,
                    )
                    if epoch_history and len(epoch_history) > snapshot_index:
                        mlflow.log_param(
                            f"bde_{loss_name}_epoch_value_at_history_idx_{snapshot_index}_model0",
                            int(epoch_history[snapshot_index]),
                        )

                # Log by *epoch value* (robust even if history is truncated by early stopping).
                if epoch_history and len(epoch_history) == len(loss_values):
                    epoch_list = [int(e) for e in epoch_history]
                    for wanted_epoch in (800, 1200):
                        if wanted_epoch in epoch_list:
                            idx = epoch_list.index(wanted_epoch)
                            mlflow.log_metric(
                                f"bde_{loss_name}_at_epoch_{wanted_epoch}_value_model0",
                                float(loss_values[idx]),
                            )
                            mlflow.log_param(
                                f"bde_{loss_name}_history_idx_at_epoch_{wanted_epoch}_model0",
                                int(idx),
                            )

                for checkpoint_epoch in checkpoint_epochs:
                    checkpoint_idx = checkpoint_epoch - 1
                    if checkpoint_idx < 0 or len(loss_values) <= checkpoint_idx:
                        continue

                    mlflow.log_metric(
                        f"bde_{loss_name}_epoch_{checkpoint_epoch}_value_model0",
                        float(loss_values[checkpoint_idx]),
                    )
                    mlflow.log_param(
                        f"bde_{loss_name}_history_idx_for_epoch_{checkpoint_epoch}_model0",
                        int(checkpoint_idx),
                    )
                    if epoch_history and len(epoch_history) > checkpoint_idx:
                        mlflow.log_param(
                            f"bde_{loss_name}_actual_epoch_value_for_epoch_{checkpoint_epoch}_model0",
                            int(epoch_history[checkpoint_idx]),
                        )

            _log_loss_history("trainloss", trainloss)
            _log_loss_history("valloss", valloss)
    except Exception:
        pass
    
    y_pred = regressor.predict(X_test_HO)
    mu,sigma = regressor.predict(X_test_HO,mean_and_std=True)
    _, intervals = regressor.predict(X_test_HO, credible_intervals=[0.05, 0.95])

    # Prediction diagnostics: metrics can look identical while predictions differ slightly
    # (or vice versa due to rounding)
    try:
        mu_np = np.asarray(mu)
        sigma_np = np.asarray(sigma)
        y_pred_np = np.asarray(y_pred)
        pi_lower_np = np.asarray(intervals[0])
        pi_upper_np = np.asarray(intervals[1])
        width_np = pi_upper_np - pi_lower_np

        def _log_pred_stats(name: str, arr: np.ndarray):
            arr = arr.astype(np.float64, copy=False).ravel()
            if arr.size == 0:
                return
            mlflow.log_metrics(
                {
                    f"bde_pred_{name}_mean": float(arr.mean()),
                    f"bde_pred_{name}_std": float(arr.std(ddof=0)),
                    f"bde_pred_{name}_min": float(arr.min()),
                    f"bde_pred_{name}_max": float(arr.max()),
                    f"bde_pred_{name}_l1_mean": float(np.mean(np.abs(arr))),
                    f"bde_pred_{name}_linf": float(np.max(np.abs(arr))),
                }
            )

        _log_pred_stats("y_pred", y_pred_np)
        _log_pred_stats("mu", mu_np)
        _log_pred_stats("sigma", sigma_np)
        _log_pred_stats("pi_width", width_np)

        # Stable fingerprint to quickly detect bit-identical predictions across runs.
        # (Logged as params because MLflow metrics must be numeric.)
        import hashlib

        mlflow.log_params(
            {
                "bde_pred_mu_md5": hashlib.md5(mu_np.tobytes()).hexdigest(),
                "bde_pred_sigma_md5": hashlib.md5(sigma_np.tobytes()).hexdigest(),
                "bde_pred_y_pred_md5": hashlib.md5(y_pred_np.tobytes()).hexdigest(),
            }
        )
    except Exception:
        pass
        
    return {
        "y_pred":y_pred,       
        "mu": mu,
        "sigma": sigma,
        "pi_lower": intervals[0],
        "pi_upper": intervals[1],
    }
    
def XGBoostLSSPredictor(X_train, y_train, X_test_HO,datasetname ,model_seed=12,**kwargs):
    
    dtrain = xgb.DMatrix(X_train, label=y_train.ravel())
    dtest  = xgb.DMatrix(X_test_HO)
    
    cfg = XGBOOSTLSS_CONFIG[datasetname]
    xgblss = XGBoostLSS(Gaussian(
        stabilization=cfg["stabilization"],
        response_fn=cfg["response_fn"],
        loss_fn=cfg["loss_fn"],
    )) # mention that without stabilization, weird performance
    
    params = dict(cfg["params"])
    params["seed"] = model_seed
    
    xgblss.train(params=params, dtrain=dtrain, num_boost_round=cfg["num_boost_round"])
    
    pred_params = xgblss.predict(data=dtest, pred_type="parameters")
    
    mu = pred_params["loc"].to_numpy()     
    sigma = pred_params["scale"].to_numpy()
    
    pred_q = xgblss.predict(
        data=dtest,
        pred_type="quantiles",
        quantiles=[0.05, 0.95],
    )
    
    pi_lower = pred_q.iloc[:, 0].to_numpy()
    pi_upper = pred_q.iloc[:, 1].to_numpy()
    
    return {
        "y_pred":mu,       
        "mu": mu,
        "sigma": sigma,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
    }
    
def TabICLPredictor(X_train, y_train, X_test_HO, model_seed = 12,**kwargs):

    tabicl = TabICLRegressor(random_state=model_seed)
    tabicl.fit(X_train,y_train.ravel())
   
    y_pred = tabicl.predict(X_test_HO, output_type="mean")
   
    interval = tabicl.predict(
        X_test_HO,
        output_type="quantiles",
        alphas=[0.05, 0.95],
    )
   
    iqr = tabicl.predict(
    X_test_HO,
    output_type="quantiles",
    alphas=[0.25, 0.75])
   
    sigma = (iqr[:, 1] - iqr[:, 0]) / IQR_TO_STD
    
    return {
       "y_pred":y_pred,
       "mu":y_pred,
       "sigma": sigma,#sigma #requires approx via iqr and quantile approx 
       "pi_lower":interval[:,0],
       "pi_upper":interval[:,1]
    } 
    
def LGPredictor(X_train, y_train, X_test_HO,**kwargs):
    
    lr = LinearRegression()
    lr.fit(X_train, y_train.ravel())

    # Estimate constant Gaussian noise from training residuals
    mu_train = lr.predict(X_train)
    residuals = jnp.asarray(y_train.ravel()) - jnp.asarray(mu_train)
    sigma_hat = jnp.std(residuals, ddof=1)

    mu = jnp.asarray(lr.predict(X_test_HO))
    sigma = jnp.full_like(mu, sigma_hat)
    
    return {
        "y_pred": mu,
        "mu": mu,
        "sigma": sigma
    }
    
def RFPredictor(X_train, y_train, X_test_HO, model_seed=12,**kwargs):
    
    rf =RandomForestRegressor(random_state=model_seed,
                            n_estimators=1000,
                            n_jobs=-1,
                            )
    rf.fit(X_train,y_train.ravel())
    y_pred = rf.predict(X_test_HO)
    
    y_train_pred = rf.predict(X_train)
    residuals = y_train.ravel() -  y_train_pred
    sigma_hat = jnp.std(jnp.asarray(residuals), ddof=1)
    sigma = jnp.full_like(y_pred, sigma_hat)
    
    return {
       "y_pred":y_pred,
       "mu":y_pred,
       "sigma":sigma# is constant
   } 
    

def SeedRun(model_fn,seeds,X_train_HO, X_test_HO, y_train_HO, y_test_HO,bdeparam,datasetname):    
    
    for i,seed in enumerate(seeds):            
        with mlflow.start_run(nested=True, run_name=f"seed{i}"):
            
            X_train,_,y_train,_ = train_test_split(X_train_HO,y_train_HO,random_state=seed,train_size=0.8)
            
            preds = model_fn(
                    X_train= X_train,
                    y_train = y_train,
                    X_test_HO = X_test_HO, 
                    best_child_params = bdeparam,
                    model_seed=GLOBAL_SEED,
                    datasetname=datasetname)
            
            if "y_pred" in preds:
                rmse = root_mean_squared_error(y_true=y_test_HO.ravel(),y_pred=preds.get("y_pred",""))
                mlflow.log_metrics({"RMSE": float(rmse)}) 
                
            if "pi_lower" in preds:
            #winklerscore, coverage
                meanWinkler, coverage = WinklerScore(y_test_HO.ravel(),
                                                    pi_lower=preds.get("pi_lower",""),
                                                    pi_upper=preds.get("pi_upper",""), 
                                                    returnCoverage=True)
                mlflow.log_metrics({"Mean_Winkler_Score": float(meanWinkler),
                                    "Winkler_Coverage": float(coverage)}) 
                
            if "sigma" in preds:
            #nll
                nll = GaussianNll(y_test_HO.ravel(),
                                mu=preds.get("mu",""),
                                sigma=preds.get("sigma",""))
                mlflow.log_metrics({"Negative_log_likelihood": float(nll)}) 
          
                
def runExperiment(datasetname,dataset ,seeds, global_seed, run_hpo = False, n_trials = 50):
    
    experiment_name = f"Experiment_Dataset_{datasetname}"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(MLFLOW_ARTIFACT_ROOT / experiment_name).as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    
    #we do preprocessing for LG,RF,CATBOOSTLSS
    X_train_HO, X_test_HO, y_train_HO, y_test_HO, yScaler = PreProcessing(datasetname,data=dataset,random_seed=global_seed) # we do 1 hold-out-split


    # split train in to test and val
    X_train, X_test,y_train,y_test = train_test_split(X_train_HO,y_train_HO,random_state=global_seed,train_size=0.8)
            
    if run_hpo:
        with mlflow.start_run(run_name=f"HPO_Study_{datasetname}") as run:
            mlflow.log_param("n_trials", n_trials)

            # Minimal change to avoid duplicate categorical trials: use a GridSampler.
            sampler = optuna.samplers.GridSampler(BDE_GRID)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            obj = partial(objective, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            # Continue even if individual trials fail (they will be marked as FAIL)
            study.optimize(obj, n_trials=n_trials, catch=(Exception,))

            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"best_winkler": study.best_value})

            if best_run_id := study.best_trial.user_attrs.get("run_id"):
                mlflow.log_param("best_child_run_id", best_run_id)

    
    RobustTest(
        X_train_HO=X_train_HO,
        y_train_HO=y_train_HO,
        X_test_HO=X_test_HO,
        y_test_HO=y_test_HO,
        seeds=seeds,
        run_name=f"HPO_Study_{datasetname}",
        datasetname=datasetname,
        global_seed = global_seed
    )
        
if __name__ == "__main__":
    datasets = LoadData()
    
    for datasetname, dataset in datasets.items():
        try:
            runExperiment(datasetname=datasetname,
                          dataset=dataset,
                          seeds=ROBUST_SEEDS,
                          global_seed=GLOBAL_SEED,
                          n_trials=1,
                          run_hpo=True)        
            
        except Exception as e:
            print(f"Dataset loading failed | Error: {e}")