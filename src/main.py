import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from bde import BdeRegressor
from bde.loss import GaussianNLL
import mlflow 
import mlflow.data
import os
from pathlib import Path
from PreProcessing.PreProcessing import LoadData,PreProcessing
from config.metrics import GaussianNll, WinklerScore
import optuna

ROOT = Path(__file__).resolve().parent

#mlflow db path 
MLFLOW_DB = ROOT / "data" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ROOT / "mlruns"
MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

experiment_name = "test"

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
        experiment_name,
        artifact_location=(MLFLOW_ARTIFACT_ROOT / experiment_name).as_uri(),
    )

mlflow.set_experiment(experiment_name)
mlflow.autolog()

#load data : miami_housing, wine_quality ; #TODO: no preprocessing
datasets = LoadData()
seeds = [1,8,16]
#preprocess
Rng_Ho_Split = 12

def objective(trial):
            
            with mlflow.start_run(nested=True,run_name=f"trial_{trial.number}") as child_run:
                #define params
                bde_hidden_layers = trial.suggest_categorical("hidden_layers",[[16,16],[32,32]
                                                                               #,[16,16,16],[32,32,32]
                                                                               ])
                #bde_n_members =trial.suggest_int("n_members",4,10,step =1)
                bde_desired_energy_var_start, bde_desired_energy_var_end = trial.suggest_categorical("var_start_end",[(0.5,0.1),
                                                                                                                      (0.05,0.01),
                                                                                                                      #(0.005,0.001),
                                                                                                                      #(0.0005,0.0001)
                                                                                                                      ])
                bde_warmup_steps, bde_n_samples = trial.suggest_categorical("warmup_steps_n_samples",[(1000,200),
                                                                                                      #(2000,400),
                                                                                                      #(3000,600),
                                                                                                      #(5000,1000)
                                                                                                      ])
                epochs = 30
                validation_split = 0.15
                patience = 10
                
                params = {
                    "hidden_layers": bde_hidden_layers,
                    "desired_energy_var_start": bde_desired_energy_var_start,
                    "desired_energy_var_end": bde_desired_energy_var_end,
                    "warmup_steps":bde_warmup_steps,
                    "n_samples":bde_n_samples,
                    "seed": Rng_Ho_Split,
                    "epochs": epochs,
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
                mlflow.sklearn.log_model(regressor,name="model")
                trial.set_user_attr("run_id",child_run.info.run_id)
                
                return float(meanWinkler)
                #basically, after we found best config, lets say 50 runs, 







X_train_HO, X_test_HO, y_train_HO, y_test_HO, yScaler = PreProcessing("miami_housing",data=datasets["miami_housing"],random_seed=Rng_Ho_Split) # we do 1 hold-out-split
# split train in to test and val
X_train, X_test,y_train,y_test = train_test_split(X_train_HO,y_train_HO,random_state=Rng_Ho_Split,train_size=0.8)
        #hpo
        #create study
        #initialize child ruins
        #set main seed for sklearn. 
        
with mlflow.start_run(run_name="first_hpo") as run:
    
    n_trials = 4
    mlflow.log_param("n_trials",n_trials)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=n_trials)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({"best_winkler":study.best_value})
    
    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id",best_run_id)    
                
                
