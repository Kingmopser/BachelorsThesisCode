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
from functools import partial
from mlflow.tracking import MlflowClient
import ast

ROOT = Path(__file__).resolve().parent

#mlflow db path 
MLFLOW_DB = ROOT / "data" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = ROOT / "mlruns"
MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")



def objective(trial,X_train,X_test,y_train,y_test):
            
            with mlflow.start_run(nested=True,run_name=f"trial_{trial.number}") as child_run:
                
                #define params
                #bde_n_members =trial.suggest_int("n_members",4,10,step =1)
                bde_hidden_layers = trial.suggest_categorical("hidden_layers",[[16,16],[32,32],[16,16,16],[32,32,32]])
                bde_desired_energy_var_start, bde_desired_energy_var_end = trial.suggest_categorical("var_start_end",[(0.5,0.1),(0.05,0.01),(0.005,0.001),(0.0005,0.0001)])
                bde_warmup_steps, bde_n_samples = trial.suggest_categorical("warmup_steps_n_samples",[(1000,200),(2000,400),(3000,600),(5000,1000) ])                                                                                         
                epochs = 100
                validation_split = 0.15 # or 0.0 damnnnn ahahah
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
                    "patience":int(paramdict.get("patience"))
                }
    return params

def RobustTest(X_train_HO, X_test_HO, y_train_HO, y_test_HO ,seeds,run_name,datasetname):    
    '''
    RobustTest would be a seperate run. So that we can log the performance of the best model. Ideally in Optuna we would have per dataset 1 HPO Study Experiment, and 1 Stresstest 
    '''
    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB}")
    experiment = client.get_experiment_by_name(f"Experiment_Dataset_{datasetname}")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"attributes.run_name = '{run_name}'",
                                order_by=["attributes.start_time DESC"],
                                max_results=1,
        )
    '''
    Thats for BDE HPO
    '''
    parent_run_id = runs[0].info.run_id
    best_child_run_id = client.get_run(parent_run_id).data.params["best_child_run_id"]        
    best_child_params = extractParams(client.get_run(best_child_run_id).data.params)
    
    with mlflow.start_run(run_name=f"Robustnesstest_Data_{datasetname}") as run:
        for i,seed in enumerate(seeds): 
            with mlflow.start_run(nested=True,run_name=f"test_{i}") as child_run:   
                
                X_train,_,y_train,_ = train_test_split(X_train_HO,y_train_HO,random_state=seed,train_size=0.8)
                
                #TODO: this is what need to be modular
                
                #initialize bde regressor with params and fit on train data
                regressor = BdeRegressor(**best_child_params)
                regressor.fit(X_train,y_train.ravel())
                
                #evaluate performance on test dataset
                y_pred = regressor.predict(X_test_HO)
                
                #rmse
                rmse = root_mean_squared_error(y_true=y_test_HO.ravel(),y_pred=y_pred)
            
                #winklerscore, coverage
                _, intervals = regressor.predict(X_test_HO, credible_intervals=[0.05, 0.95]) # get 90% credible interval
                meanWinkler, coverage = WinklerScore(y_test_HO.ravel(),pi_lower=intervals[0],pi_upper=intervals[1], returnCoverage=True)
            
                #nll
                mu,sigma = regressor.predict(X_test_HO,mean_and_std=True)
                nll = GaussianNll(y_test_HO.ravel(),mu=mu,sigma=sigma)
            
                mlflow.log_metrics({
                    "RMSE": float(rmse),
                    "Mean_Winkler_Score": float(meanWinkler),
                    "Winkler_Coverage": float(coverage),
                    "Negative_log_likelihood": float(nll)
                })
def runExperiment(datasetname,seeds,Rng_Ho_Split): #TODO : modular function for entire dataset, fitting TabpFN,catboostlss and so on. 
    experiment_name = f"Experiment_Dataset_{datasetname}"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(MLFLOW_ARTIFACT_ROOT / experiment_name).as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    
    #we do preprocessing for LG,RF,CATBOOSTLSS
    X_train_HO, X_test_HO, y_train_HO, y_test_HO, yScaler = PreProcessing(datasetname,data=datasets[datasetname],random_seed=Rng_Ho_Split) # we do 1 hold-out-split


    # split train in to test and val
    X_train, X_test,y_train,y_test = train_test_split(X_train_HO,y_train_HO,random_state=Rng_Ho_Split,train_size=0.8)
            
    with mlflow.start_run(run_name=f"HPO_Study_{datasetname}") as run:
        
        n_trials = 1        
        mlflow.log_param("n_trials",n_trials)
        
        study = optuna.create_study(direction="minimize")
        
        obj = partial(objective,X_train = X_train,X_test = X_test,y_train=y_train,y_test=y_test)
        
        # Continue even if individual trials fail (they will be marked as FAIL)
        study.optimize(obj, n_trials=n_trials, catch=(Exception,))
        
        mlflow.log_params(study.best_params)
        
        mlflow.log_metrics({"best_winkler":study.best_value})
        
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id",best_run_id)    
                    
    RobustTest(X_train_HO=X_train_HO,y_train_HO=y_train_HO,X_test_HO=X_test_HO,y_test_HO=y_test_HO, seeds=seeds, run_name=f"HPO_Study_{datasetname}",datasetname=datasetname)
        
if __name__ == "__main__":
    
    datasets = LoadData()
    seeds = [24,38,136]
    #preprocess
    Rng_Ho_Split = 12
    runExperiment('miami_housing',seeds=seeds,Rng_Ho_Split=Rng_Ho_Split)        
