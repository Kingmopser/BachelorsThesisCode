
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![JAX](https://img.shields.io/badge/JAX-0.7.1-A020F0?style=flat&logo=google&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.1+-0194E2?style=flat&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pixi](https://img.shields.io/badge/Pixi-env-FFD700?style=flat&logo=conda-forge&logoColor=black)

# Evaluating MILE based Bayesian Deep Ensembles: A Benchmark for Predictive Accuracy and Uncertainty Quantification

This Repository contains the code for the evaluation Benchmark of my Bachelor's Thesis. 


## Scope
The goal is to implement an end to end pipeline that covers data loading, pre processing and model training and HPO. The focus would be to discover potential make or breaks during the training as well as determining suitable hyperparameter configurations by testing on an extended dataset setting. Hence, with this we establish a reproducible evaluation and benchmark environment.

## Project Structure

## high level architecture

## setup 

#### 1. install pixi if not present 
```bash
pip install pixi
```
#### 2. activate determinstic environment
```bash
pixi install 
```
#### 3. activate mlflow server
```bash
pixi run mlflow server --port 5000
```
#### 4. run scripts for experiment

*tbd*
```bash
import openml

benchmark_suite = openml.study.get_suite("tabarena-v0.1")
task_ids = benchmark_suite.tasks  # 51 task IDs

task = openml.tasks.get_task(task_ids[0])
dataset = task.get_dataset()
X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
```

#### 1. possible dataset selection : 

```bash
all supervised regression tasks. All datasets showcase superior RF performance compared to Linear Regression 


QSAR_fish_toxicity ; n = 907.0 ; big enough = 
healthcare insurance expense; n = 1338.0; big enough = 
QSAR-TID-11; n = 5742.0 ; big enough = ok
wine_qualityn; n = 6497.0;  ; big enough = ok 
Another-Dataset-on-used-Fiat-50 ;n = 1503.0 = ok 
miami_housing; n = 13776.0 = ok

--> basically create table already from csv. Used for explaining characteristics of data.

load 3-5 datasets
```
