import pandas as pd
import numpy as np
import openml
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pathlib import Path
from config.config import metadatapath, columns
# loading datasets used by tabarena

# specify required datasets. 
ROOT = Path(__file__).resolve().parent.parent

#create latex table
    
def CreateTable(characteristics): 
    
    table_df_latex=characteristics.to_latex(caption="Dataset characterstics including shape and dimensions.")
    with open("df_table.tex","w") as f:
        f.write(table_df_latex)
    
    
def determineData(columns): 
    #get attributes of datatable
    dfs = pd.read_csv(metadatapath)

    characteristics=dfs.loc[dfs["name"].isin(columns),:][["tid","name","NumberOfFeatures","target_feature","NumberOfFeatures","NumberOfInstances","NumberOfNumericFeatures","NumberOfSymbolicFeatures"]]                                                                                                          
    return characteristics
    
def LoadData():
    
    tables = dict()
    characteristics = determineData(columns)
    print(f"LoadData: loading {len(characteristics)} configured dataset(s): {characteristics['name'].tolist()}", flush=True)
    print("LoadData: fetching OpenML suite tabarena-v0.1...", flush=True)
    '''try:
        benchmark_suite = openml.study.get_suite("tabarena-v0.1")
        task_ids = benchmark_suite.tasks  # 51 task IDs
        print(f"LoadData: fetched suite with {len(task_ids)} task ids.", flush=True)
    except Exception as exc:
        print(
            f"LoadData: suite lookup failed ({type(exc).__name__}: {str(exc).splitlines()[0]}). "
            "Continuing with local metadata task ids.",
            flush=True,
        )'''
    
    #fetching datasets and saving as dict for easier iteration
    for id in characteristics["tid"]:
        #print(f"LoadData: fetching task {id}...", flush=True)
        task = openml.tasks.get_task(id)
        print(f"LoadData: fetching dataset for task {id}...", flush=True)
        df = task.get_dataset()
        #print(f"LoadData: converting dataset {df.name} to dataframe...", flush=True)
        X, y, _, _ = df.get_data(target=task.target_name, dataset_format="dataframe")
        tables.update({df.name: {"X":X,"y":y.values.reshape(-1,1)}})
        print(f"Successfully loaded dataset {df.name} | tid : {id}", flush=True)
    return tables # dictionary{Dataset_name: "X": {X_data}, "y": {target_col}}


def PreProcessing(name,data,model_name="standard",test = 0.2,random_seed=0):
    
    X = data.get("X","")
    y = data.get("y","")
    
    is_binary = True if name == "QSAR-TID-11" else False
    is_housing = True if "month_sold" in X.columns else False     
    is_standard = True if model_name != "standard" else False        
    if is_binary: # for QSAR TID 11 Dataset   
        print("correct detected")
        X = X.loc[:, X.nunique() > 1]
    
    if is_housing:
        m = X["month_sold"].astype(int) - 1  # Jan=0 ... Dec=11
        X["month_sold_sin"] = np.sin(2 * np.pi * m / 12.0)
        X["month_sold_cos"] = np.cos(2 * np.pi * m / 12.0)
        X = X.drop(columns=["month_sold"])
                

    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test,random_state=random_seed)
    
    if is_standard:
        return X_train, X_test, y_train, y_test
        
    preprocessor = ColumnTransformer([("numeric",StandardScaler(),num_cols),
                             ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)])
    
    yScaler = StandardScaler()
    
    X_train= preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    y_train = yScaler.fit_transform(y_train)
    y_test = yScaler.transform(y_test)
    
    return X_train, X_test, y_train, y_test, yScaler
    
    
