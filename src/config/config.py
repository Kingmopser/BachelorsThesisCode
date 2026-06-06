import pandas as pd


metadatapath = "/Users/kingmopser/BachelorThesis/BachelorsThesisCode/src/data/task_metadata_tabarena51.csv"
# columns controls which datasets are loaded by LoadData().
columns = [
    #"QSAR_fish_toxicity",
    # "QSAR-TID-11", # too sparse, lol
     "wine_quality", # perfect complexity
     #"healthcare_insurance_expenses", # low complexity
     #"Another-Dataset-on-used-Fiat-500", # too low complexity
     #"miami_housing"
  #   "superconductivity"
     ## perfect complexity
]

BDE_GRID = {
    "hidden_layers": ["[16,16,16,16,16,16]"],
    "var_start_end": ["(0.5,0.1)", "(0.05,0.01)", "(0.005,0.001)"],
    "warmup_steps_n_samples": ["(50000,10000)"],
}
