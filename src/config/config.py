import pandas as pd


metadatapath = "/Users/kingmopser/BachelorThesis/BachelorsThesisCode/src/data/task_metadata_tabarena51.csv"
# columns controls which datasets are loaded by LoadData().
columns = [
    #"QSAR_fish_toxicity",
    # "QSAR-TID-11", # too sparse, lol
     "wine_quality", # perfect complexity
     "healthcare_insurance_expenses", # low complexity
     "Another-Dataset-on-used-Fiat-50", # too low complexity
     "miami_housing" # perfect complexity
]
