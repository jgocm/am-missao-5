import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
import data_analysis

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/glass.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

X = np.array(df.iloc[:, :-1])
# TURN LABELS INTO NUMERBS STARTING FROM 0
#df.Type = pd.factorize(df.Type)[0]
y = np.array(df.Type)

# DATA EXPLORATORY ANALYSIS AND PREPROCESSING
#data_analysis.run(df, False)
#data_analysis.check_class_balance(df)
#data_analysis.check_attributes_distribution(df)
df_std=data_analysis.attributes_standardizing(df)
data_analysis.check_attributes_distribution(df_std)
df_norm=data_analysis.attributes_normalizing(df_std)
data_analysis.check_attributes_distribution(df_norm)
data_analysis.check_correlation(df)

# SPLITS DATA INTO TRAIN AND VALIDATION SETS
p = 0.8 # fracao de elementos no conjunto de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)
