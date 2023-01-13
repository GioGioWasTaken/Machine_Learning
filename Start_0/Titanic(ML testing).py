import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

pd.options.display.max_columns = None  # so it shows all columns
train_data = pd.read_csv("train.csv")
print(train_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
surv_rate_women = sum(women) / len(women)
print(f'\nWomen survival rate:{surv_rate_women}')

men = train_data.loc[train_data.Sex == 'male']["Survived"]
surv_rate_men = sum(men) / len(men)
print(f"Men surival rate:{surv_rate_men}\n")