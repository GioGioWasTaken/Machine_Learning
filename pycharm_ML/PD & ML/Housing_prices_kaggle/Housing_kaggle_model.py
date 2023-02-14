import numpy as np
import pandas as pd
# data=pd.read_csv('train.csv')
# print(data[['Id','LotFrontage','LotArea','YearBuilt','SalePrice']].head(5))
# # Let's do some feature engineering, we'll take the YearBuilt, and do 2023 (current year)- YearBuilt
# # to make a new feature, house age.
# # We will also make a new feature "Lot Depth" by doing: LotArea/LotFrontage
#
# Year_built=numpy.array(data['YearBuilt'])
# house_age=np.full(len(Year_built),2023)-Year_built
#
#  lets now add this new feature to the test.csv file
#   data = data.assign(house_age=house_age)
#   data.to_csv('train_updated.csv', index=False)
#  lot_frontage=np.array(data['LotFrontage'])
#  lot_area=np.array(data['LotArea'])
#  lot_depth=lot_area/lot_frontage
#  data = data.assign(lot_depth=lot_depth)

# above are the steps I took in order to make the train_updated.csv file.

data=pd.read_csv('train_updated.csv')
print(data[['Id','LotFrontage','LotArea','lot_depth','house_age','SalePrice']].head(5))

lot_frontage=np.array(data['LotFrontage'])
lot_area=np.array(data['LotArea'])
lot_depth=np.array(data['lot_depth'])
house_age=np.array(data['house_age'])
X_train = np.column_stack((lot_frontage, lot_area, lot_depth, house_age))
# Making a 2D list that has 4 features.
print(X_train)
Y_train=np.array(data['SalePrice']) # SalePrice as the output y.
