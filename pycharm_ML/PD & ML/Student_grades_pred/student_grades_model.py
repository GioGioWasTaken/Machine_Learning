import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# The goal: Trying to engineer a model that'll predict student success based on many different parameters.

math_data=pd.read_csv('student-mat.csv',delimiter=';')
Port_data=pd.read_csv('student-por.csv',delimiter=';') # Portuguese data

# extracting features from the data
school_name=list( math_data['school'] ) # Let's assign arbitrary numbers, to make a feature from 'school'
school_name=[20 if i == 'GP' else 10 for i in school_name]
school_name=np.array( school_name ) # feature 1 engineered.
age=np.array(math_data['age'])
# same tactic as school_math
address=list(math_data['address'])
address=[20 if i=='U' else 10 for i in address]
address=np.array(address)
m_ed=np.array(math_data['Medu']) # mother's education
f_ed=np.array(math_data['Fedu']) # father's education
travel_time=np.array(math_data['traveltime']) # time it takes to get to the school
study_time=np.array(math_data['studytime'])
c_failures=np.array(math_data['failures']) #class failures
schools_help=np.array(math_data['schoolsup'])
schools_help=[1 if i=='yes' else 0 for i in schools_help] # binary: extra educational support
schools_help=np.array(schools_help)

# setting up features variable and target values
X_train=np.column_stack((school_name,age,address,m_ed,f_ed,travel_time,study_time,c_failures,schools_help))
Y_train=np.array(math_data['G3']) # final year grade.