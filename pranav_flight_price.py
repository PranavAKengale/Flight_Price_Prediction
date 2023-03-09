import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()
import pickle

df=pd.read_excel(r'C:\Users\prana\Downloads\Pranav_New\Data_Train.xlsx')

df.head(5)

 pd.set_option('display.max_columns',None)

df.head()

df.shape

df.info()

df.dropna(inplace=True)

df.shape

df.head(5)

df['Airline'].value_counts()

df['Source'].value_counts()

df['Destination'].value_counts()

### Preprcessing for Data_of_Journey
df['Day_of_Journey']=pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y" ).dt.day
df['Month_of_Journey']=pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y" ).dt.month
#df['Day_of_week']=pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y" ).dt.dayofweek
df.drop(['Date_of_Journey'],axis=1,inplace=True)


###Preprcessing for Dept time
df['Dep_Hour']=pd.to_datetime(df['Dep_Time'] ).dt.hour
df['Dep_Min']=pd.to_datetime(df['Dep_Time'] ).dt.minute
df.drop(['Dep_Time'],axis=1,inplace=True)

##Preprocessing for Arrival Time
df['Arrival_Hour']=pd.to_datetime(df['Arrival_Time'] ).dt.hour
df['Arrival_Min']=pd.to_datetime(df['Arrival_Time'] ).dt.minute
df.drop(['Arrival_Time'],axis=1,inplace=True)

df.head(2)

## Preprocessing for Duration
new = df['Duration'].str.split(" ", n = 2, expand = True)
  # making separate Hours column from new data frame
df["Duration_Hours"]= new[0]
  # making separate Minutes column from new data frame
df["Duration_Minutes"]= new[1]
#Removing letters from string
df["Duration_Hours"] = df["Duration_Hours"].str.replace(r'\D', '')
df["Duration_Minutes"] = df["Duration_Minutes"].str.replace(r'\D', '')

df['Duration_Minutes'] = df['Duration_Minutes'].fillna(0)
df["Duration_Hours"] = df["Duration_Hours"].astype(int)
df["Duration_Minutes"] = df["Duration_Minutes"].astype(int)
df.drop(["Duration"], axis = 1, inplace = True)

df.head(5)

# Nominal data --> Data not in order -->OneHotEncoder is used
# Ordinal data --> Data in order -->LabelEncode is used

df['Airline'].value_counts()

df.drop(['Route','Additional_Info'],axis=1,inplace=True)

df['Total_Stops'].value_counts()

df['Total_Stops']=df['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})

df_Airline=pd.get_dummies(df['Airline'], prefix="Airline")
df_Source=pd.get_dummies(df['Source'], prefix="Source")
df_Destination=pd.get_dummies(df['Destination'], prefix="Destination")
df = pd.concat([df, df_Airline,df_Source,df_Destination ], axis=1)
df.drop(['Airline','Destination','Source'],axis=1,inplace=True)

df.head(4)

df.columns

df_for_corr=df[['Total_Stops', 'Price', 'Day_of_Journey', 'Month_of_Journey',
       'Dep_Hour', 'Dep_Min', 'Arrival_Hour', 'Arrival_Min', 'Duration_Hours',
       'Duration_Minutes', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Destination_Banglore', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi']]

df.shape

# Test Data 

df_test=pd.read_excel(r'C:\Users\prana\Downloads\Pranav_New\Test_set.xlsx')

df_test.head(4)

df_test.shape

# All the preprocessing on test data

df_test.dropna(inplace=True)

df_test['Day_of_Journey']=pd.to_datetime(df_test.Date_of_Journey, format="%d/%m/%Y" ).dt.day
df_test['Month_of_Journey']=pd.to_datetime(df_test.Date_of_Journey, format="%d/%m/%Y" ).dt.month
#df_test['Day_of_week']=pd.to_datetime(df_test.Date_of_Journey, format="%d/%m/%Y" ).dt.dayofweek
df_test.drop(['Date_of_Journey'],axis=1,inplace=True)



df_test['Dep_Hour']=pd.to_datetime(df_test['Dep_Time'] ).dt.hour
df_test['Dep_Min']=pd.to_datetime(df_test['Dep_Time'] ).dt.minute
df_test.drop(['Dep_Time'],axis=1,inplace=True)


df_test['Arrival_Hour']=pd.to_datetime(df_test['Arrival_Time'] ).dt.hour
df_test['Arrival_Min']=pd.to_datetime(df_test['Arrival_Time'] ).dt.minute
df_test.drop(['Arrival_Time'],axis=1,inplace=True)



df_test.head(2)

df_test['Airline'].value_counts()

### Preprocessing of Duration on test data 
new=df_test['Duration'].str.split(" ",n=2,expand=True)
df_test['Duration_Hours']=new[0]
df_test["Duration_Minutes"]= new[1]
df_test["Duration_Hours"] = df_test["Duration_Hours"].str.replace(r'\D', '')
df_test["Duration_Minutes"] = df_test["Duration_Minutes"].str.replace(r'\D', '')


df_test['Duration_Minutes'] = df_test['Duration_Minutes'].fillna(0)
df_test["Duration_Hours"] = df_test["Duration_Hours"].astype(int)
df_test["Duration_Minutes"] = df_test["Duration_Minutes"].astype(int)
df_test.drop(["Duration"], axis = 1, inplace = True)

df_test.drop(['Route','Additional_Info'],axis=1,inplace=True)

df_test['Total_Stops'].value_counts()

df_test['Total_Stops']=df_test['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})

df_Airline=pd.get_dummies(df_test['Airline'], prefix="Airline")
df_Source=pd.get_dummies(df_test['Source'], prefix="Source")
df_Destination=pd.get_dummies(df_test['Destination'], prefix="Destination")
df_test = pd.concat([df_test, df_Airline,df_Source,df_Destination ], axis=1)
df_test.drop(['Airline','Destination','Source'],axis=1,inplace=True)

df_test.head(4)

df_test.shape

df_test.columns

## Feature Selection
1)heatmap
2)feature_importance
3)SelectKBest


df.shape

X = df[df.columns.difference(["Price"])]

y=df['Price']

X

## Important features using ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(X,y)


selection.feature_importances_

plt.figure(figsize=(12,9))
feature_imp=pd.Series(selection.feature_importances_,index=X.columns)
feature_imp.nlargest(20).plot(kind='bar')

feature_imp

X = X.astype(str).astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

X_train.columns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn import metrics
! pip install xgboost
import xgboost as xgb
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import cross_val_score

regg_models = [LinearRegression(), Lasso(), Ridge(), SVR(),
               RandomForestRegressor(), xgb.XGBRegressor(),Lars(),LassoLars()]

R2_score = []
Score_Train=[]
Score_Test=[]
RMSE = []
MAE=[]
Error=[]
MSE=[]
mean=[]
std=[]
    
#Making ditonary to plot bar graph
# R2_score_df=()
R2_score_dict = {}
RMSE_score_dict = {}
MAE_score_dict = {}
MSE_score_dict={}
Score_Train__dict={}
Score_Test__dict={}
Cross_Valication_score_dict={}

for i in regg_models:    
    train_model = i.fit(X_train, y_train)
    y_pred = train_model.predict(X_test)
    # score = train_model.score(X_test, y_test)
    # r2_score.append(score)
    #R2Score
    r2score=metrics.r2_score(y_test,y_pred) # (coefficient of determination) regression score function
    R2_score.append(r2score)
    #Score training dataset
    scoretrain=train_model.score(X_train,y_train)
    Score_Train.append(scoretrain)
    #Score training dataset
    scoretest=train_model.score(X_test,y_test)
    Score_Test.append(scoretest)
    
    #Normaized RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))/(max(y)-min(y))
    RMSE.append(rmse)   
    #MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    MSE.append(mse)   
    #MAE
    mae=metrics.mean_absolute_error(y_test,y_pred)
    MAE.append(mae)
    #Cross validation score
    cvs=cross_val_score(i,X,y,cv=5,scoring=make_scorer(r2_score))
    mean.append(np.mean(cvs))
    std.append(np.std(cvs))
    #Error
    # errr = [((y_pred[i]-y_test[i])/(y_test[i])) for i in range(0,len(y_test))]
    # errr = [((y_prediction[i]-actual[i])/(actual[i])) for i in range(0,len(y_test))]
    # Error.append(errr)    



print("                                         Training Data \n","="*100)
print("r2_score")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',R2_score[i])
    R2_score_dict.update({regg_models[i].__class__.__name__: R2_score[i]})

print('-'*100,'\n')
print("Score Train")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',Score_Train[i])
    Score_Train__dict.update({regg_models[i].__class__.__name__:Score_Train[i]})

print('-'*100,'\n')
print("Score Test")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',Score_Test[i])
    Score_Test__dict.update({regg_models[i].__class__.__name__:Score_Test[i]})
    
print('-'*100,'\n')
print("Normalized RMSE")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',RMSE[i])
    RMSE_score_dict.update({regg_models[i].__class__.__name__:RMSE[i]})

print('-'*100,'\n')
print("MSE")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',MSE[i])
    MSE_score_dict.update({regg_models[i].__class__.__name__:MSE[i]})

print('-'*100,'\n')
print("MAE")
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',MAE[i])
    MAE_score_dict.update({regg_models[i].__class__.__name__ :MAE[i]})


print('-'*100,'\n')
print('________________Cross Validation Score________________ \n')
print("-"*40,'\n')
for i in range(len(regg_models)):
    print(regg_models[i].__class__.__name__ ,':',mean[i])
    Cross_Valication_score_dict.update({regg_models[i].__class__.__name__ :MAE[i]})

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
reg_rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)          
rf_random.fit(X_train,y_train)
rf_random.best_params_
pred = rf_random.predict(X_test)


print("R2score:",metrics.r2_score(y_test,pred)) # (coefficient of determination) regression score function
#Score training dataset
print("scoretrain:",rf_random.score(X_train,y_train))
#Score training dataset
print("scoretest",rf_random.score(X_test,y_test))
#Normaized RMSE
rmse = np.sqrt(metrics.mean_squared_error(y_test,pred))/(max(y)-min(y))
print("Normaized RMSE:", rmse)
#MSE
print("MSE", metrics.mean_squared_error(y_test, pred))
#MAE
print("MAE:",metrics.mean_absolute_error(y_test,pred))

# Saving our Model

pickle.dump(rf_random, open('pranav_rf_model.pkl','wb'))
