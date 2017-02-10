
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(1000)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


def clean_data(df):
    #create dummies for datasources & concat
    datasources = pd.get_dummies(df['datasource'])
    df = pd.concat([df, datasources], axis = 1)
    #get rid v old & non-sensiscal year values (note: they may be meaningful)
    df=df.loc[df['YearMade']>1000]
    # df['saledate']=pd.to_datetime(df['saledate'])
    df['saledate_ym'] = pd.to_datetime(df['saledate']).year

    #create X and y
    y = df['SalePrice']
    X = df.drop(['SalePrice'], axis = 1)


    #impute values for Machine hours
    df_usage=X[['MachineHoursCurrentMeter','YearMade']]

    df_usage_vals = df_usage.dropna()

    mask_missing = df_usage.isnull().any(axis=1)
    df_usage_missing = df_usage[mask_missing]

    y_impute = df_usage_vals['MachineHoursCurrentMeter']
    x_impute = df_usage_vals.drop(['MachineHoursCurrentMeter'], axis = 1)

    #create a Random Forest to fill in machine hour nans
    rf_usage = RandomForestRegressor()
    rf_usage.fit(x_impute, y_impute)
    preds_usage  = rf_usage.predict(df_usage_missing.drop(['MachineHoursCurrentMeter'], axis = 1))

    #smoosh them back together
    df_usage_missing['MachineHoursCurrentMeter']=pd.Series(preds_usage, index=df_usage_missing.index)
    df_usage_total= pd.concat([df_usage_missing, df_usage_vals], axis=0)

    # add usages with datasources
    ds = df[[121,132,136,149,172, 'SalesID', 'ModelID', 'MachineID', 'saledate_ym', 'state', 'auctioneerID', 'Enclosure', 'Hydraulics', 'ProductGroup', 'UsageBand']]
    df1 = pd.concat([df_usage_total, ds], axis=1)
    return df1, y

def run_reg(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    model.predict(y_test)
    return model.get_params()


if __name__ == "__main__":

    #read in data
    df = pd.read_csv('train.csv')
    #clean data
    # df1, y = clean_data(df)

    #remove 2 var
    df1 = df1.drop(['Enclosure', 'Hydraulics', 'UsageBand', 'state', 'ProductGroup'], axis = 1)
    #split data
    X_train, X_test, y_train, y_test = train_test_split(df1,y, random_state=1)

    #run prelim regression
    print(run_reg(X_train, y_train, X_test, y_test))
