
import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    # Parse date to also be of dtype datetime
    holidays = pd.read_csv(filepath)

    # Make sure that DateOfDeparture is of dtype datetime
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    holidays.loc[:, 'Debut'] = pd.to_datetime(holidays.loc[:, 'Debut'])
    holidays.loc[:, 'Fin'] = pd.to_datetime(holidays.loc[:, 'Fin'])

    X = X.copy()  # modify a copy of X

    name_column_0 = ""
    name_column_1 = ""
    name_column =""
    for i in range (0,13) : 
        name_column_0 = "H"+str(i)+"_"+str(i)
        name_column_1 = "H"+str(i)+"_"+str(i+1)
        name_column = "H" + str(i)
        X[name_column_0]=  X["DateOfDeparture"] <= holidays.loc[i,'Fin'] 
        X[name_column_1]=  X["DateOfDeparture"] >= holidays.loc[i,'Debut'] 
        X[name_column] = X[name_column_0] == X[name_column_1]
        X.drop([name_column_1,name_column_0],1,inplace=True)
    
    X["isHolidays"] = 1 
    X["isHolidays"][(X["H0"]== False) & (X["H1"]== False) & (X["H2"]== False) &
                       (X["H3"]== False) & (X["H4"]== False) & (X["H5"]== False) & 
                       (X["H6"]== False) & (X["H7"]== False) & (X["H8"]== False) & 
                       (X["H9"]== False) & (X["H10"]== False) & (X["H11"]== False) &
                       (X["H12"]== False)] = 0 

    X.drop(["H0","H1","H2","H3","H4","H5","H6","H7","H8","H9","H10","H11","H12"],1,inplace=True)

    return X


def _encode_dates(X):
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def get_estimator():
    data_merger = FunctionTransformer(_merge_external_data)

    date_encoder = FunctionTransformer(_encode_dates)
    
    date_cols = ["DateOfDeparture"]

    categorical_encoder = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )
    categorical_cols = ['Arrival', 'Departure','isHolidays']

    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )

    regressor = RandomForestRegressor(
        n_estimators=49, max_depth=14, max_features=4, min_samples_split = 2 , n_jobs=4
    )

    return make_pipeline(data_merger, preprocessor, regressor)
