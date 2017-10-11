from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import featuretools as ft
from featuretools.primitives import (Day, Hour, Minute, Month, Weekday, Week, Weekend, Mean, Max, Min, Std, Skew)


def read_data(TRAIN_DIR, TEST_DIR, nrows=None):
    data_train = pd.read_csv(TRAIN_DIR,
                             parse_dates=["pickup_datetime",
                                          "dropoff_datetime"],
                             nrows=nrows)
    data_test = pd.read_csv(TEST_DIR,
                            parse_dates=["pickup_datetime"],
                            nrows=nrows)
    data_train = data_train.drop(['dropoff_datetime'], axis=1)
    data_train.loc[:, 'store_and_fwd_flag'] = data_train['store_and_fwd_flag'].map({'Y': True,
                                                                                    'N': False})
    data_test.loc[:, 'store_and_fwd_flag'] = data_test['store_and_fwd_flag'].map({'Y': True,
                                                                                  'N': False})
    data_train = data_train[data_train.trip_duration < data_train.trip_duration.quantile(0.99)]

    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]
    data_train = data_train[(data_train.pickup_longitude> xlim[0]) & (data_train.pickup_longitude < xlim[1])]
    data_train = data_train[(data_train.dropoff_longitude> xlim[0]) & (data_train.dropoff_longitude < xlim[1])]
    data_train = data_train[(data_train.pickup_latitude> ylim[0]) & (data_train.pickup_latitude < ylim[1])]
    data_train = data_train[(data_train.dropoff_latitude> ylim[0]) & (data_train.dropoff_latitude < ylim[1])]

    return (data_train, data_test)


def train_xgb(X_train, labels):
    Xtr, Xv, ytr, yv = train_test_split(X_train.values,
                                        labels,
                                        test_size=0.2,
                                        random_state=0)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    params = {
        'min_child_weight': 1, 'eta': 0.166,
        'colsample_bytree': 0.4, 'max_depth': 9,
        'subsample': 1.0, 'lambda': 57.93,
        'booster': 'gbtree', 'gamma': 0.5,
        'silent': 1, 'eval_metric': 'rmse',
        'objective': 'reg:linear',
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=227,
                      evals=evals, early_stopping_rounds=50, maximize=False,
                      verbose_eval=10)

    print('Modeling RMSE %.5f' % model.best_score)
    return model


def predict_xgb(model, X_test):
    dtest = xgb.DMatrix(X_test.values)
    ytest = model.predict(dtest)
    X_test['trip_duration'] = np.exp(ytest)-1
    return X_test[['trip_duration']]


def feature_importances(model, feature_names):
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(feature_names))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    return feature_importance[['feature_name', 'importance']].sort_values(by='importance',
                                                                          ascending=False)


def get_train_test_fm(feature_matrix, percentage):
    nrows = feature_matrix.shape[0]
    head = int(nrows * percentage)
    tail = nrows-head
    X_train = feature_matrix.head(head)
    y_train = X_train['trip_duration']
    X_train = X_train.drop(['trip_duration'], axis=1)
    X_test = feature_matrix.tail(tail)
    y_test = X_test['trip_duration']
    X_test = X_test.drop(['trip_duration'], axis=1)

    return (X_train, y_train, X_test,y_test)


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict(orient="list")

        vs = dcols.values()
        ks = dcols.keys()
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]:
                    dups.append(ks[i])
                    break
    return dups

def find_training_examples(item_purchases, invoices, prediction_window, training_window, lead,threshold):
    niter = 2 #hard coded number of cutoffs we will search starting with
    cutoff_time = pd.Timestamp("2011-05-01") # hard coded start date
    label_times=pd.DataFrame()
    for k in range(1,niter):
        cutoff_time = cutoff_time + pd.Timedelta("45d")
        lt = make_label_times(item_purchases, invoices, cutoff_time, prediction_window,
                                       training_window, lead,threshold)
        label_times=label_times.append(lt)

    label_times=label_times.sort_values('cutoff_time')
    return label_times

def make_label_times(item_purchases, invoices, cutoff_time, prediction_window, training_window, lead,threshold):
    data = item_purchases.merge(invoices)[["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]]
    data["amount"] = data["Quantity"] * data["UnitPrice"]

    prediction_window_start = cutoff_time
    prediction_window_end = cutoff_time + prediction_window
    cutoff_time = cutoff_time - lead
    t_start = cutoff_time - training_window

    training_data = data[(data["InvoiceDate"] <= cutoff_time) & (data["InvoiceDate"] > t_start)]
    prediction_data = data[(data["InvoiceDate"] > prediction_window_start) & (data["InvoiceDate"] < prediction_window_end)]


    # get customers in training data
    label_times = pd.DataFrame()
    label_times["CustomerID"] = training_data["CustomerID"].dropna().unique()
    label_times["t_start"] = t_start
    label_times["cutoff_time"] = cutoff_time





    labels = prediction_data.groupby("CustomerID")[["amount"]].count()


    label_times = label_times.merge(labels, how="left", left_on="CustomerID", right_index=True)

    # if the amount is nan that means the customer made no purchases in prediction window
    label_times["amount"] = label_times["amount"].fillna(0)
    label_times.rename(columns={"amount": "purchases>threshold"}, inplace=True)
    label_times['purchases>threshold']=label_times['purchases>threshold']>threshold


    return label_times

def load_nyc_taxi_data():
    trips = pd.read_csv('nyc-taxi-data/trips.csv',
                        parse_dates=["pickup_datetime","dropoff_datetime"],
                        dtype={'vendor_id':"category",'passenger_count':'int64'},
                        encoding='utf-8')
    passenger_cnt = pd.read_csv('nyc-taxi-data/passenger_cnt.csv',
                                parse_dates=["first_trips_time"],
                                dtype={'passenger_count':'int64'},
                                encoding='utf-8')
    vendors = pd.read_csv('nyc-taxi-data/vendors.csv',
                          parse_dates=["first_trips_time"],
                          dtype={'vendor_id':"category"},
                          encoding='utf-8')
    #trips.drop("id.1", axis=1, inplace=True)
    #trips['pickup_datetime'] = pd.to_datetime(trips['pickup_datetime'] , format="%Y-%m-%d %H:%M:%S")
    #vendors['first_trips_time'] = pd.to_datetime(vendors['first_trips_time'] , format="%Y-%m-%d %H:%M:%S")
    return trips, passenger_cnt, vendors

def load_uk_retail_data():
    item_purchases = pd.read_csv('uk-retail-data/item_purchases.csv')
    invoices = pd.read_csv('uk-retail-data/invoices.csv')
    items = pd.read_csv('uk-retail-data/items.csv')
    customers = pd.read_csv('uk-retail-data/customers.csv')
    invoices['first_item_purchases_time'] = pd.to_datetime(invoices['first_item_purchases_time'] , format="%m/%d/%y %H:%M")
    item_purchases['InvoiceDate'] = pd.to_datetime(item_purchases['InvoiceDate'] , format="%m/%d/%y %H:%M")
    customers['first_invoices_time'] = pd.to_datetime(customers['first_invoices_time'] , format="%m/%d/%y %H:%M")
    items['first_item_purchases_time'] = pd.to_datetime(items['first_item_purchases_time'], format="%m/%d/%y %H:%M")
    return item_purchases, invoices, items,customers

def compute_features(features,cutoff_time):
    feature_matrix = ft.calculate_feature_matrix(features,
                                                 cutoff_time=cutoff_time,
                                                 approximate='36d')
    return feature_matrix

def engineer_features_uk_retail(entities,relationships,label_times,training_window):
    trans_primitives = [Minute, Hour, Day, Week, Month, Weekday, Weekend]

    feature_matrix,features = ft.dfs(entities=entities,
                                     relationships=relationships,
                                     target_entity="customers",
                                     trans_primitives=trans_primitives,
                                     agg_primitives=[Mean,Max,Std],
                                     cutoff_time=label_times,
                                     training_window=training_window)
    feature_matrix.drop("Country", axis=1, inplace=True)
    feature_matrix=feature_matrix.sort_index()
    return feature_matrix