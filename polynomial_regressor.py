import math
import numpy as np
import pandas as pd
import csv

import theano
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import pymc3 as pm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from metric_functions import root_mean_squared_percentage_error, mean_absolute_percentage_error
import uuid
from pymc3 import summary

import pickle

seed = 42
np.random.seed(seed)
globalModel = None;
globalScaler = None;
globalEncoder = None;
f_pred = None;
X_New_shared = None;
globalTrace = None;
globalGp = None;

def load_model():
    with open('save/saved_model.p', 'rb') as buff:
        data = pickle.load(buff)
        global globalModel
        global globalTrace
        global X_New_shared
        global f_pred
        global globalScaler
        global globalEncoder
        global globalGp
        globalModel = data['model']
        globalTrace = data['trace']
        X_New_shared = data['X_New_shared']
        f_pred = data['f_pred']
        globalScaler = data['scaler']
        globalEncoder = data['encoder']
        globalGp = data['gp']


def predict(sample_count=2000):
    with globalModel:
        global f_pred;
        global globalTrace
        pred_samples = pm.sample_posterior_predictive(globalTrace, vars=[f_pred], samples=sample_count, random_seed=42)
        y_pred, uncer = pred_samples["f_pred"].mean(axis=0), pred_samples["f_pred"].std(axis=0)
        print(y_pred)

def predict_gp():
    with globalModel:
        global X_New_shared
        mu, var = globalGp.predict(Xnew=X_New_shared, point=globalTrace[0], diag=True)
        print(mu)

        file_name = "results/" + "test1"+".csv"
        with open(file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(mu))


def predict_list(method = None, sample_count=2000):
    #dataset = pd.read_csv('dataset/custom.csv');
    dataset = pd.read_csv('dataset/MessageSizeVariationDataset.csv');
    X_Custom = dataset.iloc[:, [0, 1, 2]].values;
    X_Custom[:, 0] = globalEncoder.transform(X_Custom[:, 0]);
    X_Custom = globalScaler.transform(X_Custom);

    global X_New_shared
    X_New_shared.set_value(X_Custom)

    if method == "NS":
        predict_gp()
    else:
        predict(sample_count=sample_count)

def predict_point(data, method = None, sample_count=2000):
    global globalEncoder
    global globalScaler
    # X_Vals = ['Transformation', 1000, 102400]
    X_Vals = data;

    X_Vals[0] = globalEncoder.transform([X_Vals[0]])[0]
    X_Vals = globalScaler.transform([X_Vals])

    global X_New_shared
    X_New_shared.set_value(X_Vals)

    if method == "NS":
        predict_gp()
    else:
        predict(sample_count=sample_count)