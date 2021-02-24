import numpy as np
import pandas as pd
import theano
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pymc3 as pm
import pickle

seed = 42
np.random.seed(seed)

globalModel = None
globalScaler = None
globalEncoder = None
f_pred = None
X_New_shared = None
Y_New_shared = None

class BayesianPolyRegression:
    def fit(self, X, Y):
        with pm.Model() as self.model:
            global X_New_shared
            global Y_New_shared
            X_New_shared = theano.shared(X)
            Y_New_shared = theano.shared(Y)
            lm = pm.Gamma("l", alpha=2, beta=1)
            offset = 0.1
            nu = pm.HalfCauchy("nu", beta=1)
            d = 2

            cov = nu ** 2 * pm.gp.cov.Polynomial(X_New_shared.get_value().shape[1], lm, d, offset)

            self.gp = pm.gp.Marginal(cov_func=cov)

            sigma = pm.HalfCauchy("sigma", beta=1)
            y_ = self.gp.marginal_likelihood("y", X=X_New_shared, y=Y_New_shared, noise=sigma)

            self.map_trace = [pm.find_MAP()]

            global f_pred
            f_pred = self.gp.conditional("f_pred", X_New_shared, shape=X_New_shared.get_value().shape[0])

    def save_model(self):
        with open('save/saved_model.p', 'wb') as buff:
            global X_New_shared
            global f_pred
            global globalScaler
            global globalEncoder
            pickle.dump({'model': self.model, 'trace': self.map_trace, 'X_New_shared': X_New_shared, 'f_pred': f_pred, 'scaler': globalScaler, 'encoder': globalEncoder, 'gp': self.gp}, buff)


def read_training_data():
    predict_label = 9 #9 for TPS

    # Read Data
    dataset = pd.read_csv('dataset/dataset.csv')

    # Ignore Errors
    dataset = dataset.loc[dataset["Error %"] < 5]

    # Define X and Y columns
    X = dataset.iloc[:, [0, 2, 3]].values
    Y = dataset.iloc[:, predict_label].values  # 10 for Latency, 9 for TPS

    # Encode 'Scenario Name'
    le_X_0 = LabelEncoder()
    X[:, 0] = le_X_0.fit_transform(X[:, 0])
    global globalEncoder
    globalEncoder = le_X_0

    # Create Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    global globalScaler
    globalScaler = scaler

    # Apply Scaler on X
    scaler.fit(X)
    X = scaler.transform(X)

    # Convert Y to 1D Array - Not necessary
    Y = Y.flatten()

    # Shuffle Data
    X, Y = shuffle(X, Y, random_state=42)

    return X, Y

def train_poly_regressor():
    X, Y = read_training_data()
    global globalModel
    globalModel = BayesianPolyRegression()
    globalModel.fit(X, Y)


def pickel_model():
    global globalModel
    globalModel.save_model()


train_poly_regressor()
pickel_model()
