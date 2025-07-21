import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#... import sklearn stuff...
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from tqdm import tqdm
from boruta import BorutaPy
from sklearn.neural_network import MLPClassifier

class Model():
    def __init__(self, type):
        self.type = type
    
    def fit(self, X, Y, penalty = None, ipw = False, target = None):
        if self.type == "rf":
            if ipw:
                model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
                model.fit(np.array(X), np.array(Y))
                return model
            else:
                if target == 'Y':
                    model = RandomForestRegressor(n_jobs=-1, n_estimators=100)
                else:
                    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
                feat_selector = BorutaPy(
                    verbose=0,
                    estimator=model,
                    n_estimators='auto'
                )
                feat_selector.fit(np.array(X), np.array(Y))
                self.best_features = []
                for i in range(len(feat_selector.support_)):
                    if feat_selector.support_[i]:
                        self.best_features.append(X.columns[i])
                return model
        if self.type == "nn":
            if ipw:
                model = MLPClassifier(random_state=42, hidden_layer_sizes=[100,]).fit(np.array(X), np.array(Y))
                return model
            else:
                if target == 'Y':
                    model = RandomForestRegressor(n_jobs=-1, n_estimators=100)
                else:
                    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
                feat_selector = BorutaPy(
                    verbose=0,
                    estimator=model,
                    n_estimators='auto'
                )
                feat_selector.fit(np.array(X), np.array(Y))
                self.best_features = []
                for i in range(len(feat_selector.support_)):
                    if feat_selector.support_[i]:
                        self.best_features.append(X.columns[i])
                return model
                        
        elif self.type == "log":
            if penalty == None:
                return LogisticRegression(penalty=None).fit(X, Y)
            else:
                if target == 'Y':
                    model = LinearRegression().fit(X,Y)
                else:
                    model = LogisticRegression(penalty=penalty, solver="liblinear").fit(X, Y)
                self.best_features = []
                for feature, coef in zip(X.columns, model.coef_[0] if target == 'A' else model.coef_):
                    if abs(coef) > 0.1:
                        self.best_features.append(feature)
                return model
    
    def ipw(self, data, A, Y, Z, trim = False):
        self.model = self.fit(data[Z], data[A], penalty = None, ipw = True)
        if self.type == "rf" or self.type == "nn":
            propensity = self.model.predict_proba(np.array(data[Z]))
        else:
            propensity = self.model.predict_proba(data[Z])

        props = propensity[:,1]
        
        IPW_1 = data[Y]*data[A]/(props)
        IPW_0 = (data[Y]*(1-data[A]))/(1 - props)

        if trim:
            trim_arr = []
            for i, val in enumerate(data.index):
                if props[i] <= 0.05 or props[i] >= 0.95:
                    trim_arr.append(val)

            IPW_0.drop(trim_arr, inplace = True)
            IPW_1.drop(trim_arr, inplace = True)

        # Compute the ACE
        ACE = (diff := IPW_1 - IPW_0).sum() / len(diff)

        return ACE
    
    def accuracy(self, data, A, Z):
        return self.model.score(data[Z], data[A])
    
    def conf_int(self, data, A, Y, Z, num_bootstraps=200, alpha=0.05):
        """
        Compute confidence intervals for IPW  via bootstrap.
        The input method_name can be used to decide how to compute the confidence intervals.

        Returns variance estimate.
        """
        estimates = []

        for i in range(num_bootstraps):
            # Resample the data with replacement
            resampled_data = data.sample(n = data.shape[0], replace = True, ignore_index = True)
            # Get the ace for the resampled data
            ACE= self.ipw(resampled_data, A, Y, Z, True)
            # Add the ace to the estimates
            estimates.append(ACE)
            pass
        # Return the variance
            # Get the lower quantile
        q_low = np.quantile(estimates, 0.025)
        # Get the upper quantile
        q_up = np.quantile(estimates, .975)
        # Return the quantile estimates
        return np.var(estimates), q_low, q_up
            