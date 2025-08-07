import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#... import sklearn stuff...
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
from tqdm import tqdm
from boruta import BorutaPy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split


class Model():
    def __init__(self, data, A, Y, Z, model_type, covs = None):
        self.type = model_type
        self.data = data
        self.A = A
        self.Y = Y
        self.Z = Z
        self.covs = covs

        df1, df2 = train_test_split(data, train_size=.33)
        df2, df3 = train_test_split(df2, train_size=.5)
    
    def fit(self, X, Y, penalty = None, ipw = False, target = None):
        if self.type == "rf":
            if ipw:
                model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
                model.fit(np.array(X), np.array(Y))
                return model
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
                if np.array(Y).size > 1000:
                    model = MLPClassifier(random_state=42, hidden_layer_sizes=[100],alpha=0).fit(np.array(X), np.array(Y))
                else:
                    model = MLPClassifier(random_state=42, hidden_layer_sizes=[100]).fit(np.array(X), np.array(Y))
                return model
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
                model = LogisticRegression(penalty=penalty, solver="liblinear").fit(X, Y)
                self.best_features = []
                for feature, coef in zip(X.columns, model.coef_[0] if target == 'A' else model.coef_):
                    if abs(coef) > 0.1:
                        self.best_features.append(feature)
                return model
        elif self.type == "linear":
            if penalty == None:
                return LinearRegression().fit(X, Y)
            else:
                model = Ridge().fit(X, Y)
                self.best_features = []
                for feature, coef in zip(X.columns, model.coef_[0] if target == 'A' else model.coef_):
                    if abs(coef) > 0.1:
                        self.best_features.append(feature)
                return model
        elif self.type == "nn_linear":
            if penalty == None:
                self.model = MLPRegressor(random_state=42, hidden_layer_sizes=[100]).fit(np.array(X), np.array(Y))
                return self.model
            else:
                model = RandomForestRegressor(n_jobs=-1, n_estimators=100)
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
    
    def aipw(self, data, A, Y, Z):
        self.model = self.fit(data[Z], data[A], penalty = None, ipw = True)
        if self.type == "rf" or self.type == "nn":
            propensity = self.model.predict_proba(np.array(data[Z]))
        else:
            propensity = self.model.predict_proba(data[Z])
        covs = data[Z + [A]]
        if self.type == 'log':
            y_model = LinearRegression().fit(covs, data[Y])
        else:
            y_model = MLPRegressor(random_state=42, hidden_layer_sizes=[100]).fit(np.array(covs), np.array(data[Y]))
        
        y_errs = data[Y] - y_model.predict(np.array(covs) if self.type == 'nn' else covs)

        props = propensity[:,1]

        covs_0, covs_1 = covs.copy(), covs.copy()
        covs_0[A] = 0
        covs_1[A] = 1
        
        IPW_1 = y_errs*data[A]/(props) + y_model.predict(np.array(covs_1) if self.type == 'nn' else covs_1)
        IPW_0 = (y_errs*(1-data[A]))/(1 - props) + y_model.predict(np.array(covs_0) if self.type == 'nn' else covs_0)

        # Compute the ACE
        ACE = (diff := IPW_1 - IPW_0).sum() / len(diff)

        return ACE
    
    def doubleml(self, data, A, Y, Z, type):
        """
        Get average treatment effect via Double ML
        """
        if type == 'linear':
            debias_m = LogisticRegression(penalty=None)
            denoise_m = LinearRegression()
        elif type == 'nn':
            debias_m = MLPClassifier(random_state=42, hidden_layer_sizes=[100])
            denoise_m = MLPRegressor(random_state=42, hidden_layer_sizes=[100])
        print(cross_val_predict(debias_m, np.array(data[Z]),np.array(data[A]), cv=10))
        t_res = pd.DataFrame(data[A] - cross_val_predict(debias_m, np.array(data[Z]), np.array(data[A]),method='predict_proba', cv=10)[:,1])
        
        out_res =  pd.DataFrame(data[Y] -cross_val_predict(denoise_m, np.array(data[Z]), np.array(data[Y]), cv=10))

        final_model = LinearRegression().fit(t_res, out_res)
        return final_model.coef_[0][0]

    
    def accuracy(self, data, A, Z, metric):
        if metric == 'RMSE':
            return root_mean_squared_error(data[A], self.model.predict(data[Z]))
        return self.model.score(data[Z], data[A])
    
    def conf_int(self, data, A, Y, Z, num_bootstraps=200, trim = False, model = "ipw"):
        """
        Compute confidence intervals for IPW or backdoor via bootstrap.
        The input method_name can be used to decide how to compute the confidence intervals.

        Returns variance estimate.
        """
        estimates = []

        for i in range(num_bootstraps):
            # Resample the data with replacement
            resampled_data = data.sample(n = data.shape[0], replace = True, ignore_index = True)
            # Get the ace for the resampled data
            if model == "ipw":
                ACE= self.ipw(resampled_data, A, Y, Z, trim)
            elif model == "ACE":
                ACE =self.backdoor_adjustment(resampled_data, A, Y, Z)
            elif model == "aipw":
                ACE = self.aipw(resampled_data, A, Y, Z)
            elif model == "dml":
                ACE = self.doubleml(resampled_data, A, Y, Z, "linear")
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
    
    def backdoor_adjustment(self, data, A,  Y, Z):
        Z.insert(0, A)
        # Run the linear regression
        reg = self.fit(data[Z], data[Y])
        self.model = reg

        # Create fragmented datasets
        df_0, df_1 = data.copy(), data.copy()
        df_0[A] = 0 
        df_1[A] = 1

        # Apply the model to the data and compute the ACE
        result_0 = reg.predict(df_0[Z])
        result_1 = reg.predict(df_1[Z])

        # Calculate the difference between the two fragmented datasets
        difference = result_1 - result_0

        # Compute the ACE
        ACE = difference.sum() / len(difference)
        return ACE
                