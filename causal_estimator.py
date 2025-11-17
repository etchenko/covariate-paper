def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, List
from feature_selector import *

#import sklearn stuff
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

@dataclass
class ModelBundle:
    treatment_selector_model: object            # Penalized model to select adjustment set based on treatment
    treatment_estimator_model: object           # Unpenalized model to compute propensity scores
    outcome_selector_model: object              # Penalized model to select adjustment set based on outcome
    outcome_estimator_model: object             # Unpenalized model to estimate outcome 
    treatment_feature_selector: FeatureSelector # Feature Selector to use for treatment model
    outcome_feature_selector: FeatureSelector  # Feature Selector to use for outcome model

def get_model_bundle(model_type: str) -> ModelBundle:
    """
    Given a model_type (e.g. 'linear', 'mlp', 'forest'), returns:
    - A ModelBundle with penalized/unpenalized models and feature selectors
    """
    model_type = model_type.lower()

    if model_type == "linear" or model_type == None:
        bundle = ModelBundle(
            treatment_selector_model=LogisticRegression(penalty="l1", solver="liblinear"),
            treatment_estimator_model=LogisticRegression(penalty=None, solver="lbfgs"),
            outcome_selector_model=Ridge(),
            outcome_estimator_model=LinearRegression(),
            treatment_feature_selector = CoefficientThresholdSelector(threshold=0.2),
            outcome_feature_selector = CoefficientThresholdSelector(threshold=0.2)
        )

    elif model_type == "nn":
        bundle = ModelBundle(
            treatment_selector_model=BorutaPy(
                    verbose=0,
                    estimator=RandomForestClassifier( n_jobs=-1),
                    random_state=42,
                ),
            treatment_estimator_model=MLPClassifier(hidden_layer_sizes=[100, 100, 100],learning_rate='adaptive',early_stopping=True,),
            outcome_selector_model=BorutaPy(
                    verbose=0,
                    estimator=RandomForestRegressor(n_jobs=-1),
                    random_state=42,
                ),
            outcome_estimator_model=MLPRegressor(hidden_layer_sizes=[100, 100, 100],learning_rate='adaptive',early_stopping=True,),
            treatment_feature_selector = SupportMaskSelector(),
            outcome_feature_selector = SupportMaskSelector()
        )

    elif model_type == "rf":
        bundle = ModelBundle(
            treatment_selector_model=BorutaPy(
                    verbose=0,
                    estimator=RandomForestClassifier(n_estimators=100),
                    n_estimators='auto'
                ),
            treatment_estimator_model=RandomForestClassifier(n_estimators=100),
            outcome_selector_model=BorutaPy(
                    verbose=0,
                    estimator=RandomForestRegressor(n_estimators=100),
                    n_estimators='auto'
                ),
            outcome_estimator_model=RandomForestRegressor(n_estimators=100),
            treatment_feature_selector = SupportMaskSelector(),
            outcome_feature_selector = SupportMaskSelector()
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return bundle


class CausalEstimator():
    def __init__(self, A, Y, Z, type = None):
        self.A = A
        self.Y = Y
        self.Z = Z
        self.type = type

    def split_data(self, data, train_size=0.4, val_size=0.4, test_size=0.2, random_state=None):
        train_data, val_data = train_test_split(data, train_size=train_size, random_state=random_state)
        val_data, test_data = train_test_split(val_data, train_size=val_size/(val_size +test_size), random_state=random_state)
        return train_data, val_data, test_data
    
    def find_best_features(self, data, model, selector: FeatureSelector, target):
        if target != 'treatment':
            X = data[self.Z + [self.A]]
        else:
            X = data[self.Z]
        
        y = data[self.A if target == 'treatment' else self.Y]
        features = selector.select_features(model, X, y, feature_names=self.Z)
        if target != "treatment" and (self.A in features):
            features.remove(self.A)
        return features

    
    def fit_model(self, data, model, features, target):
        """Fit a model to predict treatment or outcome"""
        X = data[features]
        y = data[target]
        model.fit(X, y)
        return model
    
    def run_estimation(self,data, model: ModelBundle | Literal["linear", "nn", "rf"] = "linear", criterion: Literal["treatment","outcome","union","intersection","different","covs"] = "treatment", method: Literal["ipw","aipw","dml", "backdoor"] = "aipw", save: bool = True, covs = None):
        self.method = method
        # Get the model bundle of not defined
        if type(model) != ModelBundle:
            models = get_model_bundle(model)
        else:
            models = model
        
        # Data splitting
        train_data, val_data, test_data = self.split_data(data)
        adjustment_set = []
        if criterion != 'covs':
            # Feature selection
            if criterion in ["treatment", "union","intersection","different"]:
                treatment_features = self.find_best_features(
                    data=train_data,
                    model=models.treatment_selector_model,
                    selector=models.treatment_feature_selector,
                    target='treatment'
                )
                self.treatment_features = treatment_features  
            if criterion in ["outcome", "union","intersection","different"]:
                outcome_features = self.find_best_features(
                    data=train_data,
                    model=models.outcome_selector_model,
                    selector=models.outcome_feature_selector,
                    target='outcome'
                )
                self.outcome_features = outcome_features
            # Find adjustment set based on criterion
            if criterion == "treatment":
                adjustment_set = treatment_features
                treatment_covs = adjustment_set
                outcome_covs = adjustment_set
            elif criterion == "outcome":
                adjustment_set = outcome_features
                treatment_covs = adjustment_set
                outcome_covs = adjustment_set
            elif criterion == "union":
                adjustment_set = list(set(outcome_features + treatment_features))
                treatment_covs = adjustment_set
                outcome_covs = adjustment_set
            elif criterion == "intersection":
                adjustment_set = list(set(outcome_features).intersection(set(treatment_features)))
                treatment_covs = adjustment_set
                outcome_covs = adjustment_set
            elif criterion == "different":
                treatment_covs = treatment_features
                outcome_covs = outcome_features
                adjustment_set = treatment_features + outcome_features
        else:
            adjustment_set = covs
            treatment_covs = covs
            outcome_covs = covs
            treatment_features = []
            outcome_features = []

        # Fit treatment and outcome models using adjustment set
        if method != "backdoor":
            treatment_model = self.fit_model(
                data=val_data,
                model=models.treatment_estimator_model,
                features= treatment_covs,
                target=self.A
            )
        outcome_covs = outcome_covs if method == 'dml' else outcome_covs + [self.A]
        outcome_model = self.fit_model(
            data=val_data,
            model=models.outcome_estimator_model,
            features=outcome_covs,
            target=self.Y
        )

        # Estimation
        if method == 'ipw':
            ace = self.estimate_ipw(test_data, treatment_model, treatment_covs)
        elif method == 'aipw':
            ace = self.estimate_aipw(test_data, treatment_model, outcome_model, treatment_covs, outcome_covs)
        elif method == 'dml':
            ace = self.estimate_dml(test_data, treatment_model, outcome_model, treatment_covs, outcome_covs)
        elif method == 'backdoor':
            ace = self.estimate_backdoor(test_data, outcome_model, treatment_covs, outcome_covs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if save:
            self.treatment_covs = treatment_covs
            if method != "backdoor":
                self.treatment_model = treatment_model
            self.outcome_model = outcome_model
            self.adjustment_set = adjustment_set
            self.train_data = train_data
            self.test_data = test_data
            self.val_data = val_data
            self.outcome_covs = outcome_covs
            self.treatment_selector_model = models.treatment_selector_model
            self.outcome_selector_model = models.outcome_selector_model
        return ace

    def calculate_accuracy(self):
        if self.method != "backdoor":
            acc = self.treatment_model.score(self.test_data[self.treatment_covs], self.test_data[self.A])
        else:
            acc = None
        rmse = root_mean_squared_error(self.test_data[self.Y], self.outcome_model.predict(self.test_data[self.outcome_covs]))
        return (acc, rmse)
    
    def estimate_ipw(self, data, treatment_model, adjustment_set):
        propensity = treatment_model.predict_proba(data[adjustment_set])

        props = propensity[:,1]
        
        IPW_1 = data[self.Y]*data[self.A]/(props)
        IPW_0 = (data[self.Y]*(1-data[self.A]))/(1 - props)

        # Compute the ACE
        return (diff := IPW_1 - IPW_0).sum() / len(diff)
    
    def estimate_aipw(self, data, treatment_model, outcome_model, adjustment_set, outcome_covs):
        propensity = treatment_model.predict_proba(data[adjustment_set])
        y_errs = data[self.Y] - outcome_model.predict(data[outcome_covs])
        props = propensity[:,1]

        covs_0, covs_1 = data[outcome_covs].copy(), data[outcome_covs].copy()
        covs_0[self.A] = 0
        covs_1[self.A] = 1
        
        IPW_1 = y_errs*data[self.A]/(props) + outcome_model.predict(covs_1)
        IPW_0 = (y_errs*(1-data[self.A]))/(1 - props) + outcome_model.predict(covs_0)

        # Compute the ACE
        return (diff := IPW_1 - IPW_0).sum() / len(diff)
    
    def estimate_dml(self, data, treatment_model, outcome_model, adjustment_set, outcome_covs):
        """
        Get average treatment effect via Double ML
        """
        t_res = pd.DataFrame(data[self.A] - treatment_model.predict_proba(data[adjustment_set])[:,1])
        out_res =  pd.DataFrame(data[self.Y] - outcome_model.predict(data[outcome_covs]))

        final_model = LinearRegression().fit(t_res, out_res)
        return final_model.coef_[0][0]
    
    def estimate_backdoor(self, data, outcome_model, adjustment_set, outcome_covs):
        """
        Get the backdoor adjustment estimate
        """
        covs_0, covs_1 = data[outcome_covs].copy(), data[outcome_covs].copy()
        covs_0[self.A] = 0
        covs_1[self.A] = 1

        # Apply the model to the data and compute the ACE
        result_0 = outcome_model.predict(covs_0)
        result_1 = outcome_model.predict(covs_1)

        # Calculate the difference between the two fragmented datasets
        difference = result_1 - result_0

        # Compute the ACE
        ACE = difference.sum() / len(difference)
        return ACE
