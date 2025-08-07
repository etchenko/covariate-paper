from sklearn.inspection import permutation_importance

class FeatureSelector:
    def select_features(self, model, X, y, feature_names):
        """
        Returns a list of selected feature names.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
class CoefficientThresholdSelector(FeatureSelector):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def select_features(self, model, X, y, feature_names):
        model.fit(X, y)
        coefs = model.coef_.flatten()
        selected = [name for coef, name in zip(coefs, feature_names) if abs(coef) > self.threshold]
        return selected

class SupportMaskSelector(FeatureSelector):
    def select_features(self, model, X, y, feature_names):
        model.fit(X, y)
        support = model.support_
        selected = [name for mask, name in zip(support, feature_names) if mask]
        return selected


class PermutationImportanceSelector(FeatureSelector):
    def __init__(self, threshold=0.01, scoring='accuracy'):
        self.threshold = threshold
        self.scoring = scoring

    def select_features(self, model, X, y, feature_names):
        model.fit(X, y)
        result = permutation_importance(model, X, y, scoring=self.scoring)
        importances = result.importances_mean
        selected = [name for imp, name in zip(importances, feature_names) if imp > self.threshold]
        return selected
    
