import numpy as np
from sklearn.linear_model import LinearRegression

def check_adjustment_validity(df, all_covs, Y, Z, A):
    """
    Return 1 if the adjustment set is valid, and 0 otherwise
    """
    def X_effect(X, Y, Z, data):

        # Create the predictor variables list
        Z.insert(0,X)
        # Run the logistic regression
        model = LinearRegression()
        model.fit(data[Z], data[Y])
        # Return the odds ratio
        return model.coef_[0]
    used = Z + [A] + [Y]
    for w in [x for x in all_covs if x not in used]:
        odds = X_effect(w, Y, Z, df)
        if  0.1 < odds or -0.1 > odds:
            odds2 = X_effect(w, Y, Z + [A], df)
            if -0.05 < odds2 and 0.05 > odds2:
                return 1
    return 0