import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from causallearn.utils.cit import CIT
from causallearn.search.ConstraintBased.PC import pc
from sklearn.linear_model import LogisticRegression, LinearRegression

def check_adjustment_validity(df, Y, Z, A, w, type):
    """
    Return 1 if the adjustment set is valid, and 0 otherwise
    """
    '''
    #y_errs = data[Y] - LinearRegression().fit(data, data[Y]).predict(data)
    fisherz_obj = CIT(np.array(df), "fastkci")
    w = df.columns.get_loc(w)
    y = df.columns.get_loc("Y")
    cond = Z + [A]
    cond = [df.columns.get_loc(c) for c in cond]
    pValue = fisherz_obj(w, y, cond)
    return pValue
        
    sm.add_constant(data)
    sm.OLS(df[Y], data).fit'''
    if type == "linear":
        used = Z + [A] + [Y]
        ws = {}

        data = df[[w] + Z + [A]]
        data = sm.add_constant(data)
        results = sm.OLS(df[Y], data).fit()
        p_value = results.pvalues[1]

        return p_value
    else:
        kci_obj = CIT(np.array(df), "fastkci")
        w = df.columns.get_loc(w)
        y = df.columns.get_loc("Y")
        cond = Z + [A]
        cond = [df.columns.get_loc(c) for c in cond]
        pValue = kci_obj(w, y, cond)
        return pValue





    def X_effect(X, Y, Z, data):
        
        # Create the predictor variables list
        Z.insert(0,X)
        # Run the logistic regression
        model = LinearRegression()
        model.fit(data[Z], data[Y])
        # Return the odds ratio
        return model.coef_[0]
    
    for w in [x for x in all_covs if x not in used]:
        odds = X_effect(w, Y, Z, df)
        if  0.1 < odds or -0.1 > odds:
            odds2 = X_effect(w, Y, Z + [A], df)
            if -0.1 < odds2 and 0.1 > odds2:
                return 1
    return 0