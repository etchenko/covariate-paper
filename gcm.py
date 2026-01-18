import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

"""
This is an implementation for Generalized Covariance Measure in python.
Lasso Version Implemented by Xueda Shen. shenxueda@berkeley.edu
"""


def GCM_translation(X, Y, Z, alpha = 0.05, nsim = 1000, res_X = None, res_Y = None, model = "linear", tolerance_threshold = None):
    """
    This is a direct translation from GCM Testing in R. 
    
    Inputs:
        X, Y, Z: iid data for testing conditional independence.
        alpha: Pre-specified level of cutoff.
        nsim: Number of monte-carlo simulations.
        res_X, res_Y: corresponding to resid.XonZ, resid.YonZ respectively in original implementation. 
        tolerance_threshold: When set, used to compute the inverted test as proposed in Malinsky (2024)
    """

    # Checking if providing residual or computing residual
    if Z is None:
        # Is residual provided
        if res_X is None:
            res_X = X
        if res_Y is None:
            res_Y = Y
    else:
        if res_X is None:
            if X is None:
                raise ValueError('Either X on residual of X given Z has to be provided.')
            if model == "linear":
                if len(X.shape) == 1 or X.shape[1] == 1:
                    if len(Z.shape) == 1:
                        Z = np.expand_dims(Z, axis = 1)
                    # Compute residuals of X|Z. I chose Lasso as estimator. 
                    model_X = LassoCV(cv = 5).fit(Z, X)
                    res_X = X - model_X.predict(Z)
                else:
                    model_X = MultiTaskLassoCV(cv = 5).fit(Z, X)
                    res_X = X - model_X.predict(Z)
            else:
                model_X = MLPRegressor(hidden_layer_sizes=[100,100],learning_rate='adaptive',early_stopping=True,validation_fraction=0.2,
                alpha=0.01).fit(Z, X)
                res_X = X - model_X.predict(Z)
    if Y is None:
        raise ValueError ('Either Y or residual of Y given Z has to be provided.')
    if res_Y is None:
        if model == "linear":
            if len(Y.shape) == 1 or Y.shape[1] == 1:
                model_Y = LassoCV(cv = 5).fit(Z, Y)
                res_Y = Y - model_Y.predict(Z)
            else:
                model_Y = MultiTaskLassoCV(cv = 5).fit(Z, Y)
                res_Y = Y - model_Y.predict(Z)
        else:
            model_Y = MLPRegressor(hidden_layer_sizes=[100,100],learning_rate='adaptive',early_stopping=True,validation_fraction=0.2,
                alpha=0.01).fit(Z, Y)
            res_Y = Y - model_Y.predict(Z)

    if (len(res_X.shape) > 1 or len(res_Y.shape) > 1):
        # Obtaining covariance and test statistics
        d_X = res_X.shape[1]; d_Y = res_Y.shape[1]; nn = res_X.shape[0]

        # rep(times) in R is really np.tile. 
        # Translating R_mat = rep(....) * as.numeric(...)[, rep(....)]
        left = np.tile(res_X, reps = d_Y)  # rep(resid.XonZ, times = d_Y)
        left = left.flatten('F')
        right = res_Y[:, np.tile(np.arange(d_Y), reps = d_X)].flatten(order = 'F')   # as.numeric(as.matrix(resid.YonZ)[, rep(seq_len(d_Y), each=d_X)])
        R_mat = np.multiply(left, right)
        R_mat = R_mat.flatten(order = 'F')
        R_mat = np.reshape(R_mat, (nn, d_X * d_Y), order = 'F')
        R_mat = np.transpose(R_mat)

        norm_con = np.sqrt(np.mean(R_mat ** 2, axis = 1) - np.mean(R_mat, axis = 1)**2)
        norm_con = np.expand_dims(norm_con, axis = 1)
        if tolerance_threshold is  None:
            R_mat = R_mat / norm_con
        else:
            R_mat = (R_mat - tolerance_threshold)/  norm_con

        # Test statistics
        test_stat = np.max(np.abs(np.mean(R_mat, axis = 1))) * np.sqrt(nn)
        noise = np.random.randn(nn, nsim)
        test_stat_sim = np.abs(R_mat @ noise)
        test_stat_sim = np.amax(test_stat_sim, axis = 0) / np.sqrt(nn)

        # p value
        pval = (np.sum(test_stat_sim >= test_stat) + 1) / (nsim + 1)


    else:
        if (len(res_X.shape) == 1):  # ifesle(is.null()...)
            nn = res_X.shape[0]
        else:
            nn = res_X.shape[0]
        R = np.multiply(res_X, res_Y)
        R_sq = R ** 2
        meanR = np.mean(R)
        if tolerance_threshold is None:
            test_stat = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR ** 2)


            pval = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))
        else:
            test_stat1 = np.sqrt(nn) * (meanR - tolerance_threshold) / np.sqrt(np.mean(R_sq) - meanR ** 2)
            test_stat2 = np.sqrt(nn) * (meanR + tolerance_threshold) / np.sqrt(np.mean(R_sq) - meanR ** 2)
            pval = stats.norm.cdf(test_stat1)


    return pval