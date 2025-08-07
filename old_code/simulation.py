import numpy as np
from tqdm import tqdm
from old_code.model import Model
from graphs import generateDag
from sklearn.model_selection import train_test_split
from causal_estimator import *

def run_simulation(df, Z, A, Y, model_type, selected_features, verbose, target, trim = False, est_type = True, metric = 'accuracy'):

    # Split data into 3 sets of .4, .4, .2
    df1, df2 = train_test_split(df, train_size=.4)
    df2, df3 = train_test_split(df2, train_size=.67) 

    # Find Adjustment set
    model = Model(model_type)
    model.fit(df1[Z], df1[target], "l1", False, target)
    best_features = model.best_features
    if est_type == "ipw":
        # Compute ACE, accuracy, and variance using found adjustment set
        ACE = model.ipw(df2, A, Y, best_features, trim)
        acc = model.accuracy(df3, A, best_features, metric)
        var, q_low1, q_up1 = model.conf_int(df2, A, Y, best_features, 100, trim, "ipw")

        # Compute ACE, accuracy, and varance using inputted adjustment set
        ACE2= model.ipw(df2, A, Y, selected_features, trim)
        acc2 = model.accuracy(df3, A, selected_features, metric)
        var2, q_low2, q_up2 = model.conf_int(df2, A, Y, selected_features, 100, trim, "ipw")
    elif est_type == "aipw":
            # Compute ACE, accuracy, and variance using found adjustment set
        ACE = model.aipw(df2, A, Y, best_features)
        acc = model.accuracy(df3, A, best_features, metric)
        var, q_low1, q_up1 = model.conf_int(df2, A, Y, best_features, 100, trim, "aipw")

        # Compute ACE, accuracy, and varance using inputted adjustment set
        ACE2= model.aipw(df2, A, Y, selected_features)
        acc2 = model.accuracy(df3, A, selected_features, metric)
        var2, q_low2, q_up2 = model.conf_int(df2, A, Y, selected_features, 100, trim, "aipw")
    elif est_type == 'backdoor' :
        ACE = model.backdoor_adjustment(df2, A, Y, best_features)
        acc = model.accuracy(df3, Y, best_features, metric)
        var, q_low1, q_up1 = model.conf_int(df2, A, Y, best_features, 100, trim, "ACE")
        ACE2 = model.backdoor_adjustment(df2, A, Y, [])
        acc2 = model.accuracy(df3, Y, [A], metric)
        var2, q_low2, q_up2 = model.conf_int(df2, A, Y, [], 100, trim, "ACE")
    elif est_type == 'dml':
        ACE = model.doubleml(df2, A, Y, best_features, model_type)
        acc = model.accuracy(df3, Y, best_features, metric)
        var, q_low1, q_up1 = model.conf_int(df2, A, Y, best_features, 100, trim, "ACE")
        ACE2 = model.backdoor_adjustment(df2, A, Y, [])
        acc2 = model.accuracy(df3, Y, [A], metric)
        var2, q_low2, q_up2 = model.conf_int(df2, A, Y, [], 100, trim, "ACE")

    if verbose:
        print(f'ACE: {round(ACE, 2)}, {metric}: {round(acc, 2)}, Variance: {round(var, 2)}, 95% Confidence Interval: {round(q_low1, 2)} - {round(q_up1, 2)} ')
        print(f'ACE: {round(ACE2, 2)}, {metric}: {round(acc2, 2)}, Variance: {round(var2, 2)}, 95% Confidence Interval: {round(q_low2, 2)} - {round(q_up2, 2)} ')

    return ACE, acc, var, q_low1, q_up1, ACE2, acc2, var2, q_low2, q_up2
    
def run_multiple_sims(type, graph, times = 1, ipw = "ipw", nums = None, metric = 'Accuracy'):
    # Collect results with a progress bar
    results = []
    for _ in tqdm(range(times), desc=f"Running Simulations for {type}"):
        if not nums:
            nums = 1000 if graph == 1 else 2000 if graph == 3 else 7500
        df, za, zy, z = generateDag(nums, graph, 100)
        results.append(run_simulation(df, z, 'A', 'Y', type, zy, verbose = False, target = 'Y' if graph == 3 else 'A', trim = True if (graph == 2 and type == "log") else False, est_type = ipw, metric = metric))

    # Convert to array and compute column-wise averages
    results_array = np.array(results)
    averages = results_array.mean(axis=0)

    print(f"Model 1: Ace - {round(averages[0], 2)}, {metric} - {round(averages[1], 2)}, Variance - {round(averages[2], 2)}, 95% Confidence Interval: {round(averages[3], 2)} - {round(averages[4], 2)}")
    print(f"Model 2: Ace - {round(averages[5], 2)}, {metric} - {round(averages[6], 2)}, Variance - {round(averages[7], 2)}, 95% Confidence Interval: {round(averages[8], 2)} - {round(averages[9], 2)}")



if __name__ == "__main__":
    np.random.seed(42)
    '''
    print(" --- Simple Graph --- ")
    print(" Log Classifier:")
    run_multiple_sims("log", 1, 20)
    
    print(" Neural Network:")
    run_multiple_sims("nn", 1, 10)
    
    
    print(" --- High Dim Graph --- ")
    print(" Log Classifier:")
    run_multiple_sims("log", 2, 5)
    

    print(" Neural Network:")
    run_multiple_sims("nn", 2, 5, "ipw", 7500)
    
    print(" --- Outcome Graph --- ")
    print(" Linear Model:")
    run_multiple_sims("linear", 3, 10, False, None, "RMSE")

    print(" Neural Network:")
    run_multiple_sims("nn_linear", 3, 10, False, None, 'RMSE')
    '''
    #run_multiple_sims("nn", 2, 5, "aipw", 5000)
    
    results = []
    estimates1 = {"ace":[], "var":[], "ci":[],"treat_acc":[],"out_rmse":[]}
    estimates2 = {"ace":[], "var":[], "ci":[],"treat_acc":[],"out_rmse":[]}
    for _ in tqdm(range(5), desc=f"Running Simulations for DML"):
        df, za, zy, z = generateDag(1250, 1)
        causal_estimator = CausalEstimator('A','Y', z)
        output_treat = causal_estimator.run_estimation_with_ci(df, 'nn','treatment','aipw')
        output_out = causal_estimator.run_estimation_with_ci(df, 'nn', 'outcome','aipw')
        for key in estimates1.keys():
            estimates1[key].append(output_treat[key])
            estimates2[key].append(output_out[key])
    for key in estimates1.keys():
        estimates1[key] = np.array(estimates1[key]).mean(axis=0)
        estimates2[key] = np.array(estimates2[key]).mean(axis=0)

    print(f"Treatment Model: Ace - {round(estimates1["ace"], 2)}, Variance - {round(estimates1["var"], 2)}, Treatment Accuracy - {round(estimates1["treat_acc"],2)}, Outcome RMSE - {round(estimates1["out_rmse"],2)}, 95% Confidence Interval: {round(estimates1["ci"][0], 2)} - {round(estimates1["ci"][1], 2)}")
    print(f"Outcome Model: Ace - {round(estimates2["ace"], 2)}, Variance - {round(estimates2["var"], 2)}, Treatment Accuracy - {round(estimates2["treat_acc"],2)}, Outcome RMSE - {round(estimates2["out_rmse"],2)}, 95% Confidence Interval: {round(estimates2["ci"][0], 2)} - {round(estimates2["ci"][1], 2)}")







