import numpy as np
from tqdm import tqdm
from graphs import generateDag
from causal_estimator import *
from indtests import *
import sys

from multiprocess import Pool

import time

linear_sims = {
    "AIPW Treatment Linear Graph 1": ["linear","aipw","treatment",1,1500,200],
    "AIPW Outcome Linear Graph 1": ["linear","aipw","outcome",1,1500,200],
    "AIPW Different Linear Graph 1": ["linear","aipw","different",1,1500,200],
    "AIPW All Linear Graph 1": ["linear","aipw","all",1,1500,200],
    "DML Treatment Linear Graph 2": ["linear","dml","treatment",2,1500,200],
    "DML Outcome Linear Graph 2": ["linear","dml","outcome",2,1500,200],
    "DML All Linear Graph 2": ["linear","dml","all",2,1500,200],
    "DML Different Linear Graph 2": ["linear","dml","different",2,1500,200],
    "Backdoor All Linear Graph 3": ["linear","backdoor","all",3, 5000, 200],
    "Backdoor None Linear Graph 3": ["linear","backdoor","none",3, 5000, 200],
    "Backdoor Outcome Linear Graph 3": ["linear","backdoor","outcome",3, 5000, 200],
}

nn_sims ={
    "Backdoor All Non-linear Graph 3": ["nn","backdoor","all",3, 25000, 100],
    "Backdoor None Non-linear Graph 3": ["nn","backdoor","none",3, 25000, 100],
    "Backdoor Outcome Non-linear Graph 3": ["nn","backdoor","outcome",3, 25000, 100]  
}
# Not including for now since this takes so long
'''
    "DML Treatment Non-linear Graph 2": ["nn","dml","treatment",2,5000,5],
    "DML Outcome Non-linear Graph 2": ["nn","dml","outcome",2,5000,5],
    "DML All Non-linear Graph 2": ["nn","dml","all",2,5000,5],
    "DML Different Non-linear Graph 2": ["nn","dml","different",2,5000,5],
    "AIPW Treatment Non-linear Graph 1": ["nn","aipw","treatment",1,7500,100],
    "AIPW Outcome Non-linear Graph 1": ["nn","aipw","outcome",1,7500,100],
    "AIPW Different Non-linear Graph 1": ["nn","aipw","different",1,7500,100],
    "AIPW All Non-linear Graph 1": ["nn","aipw","all",1,7500,100], 
'''

def simulations(sims):
    for key, item in sims.items():
        average_multiple_sims(item[4],item[3],item[0],item[2],item[1],n_jobs = 4, n_runs = item[5], title = key)

def average_multiple_sims(nums, graph, model, criterion, method, n_jobs, n_runs, title):
    if method != "backdoor":
        estimates = {"ace": [], "acc": [], "validity": [], "pval": []}
    else:
        estimates = {"ace": [], "rmse": [], "validity": [], "pval": []}

    def run_one(_):
        df, za, zy, z, w = generateDag(nums, graph, 66, model)
        causal_estimator = CausalEstimator('A', 'Y', z)
        if criterion == "all":
            ace = causal_estimator.run_estimation(df, model, 'covs', method, covs=z)
        elif criterion == "none":
            ace = causal_estimator.run_estimation(df, model, 'covs', method, covs=[])
        else:
            ace = causal_estimator.run_estimation(df, model, criterion, method)

        acc, rmse = causal_estimator.calculate_accuracy()
        adj = causal_estimator.adjustment_set
        res_Y = df['Y'] - causal_estimator.outcome_model.predict(df[causal_estimator.outcome_covs])
        pval = check_adjustment_validity(df, 'Y', adj, 'A', w, model, res_Y)
        validity = 1 if pval > 0.25 else 0

        return {"acc": acc, "rmse": rmse, "ace": ace, "validity": validity, "pval": pval}

    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(run_one, range(n_runs)), total=n_runs,
                            desc=f"{model} {criterion} {method} {graph}"))

    for res in results:
        for k in estimates.keys():
            estimates[k].append(res[k])
    
    variance = np.var(estimates["ace"])
    lower = np.percentile(estimates["ace"], (1 - 0.95) / 2 * 100)
    upper = np.percentile(estimates["ace"], (1 + 0.95) / 2 * 100)
    for k in estimates.keys():
        estimates[k] = np.mean(estimates[k])

    if graph == 3:
        print(f"{title}: Outcome RMSE - {round(estimates['rmse'],2)}, "
              f"Ace - {round(estimates['ace'],2)}, CI: {round(lower,2)}–{round(upper,2)}, "
              f"Validity: {round(estimates['validity'],2)}, p-value: {round(estimates['pval'],3)}")
    else:
        print(f"{title}: Accuracy - {round(estimates['acc'],2)}, Var - {round(variance,2)}, "
              f"CI: {round(lower,2)}–{round(upper,2)}, Validity: {round(estimates['validity'],2)}, "
              f"p-value: {round(estimates['pval'],3)}")

if __name__ == "__main__":
    np.random.seed(23)
    
    # If run with a Slurm array index, only run that simulation
    if len(sys.argv) > 2:
        sim_idx = int(sys.argv[2]) - 1  # SLURM_ARRAY_TASK_ID starts at 1
        sims = linear_sims if sys.argv[1] == "linear" else nn_sims
        keys = list(sims.keys())
        key = keys[sim_idx]
        item = sims[key]
        print(f"Running simulation {sim_idx + 1}/{len(sims)}: {key}")
        average_multiple_sims(
            item[4], item[3], item[0], item[2], item[1],
            n_jobs=4 if item[0] == "nn" else 1,
            n_runs=item[5],
            title=key
        )
    else:
        # Default: run all (useful for local testing)
        sims = linear_sims if sys.argv[1] == "linear" else nn_sims
        simulations(sims)
