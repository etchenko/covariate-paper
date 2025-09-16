import numpy as np
from tqdm import tqdm
from graphs import generateDag
from causal_estimator import *

sims = {
    "DML Treatment Linear Graph 1": ["linear","dml","treatment",1,1000,10],
    "DML Outcome Linear Graph 1": ["linear","dml","outcome",1,1000,10],
    "DML All Linear Graph 1": ["linear","dml","all",1,1000,10],
    "AIPW Treatment Linear Graph 1": ["linear","aipw","treatment",1,2500,10],
    "AIPW Outcome Linear Graph 1": ["linear","aipw","outcome",1,2500,10],
    "AIPW All Linear Graph 1": ["linear","aipw","all",1,2500,10],
    "DML Treatment NN Graph 1": ["nn","dml","treatment",1,2500,10],
    "DML Outcome NN Graph 1": ["nn","dml","outcome",1,2500,10],
    "DML All NN Graph 1": ["nn","dml","all",1,2500,10],
    "AIPW Treatment NN Graph 1": ["nn","aipw","treatment",1,4000,10],
    "AIPW Outcome NN Graph 1": ["nn","aipw","outcome",1,4000,10],
    "AIPW All NN Graph 1": ["nn","aipw","all",1,4000,10],
    "DML Treatment Linear Graph 2": ["linear","dml","treatment",2,4000,10,100],
    "DML Outcome Linear Graph 2": ["linear","dml","outcome",2,4000,10,100],
    "DML All Linear Graph 2": ["linear","dml","all",2,4000,10,100],
    "DML Treatment NN Graph 2": ["nn","dml","treatment",2,12000,1],
    "DML Outcome NN Graph 2": ["nn","dml","outcome",2,12000,1],
    "DML All NN Graph 2": ["nn","dml","all",2,12000,1],
    "Backdoor All Linear Graph 3": ["linear","backdoor","all",3,3000, 10],
    "Backdoor None Linear Graph 3": ["linear","backdoor","none",3,3000, 10],
    "Backdoor All NN Graph 3": ["nn","backdoor","all",3,5000, 1],
    "Backdoor None NN Graph 3": ["nn","backdoor","none",3,5000, 1],

}


models = ["linear","nn"]
criteria = ["outcome","treatment","union","intersection"]
methods = ["aipw","dml"]
graphs = {1: 2000, 2: 7500, 3: 2500}


def simulations2():

    for key, item in sims.items():
        average_multiple_sims(item[4],item[3],item[0],item[2],item[1],n_jobs = 4 if item[0] == "nn" else 1, n_runs = item[5], title = key)

def simulations():
    for graph, nums in graphs.items():
        print(f"Simulations for Graph {graph}:")
        
        for model in models:
            print(f"    {model.capitalize()} Simulations:")
            for method in methods:
                for criterion in criteria:
                    average_multiple_sims(nums if method != 'nn' else 2000, graph, model, criterion, method, n_jobs = 4 if method == "nn" else 1, n_runs = 5)
                    '''
                    df, za, zy, z = generateDag(nums, graph, 100)
                    causal_estimator = CausalEstimator('A','Y', z)
                    output = causal_estimator.run_estimation_with_ci(df, model,criterion,method, n_jobs = 4 if model == 'nn' else 1)
                    print(f"        {method.capitalize()} {criterion.capitalize()} Model: Ace - {round(output["ace"], 2)}, Variance - {round(output["var"], 2)}, Treatment Accuracy - {round(output["treat_acc"],2)}, Outcome RMSE - {round(output["out_rmse"],2)}, 95% Confidence Interval: {round(output["ci"][0], 2)} - {round(output["ci"][1], 2)}")
                    '''

def average_multiple_sims(nums, graph, model, criterion, method, n_jobs, n_runs, title):
    estimates = {"ace":[],"var":[],"treat_acc":[],"out_rmse":[],"ci":[]}
    for _ in range(n_runs):
        df, za, zy, z = generateDag(nums, graph, 100 if model != 'nn' else 66)
        causal_estimator = CausalEstimator('A','Y', z)
        if criterion == "all":
            output = causal_estimator.run_estimation_with_ci(df, model, z,method, n_jobs = n_jobs)
        elif criterion == "none":
            output = causal_estimator.run_estimation_with_ci(df, model, ['A'],method, n_jobs = n_jobs)
        else:
            output = causal_estimator.run_estimation_with_ci(df, model, criterion,method, n_jobs = n_jobs)
        for key in estimates.keys():
            estimates[key].append(output[key])
    for key in estimates.keys():
        estimates[key] = np.array(estimates[key]).mean(axis=0)
    print(f"{(method.capitalize() + ' ' + criterion.capitalize()) if (not title) else title} Model: Ace - {round(estimates['ace'], 2)}, Variance - {round(estimates['var'], 2)}, Treatment Accuracy - {round(estimates['treat_acc'],2)}, Outcome RMSE - {round(estimates['out_rmse'],2)}, 95% Confidence Interval: {round(estimates['ci'][0], 2)} - {round(estimates['ci'][1], 2)}")

if __name__ == "__main__":
    np.random.seed(42)
    simulations2()