import numpy as np
from tqdm import tqdm
from graphs import generateDag
from causal_estimator import *

models = ["linear","nn"]
criteria = ["treatment","outcome","union","intersection"]
methods = ["ipw","aipw","dml"]
graphs = {1: 2000, 2: 5000, 3: 2500}

def simulations():
    for graph, nums in graphs.items():
        print(f"Simulations for Graph {graph}:")
        df, za, zy, z = generateDag(nums, graph, 100)
        for model in models:
            print(f"    {model.capitalize()} Simulations:")
            for method in methods:
                for criterion in criteria:
                    causal_estimator = CausalEstimator('A','Y', z)
                    output = causal_estimator.run_estimation_with_ci(df, model,criterion,method, n_jobs = 4 if model == 'nn' else 1)
                    print(f"        {method.capitalize()} {criterion.capitalize()} Model: Ace - {round(output["ace"], 2)}, Variance - {round(output["var"], 2)}, Treatment Accuracy - {round(output["treat_acc"],2)}, Outcome RMSE - {round(output["out_rmse"],2)}, 95% Confidence Interval: {round(output["ci"][0], 2)} - {round(output["ci"][1], 2)}")



if __name__ == "__main__":
    np.random.seed(42)
    simulations()