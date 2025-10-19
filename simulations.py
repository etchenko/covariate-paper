import numpy as np
from tqdm import tqdm
from graphs import generateDag
from causal_estimator import *
from indtests import *

try:
    import multiprocess
except:
    import multiprocessing as multiprocess
import time

sims = {
    "AIPW Treatment Linear Graph 1": ["nn","aipw","treatment",1,5000,10],
    "AIPW Outcome Linear Graph 1": ["nn","aipw","outcome",1,5000,10],
    "AIPW Different Linear Graph 1": ["nn","aipw","different",1,5000,10],
    "AIPW All Linear Graph 1": ["nn","aipw","all",1,5000,10],
    "DML Treatment Linear Graph 2": ["nn","dml","treatment",2,2500,1],
    "DML Outcome Linear Graph 2": ["nn","dml","outcome",2,2500,1],
    "DML All Linear Graph 2": ["nn","dml","all",2,2500,1],
    "DML Different Linear Graph 2": ["nn","dml","different",2,2500,1],
    "Backdoor All Linear Graph 3": ["nn","backdoor","all",3, 5000, 1],
    "Backdoor None Linear Graph 3": ["nn","backdoor","none",3, 5000, 1],
    "Backdoor Outcome Linear Graph 3": ["nn","backdoor","outcome",3, 5000, 1]
}
'''
    "AIPW Treatment Linear Graph 1": ["linear","aipw","treatment",1,1500,200],
    "AIPW Outcome Linear Graph 1": ["linear","aipw","outcome",1,1500,200],
    "AIPW Different Linear Graph 1": ["linear","aipw","different",1,1500,200],
    "AIPW All Linear Graph 1": ["linear","aipw","all",1,1500,200],
    "DML Treatment Linear Graph 2": ["linear","dml","treatment",2,2500,200],
    "DML Outcome Linear Graph 2": ["linear","dml","outcome",2,2500,200],
    "DML All Linear Graph 2": ["linear","dml","all",2,2500,200],
    "DML Different Linear Graph 2": ["linear","dml","different",2,2500,200],
    "Backdoor All Linear Graph 3": ["linear","backdoor","all",3, 5000, 200],
    "Backdoor None Linear Graph 3": ["linear","backdoor","none",3, 5000, 200],
    "Backdoor Outcome Linear Graph 3": ["linear","backdoor","outcome",3, 5000, 200],
    
}'''


models = ["linear","nn"]
criteria = ["outcome","treatment","union","intersection"]
methods = ["aipw","dml"]
graphs = {1: 2000, 2: 7500, 3: 2500}


def simulations():

    for key, item in sims.items():
        average_multiple_sims(item[4],item[3],item[0],item[2],item[1],n_jobs = 4 if item[0] == "nn" else 1, n_runs = item[5], title = key)

def multiple_sims():
    graphs = {
        (1, "aipw"): [["treatment","outcome","different","all"],[20,5], [750, 1500],["ace","var","treat_acc","ci","validity"]],
        (1, "dml"): [["treatment","outcome","different","all"],[20,5], [500, 1500],["ace","var","treat_acc","ci","validity"]],
        (2, "dml"): [["treatment","outcome","different","all"],[20,1], [1500, 3000], ["ace","var","treat_acc","ci","validity"]],
        (3, "backdoor"): [["all","none","outcome"],[10,1], [500, 500], ["ace","var","out_rmse","ci","validity"]]
    }

    for (graph, method), values in graphs.items():
        for i, num in enumerate(values[1]):
            results = {}
            for _ in tqdm(range(num), desc=f"Running Simulations for Graph {graph} {"Linear" if i ==0 else "NN"}"):
                df, za, zy, z, w = generateDag(values[2][i], graph, 66)
                for criterion in values[0]:
                    causal_estimator = CausalEstimator('A','Y',z)
                    if criterion == "all":
                        output = causal_estimator.run_estimation(df, "linear" if i == 0 else "nn", z,method)
                    elif criterion == "none":
                        output = causal_estimator.run_estimation(df, "linear" if i == 0 else "nn", ['A'],method)
                    else:
                        output = causal_estimator.run_estimation(df, "linear" if i == 0 else "nn", criterion,method)
                    adj = causal_estimator.adjustment_set
                    p_val = check_adjustment_validity(df, z, 'Y', adj, 'A', w)
                    output['validity'] = 1 if p_val < 0.05 else 0
                    output['pvals'] = p_val
                    for key in values[3]:
                        if not criterion in results:
                            results[criterion] = {key:[output[key]]}
                        else:
                            if not key in results[criterion]:
                                results[criterion][key] = [output[key]]
                            else:
                                results[criterion][key].append(output[key])
            for key, items in results.items():
                for value_key in items.keys():
                    items[value_key] = np.array(items[value_key]).mean(axis=0)
                if graph == 3:
                    print(f"{method.upper()}  {key.capitalize()} Graph {graph} {"Linear" if i == 0 else "NN"} Model: Outcome RMSE - {round(items['out_rmse'],2)}, Ace - {round(items['ace'], 2)}, 95% Confidence Interval: {round(items['ci'][0], 2)} - {round(items['ci'][1], 2)}, Average Validity: {round(items['validity'], 2)}")
                else:
                    print(f"{method.upper()}  {key.capitalize()} Graph {graph} {"Linear" if i == 0 else "NN"} Model: Treatment Accuracy - {round(items['treat_acc'],2)}, Variance - {round(items['var'], 2)}, 95% Confidence Interval: {round(items['ci'][0], 2)} - {round(items['ci'][1], 2)}, Average Validity: {round(items['validity'], 2)}")





def average_multiple_sims(nums, graph, model, criterion, method, n_jobs, n_runs, title):
    if method != "backdoor":
        estimates = {"ace":[],"acc":[],"validity":[], "pval":[]}
    else:
        estimates = {"ace":[],"rmse":[],"validity":[], "pval":[]}
    for _ in range(n_runs):
        df, za, zy, z, w = generateDag(nums, graph, 100, model)
        causal_estimator = CausalEstimator('A','Y', z)
        if criterion == "all":
            ace = causal_estimator.run_estimation(df, model, 'covs',method, covs = z)
        elif criterion == "none":
            ace = causal_estimator.run_estimation(df, model, 'covs',method, covs = ['A'])
        else:
            ace = causal_estimator.run_estimation(df, model, criterion,method)

        # TODO: Bake accuracy tests into code
        (acc, rmse) = causal_estimator.calculate_accuracy()

        adj = causal_estimator.adjustment_set
        pval = check_adjustment_validity(df, 'Y', adj, 'A', w)
        validity = 1 if pval > 0.05 else 0
        output = {"acc": acc, "rmse":rmse, "ace":ace,"validity":validity, "pval": pval}
        #output['validity'] = validity
        for key in estimates.keys():
            estimates[key].append(output[key])
    variance = np.var(estimates["ace"])
    lower = np.percentile(estimates["ace"], (1 - 0.95) / 2 * 100)
    upper = np.percentile(estimates["ace"], (1 + 0.95) / 2 * 100)
    for key in estimates.keys():
        estimates[key] = np.array(estimates[key]).mean(axis=0)
    if graph == 3:
        print(f"{(method.capitalize() + ' ' + criterion.capitalize()) if (not title) else title} Model: Outcome RMSE - {round(estimates['rmse'],2)}, Ace - {round(estimates['ace'], 2)}, 95% Confidence Interval: {round(lower, 2)} - {round(upper, 2)}, Average Validity: {round(estimates['validity'], 2)}, Average p-value: {estimates['pval']}")
    else:
        print(f"{(method.capitalize() + ' ' + criterion.capitalize()) if (not title) else title} Model: Treatment Accuracy - {round(estimates['acc'],2)}, Variance - {round(variance, 2)}, 95% Confidence Interval: {round(lower, 2)} - {round(upper, 2)}, Average Validity: {round(estimates['validity'], 2)}, Average p-value: {estimates['pval']}")

def test():
    num = 1
    df, za, zy, z, w = generateDag(1000, 1, 33, "nonlinear")
    #print(np.array(df))
    '''s = time.time() # start time
    samples = []
    arg_iterable = [(df, 'nn', 'outcome', 'dml') for i in range(num)]
    with multiprocess.Pool(num) as pool:
        estimates = pool.starmap(CausalEstimator('A','Y',z).run_estimation, arg_iterable)
    e = time.time() # end time
    print(estimates)
    print(e - s, "seconds")'''

    s = time.time() # start time
    estimates = []
    for i in range(num):
        ce = CausalEstimator('A','Y',z)
        estimates.append(ce.run_estimation(df, 'nn', 'treatment', 'dml'))
        #val = check_adjustment_validity(df, 'Y', [], 'A', w)
        #print(val)
    e = time.time() # end time
    print(estimates)
    print(ce.treatment_features)
    print(ce.outcome_features)
    print(e - s, "seconds")





if __name__ == "__main__":
    np.random.seed(23)
    simulations()
    #test()
    #multiple_sims()