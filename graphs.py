'''
Graph Generation code
'''
import numpy as np
import pandas as pd

def generateDag(nums, graph = 1, zs = None):
    def generate_gauss(mu, sigma, nums):
        '''
        Generate an array from a gaussian distribution
        '''
        return np.random.normal(loc=mu, scale=sigma, size=nums)
    
    def generate_treatment(data):
        '''
        Generate a binary treatment from a gaussian distribution using binomial sampling
        over a sigmoid function
        '''
        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        prob = sigmoid(data)
        output = np.random.binomial([1 for i in prob], prob, len(data))
        return output
    
    def generate_random_covariance(n, epsilon=1e-6):
        '''
        Generate a covariance matrix of size nxn
        '''
        A = np.random.randn(n, n)  # Random matrix
        cov = A @ A.T              # Symmetric, positive semi-definite
        cov += epsilon * np.eye(n)  # Make it positive definite

        # Normalize to unit variance
        D_inv = np.diag(1 / np.sqrt(np.diag(cov)))
        corr = D_inv @ cov @ D_inv  # This is now a correlation matrix (diag = 1)
    
        return corr
    data = pd.DataFrame()
    # Generate the DAG
    if graph == 1:
        '''
        Generate the low-dimensional graph from Figure 1
        '''
        data['W1'] = generate_gauss(0, 1, nums)
        data['W2'] = generate_gauss(0, 1, nums)
        aw1 = aw2 = o1w1 = o2w2 = mo1 = mo2 = mo3 = yo3 = yo1 = ym = ma = 1
        ym = 0.25
        data['A'] = generate_treatment(aw1*data['W1']+aw2*data['W2']+generate_gauss(0,1,nums))
        data['O1'] =  o1w1*data['W1']+generate_gauss(0, 1, nums)
        data['O2'] = o2w2*data['W2']+generate_gauss(0, 1, nums)
        data['O3'] = generate_gauss(0, 1, nums)
        data['M'] = mo1*data['O1']+ mo2*data['O2']+ mo3*data['O3']+ ma*data['A']+ generate_gauss(0, 1, nums)
        data['Y'] = yo3*data['O3'] + yo1*data['O1'] + ym*data['M']  + generate_gauss(0, 1, nums)
        return data, ['W1','W2'], ['O1','O2','O3'], ['W1','W2','O1','O2','O3']
    elif graph == 2:
        '''
        Generate the high-dimensional Graph from Figure 2
        '''
        z = np.random.multivariate_normal([0 for i in range(zs)],generate_random_covariance(zs),nums)
        z_array = list(zip(*z))
        pred = np.random.choice([1, 2, 3], size=zs)
        za = [f'Z{i}' for i in range(len(pred)) if pred[i] == 1]
        zy = [f'Z{i}' for i in range(len(pred)) if pred[i] == 2]
        zay = [f'Z{i}' for i in range(len(pred)) if pred[i] == 3]
        for i in range(len(z_array)):
            data[f'Z{i}'] = z_array[i]
        data['A'] = generate_treatment(1.25*data[za].sum(axis=1) + 1.25*data[zay].sum(axis=1)+ 0.25*generate_gauss(0,1,nums))
        data['Y'] = 1.25*data[zy].sum(axis = 1) +0.75*data[zay].sum(axis=1) + 1*data['A'] +0.25*generate_gauss(0,1,nums)
        return data, za+ zay, zay+zy, zay+za+zy
    elif graph == 3:
        '''
        Generate the graph from Figure 3 (For outcome regression)
        '''
        au1 = c1u1 = c1u2 = c2u2 = c2u3= yu3 = ya = 1
        c1u2 = -4
        c2u2 = 3
        yu3 = 4
        ya = 1
        data['U1'] = generate_gauss(0,1, nums)
        data['U2'] = generate_gauss(0,1, nums)
        data['U3'] = generate_gauss(0,1, nums)
        data['U4'] = generate_gauss(0,1, nums)
        data['A'] = generate_treatment(au1*data['U1'])
        data['C1'] = c1u1*data['U1'] + c1u2*data['U2'] + generate_gauss(0,1,nums)
        data['C2'] = c2u2*data['U2'] + c2u3*data['U3'] + generate_gauss(0,1,nums)
        data['Y'] = ya*data['A'] + yu3*data['U3'] + generate_gauss(0,1,nums)
        return data, ['C1','C2'], [], ['C1','C2']