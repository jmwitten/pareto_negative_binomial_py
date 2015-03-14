"""
Implementation of the Pareto/NBD model from "RFM and CLV: Using Iso-Value Curves for Customer Base Analysis" by Fader, Hardie, and Lee (2005). 
Details of the implementation were taken from A note on Implementing the Pareto/NBD Model in MATLAB and as well as accompanying data (http://www.brucehardie.com/notes/008/)
"""
from math import log
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1
 
__author__ = 'Joel Witten'
 
 
def log_likelihood_individual(r, alpha, s, beta, x, tx, T):
    """Log of the likelihood function for a rnadomly chosen person with purchase history 
    (X = x, tx, T), where x is the number of purchases, tx is the time of the last purchase, and T is the current time
    """

    part_1 = gammaln(r+x) - gammaln(r) + r*log(alpha) + s*log(beta)
    part_2_1 = 1.0/((alpha+T)**(r+x) *(beta+T)**s)
    part_2_2 = s/(r+s+x)

    maxab = max(alpha, beta)
    absab = abs(alpha-beta)

    param2 = (s + 1) if alpha >= beta else (r+x)
    
    if alpha == beta:
        F1 = 1.0/((maxab+tx)**(r+s+x))
        F2 = 1.0/((maxab+T)**(r+s+x))
    else:
        F1 = 1.0/((maxab+tx)**(r+s+x))*hyp2f1(r+s+x, param2, r+s+x+1, absab/(maxab+tx))
        F2 = 1.0/((maxab+T)**(r+s+x))*hyp2f1(r+s+x, param2, r+s+x+1, absab/(maxab+T))

    return part_1 + log(part_2_1+part_2_2*(F1-F2))
 
 
def log_likelihood(r, alpha, s, beta, customers):
    """Sum of the individual log likelihoods"""
    if r <= 0 or alpha <= 0 or s <= 0 or beta <= 0:
        return -np.inf
    return sum([log_likelihood_individual(r, alpha, s, beta, x, tx, T) for x, tx, T in customers])
 
 
def maximize(customers):
    '''Maximizing the likelihood by minimizing the negative likelihood'''
    
    negative_ll = lambda params: -log_likelihood(*params, customers=customers)
    params0 = np.array([1., 1., 1., 1.])
    res = minimize(negative_ll, params0, method='nelder-mead', options={'xtol': 1e-8})
    return res
 
 
def fit(customers):
    res = maximize(customers)
    if res.status != 0:
        raise Exception(res.message)
    return res.x
 
 
def cdnow_customer_data(fname):
    data = []
    with open(fname) as f:
        f.readline()
        for line in f:
            data.append(map(float, line.strip().split(',')[1:4]))
    return data
 
 
def main():
    data = cdnow_customer_data('cdnow_customers.csv')
    r, alpha, s, beta = fit(data)
    print r, alpha, s, beta
    #Comparing values to those found in the paper
    print np.allclose([r, alpha, s, beta], [.55, 10.58, .61, 11.67], 1e-2)
 
 
if __name__ == '__main__':
    main()