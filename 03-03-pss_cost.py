import math
import numpy as np
from debug import *
from bayes import *
from results_visualization import *
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal


def split_rows_on_class_labels( sample_array, debug=False ):
    index_positions = np.argsort( sample_array[:, -1] )
    sorted_array = sample_array[ index_positions ]

    # Collect feature vectors and corresponding labels AFTER sort 
    X = sorted_array[:,:-1]
    y = sorted_array[:,-1]

    # Produce *list* of data arrays w. one array per class (( represented by list index ))
    # * np.diff( data[:,2] ) -- computes differences between rows in the label column
    # * np.flatnonzero -- returns vector of INDICES for non-zero elements in vector of differences
    #     NOTE: +1 corrects index offsets (N item list -> N-1 differences)
    label_diffs = np.diff( sorted_array[:,-1] )
    split_indices = np.flatnonzero( label_diffs ) + 1

    sorted_no_labels = sorted_array[:,:-1]
    split_data = np.vsplit( sorted_no_labels, split_indices )

    # Debugging
    if debug:
        dcheck( 'split shape', [ np.shape(a) for a in split_data ] )
        dnpcheck( 'Top elements from list of arrays in split', [ a[:3,:] for a in split_data ] )

    # Return split data, along with data matrix and corresponding labels in y
    return ( split_data, X, y )


################################################################
# Main Program
################################################################
    
def main():

    D = np.array([[3,3,1],[1,1,1],[-1,0,1],[2,2,0],[-2,2,0],[-2,-2,0],[0,-2,0],[1,5,2],[2,4,2],[3,4,2]])
    ( split_data, _, _ ) = split_rows_on_class_labels( D )
    sigmas = [np.cov(X.T) for X in split_data]
    mus = [np.mean(X, axis=0) for X in split_data]
    sigma_invs = [np.linalg.inv(sigma) for sigma in sigmas]
    sigma_dets = [np.linalg.det(sigma) for sigma in sigmas]
    priors = [X.shape[0]/len(D) for X in split_data]
    x_test = np.array([[2,2],[2,3]])
    likelihoods = [multivariate_normal(mu,sigma).pdf(x_test) for (mu,sigma) in zip(mus,sigmas)]
    posteriors = np.array([likelihood*prior for (likelihood,prior) in zip(likelihoods,priors)])
    map_predictions = np.argmax(posteriors, axis=0)
    print(f'priors={priors}\nx_test={x_test}\nlikelihoods={likelihoods}\nposteriors={posteriors}\nmap_predictions={map_predictions}')
    cost_table = np.array([[0.3,0.3,0.3],[0.3,-0.5,0.3],[0.1,0.1,-0.9]])
    # risks = [np.array(posteriors[i][0])  cost_table[i,:].T for i in range(len(posteriors))]
    risks_x0 = [cost_table[i,:] @ posteriors[:,0] for i in range(3)]
    risks_x1 = [cost_table[i,:] @ posteriors[:,1] for i in range(3)]
    risks = np.column_stack([risks_x0,risks_x1])
    risk_predictions = np.argmin(np.array(risks), axis=0)
    print(f'risks={risks}\nrisk_predictions={risk_predictions}')

if __name__ == "__main__":
    main()
