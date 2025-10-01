import numpy as np


"""
Use this file to analyse results.
"""

def get_distribution_parameters(arr):
    # Mean
    mean = np.mean(arr)

    # Standard Deviation
    std_dev = np.std(arr)

    print("Mean:", mean)
    print("Standard Deviation:", std_dev)

    

