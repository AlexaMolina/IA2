import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    """This function compute the Manhattan / citybloc distance
    V1 and V2 of the same size

    Args:
        v1 (list): list of the first object
        v2 (list): list of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1-v2))
    return dist

def euclidean(v1, v2):
    """This function compute the Euclidean distance
    V1 and V2 of the same size

    Args:
        v1 (list): list of the first object
        v2 (list): list of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum(v1-v2)**2)
    return dist
    
def chebyshev(v1, v2):
    """This function compute the Manhattan / citybloc distance
    V1 and V2 of the same size

    Args:
        v1 (list): list of the first object
        v2 (list): list of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1-v2))
    return dist

def canberra(v1, v2):
    """This function compute the Manhattan / citybloc distance
    V1 and V2 of the same size

    Args:
        v1 (list): list of the first object
        v2 (list): list of the second object
    """
    
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db, query_features, distance, num_results):
    distances = []
    for instance in features_db:
        features, label, img_path = instance[ : -2], instance[-2], instance[-1]
        if distance == 'manhattan':
            dist = manhattan(query_features, features)
        if distance == 'euclidean':
            dist = euclidean(query_features, features)
        if distance == 'chebyshev':
            dist = chebyshev(query_features, features)
        if distance == 'canberra':
            dist = canberra(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[ : num_results]
            