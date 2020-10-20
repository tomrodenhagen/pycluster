#Metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import numpy as np
def mean_hom_comp(ids1, ids2):
  return (homogeneity_score(ids1, ids2) + completeness_score(ids1, ids2)) / 2
def mae(id1, id2):
  return np.abs(len(set(id1)) - len(set(id2)))
