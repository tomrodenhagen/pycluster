#Metrics
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import numpy as np
def c_score(ids1, ids2):
  h = homogeneity_score(ids1, ids2)
  c = completeness_score(ids1, ids2)
  return h * c / (h + c)
def mae(id1, id2):
  return np.abs(len(set(id1)) - len(set(id2)))
