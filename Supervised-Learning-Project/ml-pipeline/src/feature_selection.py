
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X, y, k=20):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    return selector.fit_transform(X, y), selector
