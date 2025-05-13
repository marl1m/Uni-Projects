
from sklearn.model_selection import RandomizedSearchCV

def tune_model(model, param_grid, X, y, cv=5):
    search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=50)
    search.fit(X, y)
    return search.best_estimator_
