from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import pandas as pd
import numpy as np

# load the data trainging set
X, y = load_breast_cancer(return_X_y=True)
data = load_breast_cancer()



from sklearn.ensemble import RandomForestClassifier

# define random forest classifier
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
forest.fit(X, y)



from boruta import BorutaPy

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X, y)


# check selected features
print(feat_selector.support_)

# check ranking of features
print(feat_selector.ranking_)

print('Selected vs. total: {s} vs. {t}'.format(s=np.sum(feat_selector.support_), t=X.shape[1]))

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

