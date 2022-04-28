import numpy as np
import pandas as pd
from numerapi import NumerAPI
from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)
import sklearn
import lightgbm
import json

napi = NumerAPI()

current_round = napi.get_current_round(tournament=8)
with open("v4/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["medium"] # get the medium feature set
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]

print('Reading trading data')
df = pd.read_parquet('v4/train.parquet', columns=read_columns)
df["erano"] = df.era.astype(int)
eras = df.erano
X, y = df.filter(like='feature_', axis='columns'), df[TARGET_COL]



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
rint(feat_selector.support_)

# check ranking of features
print(feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

