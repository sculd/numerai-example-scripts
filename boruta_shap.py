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
from BorutaShap import BorutaShap

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

# create model to be used by BorutaShap feature selector
# changes to the model choice affect the features that are chosen so there's lot's of room to experiment here
model = lightgbm.LGBMRegressor(n_jobs=-1, colsample_bytree=0.1, learning_rate=0.01, n_estimators=2000, max_depth=5, device='gpu')

# initialize the feature selector
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

# here I iterate over the 4 non-overlapping sets of eras and perform feature selection in each, then take the union of the selected features
# I'm just using standard 'target' for now, but it would be interesting to investigate other targets as well
# It may also be useful to look at the borderline features that aren't accepted or eliminated
good_features = []
df_tmp = df
eras_tmp = eras
Feature_Selector.fit(X=df_tmp.filter(like='feature', axis='columns'), y=df_tmp[TARGET_COL], n_trials=5, sample=False, train_or_test = 'test', normalize=True, verbose=True)
good_features+=Feature_Selector.accepted
good_features = list(set(good_features))

print(good_features)