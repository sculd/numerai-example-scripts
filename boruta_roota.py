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
X, y = df.filter(like='feature_', axis='columns'), df[TARGET_COL]

from boostaroota import BoostARoota

#Specify the evaluation metric: can use whichever you like as long as recognized by XGBoost
  #EXCEPTION: multi-class currently only supports "mlogloss" so much be passed in as eval_metric
br = BoostARoota(metric='logloss')

#Fit the model for the subset of variables
br.fit(X, y)

#Can look at the important variables - will return a pandas series
br.keep_vars_

#Then modify dataframe to only include the important variables
br.transform(x)
