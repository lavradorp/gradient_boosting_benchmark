from typing import Any, Dict

import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from utils.metrics import benchmark_metrics
from utils.pipeline import pipeline

RANDOM_STATE = 42
N_JOBS = 4

X_train, X_test, y_train, y_test = pipeline()

MODELS: Dict[str, Any] = {
    "XGBoost": XGBRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
    "LightGBM": LGBMRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=-1),
    "CatBoost": CatBoostRegressor(thread_count=N_JOBS, random_state=RANDOM_STATE, verbose=0)
}

results = []
for name, model_instance in MODELS.items():
    res = benchmark_metrics(name, model_instance, X_train, y_train, X_test, y_test)
    results.append(res)

df_results = pd.DataFrame(results).set_index('Model')

print(df_results)
df_results.to_csv('results.csv', sep=';', decimal=',')