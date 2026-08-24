from datetime import datetime
import os
from typing import Any, Dict

import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from utils.metrics import benchmark_metrics
from utils.pipeline import pipeline

RANDOM_STATE = 42
N_RUNS = 30
N_JOBS = 4

X_train, X_test, y_train, y_test = pipeline()

MODELS: Dict[str, Any] = {
    "XGBoost": XGBRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
    "LightGBM": LGBMRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=-1),
    "CatBoost": CatBoostRegressor(thread_count=N_JOBS, random_state=RANDOM_STATE, verbose=0)
}

results = []
for name, model_instance in MODELS.items():
    res = benchmark_metrics(name, model_instance, X_train, y_train, X_test, y_test, N_RUNS)
    results.append(res)

df_results = pd.DataFrame(results).set_index('Model')

version = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

results_path = './results'

if not os.path.exists(results_path):
    os.mkdir(results_path)

df_results.to_csv(f'{results_path}/results_{version}.csv', sep=';', decimal=',')