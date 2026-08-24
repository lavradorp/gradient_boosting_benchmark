import gc
import psutil
import time
import tracemalloc
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


_process = psutil.Process()


def benchmark_metrics(model_name: str, model: Any, X_train, y_train, X_test, y_test, runs: int) -> dict:
    train_times = []
    rss_deltas = []
    infer_latencies = []
 
    last_y_pred = None
 
    for _ in range(runs):
        run_model = clone(model)
 
        gc.collect()
 
        rss_before = _process.memory_info().rss
 
        start_train = time.perf_counter()
        run_model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train
 
        rss_after = _process.memory_info().rss
 
        start_infer = time.perf_counter()
        y_pred = run_model.predict(X_test)
        infer_time_ms = (time.perf_counter() - start_infer) * 1000
 
        train_times.append(train_time)
        rss_deltas.append(max(rss_after - rss_before, 0) / (1024 * 1024))
        infer_latencies.append(infer_time_ms)
 
        last_y_pred = y_pred
 
    mae = mean_absolute_error(y_test, last_y_pred)
    rmse = root_mean_squared_error(y_test, last_y_pred)
    r2 = r2_score(y_test, last_y_pred)


    return {
        "Model": model_name,
        "Runs": runs,
        "Avg Train Time (s)": round(float(np.mean(train_times)), 3),
        "Std Train Time (s)": round(float(np.std(train_times)), 3),
        "Avg Infer Latency (ms)": round(float(np.mean(infer_latencies)), 2),
        "Std Infer Latency (ms)": round(float(np.std(infer_latencies)), 2),
                "Peak RAM - RSS delta (MB)": round(float(np.max(rss_deltas)), 2),
        "Total RAM Allocated (MB)": round(float(np.sum(rss_deltas)), 2),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    }

