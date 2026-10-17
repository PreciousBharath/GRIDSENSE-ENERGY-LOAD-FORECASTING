"""GridSense engine: synthetic smart-grid data, feature engineering, forecasting models, anomalies, tariffs."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOLIDAYS = ["01-01", "01-15", "01-26", "04-14", "05-01", "08-15", "10-02", "10-20", "11-01", "12-25"]


def tariff(hour):
    """Time-of-day tariff (₹/kWh): peak 18–22, shoulder 9–17, off-peak night."""
    return np.where((hour >= 18) & (hour < 22), 9.5, np.where((hour >= 9) & (hour < 18), 7.0, 4.5))


def carbon(hour):
    """Grid carbon intensity (g CO₂/kWh) — lower midday thanks to solar."""
    return 820 - 260 * np.exp(-((hour - 13) ** 2) / 10)


def generate(days=730, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=days * 24, freq="h")
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    temp = 28 + 5 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.3, len(idx))
    holiday = idx.strftime("%m-%d").isin(HOLIDAYS).astype(int)
    weekend = (dow >= 5).astype(int)
    base = 420 + 180 * np.exp(-((h - 10.5) ** 2) / 8) + 260 * np.exp(-((h - 19.5) ** 2) / 5) - 120 * np.exp(-((h - 3) ** 2) / 6)
    work = np.where(weekend | holiday, 0.78, 1.0)
    cooling = 22 * np.clip(temp - 26, 0, None) ** 1.25
    trend = np.linspace(0, 60, len(idx))
    load = base * work + cooling + trend + rng.normal(0, 18, len(idx))
    cloud = np.clip(rng.beta(2, 5, days).repeat(24), 0, 1)
    solar = np.clip(np.sin(np.pi * (h - 6) / 13), 0, None) * 260 * (1 - cloud * 0.8) * (1 + 0.1 * np.sin(2 * np.pi * doy / 365))
    an = rng.random(len(idx)) < 0.004
    load[an] *= rng.choice([0.45, 1.6], an.sum())
    df = pd.DataFrame({"load_kw": load.round(1), "solar_kw": solar.round(1), "temp_c": temp.round(1), "holiday": holiday, "is_anomaly": an.astype(int)}, index=idx)
    df["net_kw"] = (df.load_kw - df.solar_kw).round(1)
    return df


FEATS = ["hour", "dow", "month", "weekend", "holiday", "temp_c", "temp_sq", "lag_24", "lag_48", "lag_168", "roll_24", "hour_sin", "hour_cos"]


def features(df):
    f = pd.DataFrame(index=df.index)
    f["hour"], f["dow"], f["month"] = df.index.hour, df.index.dayofweek, df.index.month
    f["weekend"] = (f.dow >= 5).astype(int)
    f["holiday"] = df.holiday
    f["temp_c"] = df.temp_c
    f["temp_sq"] = np.clip(df.temp_c - 26, 0, None) ** 2
    for l in (24, 48, 168):
        f[f"lag_{l}"] = df.load_kw.shift(l)
    f["roll_24"] = df.load_kw.shift(24).rolling(24).mean()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f.hour / 24), np.cos(2 * np.pi * f.hour / 24)
    return f


MODELS = {
    "Gradient Boosting (Hist)": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=-1, random_state=0),
    "Neural Net (MLP)": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=0)),
}


def backtest(df, test_days=30):
    X, y = features(df), df.load_kw
    ok = X.notna().all(1)
    X, y = X[ok], y[ok]
    cut = len(X) - test_days * 24
    out = {}
    for name, mk in MODELS.items():
        m = mk().fit(X.iloc[:cut], y.iloc[:cut])
        p = m.predict(X.iloc[cut:])
        out[name] = {"model": m, "pred": pd.Series(p, index=X.index[cut:]), "mae": mean_absolute_error(y.iloc[cut:], p),
                     "mape": mean_absolute_percentage_error(y.iloc[cut:], p) * 100, "r2": r2_score(y.iloc[cut:], p)}
    best = min(out, key=lambda k: out[k]["mae"])
    pi = permutation_importance(out[best]["model"], X.iloc[cut:], y.iloc[cut:], n_repeats=3, random_state=0)
    imp = pd.Series(pi.importances_mean, index=FEATS).sort_values()
    q = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=150, max_depth=4, random_state=0).fit(X.iloc[-24 * 120:cut], y.iloc[-24 * 120:cut]) for a in (0.1, 0.9)}
    return {"results": out, "y_test": y.iloc[cut:], "best": best, "importance": imp, "quantiles": q, "X": X}


def forecast(df, bundle, model_name, hours=48, temp_shift=0.0):
    """Recursive multi-step forecast using weather-shifted temperature profile from the same hours last week."""
    hist = df[["load_kw", "holiday", "temp_c"]].copy()
    m = bundle["results"][model_name]["model"]
    rows = []
    for i in range(hours):
        t = hist.index[-1] + pd.Timedelta(hours=1)
        temp = hist.temp_c.iloc[-168] + temp_shift
        hist.loc[t] = [np.nan, int(t.strftime("%m-%d") in HOLIDAYS), temp]
        x = features(hist).iloc[[-1]]
        yhat = float(m.predict(x)[0])
        lo, hi = (float(bundle["quantiles"][a].predict(x)[0]) for a in (0.1, 0.9))
        spread = max(hi - lo, 30)
        hist.loc[t, "load_kw"] = yhat
        rows.append({"time": t, "forecast": yhat, "lower": yhat - spread / 2 * (1 + i / hours), "upper": yhat + spread / 2 * (1 + i / hours), "temp_c": temp})
    return pd.DataFrame(rows).set_index("time")


def anomalies(df, bundle):
    best = bundle["results"][bundle["best"]]
    resid = (bundle["y_test"] - best["pred"]).to_frame("resid")
    resid["load"] = bundle["y_test"]
    iso = IsolationForest(contamination=0.01, random_state=0).fit(resid[["resid"]])
    resid["flag"] = iso.predict(resid[["resid"]]) == -1
    resid["pred"] = best["pred"]
    return resid
