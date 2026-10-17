"""GridSense — smart energy load forecasting & real-time grid monitoring (Streamlit)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import auth, ui
from core.engine import anomalies, backtest, carbon, forecast, generate, tariff

st.set_page_config(page_title="GridSense · Energy Forecasting", page_icon="⚡", layout="wide")
ui.inject_css()
APP, ICON = "GridSense", "⚡"
if not auth.auth_page(APP, "AI-powered smart-grid monitoring, load forecasting and energy-cost optimisation.",
                      ["📡 Real-time smart-meter stream with anomaly alerts", "🔮 48-hour to 7-day forecasts with confidence bands",
                       "🌡️ Weather what-if simulation", "💸 Tariff-aware savings & carbon planner"], ICON,
                      demo={"name": "Grid Operator", "email": "demo@gridsense.ai", "password": "Demo@1234"}):
    st.stop()


@st.cache_data(show_spinner="Loading 2 years of smart-meter data…")
def data():
    return generate()


@st.cache_resource(show_spinner="Training forecasting models (≈20 s, once)…")
def models():
    return backtest(data())


df, bundle = data(), models()
names = list(bundle["results"])
page = ui.sidebar(APP, ICON, ["📡 Live Grid", "🔮 Forecast", "🧪 Model Backtest", "💸 Savings Planner", "📊 Consumption Analytics", "🚨 Anomalies", "👤 Account"])

if page == "📡 Live Grid":
    ui.hero("Live Grid Monitor", "Smart-meter readings stream in every second and are compared to the AI forecast", "📡")
    ss = st.session_state
    ss.setdefault("live_on", True)
    ss.setdefault("live", pd.DataFrame(columns=["t", "load", "solar", "expected", "freq"]))
    ss.setdefault("rng", np.random.default_rng())
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button("⏸ Pause" if ss.live_on else "▶ Resume", type="primary", width="stretch", on_click=lambda: ss.update(live_on=not ss.live_on))
    sens = c2.slider("Alert sensitivity (σ)", 1.5, 4.0, 2.5, 0.5)
    last_week = df.tail(24 * 7)
    profile = last_week.groupby(last_week.index.hour)[["load_kw", "solar_kw"]].mean()

    @st.fragment(run_every=1.0 if ss.live_on else None)
    def live():
        rng = ss.rng
        if ss.live_on:
            now = pd.Timestamp.now()
            hr = now.hour + now.minute / 60
            h0, h1 = int(hr) % 24, (int(hr) + 1) % 24
            w = hr - int(hr)
            exp_load = profile.load_kw[h0] * (1 - w) + profile.load_kw[h1] * w
            exp_solar = profile.solar_kw[h0] * (1 - w) + profile.solar_kw[h1] * w
            spike = rng.random() < 0.05
            load = exp_load * (1 + rng.normal(0, 0.025)) * (rng.choice([1.35, 0.7]) if spike else 1)
            solar = max(0, exp_solar * (1 + rng.normal(0, 0.08)))
            row = pd.DataFrame([{"t": now, "load": load, "solar": solar, "expected": exp_load, "freq": 50 + rng.normal(0, 0.03)}])
            ss.live = pd.concat([ss.live, row]).tail(90)
        live_df = ss.live
        if live_df.empty:
            st.info("Waiting for meter data…"); return
        cur = live_df.iloc[-1]
        resid = live_df.load - live_df.expected
        z = (cur.load - cur.expected) / (resid.std() + 1e-6) if len(live_df) > 10 else 0
        price = float(tariff(np.array([pd.Timestamp.now().hour]))[0])
        k = st.columns(5)
        ui.kpi(k[0], "Load now", f"{cur.load:,.0f} kW", "🏭", f"{abs(cur.load / cur.expected - 1) * 100:.1f}% vs forecast", cur.load >= cur.expected)
        ui.kpi(k[1], "Solar generation", f"{cur.solar:,.0f} kW", "☀️")
        ui.kpi(k[2], "Net grid import", f"{cur.load - cur.solar:,.0f} kW", "🔌")
        ui.kpi(k[3], f"Tariff · {'peak' if price > 9 else 'off-peak' if price < 5 else 'shoulder'}", f"₹{price:.1f}", "💰")
        ui.kpi(k[4], "Grid frequency", f"{cur.freq:.2f}Hz", "〰️")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_df.t, y=live_df.expected, name="AI expected", line=dict(dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=live_df.t, y=live_df.load, name="Actual load", line=dict(width=3, shape="spline"), fill="tonexty", fillcolor="rgba(124,58,237,.12)"))
        fig.add_trace(go.Scatter(x=live_df.t, y=live_df.solar, name="Solar", line=dict(width=2, color="#f59e0b"), fill="tozeroy", fillcolor="rgba(245,158,11,.10)"))
        outl = live_df[(resid.abs() > sens * (resid.std() + 1e-6))] if len(live_df) > 10 else live_df.iloc[0:0]
        fig.add_trace(go.Scatter(x=outl.t, y=outl.load, mode="markers", name="Anomaly", marker=dict(size=13, color="#ef4444", symbol="x")))
        fig.update_layout(title="Real-time load vs forecast", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        status = '<span class="pulse"></span><b>LIVE</b>' if ss.live_on else "⏸ Paused"
        alert = f"<span class='badge red'>⚠️ Deviation {z:+.1f}σ — check feeders</span>" if abs(z) > sens else "<span class='badge green'>All feeders normal</span>"
        st.markdown(f"<div class='card'>{status} · {len(live_df)} readings · {alert} · Carbon intensity now <b>{carbon(np.array([pd.Timestamp.now().hour]))[0]:.0f} g/kWh</b></div>", unsafe_allow_html=True)

    live()

elif page == "🔮 Forecast":
    ui.hero("Load Forecast", "Recursive multi-step forecasting with 80 % confidence bands and weather what-if", "🔮")
    c = st.columns(3)
    hours = c[0].select_slider("Horizon", [24, 48, 72, 96, 168], 48, format_func=lambda h: f"{h} h ({h // 24} d)")
    model = c[1].selectbox("Model", names, index=names.index(bundle["best"]))
    shift = c[2].slider("🌡️ Temperature what-if (°C)", -5.0, 5.0, 0.0, 0.5)
    with st.spinner("Forecasting…"):
        fc = forecast(df, bundle, model, hours, shift)
        base = forecast(df, bundle, model, hours, 0.0) if shift else fc
    hist = df.load_kw.tail(24 * 3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(width=2, color="#94a3b8")))
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.upper) + list(fc.lower[::-1]), fill="toself", fillcolor="rgba(124,58,237,.18)", line=dict(width=0), name="80% band"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.forecast, name="Forecast", line=dict(width=3)))
    if shift:
        fig.add_trace(go.Scatter(x=base.index, y=base.forecast, name="Baseline weather", line=dict(width=2, dash="dot")))
    fig.update_layout(height=430, title=f"{hours}-hour load forecast · {model}", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    peak_t = fc.forecast.idxmax()
    cost = (fc.forecast * tariff(fc.index.hour.values)).sum()
    k = st.columns(4)
    ui.kpi(k[0], "Forecast energy", f"{fc.forecast.sum() / 1000:,.1f} MWh", "🔋")
    ui.kpi(k[1], "Expected peak", f"{fc.forecast.max():,.0f} kW", "📈")
    ui.kpi(k[2], "Peak at", peak_t.strftime("%a %H:%M"), "⏰")
    ui.kpi(k[3], "Estimated cost", f"₹{cost / 1e5:,.2f} L", "💸", f"{abs(fc.forecast.sum() / base.forecast.sum() - 1) * 100:.1f}% vs baseline" if shift else None, shift >= 0)
    st.download_button("⬇️ Download forecast CSV", fc.round(1).to_csv(), "gridsense_forecast.csv", "text/csv")
    auth.log("forecast", f"{hours}h {model} Δ{shift}°C")

elif page == "🧪 Model Backtest":
    ui.hero("Model Backtest", "Every model trained on history and evaluated on the last 30 days it never saw", "🧪")
    res = bundle["results"]
    t = pd.DataFrame({n: {"MAE (kW)": r["mae"], "MAPE (%)": r["mape"], "R²": r["r2"]} for n, r in res.items()}).T
    st.dataframe(t.style.format("{:.3f}").highlight_min(subset=["MAE (kW)", "MAPE (%)"], color="#065f46").highlight_max(subset=["R²"], color="#065f46"), width="stretch")
    st.markdown(f"🏆 Best model: **{bundle['best']}**")
    days = st.slider("Days to show", 3, 30, 10)
    y = bundle["y_test"].tail(days * 24)
    fig = go.Figure(go.Scatter(x=y.index, y=y, name="Actual", line=dict(width=3, color="#94a3b8")))
    for n, r in res.items():
        fig.add_trace(go.Scatter(x=y.index, y=r["pred"].tail(days * 24), name=n, line=dict(width=2)))
    fig.update_layout(height=420, title="Actual vs predicted", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    resid = bundle["y_test"] - res[bundle["best"]]["pred"]
    c1.plotly_chart(px.histogram(resid, nbins=60, title="Residual distribution (best model)"), width="stretch")
    imp = bundle["importance"]
    c2.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Permutation feature importance"), width="stretch")

elif page == "💸 Savings Planner":
    ui.hero("Savings Planner", "Shift flexible loads to cheaper, greener hours and see the impact", "💸")
    st.markdown("Configure flexible appliances — the planner finds the cheapest start time for each one using the time-of-day tariff and grid carbon intensity.")
    defaults = pd.DataFrame([
        {"appliance": "EV charging", "kW": 7.2, "hours": 4, "current_start": 19},
        {"appliance": "Washing machine", "kW": 2.0, "hours": 2, "current_start": 18},
        {"appliance": "Water pump", "kW": 1.5, "hours": 2, "current_start": 20},
        {"appliance": "Dishwasher", "kW": 1.8, "hours": 1, "current_start": 21},
        {"appliance": "Water heater", "kW": 3.0, "hours": 1, "current_start": 7},
    ])
    tbl = st.data_editor(defaults, num_rows="dynamic", width="stretch", column_config={"hours": st.column_config.NumberColumn(min_value=1, max_value=12), "current_start": st.column_config.NumberColumn("current start (h)", min_value=0, max_value=23)})
    goal = st.radio("Optimise for", ["💰 Cost", "🌱 Carbon", "⚖️ Balanced"], index=2, horizontal=True)
    hours = np.arange(24)
    price, co2 = tariff(hours), carbon(hours)
    score = {"💰 Cost": price / price.max(), "🌱 Carbon": co2 / co2.max(), "⚖️ Balanced": price / price.max() + co2 / co2.max()}[goal]
    rows = []
    for _, r in tbl.dropna().iterrows():
        dur = int(r.hours)
        win = lambda s: [(s + i) % 24 for i in range(dur)]
        best = min(range(24), key=lambda s: score[win(s)].sum())
        cc, bc = price[win(int(r.current_start))].sum() * r.kW, price[win(best)].sum() * r.kW
        ce, be = co2[win(int(r.current_start))].sum() * r.kW / 1000, co2[win(best)].sum() * r.kW / 1000
        rows.append({"appliance": r.appliance, "current": f"{int(r.current_start):02d}:00", "suggested": f"{best:02d}:00", "daily ₹ now": cc, "daily ₹ optimised": bc, "₹ saved / month": (cc - bc) * 30, "kg CO₂ saved / month": (ce - be) * 30})
    res = pd.DataFrame(rows)
    k = st.columns(3)
    ui.kpi(k[0], "Monthly savings", f"₹{res['₹ saved / month'].sum():,.0f}", "💰")
    ui.kpi(k[1], "Yearly savings", f"₹{res['₹ saved / month'].sum() * 12:,.0f}", "📅")
    ui.kpi(k[2], "CO₂ avoided / year", f"{res['kg CO₂ saved / month'].sum() * 12:,.0f} kg", "🌱")
    st.dataframe(res.round(1), width="stretch", hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=price, name="Tariff ₹/kWh", marker_color=["#ef4444" if p > 9 else "#f59e0b" if p > 5 else "#10b981" for p in price]))
    fig.add_trace(go.Scatter(x=hours, y=co2, name="Carbon g/kWh", yaxis="y2", line=dict(width=3)))
    fig.update_layout(title="Tariff & carbon intensity by hour", yaxis2=dict(overlaying="y", side="right"), height=340)
    st.plotly_chart(fig, width="stretch")

elif page == "📊 Consumption Analytics":
    ui.hero("Consumption Analytics", "Two years of smart-meter history", "📊")
    k = st.columns(4)
    ui.kpi(k[0], "Total consumption", f"{df.load_kw.sum() / 1e6:,.2f} GWh", "🔋")
    ui.kpi(k[1], "Solar share", f"{df.solar_kw.sum() / df.load_kw.sum() * 100:.1f}%", "☀️")
    ui.kpi(k[2], "All-time peak", f"{df.load_kw.max():,.0f} kW", "📈")
    yoy = df.load_kw.tail(24 * 365).mean() / df.load_kw.head(24 * 365).mean() - 1
    ui.kpi(k[3], "Year-on-year growth", f"{yoy * 100:+.1f}%", "📆", None)
    c1, c2 = st.columns(2)
    hm = df.tail(24 * 90).pivot_table(index=df.tail(24 * 90).index.dayofweek, columns=df.tail(24 * 90).index.hour, values="load_kw")
    hm.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    c1.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="Plasma", title="Average load heatmap · weekday × hour (90 days)", labels=dict(x="hour", y="", color="kW")), width="stretch")
    s = df.sample(4000, random_state=0)
    c2.plotly_chart(px.scatter(s, x="temp_c", y="load_kw", color=s.index.hour, opacity=0.6, title="Temperature drives cooling load", labels={"color": "hour"}), width="stretch")
    m = df.resample("MS")[["load_kw", "solar_kw"]].sum() / 1000
    st.plotly_chart(px.bar(m, barmode="group", title="Monthly energy (MWh): consumption vs solar"), width="stretch")

elif page == "🚨 Anomalies":
    ui.hero("Anomaly Detection", "Isolation Forest on forecast residuals flags unusual consumption", "🚨")
    a = anomalies(df, bundle)
    flagged = a[a.flag]
    k = st.columns(3)
    ui.kpi(k[0], "Hours analysed", f"{len(a):,}", "⏱️")
    ui.kpi(k[1], "Anomalies found", len(flagged), "🚨")
    ui.kpi(k[2], "Largest deviation", f"{a.resid.abs().max():,.0f} kW", "📏")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.index, y=a.load, name="Actual", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=a.index, y=a.pred, name="Expected", line=dict(width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=flagged.index, y=flagged.load, mode="markers", name="Anomaly", marker=dict(size=11, color="#ef4444", line=dict(width=2, color="white"))))
    fig.update_layout(height=420, title="Last 30 days with detected anomalies", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(flagged.assign(direction=np.where(flagged.resid > 0, "⬆️ surge", "⬇️ drop")).round(1)[["load", "pred", "resid", "direction"]].sort_index(ascending=False), width="stretch")
else:
    ui.account_page()
