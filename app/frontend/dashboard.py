from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
import httpx

from src.config import load_config, project_path
from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.features.transformers import prepare_frame
from src.viz.plots import (
    correlation_heatmap,
    daily_trend,
    exam_vs_normal,
    missingness_bar,
    occupancy_by_hour,
    occupancy_by_weekday,
    occupancy_distribution,
    weekday_vs_weekend,
)

st.set_page_config(page_title="Smart Library Intelligence", layout="wide", page_icon="📚")

cfg = load_config()
API = f"http://{cfg['api']['host']}:{cfg['api']['port']}"


@st.cache_data(show_spinner=False)
def load_frame() -> pd.DataFrame:
    return prepare_frame(clean_occupancy(load_raw()))


def api_get(path: str, params: dict | None = None):
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def read_json(key: str):
    path = project_path(cfg["paths"][key])
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    h1 {letter-spacing: -0.03em;}
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Pages",
    ["Overview", "Forecast", "Analytics", "Resources", "Explainability", "Model Performance"],
)
st.sidebar.caption("Smart Library Occupancy & Resource Intelligence")
st.sidebar.code(f"API {API}", language=None)

frame = load_frame()
libs = sorted(frame["library_id"].unique())
lib_names = frame.drop_duplicates("library_id").set_index("library_id")["library_name"].to_dict()

if page == "Overview":
    st.title("Library occupancy overview")
    st.caption("Latest observed occupancy across libraries and zones, with live model scores when the API is up.")
    status = api_get("/status")
    if "locations" in status:
        loc = pd.DataFrame(status["locations"])
        st.caption(f"As of {status.get('as_of')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Locations", len(loc))
        c2.metric("Mean utilization", f"{loc['predicted_utilization'].mean():.0%}")
        c3.metric("Seats occupied", int(loc["occupied_seats"].sum()))
        c4.metric("Seats available", int(loc["available_seats"].sum()))
        st.dataframe(loc, use_container_width=True, hide_index=True)
        fig = px.bar(
            loc,
            x="zone_name",
            y="predicted_utilization",
            color="library_name",
            title="Predicted utilization by zone",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"API unavailable ({status.get('error')}). Showing processed data instead.")
        latest = frame[frame["timestamp"] == frame["timestamp"].max()]
        st.dataframe(latest, use_container_width=True, hide_index=True)

elif page == "Forecast":
    st.title("Multi-horizon occupancy forecast")
    horizon = st.selectbox("Horizon (hours)", [1, 3, 6], index=1)
    lib = st.selectbox("Library", libs, format_func=lambda x: lib_names.get(x, x))
    zones = sorted(frame.loc[frame["library_id"] == lib, "zone_id"].unique())
    zone = st.selectbox("Zone", zones)
    data = api_get("/forecast", {"horizon": horizon, "library_id": lib, "zone_id": zone})
    if "forecasts" in data:
        fc = pd.DataFrame(data["forecasts"])
        st.dataframe(fc, use_container_width=True, hide_index=True)
        fig = px.line(fc, x="timestamp", y="predicted_occupied", markers=True, title=f"{horizon}h forecast")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(data.get("error", "Forecast failed. Train the pipeline and start the API."))

elif page == "Analytics":
    st.title("Exploratory analytics")
    st.plotly_chart(daily_trend(frame), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(occupancy_by_hour(frame), use_container_width=True)
    b.plotly_chart(occupancy_by_weekday(frame), use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(weekday_vs_weekend(frame), use_container_width=True)
    d.plotly_chart(exam_vs_normal(frame), use_container_width=True)
    st.plotly_chart(occupancy_distribution(frame), use_container_width=True)
    st.plotly_chart(correlation_heatmap(frame), use_container_width=True)
    raw = load_raw()
    st.plotly_chart(missingness_bar(raw), use_container_width=True)

elif page == "Resources":
    st.title("Resource recommendations")
    st.caption("When a zone is near capacity, alternatives are ranked by remaining seats, preferring the same library.")
    lib = st.selectbox("Library", libs, format_func=lambda x: lib_names.get(x, x), key="rlib")
    zones = sorted(frame.loc[frame["library_id"] == lib, "zone_id"].unique())
    zone = st.selectbox("Zone", zones, key="rzone")
    horizon = st.selectbox("Horizon", [1, 3, 6], key="rhor")
    data = api_get("/recommend", {"library_id": lib, "zone_id": zone, "horizon": horizon})
    if "recommendations" in data:
        recs = data["recommendations"]
        if recs:
            payload = recs[0]
            st.metric("Predicted utilization", f"{payload.get('utilization', 0):.0%}")
            st.write(payload.get("reason"))
            alts = payload.get("alternatives") or []
            if alts:
                st.dataframe(pd.DataFrame(alts), use_container_width=True, hide_index=True)
            else:
                st.success("No diversion needed at this horizon.")
    else:
        st.error(data.get("error", "Recommendation service unavailable."))

elif page == "Explainability":
    st.title("Why the model predicted this occupancy")
    shap_summary = read_json("shap_path")
    if shap_summary and "mean_abs_shap" in shap_summary:
        imp = pd.Series(shap_summary["mean_abs_shap"]).reset_index()
        imp.columns = ["feature", "mean_abs_shap"]
        fig = px.bar(imp.head(15), x="mean_abs_shap", y="feature", orientation="h", title="Global mean |SHAP|")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Base value {shap_summary.get('base_value'):.2f} · {shap_summary.get('n_samples')} sample rows")
    lib = st.selectbox("Library", libs, format_func=lambda x: lib_names.get(x, x), key="elib")
    zones = sorted(frame.loc[frame["library_id"] == lib, "zone_id"].unique())
    zone = st.selectbox("Zone", zones, key="ezone")
    expl = api_get("/explain", {"library_id": lib, "zone_id": zone})
    if "contributions" in expl:
        st.subheader("Per-prediction contributions")
        st.metric("Prediction", f"{expl['prediction']:.1f} seats")
        contrib = pd.Series(expl["contributions"]).reset_index()
        contrib.columns = ["feature", "shap"]
        fig = px.bar(contrib.head(12), x="shap", y="feature", orientation="h", title="SHAP contributions")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    elif expl.get("error"):
        st.info("Local explanation needs a trained tree model and a running API.")

else:
    st.title("Model performance")
    metrics = read_json("metrics_path")
    ablation = read_json("ablation_path")
    errors = read_json("error_analysis_path")
    params = read_json("best_params_path")
    if not metrics:
        st.warning("No metrics yet. Run `python -m src.pipeline.train`.")
    else:
        st.subheader("Regression")
        st.caption(f"Best model: **{metrics.get('best_regressor')}**")
        st.dataframe(pd.DataFrame(metrics["regression"]).T, use_container_width=True)
        st.subheader("Baselines")
        st.dataframe(pd.DataFrame(metrics["baselines"]).T, use_container_width=True)
        st.subheader("Peak classification")
        st.json(metrics["classification"])
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["levels"], columns=metrics["levels"])
        fig = px.imshow(cm, text_auto=True, title="Confusion matrix", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Chronological split · train {metrics['n_train']:,} · val {metrics['n_val']:,} · test {metrics['n_test']:,}"
        )
    if ablation:
        st.subheader("Feature ablation (A→D)")
        st.dataframe(pd.DataFrame(ablation).T, use_container_width=True)
    if params:
        st.subheader("Tuned hyperparameters")
        st.json(params)
    if errors:
        st.subheader("Error analysis")
        st.write("By weekday")
        st.json(errors.get("by_weekday", {}))
        st.write("Exam vs normal")
        st.json(errors.get("by_regime", {}))
        hour_err = pd.Series(errors.get("by_hour", {})).reset_index()
        hour_err.columns = ["hour", "mae"]
        if len(hour_err):
            hour_err["hour"] = hour_err["hour"].astype(int)
            st.plotly_chart(px.bar(hour_err, x="hour", y="mae", title="MAE by hour"), use_container_width=True)
