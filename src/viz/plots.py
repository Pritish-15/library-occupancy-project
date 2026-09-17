"""Reusable Plotly/matplotlib helpers for EDA notebooks and the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def occupancy_by_hour(df: pd.DataFrame):
    g = df.groupby("hour", as_index=False)["occupied_seats"].mean()
    return px.bar(g, x="hour", y="occupied_seats", title="Average occupancy by hour")


def occupancy_by_weekday(df: pd.DataFrame):
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    tmp = df.copy()
    if "weekday" not in tmp.columns:
        tmp["weekday"] = pd.to_datetime(tmp["timestamp"]).dt.weekday
    g = tmp.groupby("weekday", as_index=False)["occupied_seats"].mean()
    g["weekday_name"] = g["weekday"].map(names)
    return px.bar(g, x="weekday_name", y="occupied_seats", title="Average occupancy by weekday")


def weekday_vs_weekend(df: pd.DataFrame):
    tmp = df.copy()
    if "is_weekend" not in tmp.columns:
        tmp["is_weekend"] = pd.to_datetime(tmp["timestamp"]).dt.weekday.ge(5).astype(int)
    tmp["period"] = tmp["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    g = tmp.groupby("period", as_index=False)["occupied_seats"].mean()
    return px.bar(g, x="period", y="occupied_seats", title="Weekday vs weekend occupancy")


def exam_vs_normal(df: pd.DataFrame):
    tmp = df.copy()
    tmp["period"] = tmp["is_exam_period"].map({1: "Exam", 0: "Normal"})
    g = tmp.groupby("period", as_index=False)["occupied_seats"].mean()
    return px.bar(g, x="period", y="occupied_seats", title="Exam period vs normal occupancy")


def occupancy_distribution(df: pd.DataFrame):
    return px.histogram(df, x="occupied_seats", nbins=40, title="Occupancy distribution")


def correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None):
    use = cols or [
        "occupied_seats",
        "computer_usage",
        "study_room_usage",
        "book_demand",
        "charging_station_usage",
        "hour",
        "is_exam_period",
    ]
    use = [c for c in use if c in df.columns]
    corr = df[use].corr()
    return px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature correlations")


def missingness_bar(df: pd.DataFrame):
    miss = df.isna().mean().reset_index()
    miss.columns = ["column", "missing_rate"]
    return px.bar(miss, x="column", y="missing_rate", title="Missingness by column")


def daily_trend(df: pd.DataFrame):
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
    g = tmp.groupby("date", as_index=False)["occupied_seats"].mean()
    return px.line(g, x="date", y="occupied_seats", title="Daily mean occupancy")
