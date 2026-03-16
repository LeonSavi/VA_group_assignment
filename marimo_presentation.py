import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import json
    import altair as alt
    import os


    import pandas as pd 
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from typing import Callable, Union, Literal
    from scipy.stats import entropy
    import desbordante as db
    import itertools
    from tqdm import tqdm #usually this library slow down loops
    import plotly.graph_objects as go

    from collections import defaultdict

    ROOT_DATA = 'data'
    BASE = os.path.dirname(os.path.abspath(__file__))
    return BASE, defaultdict, go, json, mo, np, os, pd


@app.cell
def _(BASE, json, os, pd):
    def load_graph(filename, to_pandas:bool=False):
        with open(os.path.join(BASE, 'data',filename)) as f:
            d = json.load(f)
        nodes = {n["id"]: n for n in d["nodes"]}
        edges = d.get("links", d.get("edges", []))
        if not to_pandas:
            return nodes, edges
        else:
            df_nodes = pd.DataFrame(d["nodes"])
            df_links = pd.DataFrame(d.get("links", d.get("edges", [])))
            return df_nodes,df_links

    DATASETS = {
        "FILAH":      load_graph("FILAH.json"),
        "TROUT":      load_graph("TROUT.json"),
        "journalist": load_graph("journalist.json")
    }

    DATASETS_DF = {
        "FILAH":      load_graph("FILAH.json",True),
        "TROUT":      load_graph("TROUT.json",True),
        "journalist": load_graph("journalist.json",True)
    }

    ALL_PERSONS = sorted(
        n["id"] for n in DATASETS["journalist"][0].values()
        if n.get("type") == "entity.person"
    )
    return ALL_PERSONS, DATASETS


@app.cell
def _(DATASETS, defaultdict, np, pd):
    # ─────────────────────────────────────────────────────────────────────────────
    #  DATA HELPERS  (work from dict-based loader — same as Dash)
    # ─────────────────────────────────────────────────────────────────────────────

    def compute_sentiment(nodes: dict, links: list) -> pd.DataFrame:
        """
        Returns: person | industry | avg_sentiment | n
        Uses the raw dict/list representation to avoid pandas list-in-cell issues.
        """
        rows = []
        for l in links:
            if l.get("role") != "participant":
                continue
            s = l.get("sentiment")
            if s is None or s is np.nan:
                continue
            pid = l.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person":
                continue
            inds = l.get("industry", [])
            if isinstance(inds, str):
                inds = [inds]
            for ind in (inds or []):
                rows.append({"person": pid, "industry": ind, "sentiment": float(s)})

        if not rows:
            return pd.DataFrame(columns=["person", "industry", "avg_sentiment", "n"])

        return (
            pd.DataFrame(rows)
            .groupby(["person", "industry"])["sentiment"]
            .agg(avg_sentiment="mean", n="count")
            .reset_index()
        )


    def compute_zones(nodes: dict, links: list) -> pd.DataFrame:
        """
        Returns: person | zone | trips
        Counts trip waypoints per person per zone.
        """
        trip_person = {}
        trip_places = defaultdict(list)

        for l in links:
            src, tgt = l["source"], l["target"]
            sn = nodes.get(src, {})
            tn = nodes.get(tgt, {})
            if sn.get("type") == "trip":
                if tn.get("type") == "entity.person":
                    trip_person[src] = tgt
                elif tn.get("zone"):
                    trip_places[src].append(tn["zone"])

        rows = [
            {"person": p, "zone": z}
            for trip, p in trip_person.items()
            for z in trip_places[trip]
        ]

        if not rows:
            return pd.DataFrame(columns=["person", "zone", "trips"])

        return (
            pd.DataFrame(rows)
            .groupby(["person", "zone"])
            .size()
            .reset_index(name="trips")
        )


    def compute_zone_pct(nodes: dict, links: list) -> pd.DataFrame:
        """
        Returns: person | zone | trips | pct
        Adds a % column so heatmap and per-person charts don't recompute it.
        """
        df = compute_zones(nodes, links)
        if df.empty:
            return df
        totals = df.groupby("person")["trips"].transform("sum")
        df["pct"] = (100 * df["trips"] / totals).round(1)
        return df


    # ─────────────────────────────────────────────────────────────────────────────
    #  PRE-COMPUTE  (use dict-based DATASETS, not DATASETS_DF)
    # ─────────────────────────────────────────────────────────────────────────────
    SENT  = {k: compute_sentiment(*v) for k, v in DATASETS.items()}
    ZONES = {k: compute_zones(*v)     for k, v in DATASETS.items()}
    ZONES_PCT = {k: compute_zone_pct(*v) for k, v in DATASETS.items()}
    return SENT, ZONES


@app.cell
def _(SENT, go):
    def fig_heatmap(title: str = "") -> go.Figure:
        sent = SENT[title]
        if sent.empty:
            return go.Figure()

        col_order = [c for c in ["tourism", "small vessel", "large vessel"] if c in sent["industry"].values]
        pivot = (sent
                 .pivot(index="person", columns="industry", values="avg_sentiment")
                 .reindex(columns=col_order)
                 .fillna(0)
                 .iloc[::-1])

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "#f87171"], [0.5, "#f5f5f0"], [1, "#4ade80"]],
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:+.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(color="black", size=11),   # ← dark text, always readable
            colorbar=dict(
                title=dict(text="sentiment", font=dict(size=10, color="black")),
                tickfont=dict(size=9, color="black"),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "+0.5", "+1"],
                thickness=12,
            ),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=f"Average sentiment by industry  — {title}",
                font=dict(size=13, color="black"),
                x=0.01,
            ),
            height=260,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            font=dict(color="black"),
        )
        return fig


    def fig_sentiment_bars(title: str = "") -> go.Figure:
        sent = SENT[title]
        if sent.empty:
            return go.Figure()

        INDUSTRY_COLORS = {
            "tourism":      "#0ea5e9",
            "large vessel": "#f97316",
            "small vessel": "#fb923c",
        }

        persons    = sorted(sent["person"].unique())
        industries = sorted(sent["industry"].unique())

        fig = go.Figure()
        for ind in industries:
            sub  = sent[sent["industry"] == ind]
            vals = [sub.loc[sub["person"]==p, "avg_sentiment"].values[0]
                    if p in sub["person"].values else None for p in persons]
            ns   = [int(sub.loc[sub["person"]==p, "n"].values[0])
                    if p in sub["person"].values else 0 for p in persons]
            fig.add_trace(go.Bar(
                name=ind,
                x=persons, y=vals,
                marker_color=INDUSTRY_COLORS.get(ind, "#94a3b8"),
                opacity=0.88,
                customdata=ns,
                hovertemplate="<b>%{x}</b><br>" + ind + "<br>avg=%{y:.2f}  n=%{customdata}<extra></extra>",
            ))

        fig.add_hline(y=0, line_color="#aaaaaa", line_width=1.2)
        fig.update_layout(
            barmode="group",
            title=dict(
                text=f"Average sentiment by industry - {title}",
                font=dict(size=13, color="black"),
                x=0.01,
            ),
            height=280,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="black")),
            yaxis=dict(
                range=[-1.25, 1.45],
                zeroline=False,
                gridcolor="#dddddd",
                title=dict(text="avg sentiment", font=dict(size=10, color="black")),
                tickfont=dict(color="black"),
            ),
            xaxis=dict(tickfont=dict(color="black")),
        )
        return fig
    return fig_heatmap, fig_sentiment_bars


@app.cell
def _(mo):
    # Cell 1 — widgets
    dataset_selector = mo.ui.dropdown(
        options=["FILAH", "TROUT", "journalist"],
        value="FILAH",
        label="Dataset"
    )

    chart_type = mo.ui.radio(
        options=["Heatmap", "Bar chart"],
        value="Heatmap",
        label="Chart type"
    )

    share_type = mo.ui.radio(
        options={"% share": True, "raw counts": False},
        value="% share",
        label="Share of Total",
    )

    adj_scale = mo.ui.radio(
        options={"Fix Scale": True, "Float Scale": False},
        value="Float Scale",
        label="Adjust Scale",
    )

    mo.hstack([dataset_selector, chart_type, share_type, adj_scale], gap="2rem")
    return adj_scale, chart_type, dataset_selector, share_type


@app.cell
def _(
    adj_scale,
    chart_type,
    dataset_selector,
    fig_heatmap,
    fig_sentiment_bars,
    mo,
    share_type,
):
    val = dataset_selector.value
    share_ = share_type.value
    adj_scale_ = adj_scale.value

    fig = fig_heatmap(val) if chart_type.value == "Heatmap" else fig_sentiment_bars(val)
    mo.ui.plotly(fig)
    return adj_scale_, share_, val


@app.cell
def _(DATASETS, defaultdict, pd):
    def compute_zones(nodes: dict, links: list) -> pd.DataFrame:
        trip_person, trip_places = {}, defaultdict(list)
        for l in links:
            src, tgt = l["source"], l["target"]
            sn, tn = nodes.get(src, {}), nodes.get(tgt, {})
            if sn.get("type") == "trip":
                if tn.get("type") == "entity.person":
                    trip_person[src] = tgt
                elif tn.get("zone"):
                    trip_places[src].append(tn["zone"])
        rows = [{"person": p, "zone": z}
                for trip, p in trip_person.items()
                for z in trip_places[trip]]
        if not rows:
            return pd.DataFrame(columns=["person", "zone", "trips"])
        return pd.DataFrame(rows).groupby(["person","zone"]).size().reset_index(name="trips")

    ZONES = {k: compute_zones(*v) for k, v in DATASETS.items()}
    
    return (ZONES,)


@app.cell
def _(ZONES, adj_scale, chart_type, dataset_selector, go, mo, share_type):
    def fig_zone_bars(title: str = "", share: bool = False, adj_scale:bool = False) -> go.Figure:
        zones = ZONES[title]
        if zones.empty:
            return go.Figure()

        ZONE_COLORS = {
            "tourism":     "#0ea5e9",
            "commercial":  "#f97316",
            "industrial":  "#fb923c",
            "government":  "#7c6af7",
            "residential": "#94a3b8",
            "connector":   "#64748b",
        }
        ZONE_ORDER = ["tourism", "commercial", "industrial", "government", "residential", "connector"]
        persons = sorted(zones["person"].unique())

        # compute totals per person for % mode
        totals = zones.groupby("person")["trips"].sum()

        fig = go.Figure()
        for z in ZONE_ORDER:
            sub = zones[zones["zone"] == z]
            if share:
                vals = [
                    round(100 * int(sub.loc[sub["person"]==p, "trips"].values[0]) / totals[p], 1)
                    if p in sub["person"].values else 0 for p in persons
                ]
                hover = "<b>%{x}</b><br>zone: " + z + "<br>%{y:.1f}%<extra></extra>"
            else:
                vals = [
                    int(sub.loc[sub["person"]==p, "trips"].values[0])
                    if p in sub["person"].values else 0 for p in persons
                ]
                hover = "<b>%{x}</b><br>zone: " + z + "<br>waypoints: %{y}<extra></extra>"

            fig.add_trace(go.Bar(
                name=z,
                x=persons, y=vals,
                marker_color=ZONE_COLORS.get(z, "#94a3b8"),
                opacity=0.88,
                hovertemplate=hover,
            ))

        fig.update_layout(
            barmode="stack",
            title=dict(
                text=f"Travel zones ({'% share' if share else 'waypoints'}) - {title}",
                font=dict(size=13, color="black"),
                x=0.01,
            ),
            height=300,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="black")),
            yaxis=dict(
                range=[0, 100] if share else ([0,350] if adj_scale else None),
                zeroline=False,
                gridcolor="#dddddd",
                ticksuffix="%" if share else "",
                title=dict(
                    text="% of trips" if share else "trip waypoints",
                    font=dict(size=10, color="black")
                ),
                tickfont=dict(color="black"),
            ),
            xaxis=dict(tickfont=dict(color="black")),
        )
        return fig
    mo.hstack([dataset_selector, chart_type, share_type,adj_scale], gap="2rem")
    return (fig_zone_bars,)


@app.cell
def _(adj_scale_, fig_zone_bars, share_, val):
    fig_zone_bars(val,share_,adj_scale_)
    return


@app.cell
def _(DATASETS, go, pd, val):
    def fig_sentiment_spread(title: str = "") -> go.Figure:
        """Shows min, mean, max sentiment per person per industry — reveals consistency of bias."""
        nodes, links = DATASETS[title]
        rows = []
        for l in links:
            if l.get("role") != "participant" or l.get("sentiment") is None: continue
            pid = l.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person": continue
            inds = l.get("industry", [])
            if isinstance(inds, str): inds = [inds]
            for ind in (inds or []):
                rows.append({"person": pid, "industry": ind, "sentiment": float(l["sentiment"])})

        if not rows:
            return go.Figure()

        df = pd.DataFrame(rows)
        stats = (df.groupby(["person", "industry"])["sentiment"]
                   .agg(mean="mean", min="min", max="max", n="count")
                   .reset_index())

        INDUSTRY_COLORS = {"tourism": "#0ea5e9", "large vessel": "#f97316", "small vessel": "#fb923c"}
        industries = sorted(stats["industry"].unique())
        persons = sorted(stats["person"].unique())

        fig = go.Figure()
        for ind in industries:
            sub = stats[stats["industry"] == ind]
            color = INDUSTRY_COLORS.get(ind, "#94a3b8")

            # range line (min → max)
            for _, row in sub.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["min"], row["max"]],
                    y=[f"{row['person']} · {ind}"] * 2,
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # mean dot
            fig.add_trace(go.Scatter(
                x=sub["mean"],
                y=[f"{row['person']} · {ind}" for _, row in sub.iterrows()],
                mode="markers",
                name=ind,
                marker=dict(color=color, size=10, line=dict(color="white", width=1.5)),
                customdata=sub[["min", "max", "n"]].values,
                hovertemplate="<b>%{y}</b><br>mean=%{x:.2f}<br>min=%{customdata[0]:.2f}  max=%{customdata[1]:.2f}  n=%{customdata[2]}<extra></extra>",
            ))

        fig.add_vline(x=0, line_color="#aaaaaa", line_width=1.2, line_dash="dash")
        fig.update_layout(
            title=dict(text=f"Sentiment spread (min/mean/max) — {title}", font=dict(size=13, color="black"), x=0.01),
            height=max(300, 40 * len(stats)),
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(range=[-1.2, 1.2], zeroline=False, gridcolor="#dddddd", tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black"), gridcolor="#dddddd"),
        )
        return fig

    fig_sentiment_spread(val)
    return


@app.cell
def _(DATASETS, chart_type, dataset_selector, go, mo, pd, share_type):
    def fig_bias_score(title: str = "") -> go.Figure:
        """
        Bias score = avg tourism sentiment − avg fishing sentiment.
        Error bars show std of individual sentiment observations per person.
        """
        nodes, links = DATASETS[title]
        fishing_inds = ["large vessel", "small vessel"]

        # ── raw rows (needed for std) ─────────────────────────────────────────────
        rows = []
        for l in links:
            if l.get("role") != "participant" or l.get("sentiment") is None:
                continue
            pid = l.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person":
                continue
            inds = l.get("industry", [])
            if isinstance(inds, str): inds = [inds]
            for ind in (inds or []):
                rows.append({"person": pid, "industry": ind, "sentiment": float(l["sentiment"])})

        if not rows:
            return go.Figure()

        raw = pd.DataFrame(rows)

        # ── per-person stats ──────────────────────────────────────────────────────
        result = []
        for person in raw["person"].unique():
            sub = raw[raw["person"] == person]
            t = sub.loc[sub["industry"] == "tourism",       "sentiment"]
            f = sub.loc[sub["industry"].isin(fishing_inds), "sentiment"]
            if t.empty or f.empty:
                continue
            result.append({
                "person":    person,
                "bias":      round(t.mean() - f.mean(), 3),
                "t_avg":     round(t.mean(), 3),
                "f_avg":     round(f.mean(), 3),
                "t_std":     round(t.std(ddof=1) if len(t) > 1 else 0.0, 3),
                "f_std":     round(f.std(ddof=1) if len(f) > 1 else 0.0, 3),
                # combined std via error propagation (tourism − fishing)
                "bias_std":  round((
                                 (t.std(ddof=1)**2 / len(t)) +
                                 (f.std(ddof=1)**2 / len(f))
                             ) ** 0.5 if len(t) > 1 and len(f) > 1 else 0.0, 3),
                "t_n": len(t),
                "f_n": len(f),
            })

        if not result:
            return go.Figure()

        df = pd.DataFrame(result).sort_values("bias")
        colors = ["#f87171" if v < 0 else "#4ade80" for v in df["bias"]]

        fig = go.Figure(go.Bar(
            x=df["bias"],
            y=df["person"],
            orientation="h",
            marker_color=colors,
            opacity=0.85,
            error_x=dict(
                type="data",
                array=df["bias_std"],
                color="#555555",
                thickness=1.5,
                width=6,
            ),
            customdata=df[["t_avg", "f_avg", "t_std", "f_std", "t_n", "f_n"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "bias = %{x:+.2f} ± %{error_x.array:.2f}<br>"
                "tourism:  avg=%{customdata[0]:.2f}  std=%{customdata[2]:.2f}  n=%{customdata[4]}<br>"
                "fishing:  avg=%{customdata[1]:.2f}  std=%{customdata[3]:.2f}  n=%{customdata[5]}"
                "<extra></extra>"
            ),
        ))

        fig.add_vline(x=0, line_color="#aaaaaa", line_width=1.5, line_dash="dash")
        fig.update_layout(
            title=dict(
                text=f"Bias score ± std  (tourism − fishing avg) — {title}",
                font=dict(size=13, color="black"), x=0.01,
            ),
            height=max(280, 50 * len(df)),
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            xaxis=dict(
                range=[-1.5, 1.5], zeroline=False, gridcolor="#dddddd",
                title=dict(text="← pro-fishing      pro-tourism →",
                           font=dict(size=10, color="black")),
                tickfont=dict(color="black"),
            ),
            yaxis=dict(tickfont=dict(color="black")),
        )
        return fig

    mo.hstack([dataset_selector, chart_type, share_type], gap="2rem")
    return (fig_bias_score,)


@app.cell
def _(dataset_selector, fig_bias_score):
    fig_bias_score(dataset_selector.value)
    return


@app.cell
def _(SENT, chart_type, dataset_selector, go, mo, pd, share_type):
    def fig_sentiment_scatter(title: str = "") -> go.Figure:
        """
        X = avg fishing sentiment, Y = avg tourism sentiment.
        Top-left quadrant = pro-tourism / anti-fishing (supports FILAH claim).
        Bottom-right = pro-fishing / anti-tourism (supports TROUT claim).
        """
        sent = SENT[title]
        if sent.empty:
            return go.Figure()

        fishing_inds = ["large vessel", "small vessel"]
        rows = []
        for person in sent["person"].unique():
            sub = sent[sent["person"] == person]
            t = sub.loc[sub["industry"] == "tourism", "avg_sentiment"]
            f = sub.loc[sub["industry"].isin(fishing_inds), "avg_sentiment"]
            rows.append({
                "person": person,
                "tourism":  float(t.mean()) if not t.empty else None,
                "fishing":  float(f.mean()) if not f.empty else None,
            })

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return go.Figure()

        fig = go.Figure()

        # quadrant shading
        fig.add_hrect(y0=0, y1=1.1,  fillcolor="#0ea5e9", opacity=0.04, line_width=0)
        fig.add_hrect(y0=-1.1, y1=0, fillcolor="#f97316", opacity=0.04, line_width=0)
        fig.add_vrect(x0=0, x1=1.1,  fillcolor="#f97316", opacity=0.04, line_width=0)
        fig.add_vrect(x0=-1.1, x1=0, fillcolor="#0ea5e9", opacity=0.04, line_width=0)

        # diagonal neutral line
        fig.add_shape(type="line", x0=-1.1, y0=-1.1, x1=1.1, y1=1.1,
                      line=dict(color="#aaaaaa", width=1, dash="dash"))

        fig.add_shape(type="line", x0=-1.1, y0=1.1, x1=1.1, y1=-1.1,
                      line=dict(color="#aaaaaa", width=1, dash="dash"))

        fig.add_trace(go.Scatter(
            x=df["fishing"], y=df["tourism"],
            mode="markers+text",
            text=df["person"],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(size=14, color=df["tourism"] - df["fishing"],
                        colorscale=[[0, "#f87171"], [0.5, "#e2e8f0"], [1, "#4ade80"]],
                        cmin=-2, cmax=2,
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{text}</b><br>fishing avg=%{x:.2f}<br>tourism avg=%{y:.2f}<extra></extra>",
            showlegend=False,
        ))

        fig.add_vline(x=0, line_color="#aaaaaa", line_width=1, line_dash="dot")
        fig.add_hline(y=0, line_color="#aaaaaa", line_width=1, line_dash="dot")

        # quadrant labels
        for (x, y, text) in [
            ( 0.7,  0.95, "pro-tourism<br>pro-fishing"),
            (-0.7,  0.95, "pro-tourism<br>anti-fishing"),
            (-0.7, -0.95, "anti-tourism<br>anti-fishing"),
            ( 0.7, -0.95, "anti-tourism<br>pro-fishing"),
        ]:
            fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                               font=dict(size=8, color="#aaaaaa"), align="center")

        fig.update_layout(
            title=dict(text=f"Tourism vs fishing sentiment — {title}", font=dict(size=13, color="black"), x=0.01),
            height=380,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            xaxis=dict(range=[-1.2, 1.2], zeroline=False, gridcolor="#dddddd",
                       title=dict(text="avg fishing sentiment", font=dict(size=10, color="black")),
                       tickfont=dict(color="black")),
            yaxis=dict(range=[-1.2, 1.2], zeroline=False, gridcolor="#dddddd",
                       title=dict(text="avg tourism sentiment", font=dict(size=10, color="black")),
                       tickfont=dict(color="black")),
        )
        return fig
    mo.hstack([dataset_selector, chart_type, share_type], gap="2rem")
    return (fig_sentiment_scatter,)


@app.cell
def _(fig_sentiment_scatter, val):
    fig_sentiment_scatter(val)
    return


@app.cell
def _(DATASETS, ZONES, go, pd):
    # ── SHARED CONSTANTS ──────────────────────────────────────────────────────────
    ZONE_ORDER  = ["tourism", "commercial", "industrial", "government", "residential", "connector"]
    ZONE_COLORS = {
        "tourism":     "#0ea5e9",
        "commercial":  "#f97316",
        "industrial":  "#fb923c",
        "government":  "#7c6af7",
        "residential": "#94a3b8",
        "connector":   "#64748b",
    }
    DS_COLORS = {"FILAH": "#f97316", "TROUT": "#0ea5e9", "journalist": "#7c6af7"}
    LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="black")),
        margin=dict(l=10, r=10, t=44, b=10),
    )

    def _title(text): 
        return dict(text=text, font=dict(size=13, color="black"), x=0.01)

    def _yaxis(label):
        return dict(zeroline=False, gridcolor="#dddddd",
                    title=dict(text=label, font=dict(size=10, color="black")),
                    tickfont=dict(color="black"))

    def _xaxis():
        return dict(tickfont=dict(color="black"))


    # ── 1. MEMBER COVERAGE ────────────────────────────────────────────────────────
    def fig_member_coverage() -> go.Figure:
        """
        Grouped bar: how many sentiment participation records each dataset
        holds per member — immediately shows who was selectively included/excluded.
        """
        rows = []
        for ds_name, (nodes, links) in DATASETS.items():
            for l in links:
                if l.get("role") != "participant" or l.get("sentiment") is None:
                    continue
                pid = l.get("target")
                if nodes.get(pid, {}).get("type") != "entity.person":
                    continue
                rows.append({"dataset": ds_name, "person": pid})

        df = pd.DataFrame(rows).groupby(["dataset", "person"]).size().reset_index(name="n")
        persons = sorted(df["person"].unique())

        fig = go.Figure()
        for ds_name, color in DS_COLORS.items():
            sub = df[df["dataset"] == ds_name]
            vals = [int(sub.loc[sub["person"]==p, "n"].values[0])
                    if p in sub["person"].values else 0 for p in persons]
            fig.add_trace(go.Bar(
                name=ds_name, x=persons, y=vals,
                marker_color=color, opacity=0.88,
                hovertemplate="<b>%{x}</b><br>" + ds_name + "<br>records: %{y}<extra></extra>",
            ))

        fig.update_layout(
            **LAYOUT,
            barmode="group",
            height=300,
            title=_title("Member coverage — participation records per dataset"),
            yaxis=_yaxis("participation records"),
            xaxis=_xaxis(),
        )
        return fig


    # ── 2. ZONE % BIAS HEATMAP ────────────────────────────────────────────────────
    def fig_zone_bias_heatmap(title: str = "") -> go.Figure:
        """
        Heatmap: % of each person's trips going to each zone.
        Makes FILAH's commercial bias and TROUT's government bias immediately visible.
        One chart per dataset — pass title = 'FILAH' / 'TROUT' / 'journalist'.
        """
        df = ZONES[title]
        if df.empty:
            return go.Figure()

        persons = sorted(df["person"].unique())
        totals  = df.groupby("person")["trips"].transform("sum")
        df      = df.copy()
        df["pct"] = (100 * df["trips"] / totals).round(1)

        # build pivot: rows = zones, cols = persons
        pivot = (df.pivot(index="zone", columns="person", values="pct")
                   .reindex(index=ZONE_ORDER)
                   .reindex(columns=persons)
                   .fillna(0))

        text = [[f"{v:.0f}%" for v in row] for row in pivot.values]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "#f0f0f0"], [1, "#7c6af7"]],
            zmin=0, zmax=100,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="black", size=11),
            colorbar=dict(
                title=dict(text="% of trips", font=dict(size=10, color="black")),
                tickfont=dict(size=9, color="black"),
                ticksuffix="%",
                thickness=12,
            ),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **LAYOUT,
            height=300,
            title=_title(f"Zone % of trips — {title}"),
            xaxis=dict(tickfont=dict(color="black"), side="bottom"),
            yaxis=dict(tickfont=dict(color="black"), autorange="reversed"),
        )
        return fig


    # ── 3. PER-MEMBER ZONE FRAMING (for marimo dropdown) ─────────────────────────
    def fig_zone_framing(person: str) -> go.Figure:
        """
        100% stacked bar: for a selected person, shows zone % breakdown
        across all three datasets side by side.
        The divergence between FILAH/TROUT/journalist is the key Part 3 story.
        """
        rows = []
        for ds_name in ["FILAH", "TROUT", "journalist"]:
            df  = ZONES[ds_name]
            sub = df[df["person"] == person]
            if sub.empty:
                continue
            tot = sub["trips"].sum()
            for _, row in sub.iterrows():
                rows.append({
                    "dataset": ds_name,
                    "zone":    row["zone"],
                    "pct":     round(100 * row["trips"] / tot, 1),
                    "raw":     int(row["trips"]),
                })

        if not rows:
            return go.Figure()

        df = pd.DataFrame(rows)
        fig = go.Figure()
        for z in ZONE_ORDER:
            sub = df[df["zone"] == z]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                name=z,
                x=sub["dataset"], y=sub["pct"],
                marker_color=ZONE_COLORS.get(z, "#94a3b8"),
                opacity=0.88,
                customdata=sub["raw"],
                hovertemplate=(f"zone: {z}<br>%{{y:.1f}}%"
                               "  (%{customdata} waypoints)<extra></extra>"),
            ))

        fig.update_layout(
            **LAYOUT,
            barmode="stack",
            height=320,
            title=_title(f"Zone framing — {person}  ·  how each dataset tells a different story"),
            yaxis=dict(range=[0, 101], ticksuffix="%", zeroline=False,
                       gridcolor="#dddddd",
                       title=dict(text="% of trips", font=dict(size=10, color="black")),
                       tickfont=dict(color="black")),
            xaxis=_xaxis(),
        )
        return fig
    return fig_member_coverage, fig_zone_bias_heatmap, fig_zone_framing


@app.cell
def _(ALL_PERSONS, mo):
    person_selector = mo.ui.dropdown(
        options=ALL_PERSONS,
        value="Simone Kat",
        label="Select member",
    )
    person_selector
    return (person_selector,)


@app.cell
def _(fig_zone_framing, mo, person_selector):
    # Cell — reactive chart
    mo.ui.plotly(fig_zone_framing(person_selector.value))
    return


@app.cell
def _(chart_type, dataset_selector, mo, share_type):
    mo.hstack([dataset_selector, chart_type, share_type], gap="2rem")
    return


@app.cell
def _(fig_zone_bias_heatmap, val):
    fig_zone_bias_heatmap(val)
    return


@app.cell
def _(fig_member_coverage):
    fig_member_coverage() 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DASHBOARD
    """)
    return


@app.cell
def _(adj_scale, chart_type, dataset_selector, mo, share_type):
    # _dataset_selector = mo.ui.dropdown(
    #     options=["FILAH", "TROUT", "journalist"],
    #     value="FILAH",
    #     label="Dataset",
    # )
    # _chart_type = mo.ui.radio(
    #     options=["Heatmap", "Bar chart"],
    #     value="Heatmap",
    #     label="Sentiment view",
    # )
    # _share_type = mo.ui.radio(
    #     options={"% share": True, "raw counts": False},
    #     value="% share",
    #     label="Zone view",
    # )
    # _adj_scale = mo.ui.radio(
    #     options={"Fix scale": True, "Float scale": False},
    #     value="Float scale",
    #     label="Y-axis scale",
    # )
    mo.hstack([dataset_selector, chart_type, share_type, adj_scale], gap="2rem")
    return


@app.cell
def _(
    SENT,
    ZONES,
    adj_scale,
    chart_type,
    dataset_selector,
    fig_bias_score,
    fig_heatmap,
    fig_sentiment_bars,
    fig_sentiment_scatter,
    fig_zone_bars,
    mo,
    share_type,
):
    # ── Cell 2: reactive dashboard ───────────────────────────────────────────────
    _ds    = dataset_selector.value
    _share = share_type.value
    _adj   = adj_scale.value

    # ── figures ───────────────────────────────────────────────────────────────────
    _fig_sent  = fig_heatmap(_ds) if chart_type.value == "Heatmap" else fig_sentiment_bars(_ds)
    _fig_bias  = fig_bias_score(_ds)
    _fig_zones = fig_zone_bars(_ds, share=_share, adj_scale=_adj)
    _fig_scat  = fig_sentiment_scatter(_ds)

    # ── info strip ────────────────────────────────────────────────────────────────
    _sent_df  = SENT[_ds]
    _n_mem    = _sent_df["person"].nunique() if not _sent_df.empty else 0
    _n_rec    = int(_sent_df["n"].sum())     if not _sent_df.empty else 0
    _n_trips  = int(ZONES[_ds]["trips"].sum()) if not ZONES[_ds].empty else 0

    _info = mo.md(
        f"Dataset **{_ds}** · "
        f"Members recorded: **{_n_mem}** · "
        f"Participation records: **{_n_rec}** · "
        f"Trip waypoints: **{_n_trips}**"
    )

    # ── layout ────────────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md("# ⚖️ **COOTEFOO Bias Analysis — Part 1**"),
        mo.md(
            "---\n\n"
            "Select a **dataset** to explore how FILAH and TROUT recorded member activity. "
            "Toggle between **Heatmap / Bar chart** for sentiment, "
            "**% share / raw counts** for travel zones, "
            "and **fix / float** the y-axis scale for cross-dataset comparison."
        ),
        mo.callout(_info, kind="neutral"),
        mo.hstack([dataset_selector, chart_type, share_type, adj_scale], gap=0.5,
            widths="equal"),
        mo.hstack(
            [mo.ui.plotly(_fig_sent),  mo.ui.plotly(_fig_scat)],
            widths="equal", gap=0.5,
        ),
        mo.hstack(
            [mo.ui.plotly(_fig_bias),  mo.ui.plotly(_fig_zones)],
            widths="equal", gap=0.5,
        ),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
