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


    # ── Cell 1: add to existing controls ─────────────────────────────────────────
    ALL_MEETINGS = [f"Meeting_{i}" for i in range(1, 17)]  # full range



    # ─────────────────────────────────────────────────────────────────────────────
    #  PRE-COMPUTE  (use dict-based DATASETS, not DATASETS_DF)
    # ─────────────────────────────────────────────────────────────────────────────
    SENT  = {k: compute_sentiment(*v) for k, v in DATASETS.items()}
    ZONES = {k: compute_zones(*v)     for k, v in DATASETS.items()}
    ZONES_PCT = {k: compute_zone_pct(*v) for k, v in DATASETS.items()}
    return SENT, ZONES, compute_sentiment, compute_zones


@app.cell
def _(SENT, go, pd):
    def fig_heatmap(sent: pd.DataFrame, title: str = "") -> go.Figure:
        # sent = SENT[title]
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

    meeting_range = mo.ui.range_slider(
        start=1,
        stop=16,
        step=1,
        value=[1, 12],
        label="Meeting range",
        show_value=True,
    )

    mo.hstack([dataset_selector, chart_type, share_type, adj_scale, meeting_range], gap="2rem")
    return adj_scale, chart_type, dataset_selector, meeting_range, share_type


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
    return (val,)


@app.cell
def _(adj_scale, chart_type, dataset_selector, go, mo, pd, share_type):
    def fig_zone_bars(zones: pd.DataFrame, title: str = "",
                      use_share: bool = False,
                      fix_scale: bool = False) -> go.Figure:
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
        totals  = zones.groupby("person")["trips"].sum()

        fig = go.Figure()
        for z in ZONE_ORDER:
            sub = zones[zones["zone"] == z]
            if use_share:
                vals  = [round(100 * int(sub.loc[sub["person"]==p, "trips"].values[0]) / totals[p], 1)
                         if p in sub["person"].values else 0 for p in persons]
                hover = "<b>%{x}</b><br>zone: " + z + "<br>%{y:.1f}%<extra></extra>"
            else:
                vals  = [int(sub.loc[sub["person"]==p, "trips"].values[0])
                         if p in sub["person"].values else 0 for p in persons]
                hover = "<b>%{x}</b><br>zone: " + z + "<br>waypoints: %{y}<extra></extra>"
            fig.add_trace(go.Bar(
                name=z, x=persons, y=vals,
                marker_color=ZONE_COLORS.get(z, "#94a3b8"), opacity=0.88,
                hovertemplate=hover,
            ))

        fig.update_layout(
            barmode="stack",
            title=dict(
                text=f"Travel zones ({'% share' if use_share else 'waypoints'}) — {title}",
                font=dict(size=13, color="black"), x=0.01,
            ),
            height=300,
            margin=dict(l=10, r=10, t=44, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="black")),
            yaxis=dict(
                range=[0, 100] if use_share else ([0, 350] if fix_scale else None),
                zeroline=False, gridcolor="#dddddd",
                ticksuffix="%" if use_share else "",
                title=dict(text="% of trips" if use_share else "trip waypoints",
                           font=dict(size=10, color="black")),
                tickfont=dict(color="black"),
            ),
            xaxis=dict(tickfont=dict(color="black")),
        )
        return fig
    mo.hstack([dataset_selector, chart_type, share_type,adj_scale], gap="2rem")
    return (fig_zone_bars,)


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
def _(chart_type, dataset_selector, go, mo, pd, share_type):
    def fig_bias_score(nodes: dict, links: list, title: str = "",) -> go.Figure:
        """
        Bias score = avg tourism sentiment − avg fishing sentiment.
        Error bars show std of individual sentiment observations per person.
        """
        # nodes, links = DATASETS[title]
    
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
def _(chart_type, dataset_selector, go, mo, pd, share_type):
    def fig_sentiment_scatter(sent, title: str = "") -> go.Figure:
        """
        X = avg fishing sentiment, Y = avg tourism sentiment.
        Top-left quadrant = pro-tourism / anti-fishing (supports FILAH claim).
        Bottom-right = pro-fishing / anti-tourism (supports TROUT claim).
        """
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


@app.cell
def _(
    DATASETS,
    compute_sentiment,
    compute_zones,
    dataset_selector,
    meeting_range,
):
    # ── Cell 2: filtered data — reacts to dataset + meeting range ────────────────
    _ds = dataset_selector.value
    _m_from, _m_to = meeting_range.value

    _nodes, _links = DATASETS[_ds]

    # meetings active in range
    _active_meetings = {
        f"Meeting_{i}" for i in range(_m_from, _m_to + 1)
        if f"Meeting_{i}" in _nodes
    }

    # discussion/plan ids belonging to active meetings
    _active_items = set()
    for _l in _links:
        if _l.get("role") != "part_of":
            continue
        _src, _tgt = _l["source"], _l["target"]
        if _src in _active_meetings:
            _active_items.add(_tgt)
        if _tgt in _active_meetings:
            _active_items.add(_src)

    # filtered links: keep participant edges from active items + all non-participant links
    _links_filtered = [
        _l for _l in _links
        if _l.get("role") != "participant" or _l.get("source") in _active_items
    ]

    # recompute SENT and ZONES from filtered links — feeds ALL charts
    sent_filtered  = compute_sentiment(_nodes, _links_filtered)
    zones_filtered = compute_zones(_nodes, _links_filtered)
    return


@app.cell
def _(defaultdict, go, pd):
    """
    Advanced COOTEFOO visualizations — add these cells to your Marimo notebook.
    Requires: DATASETS, SENT already defined in earlier cells.
    """

    # ═════════════════════════════════════════════════════════════════════════════
    #  SHARED HELPERS  (put in a single cell)
    # ═════════════════════════════════════════════════════════════════════════════

    TOPIC_INDUSTRY = {
        "expanding_tourist_wharf":  "tourism",
        "marine_life_deck":         "tourism",
        "seafood_festival":         "tourism",
        "heritage_walking_tour":    "tourism",
        "waterfront_market":        "tourism",
        "deep_fishing_dock":        "fishing",
        "new_crane_lomark":         "fishing",
        "fish_vacuum":              "fishing",
        "low_volume_crane":         "fishing",
        "affordable_housing":       "fishing",
        "statue_john_smoth":        "neutral",
        "renaming_park_himark":     "neutral",
        "name_harbor_area":         "neutral",
        "name_inspection_office":   "neutral",
        "concert":                  "neutral",
    }

    IND_COLOR = {"tourism": "#0ea5e9", "fishing": "#f97316", "neutral": "#94a3b8"}
    PERSON_COLORS = {
        "Carol Limpet":   "#f59e0b",
        "Ed Helpsford":   "#10b981",
        "Seal":           "#6366f1",
        "Simone Kat":     "#ec4899",
        "Tante Titan":    "#8b5cf6",
        "Teddy Goldstein":"#ef4444",
    }
    STATUS_COLORS = {
        "completed":   "#4ade80",
        "Completed":   "#4ade80",
        "planned":     "#60a5fa",
        "in_progress": "#fbbf24",
        "introduced":  "#a78bfa",
        "none":        "#e5e7eb",
    }
    _LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="black")),
        margin=dict(l=10, r=10, t=50, b=10),
    )


    # ═════════════════════════════════════════════════════════════════════════════
    #  1.  ORG vs PERSON SENTIMENT
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_org_vs_person(nodes: dict, links: list, title: str = "") -> go.Figure:
        """
        Scatter: each dot is an (actor, industry) pair.
        Persons = circles, Orgs = diamonds.
        Shows that organisations hold more extreme views than committee members.
        """
        rows = []
        for l in links:
            if l.get("role") != "participant" or l.get("sentiment") is None:
                continue
            tgt = l["target"]
            tgt_node = nodes.get(tgt, {})
            actor_type = tgt_node.get("type", "")
            if actor_type not in ("entity.person", "entity.organization"):
                continue
            inds = l.get("industry", [])
            if isinstance(inds, str):
                inds = [inds]
            for ind in (inds or []):
                rows.append({
                    "actor":      tgt,
                    "actor_type": "Person" if actor_type == "entity.person" else "Organisation",
                    "industry":   ind,
                    "sentiment":  float(l["sentiment"]),
                    "reason":     l.get("reason", ""),
                })

        if not rows:
            return go.Figure()

        df = (pd.DataFrame(rows)
                .groupby(["actor", "actor_type", "industry"])
                .agg(avg_sentiment=("sentiment", "mean"),
                     n=("sentiment", "count"),
                     std=("sentiment", "std"))
                .reset_index()
                .fillna({"std": 0}))

        fig = go.Figure()
        for actor_type, symbol in [("Person", "circle"), ("Organisation", "diamond")]:
            sub = df[df["actor_type"] == actor_type]
            for ind in ["tourism", "large vessel", "small vessel"]:
                s = sub[sub["industry"] == ind]
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=s["avg_sentiment"],
                    y=s["actor"],
                    mode="markers",
                    name=f"{actor_type} · {ind}",
                    marker=dict(
                        symbol=symbol,
                        size=12 + s["n"] * 1.5,
                        color=IND_COLOR.get(ind, "#94a3b8"),
                        opacity=0.85,
                        line=dict(color="white", width=1),
                    ),
                    customdata=s[["n", "std", "industry"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "industry: %{customdata[2]}<br>"
                        "avg sentiment: %{x:.2f}<br>"
                        "n=%{customdata[0]:.0f}  std=%{customdata[1]:.2f}"
                        "<extra></extra>"
                    ),
                ))

        fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa", line_width=1.5)
        fig.add_vrect(x0=-1.05, x1=-0.75, fillcolor="#f87171", opacity=0.05, line_width=0)
        fig.add_vrect(x0=0.75,  x1=1.05,  fillcolor="#4ade80", opacity=0.05, line_width=0)

        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"Org vs Person sentiment — {title}  ·  orgs hold more extreme views",
                       font=dict(size=13, color="black"), x=0.01),
            height=420,
            xaxis=dict(range=[-1.2, 1.2], zeroline=False, gridcolor="#dddddd",
                       title=dict(text="← pro-fishing      neutral      pro-tourism →",
                                  font=dict(size=10, color="black")),
                       tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
        )
        return fig


    # ═════════════════════════════════════════════════════════════════════════════
    #  2.  PLAN STATUS BY TOPIC
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_plan_status(nodes: dict, links: list, title: str = "") -> go.Figure:
        """
        Stacked bar: for each topic, how many plans reached each status.
        Reveals which industry's proposals actually got completed.
        """
        # map plan → topic
        plan_topic = {}
        for l in links:
            if l.get("role") == "plan":
                src, tgt = l["source"], l["target"]
                if nodes.get(src, {}).get("type") == "plan":
                    plan_topic[src] = tgt

        # map discussion/plan → status (from 'about' edges)
        item_status = {}
        for l in links:
            if l.get("role") == "about" and l.get("status"):
                item_status[l["source"]] = l["status"]

        # build rows
        rows = []
        for plan_id, topic_id in plan_topic.items():
            topic_node = nodes.get(topic_id, {})
            short = topic_node.get("short_topic", str(topic_id))
            status = item_status.get(plan_id, "none")
            rows.append({
                "topic":    short,
                "status":   status,
                "industry": TOPIC_INDUSTRY.get(short, "neutral"),
            })

        if not rows:
            return go.Figure()

        df = pd.DataFrame(rows)
        pivot = (df.groupby(["topic", "status"])
                   .size()
                   .unstack(fill_value=0)
                   .reset_index())

        # sort topics by industry
        topic_order = (df[["topic", "industry"]]
                       .drop_duplicates()
                       .sort_values(["industry", "topic"])["topic"]
                       .tolist())

        status_order = ["completed", "Completed", "in_progress", "planned", "introduced", "none"]
        fig = go.Figure()
        for status in status_order:
            if status not in pivot.columns:
                continue
            vals = [int(pivot.loc[pivot["topic"]==t, status].values[0])
                    if t in pivot["topic"].values else 0 for t in topic_order]
            colors = [IND_COLOR[TOPIC_INDUSTRY.get(t, "neutral")] for t in topic_order]
            fig.add_trace(go.Bar(
                name=status,
                x=topic_order, y=vals,
                marker_color=STATUS_COLORS.get(status, "#e5e7eb"),
                opacity=0.88,
                hovertemplate="<b>%{x}</b><br>status: " + status + "<br>plans: %{y}<extra></extra>",
            ))

        # add industry colour band underneath via annotation
        fig.update_layout(
            **_LAYOUT,
            barmode="stack",
            title=dict(text=f"Plan status by topic — {title}  ·  what actually got done?",
                       font=dict(size=13, color="black"), x=0.01),
            height=380,
            xaxis=dict(tickfont=dict(color="black"), tickangle=-35),
            yaxis=dict(zeroline=False, gridcolor="#dddddd",
                       title=dict(text="number of plans", font=dict(size=10, color="black")),
                       tickfont=dict(color="black")),
        )
        return fig


    # ═════════════════════════════════════════════════════════════════════════════
    #  3.  CO-PARTICIPATION MATRIX
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_coparticipation(nodes: dict, links: list, title: str = "") -> go.Figure:
        """
        Heatmap: how many discussions two people attended together.
        Reveals alliances, coalitions and who avoids whom.
        """
        persons = sorted(n["id"] for n in nodes.values()
                         if n.get("type") == "entity.person")

        disc_parts = defaultdict(set)
        for l in links:
            if l.get("role") == "participant":
                tgt = l["target"]
                if nodes.get(tgt, {}).get("type") == "entity.person":
                    disc_parts[l["source"]].add(tgt)

        co = defaultdict(int)
        for parts in disc_parts.values():
            parts = list(parts)
            for i in range(len(parts)):
                for j in range(len(parts)):
                    co[(parts[i], parts[j])] += 1

        z = [[co.get((p1, p2), 0) for p2 in persons] for p1 in persons]
        text = [[str(co.get((p1, p2), 0)) for p2 in persons] for p1 in persons]

        fig = go.Figure(go.Heatmap(
            z=z, x=persons, y=persons,
            colorscale=[[0, "#f0f0f0"], [1, "#7c6af7"]],
            text=text, texttemplate="%{text}",
            textfont=dict(color="black", size=11),
            colorbar=dict(
                title=dict(text="shared discussions", font=dict(size=10, color="black")),
                tickfont=dict(size=9, color="black"),
                thickness=12,
            ),
            hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>shared discussions: %{z}<extra></extra>",
        ))
        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"Co-participation matrix — {title}  ·  who appears in discussions together?",
                       font=dict(size=13, color="black"), x=0.01),
            height=380,
            xaxis=dict(tickfont=dict(color="black"), tickangle=-30),
            yaxis=dict(tickfont=dict(color="black"), autorange="reversed"),
        )
        return fig


    # ═════════════════════════════════════════════════════════════════════════════
    #  4.  MEETING AGENDA TIMELINE
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_meeting_timeline(nodes: dict, links: list, title: str = "") -> go.Figure:
        """
        Heatmap: meetings (x) × topics (y), coloured by industry.
        Shows how the agenda shifted from fishing-heavy → tourism-heavy over time.
        """
        meet_disc, disc_topic = defaultdict(set), {}
        for l in links:
            src, tgt, role = l["source"], l["target"], l.get("role")
            if role == "part_of":
                if nodes.get(src, {}).get("type") == "meeting":
                    meet_disc[src].add(tgt)
                if nodes.get(tgt, {}).get("type") == "meeting":
                    meet_disc[tgt].add(src)
            elif role == "about":
                if (nodes.get(src, {}).get("type") == "discussion" and
                        nodes.get(tgt, {}).get("type") == "topic"):
                    disc_topic[src] = tgt

        rows = []
        meetings = sorted(meet_disc.keys(), key=lambda x: int(x.split("_")[1]))
        for meet_id in meetings:
            meet_num = int(meet_id.split("_")[1])
            for disc_id in meet_disc[meet_id]:
                topic_id = disc_topic.get(disc_id)
                if topic_id:
                    topic_node = nodes.get(topic_id, {})
                    short = topic_node.get("short_topic", str(topic_id))
                    rows.append({
                        "meeting": meet_num,
                        "topic":   short,
                        "industry": TOPIC_INDUSTRY.get(short, "neutral"),
                    })

        if not rows:
            return go.Figure()

        df = pd.DataFrame(rows).drop_duplicates()

        # count discussions per meeting×topic
        counts = (df.groupby(["meeting", "topic"])
                    .size()
                    .reset_index(name="n"))

        all_meetings = sorted(df["meeting"].unique())
        topic_order = (df[["topic", "industry"]]
                       .drop_duplicates()
                       .sort_values(["industry", "topic"])["topic"]
                       .tolist())

        pivot = (counts.pivot(index="topic", columns="meeting", values="n")
                       .reindex(index=topic_order, columns=all_meetings)
                       .fillna(0))

        # colour cells by industry
        ind_vals = {"fishing": 1.0, "neutral": 0.5, "tourism": 0.0}
        color_z  = [[ind_vals.get(TOPIC_INDUSTRY.get(t, "neutral"), 0.5)
                     for _ in all_meetings] for t in topic_order]

        text = [[f"{int(pivot.loc[t, m])}" if pivot.loc[t, m] > 0 else ""
                 for m in all_meetings] for t in topic_order]

        fig = go.Figure(go.Heatmap(
            z=color_z,
            x=[f"M{m}" for m in all_meetings],
            y=topic_order,
            colorscale=[[0, "#0ea5e9"], [0.5, "#94a3b8"], [1, "#f97316"]],
            zmin=0, zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="black", size=10),
            showscale=False,
            hovertemplate="<b>%{y}</b><br>Meeting %{x}<br>discussions: %{text}<extra></extra>",
            opacity=0.75,
        ))

        # overlay discussion count as marker size via scatter
        xs, ys, sizes, hovers = [], [], [], []
        for ti, t in enumerate(topic_order):
            for mi, m in enumerate(all_meetings):
                n = int(pivot.loc[t, m])
                if n > 0:
                    xs.append(f"M{m}")
                    ys.append(t)
                    sizes.append(n * 8)
                    hovers.append(f"{t}<br>Meeting {m}: {n} discussion(s)")

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color="rgba(0,0,0,0.25)",
                        line=dict(color="white", width=1)),
            hovertext=hovers, hoverinfo="text",
            showlegend=False,
        ))

        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"Meeting agenda timeline — {title}  ·  🟠 fishing  ·  ⬜ neutral  ·  🔵 tourism",
                       font=dict(size=13, color="black"), x=0.01),
            height=420,
            xaxis=dict(tickfont=dict(color="black"),
                       title=dict(text="meeting", font=dict(size=10, color="black"))),
            yaxis=dict(tickfont=dict(color="black", size=9)),
        )
        return fig


    # ═════════════════════════════════════════════════════════════════════════════
    #  5.  SANKEY: MEETING → TOPIC → PERSON → SENTIMENT BUCKET
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_sankey(nodes: dict, links: list, title: str = "") -> go.Figure:
        """
        Sankey flow: Meeting → Topic → Person → Sentiment bucket (positive/negative/neutral).
        Reveals the full chain: which meetings drove which topics,
        and how each person reacted to them.
        """
        meet_disc, disc_topic = defaultdict(set), {}
        disc_person = []

        for l in links:
            src, tgt, role = l["source"], l["target"], l.get("role")
            if role == "part_of":
                if nodes.get(src, {}).get("type") == "meeting":
                    meet_disc[src].add(tgt)
                if nodes.get(tgt, {}).get("type") == "meeting":
                    meet_disc[tgt].add(src)
            elif role == "about":
                if (nodes.get(src, {}).get("type") == "discussion" and
                        nodes.get(tgt, {}).get("type") == "topic"):
                    disc_topic[src] = tgt
            elif role == "participant" and l.get("sentiment") is not None:
                if nodes.get(tgt, {}).get("type") == "entity.person":
                    disc_person.append({
                        "disc": src, "person": tgt, "sentiment": float(l["sentiment"])
                    })

        # sentiment bucket
        def bucket(s):
            if s >= 0.3:   return "positive"
            if s <= -0.3:  return "negative"
            return "neutral"

        # build flows: meeting → topic → person → bucket
        flows = defaultdict(int)
        for dp in disc_person:
            disc  = dp["disc"]
            person = dp["person"]
            sent_b = bucket(dp["sentiment"])
            topic_id = disc_topic.get(disc)
            if not topic_id:
                continue
            topic_node = nodes.get(topic_id, {})
            topic = topic_node.get("short_topic", str(topic_id))
            # find meeting for this disc
            meeting = None
            for meet_id, discs in meet_disc.items():
                if disc in discs:
                    meeting = f"M{meet_id.split('_')[1]}"
                    break
            if not meeting:
                continue
            flows[(meeting, topic, person, sent_b)] += 1

        if not flows:
            return go.Figure()

        # build node list
        meetings = sorted(set(f[0] for f in flows))
        topics   = sorted(set(f[1] for f in flows))
        persons  = sorted(set(f[2] for f in flows))
        buckets  = ["positive", "neutral", "negative"]

        all_nodes = meetings + topics + persons + buckets
        idx = {n: i for i, n in enumerate(all_nodes)}

        BUCKET_COLOR = {"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"}
        node_colors = (
            ["#818cf8"] * len(meetings) +
            [IND_COLOR.get(TOPIC_INDUSTRY.get(t, "neutral"), "#94a3b8") for t in topics] +
            [PERSON_COLORS.get(p, "#60a5fa") for p in persons] +
            [BUCKET_COLOR[b] for b in buckets]
        )

        src_list, tgt_list, val_list, colors = [], [], [], []

        # aggregate: meeting → topic
        mt_flow = defaultdict(int)
        tp_flow = defaultdict(int)
        pb_flow = defaultdict(int)
        for (meet, topic, person, bkt), cnt in flows.items():
            mt_flow[(meet, topic)] += cnt
            tp_flow[(topic, person)] += cnt
            pb_flow[(person, bkt)] += cnt

        def hex_to_rgba(hex_color, alpha=0.4):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        for (a, b), v in mt_flow.items():
            src_list.append(idx[a]); tgt_list.append(idx[b]); val_list.append(v)
            colors.append(hex_to_rgba("#818cf8"))
        for (a, b), v in tp_flow.items():
            src_list.append(idx[a]); tgt_list.append(idx[b]); val_list.append(v)
            colors.append(hex_to_rgba(IND_COLOR.get(TOPIC_INDUSTRY.get(a, "neutral"), "#94a3b8")))
        for (a, b), v in pb_flow.items():
            src_list.append(idx[a]); tgt_list.append(idx[b]); val_list.append(v)
            colors.append(hex_to_rgba(PERSON_COLORS.get(a, "#60a5fa")))

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=12, thickness=16,
                label=all_nodes,
                color=node_colors,
                hovertemplate="%{label}: %{value} flows<extra></extra>",
            ),
            link=dict(
                source=src_list, target=tgt_list, value=val_list,
                color=colors,
                hovertemplate="from %{source.label} → %{target.label}: %{value}<extra></extra>",
            ),
        ))
        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"Meeting → Topic → Person → Sentiment — {title}",
                       font=dict(size=13, color="black"), x=0.01),
            height=600,
        )
        return fig


    # ═════════════════════════════════════════════════════════════════════════════
    #  6.  GEOGRAPHIC MAP OF TRIPS
    # ═════════════════════════════════════════════════════════════════════════════
    def fig_trip_map(nodes: dict, links: list, title: str = "",
                     selected_person: str = "All") -> go.Figure:
        """
        Scatter map: each waypoint a member visited during official trips.
        Dot size = visit frequency, colour = zone type.
        """
        ZONE_COLORS_MAP = {
            "tourism":     "#0ea5e9",
            "commercial":  "#f97316",
            "industrial":  "#ef4444",
            "government":  "#7c6af7",
            "residential": "#94a3b8",
            "connector":   "#64748b",
        }

        trip_person, trip_places = {}, defaultdict(list)
        for l in links:
            src, tgt = l["source"], l["target"]
            sn, tn = nodes.get(src, {}), nodes.get(tgt, {})
            if sn.get("type") == "trip":
                if tn.get("type") == "entity.person":
                    trip_person[src] = tgt
                elif tn.get("lat") and tn.get("lon"):
                    trip_places[src].append({
                        "label": tn.get("label", "?"),
                        "lat":   tn["lat"],
                        "lon":   tn["lon"],
                        "zone":  tn.get("zone", "unknown"),
                    })

        rows = []
        for trip, person in trip_person.items():
            if selected_person != "All" and person != selected_person:
                continue
            for place in trip_places[trip]:
                rows.append({"person": person, **place})

        if not rows:
            return go.Figure()

        df = pd.DataFrame(rows)
        counts = (df.groupby(["person", "label", "lat", "lon", "zone"])
                    .size()
                    .reset_index(name="visits"))

        fig = go.Figure()
        for zone, color in ZONE_COLORS_MAP.items():
            sub = counts[counts["zone"] == zone]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["lon"], y=sub["lat"],
                mode="markers",
                name=zone,
                marker=dict(
                    size=4 + sub["visits"] * 1.5,
                    color=color,
                    opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                customdata=sub[["person", "label", "visits"]].values,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "zone: " + zone + "<br>"
                    "person: %{customdata[0]}<br>"
                    "visits: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))

        # person labels at centroid
        if selected_person == "All":
            centroids = counts.groupby("person")[["lat", "lon"]].mean().reset_index()
            for _, row in centroids.iterrows():
                fig.add_annotation(
                    x=row["lon"], y=row["lat"],
                    text=row["person"].split()[0],
                    showarrow=False,
                    font=dict(size=9, color=PERSON_COLORS.get(row["person"], "black"),
                              family="monospace"),
                    bgcolor="rgba(255,255,255,0.7)",
                    borderpad=2,
                )

        person_label = f" · {selected_person}" if selected_person != "All" else " · all members"
        fig.update_layout(
            **_LAYOUT,
            title=dict(text=f"Trip waypoints map — {title}{person_label}",
                       font=dict(size=13, color="black"), x=0.01),
            height=440,
            xaxis=dict(title=dict(text="longitude", font=dict(size=10, color="black")),
                       tickfont=dict(color="black"), gridcolor="#eeeeee", zeroline=False),
            yaxis=dict(title=dict(text="latitude",  font=dict(size=10, color="black")),
                       tickfont=dict(color="black"), gridcolor="#eeeeee", zeroline=False,
                       scaleanchor="x", scaleratio=1),
        )
        return fig
    return (
        fig_coparticipation,
        fig_meeting_timeline,
        fig_org_vs_person,
        fig_plan_status,
        fig_sankey,
        fig_trip_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DASHBOARD
    """)
    return


@app.cell
def _(adj_scale, chart_type, dataset_selector, meeting_range, mo, share_type):
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
    mo.hstack([dataset_selector, chart_type, share_type, adj_scale,meeting_range], gap="2rem")
    return


@app.cell
def _(
    DATASETS,
    adj_scale,
    chart_type,
    compute_sentiment,
    compute_zones,
    dataset_selector,
    fig_bias_score,
    fig_heatmap,
    fig_sentiment_bars,
    fig_sentiment_scatter,
    fig_zone_bars,
    meeting_range,
    mo,
    share_type,
):
    # ── Cell 2: filtered data (reacts to dataset + meeting range) ────────────────
    _ds = dataset_selector.value
    _m_from, _m_to = meeting_range.value

    _nodes, _links = DATASETS[_ds]

    # meetings active in range
    _active_meetings = {
        f"Meeting_{i}" for i in range(_m_from, _m_to + 1)
        if f"Meeting_{i}" in _nodes
    }

    # discussion/plan ids belonging to active meetings
    _active_items = set()
    for _l in _links:
        if _l.get("role") != "part_of":
            continue
        if _l["source"] in _active_meetings:
            _active_items.add(_l["target"])
        if _l["target"] in _active_meetings:
            _active_items.add(_l["source"])

    # keep only participant edges from active items
    _links_filtered = [
        _l for _l in _links
        if _l.get("role") != "participant" or _l.get("source") in _active_items
    ]

    # recompute from filtered links — single source of truth for all charts
    _sent_filtered  = compute_sentiment(_nodes, _links_filtered)
    _zones_filtered = compute_zones(_nodes, _links_filtered)


    # ── Cell 3: dashboard ────────────────────────────────────────────────────────
    _ds             = dataset_selector.value
    _share          = share_type.value
    _adj            = adj_scale.value
    _m_from, _m_to  = meeting_range.value

    # ── figures — all use filtered data ──────────────────────────────────────────
    _fig_sent  = (fig_heatmap(_sent_filtered, _ds)
                  if chart_type.value == "Heatmap"
                  else fig_sentiment_bars(_sent_filtered, _ds))
    _fig_scat  = fig_sentiment_scatter(_sent_filtered, _ds)
    _fig_bias  = fig_bias_score(_nodes, _links_filtered, _ds)
    _fig_zones = fig_zone_bars(_zones_filtered, _ds,
                                use_share=_share,
                                fix_scale=_adj)

    # ── info strip ────────────────────────────────────────────────────────────────
    _n_mem   = _sent_filtered["person"].nunique()  if not _sent_filtered.empty  else 0
    _n_rec   = int(_sent_filtered["n"].sum())      if not _sent_filtered.empty  else 0
    _n_trips = int(_zones_filtered["trips"].sum()) if not _zones_filtered.empty else 0

    _info = mo.md(
        f"Dataset **{_ds}** · "
        f"Meetings **{_m_from} → {_m_to}** · "
        f"Members: **{_n_mem}** · "
        f"Records: **{_n_rec}** · "
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
        mo.hstack([dataset_selector, chart_type, share_type, adj_scale, meeting_range],
                  gap=0.5, widths="equal"),
        mo.hstack([mo.ui.plotly(_fig_sent),  mo.ui.plotly(_fig_scat)],  widths="equal", gap=0.5),
        mo.hstack([mo.ui.plotly(_fig_bias),  mo.ui.plotly(_fig_zones)], widths="equal", gap=0.5),
    ], gap=0.5)
    return


@app.cell
def _(ALL_PERSONS, mo):
    # ── widgets for advanced viz ─────────────────────────────────────────────────
    ds_adv = mo.ui.dropdown(
        options=["FILAH", "TROUT", "journalist"],
        value="journalist",
        label="Dataset",
    )
    person_map = mo.ui.dropdown(
        options=["All"] + ALL_PERSONS,
        value="All",
        label="Member (map)",
    )
    mo.hstack([ds_adv, person_map], gap="2rem")
    return ds_adv, person_map


@app.cell
def _(
    DATASETS,
    ds_adv,
    fig_coparticipation,
    fig_meeting_timeline,
    fig_org_vs_person,
    fig_plan_status,
    fig_sankey,
    fig_trip_map,
    mo,
    person_map,
):
    # ── advanced dashboard ────────────────────────────────────────────────────────
    _nodes, _links = DATASETS[ds_adv.value]
    _ds = ds_adv.value

    mo.vstack([
        mo.md("# 🔍 **COOTEFOO — Advanced Analysis**"),
        mo.hstack([ds_adv, person_map], gap="2rem"),

        mo.md("### Meeting → Topic → Person → Sentiment"),
        mo.ui.plotly(fig_sankey(_nodes, _links, _ds)),

        mo.md("### Meeting agenda over time"),
        mo.ui.plotly(fig_meeting_timeline(_nodes, _links, _ds)),

        mo.hstack([
            mo.vstack([mo.md("### Co-participation"),
                       mo.ui.plotly(fig_coparticipation(_nodes, _links, _ds))]),
            mo.vstack([mo.md("### Org vs Person sentiment"),
                       mo.ui.plotly(fig_org_vs_person(_nodes, _links, _ds))]),
        ], widths="equal", gap=0.5),

        mo.md("### Plan status by topic — what got done?"),
        mo.ui.plotly(fig_plan_status(_nodes, _links, _ds)),

        mo.md("### Trip waypoints map"),
        mo.ui.plotly(fig_trip_map(_nodes, _links, _ds, selected_person=person_map.value)),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
