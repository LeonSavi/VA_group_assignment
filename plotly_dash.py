"""
COOTEFOO Bias Analysis Dashboard
Run: python cootefoo_dash.py
Then open: http://127.0.0.1:8050

Place this file in the same folder as:
  FILAH.json, TROUT.json, journalist.json
"""

import json, os
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#080c14",
    "surface":  "#0e1420",
    "card":     "#131929",
    "border":   "#1e2d45",
    "tourism":  "#22d3ee",   # cyan
    "fishing":  "#fb923c",   # amber-orange
    "neutral":  "#94a3b8",
    "positive": "#4ade80",
    "negative": "#f87171",
    "accent":   "#7c6af7",   # violet
    "text":     "#e2e8f0",
    "muted":    "#64748b",
    "FILAH":    "#fb923c",
    "TROUT":    "#22d3ee",
    "journalist": "#7c6af7",
}

INDUSTRY_COLORS = {
    "tourism":      C["tourism"],
    "large vessel": C["fishing"],
    "small vessel": "#f97316",
}

ZONE_COLORS = {
    "tourism":     C["tourism"],
    "commercial":  "#fb923c",
    "industrial":  "#f97316",
    "government":  C["accent"],
    "residential": "#94a3b8",
    "connector":   "#334155",
}

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_graph(filename):
    with open(os.path.join(BASE, 'data',filename)) as f:
        d = json.load(f)
    nodes = {n["id"]: n for n in d["nodes"]}
    links = d.get("links", d.get("edges", []))
    return nodes, links

DATASETS = {
    "FILAH":      load_graph("FILAH.json"),
    "TROUT":      load_graph("TROUT.json"),
    "journalist": load_graph("journalist.json"),
}

def sentiment_df(nodes, links):
    rows = []
    for l in links:
        if l.get("role") != "participant":
            continue
        s = l.get("sentiment")
        if s is None:
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
    df = pd.DataFrame(rows)
    return (df.groupby(["person", "industry"])["sentiment"]
              .agg(avg_sentiment="mean", n="count")
              .reset_index())

def zone_df(nodes, links):
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

# Pre-compute all frames
sent = {k: sentiment_df(*v) for k, v in DATASETS.items()}
zones = {k: zone_df(*v) for k, v in DATASETS.items()}

ALL_PERSONS = sorted(
    n["id"] for n in DATASETS["journalist"][0].values()
    if n.get("type") == "entity.person"
)
INDUSTRIES = ["tourism", "large vessel", "small vessel"]
ZONE_ORDER  = ["tourism", "commercial", "industrial", "government", "residential", "connector"]

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor=C["card"],
    plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="'DM Mono', monospace"),
    margin=dict(l=16, r=16, t=36, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
)

def apply_base(fig, height=300, title=None):
    kw = dict(**LAYOUT_BASE, height=height)
    if title:
        kw["title"] = dict(text=title, font=dict(color=C["muted"], size=12), x=0.01)
    fig.update_layout(**kw)
    return fig


def fig_heatmap(ds_name):
    df = sent[ds_name]
    if df.empty:
        return go.Figure()
    pivot = df.pivot(index="person", columns="industry", values="avg_sentiment").fillna(0)
    colorscale = [
        [0.0, C["fishing"]],
        [0.5, "#1a2035"],
        [1.0, C["tourism"]],
    ]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=colorscale,
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(
            title=dict(text="sentiment", font=dict(color=C["muted"], size=10)),
            tickfont=dict(color=C["muted"], size=9),
            thickness=10,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(tickfont=dict(color=C["text"]))
    return apply_base(fig, height=220,
                      title=f"{ds_name}  ·  avg sentiment by industry")


def fig_sentiment_bars(ds_name):
    df = sent[ds_name]
    if df.empty:
        return go.Figure()
    persons = sorted(df["person"].unique())
    fig = go.Figure()
    for ind in INDUSTRIES:
        sub = df[df["industry"] == ind]
        vals = [sub.loc[sub["person"]==p, "avg_sentiment"].values[0]
                if p in sub["person"].values else None for p in persons]
        counts = [int(sub.loc[sub["person"]==p, "n"].values[0])
                  if p in sub["person"].values else 0 for p in persons]
        fig.add_trace(go.Bar(
            name=ind,
            x=persons, y=vals,
            marker_color=INDUSTRY_COLORS.get(ind, C["neutral"]),
            opacity=0.88,
            customdata=counts,
            hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>avg=%{y:.2f}  n=%{customdata}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(range=[-1.25, 1.45], zeroline=False, gridcolor=C["border"],
                     title=dict(text="avg sentiment", font=dict(color=C["muted"], size=10)))
    return apply_base(fig.update_layout(barmode="group"), height=300,
                      title=f"{ds_name}  ·  per-member sentiment by industry")


def fig_zone_bars(ds_name):
    df = zones[ds_name]
    if df.empty:
        return go.Figure()
    persons = sorted(df["person"].unique())
    fig = go.Figure()
    for z in ZONE_ORDER:
        sub = df[df["zone"] == z]
        vals = [int(sub.loc[sub["person"]==p, "trips"].values[0])
                if p in sub["person"].values else 0 for p in persons]
        fig.add_trace(go.Bar(
            name=z, x=persons, y=vals,
            marker_color=ZONE_COLORS.get(z, C["neutral"]),
            opacity=0.88,
            hovertemplate="<b>%{x}</b><br>zone: " + z + "<br>waypoints: %{y}<extra></extra>",
        ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(zeroline=False, gridcolor=C["border"],
                     title=dict(text="trip waypoints", font=dict(color=C["muted"], size=10)))
    return apply_base(fig.update_layout(barmode="stack"), height=300,
                      title=f"{ds_name}  ·  travel zones recorded")


# ── NOTE on journalist data ───────────────────────────────────────────────────
# The journalist graph is the UNION of FILAH + TROUT + extra records.
# Sentiment values for shared members are IDENTICAL across datasets — the
# journalist adds no new meeting/discussion records for existing members.
# What the journalist DOES add:
#   1. Tante Titan (entirely absent from both subsets)
#   2. ~10x more trip records — especially government-zone travel that neither
#      FILAH nor TROUT captured, which radically changes the travel profile.
# Part 3 therefore focuses on:
#   • Who each dataset chose to INCLUDE vs EXCLUDE  (member coverage)
#   • The TRIP VOLUME gap between subset and full data  (zone bias)
#   • Tante Titan's profile (the hidden member)
# ─────────────────────────────────────────────────────────────────────────────

def fig_member_coverage():
    """Stacked bar: how many sentiment records does each dataset hold per member."""
    rows = []
    for ds_name in ["FILAH", "TROUT", "journalist"]:
        nodes, links = DATASETS[ds_name]
        for l in links:
            if l.get("role") != "participant" or l.get("sentiment") is None:
                continue
            pid = l.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person":
                continue
            rows.append({"dataset": ds_name, "person": pid})
    df = pd.DataFrame(rows).groupby(["dataset","person"]).size().reset_index(name="n")

    fig = go.Figure()
    for ds_name, color in [("FILAH", C["FILAH"]), ("TROUT", C["TROUT"]), ("journalist", C["journalist"])]:
        sub = df[df["dataset"] == ds_name]
        fig.add_trace(go.Bar(
            name=ds_name, x=sub["person"], y=sub["n"],
            marker_color=color, opacity=0.88,
            hovertemplate="<b>%{x}</b><br>" + ds_name + "<br>records: %{y}<extra></extra>",
        ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(zeroline=False, gridcolor=C["border"],
                     title=dict(text="participation records", font=dict(color=C["muted"], size=10)))
    return apply_base(fig.update_layout(barmode="group"), height=320,
                      title="member coverage  ·  who did each dataset choose to include?")


def fig_trip_volume_gap():
    """For each person: total trip waypoints in each dataset — shows the massive gap."""
    rows = []
    for ds_name in ["FILAH", "TROUT", "journalist"]:
        df = zones[ds_name]
        if df.empty:
            continue
        totals = df.groupby("person")["trips"].sum().reset_index()
        totals["dataset"] = ds_name
        rows.append(totals)
    df = pd.concat(rows)

    fig = go.Figure()
    for ds_name, color in [("FILAH", C["FILAH"]), ("TROUT", C["TROUT"]), ("journalist", C["journalist"])]:
        sub = df[df["dataset"] == ds_name]
        fig.add_trace(go.Bar(
            name=ds_name, x=sub["person"], y=sub["trips"],
            marker_color=color, opacity=0.88,
            hovertemplate="<b>%{x}</b><br>" + ds_name + "<br>total waypoints: %{y}<extra></extra>",
        ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(zeroline=False, gridcolor=C["border"],
                     title=dict(text="total trip waypoints", font=dict(color=C["muted"], size=10)))
    return apply_base(fig.update_layout(barmode="group"), height=320,
                      title="trip volume gap  ·  how many trips did each dataset record per member?")


def fig_zone_bias_heatmap(ds_name):
    """Heatmap: % of each person's trips going to each zone — reveals selection bias."""
    df = zones[ds_name]
    if df.empty:
        return go.Figure()
    persons = sorted(df["person"].unique())
    totals = df.groupby("person")["trips"].sum()
    pivot_pct = {}
    for z in ZONE_ORDER:
        sub = df[df["zone"] == z]
        pivot_pct[z] = []
        for p in persons:
            tot = totals.get(p, 0)
            val = sub.loc[sub["person"]==p, "trips"].values[0] if p in sub["person"].values else 0
            pivot_pct[z].append(round(100*val/tot, 1) if tot else 0)

    fig = go.Figure(go.Heatmap(
        z=[pivot_pct[z] for z in ZONE_ORDER],
        x=persons,
        y=ZONE_ORDER,
        colorscale=[[0,"#0f1a2e"],[0.5,"#7c6af7"],[1,"#22d3ee"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in pivot_pct[z]] for z in ZONE_ORDER],
        texttemplate="%{text}",
        colorbar=dict(
            title=dict(text="% of trips", font=dict(color=C["muted"], size=10)),
            tickfont=dict(color=C["muted"], size=9),
            thickness=10,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(tickfont=dict(color=C["text"]))
    return apply_base(fig, height=260,
                      title=f"{ds_name}  ·  % of trips per zone (zone bias heatmap)")


def fig_tante_titan():
    """Tante Titan is only in journalist — show her full profile."""
    person = "Tante Titan"
    # Sentiment
    df_s = sent["journalist"]
    sub_s = df_s[df_s["person"] == person]

    # Zones
    df_z = zones["journalist"]
    sub_z = df_z[df_z["person"] == person]

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["sentiment by industry", "trips by zone"],
    )
    # sentiment bars
    colors_s = [INDUSTRY_COLORS.get(i, C["neutral"]) for i in sub_s["industry"]]
    fig.add_trace(go.Bar(
        x=sub_s["industry"], y=sub_s["avg_sentiment"],
        marker_color=colors_s, opacity=0.88,
        customdata=sub_s["n"],
        hovertemplate="%{x}<br>avg=%{y:.2f}  n=%{customdata}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # zone bars
    colors_z = [ZONE_COLORS.get(z, C["neutral"]) for z in sub_z["zone"]]
    fig.add_trace(go.Bar(
        x=sub_z["zone"], y=sub_z["trips"],
        marker_color=colors_z, opacity=0.88,
        hovertemplate="%{x}<br>waypoints=%{y}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_color=C["border"], line_width=1, row=1, col=1)
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(gridcolor=C["border"], zeroline=False, tickfont=dict(color=C["text"]))
    fig.update_annotations(font=dict(color=C["muted"], size=11))
    return apply_base(fig, height=300,
                      title="Tante Titan  ·  the hidden member — only visible in journalist data")


def fig_zone_bias_per_person(person):
    """
    For a selected person: stacked % breakdown of travel zones per dataset.
    This is where FILAH vs TROUT vs journalist diverge most visibly.
    """
    rows = []
    for ds_name in ["FILAH", "TROUT", "journalist"]:
        df = zones[ds_name]
        sub = df[df["person"] == person]
        if sub.empty:
            continue
        tot = sub["trips"].sum()
        for _, row in sub.iterrows():
            rows.append({
                "dataset": ds_name,
                "zone": row["zone"],
                "pct": 100 * row["trips"] / tot,
                "raw": row["trips"],
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
            marker_color=ZONE_COLORS.get(z, C["neutral"]),
            opacity=0.88,
            customdata=sub["raw"],
            hovertemplate=f"zone: {z}<br>%{{y:.1f}}%  (%{{customdata}} waypoints)<extra></extra>",
        ))
    fig.update_xaxes(tickfont=dict(color=C["text"]))
    fig.update_yaxes(range=[0, 105], zeroline=False, gridcolor=C["border"],
                     title=dict(text="% of trips", font=dict(color=C["muted"], size=10)))
    return apply_base(fig.update_layout(barmode="stack"), height=300,
                      title=f"{person}  ·  zone % breakdown — how each dataset frames this member's travel")


# ─────────────────────────────────────────────────────────────────────────────
#  STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(*children, style=None):
    base = {
        "background": C["card"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "10px",
        "padding": "20px 24px",
        "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def section_title(text, accent=C["accent"]):
    return html.Div([
        html.Div(style={
            "width": "3px", "height": "22px",
            "background": accent,
            "borderRadius": "2px",
            "marginRight": "12px",
            "flexShrink": "0",
        }),
        html.Span(text, style={
            "fontFamily": "'Syne', sans-serif",
            "fontSize": "1.15rem",
            "fontWeight": "700",
            "color": C["text"],
            "letterSpacing": "0.01em",
        }),
    ], style={
        "display": "flex", "alignItems": "center",
        "margin": "36px 0 14px",
    })

def pill(text, color):
    return html.Span(text, style={
        "display": "inline-block",
        "padding": "3px 12px",
        "borderRadius": "9999px",
        "fontSize": "0.72rem",
        "fontWeight": "600",
        "letterSpacing": "0.04em",
        "marginRight": "8px",
        "background": f"{color}22",
        "color": color,
        "border": f"1px solid {color}55",
    })

def insight_box(children):
    return html.Div(children, style={
        "background": f"{C['accent']}0d",
        "border": f"1px solid {C['accent']}33",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "fontSize": "0.88rem",
        "lineHeight": "1.65",
        "color": C["text"],
        "marginTop": "8px",
        "marginBottom": "8px",
    })

def verdict_box(title, body, supported=True):
    border_color = C["negative"] if supported else C["positive"]
    icon = "⚠" if supported else "✓"
    return html.Div([
        html.Div(f"{icon}  {title}", style={
            "fontWeight": "700",
            "fontSize": "0.9rem",
            "color": border_color,
            "marginBottom": "8px",
            "fontFamily": "'Syne', sans-serif",
        }),
        html.Div(body, style={
            "fontSize": "0.84rem",
            "lineHeight": "1.6",
            "color": C["text"],
        }),
    ], style={
        "flex": "1",
        "minWidth": "260px",
        "background": f"{border_color}0d",
        "border": f"1px solid {border_color}44",
        "borderRadius": "10px",
        "padding": "16px 20px",
    })

def two_col(*children):
    return html.Div(children, style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "16px",
    })

def three_col(*children):
    return html.Div(children, style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr 1fr",
        "gap": "14px",
    })

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# Fonts & global styles injected via app.index_string below

HERO = html.Div([
    html.Div([
        html.Div([
            pill("FILAH data", C["FILAH"]),
            pill("TROUT data", C["TROUT"]),
            pill("Full journalist data", C["journalist"]),
        ], style={"marginBottom": "18px"}),
        html.H1("COOTEFOO Bias Analysis", style={
            "fontFamily": "'Syne', sans-serif",
            "fontSize": "clamp(1.8rem, 3vw, 2.8rem)",
            "fontWeight": "800",
            "margin": "0 0 10px",
            "background": f"linear-gradient(90deg, {C['tourism']} 0%, {C['accent']} 50%, {C['fishing']} 100%)",
            "WebkitBackgroundClip": "text",
            "WebkitTextFillColor": "transparent",
            "letterSpacing": "-0.02em",
        }),
        html.P(
            "Visual analytics for the Haacklee Herald · "
            "Investigating bias in COOTEFOO member actions across competing datasets",
            style={
                "color": C["muted"],
                "fontSize": "0.9rem",
                "margin": "0",
                "fontWeight": "300",
                "letterSpacing": "0.01em",
            }
        ),
    ], style={"maxWidth": "860px"}),
], style={
    "background": f"linear-gradient(135deg, {C['bg']} 0%, #0a1020 100%)",
    "borderBottom": f"1px solid {C['border']}",
    "padding": "48px 48px 36px",
})

# ─── Part 1 ──────────────────────────────────────────────────────────────────

PART1_INTRO = card(
    html.P([
        "Each dataset tags discussion-participation edges with ",
        html.Strong("sentiment (−1 → +1)"),
        " and ",
        html.Strong("industry"),
        " (tourism / large vessel / small vessel). "
        "A member who scores consistently positive toward tourism and negative toward fishing "
        "supports FILAH's bias claim; the reverse supports TROUT's. "
        "The heatmaps, bar charts and travel-zone breakdowns below surface those patterns."
    ], style={"margin": 0, "fontSize": "0.88rem", "color": C["muted"], "lineHeight": "1.7"}),
)

PART1_HEATMAPS = two_col(
    card(dcc.Graph(figure=fig_heatmap("FILAH"),  config={"displayModeBar": False})),
    card(dcc.Graph(figure=fig_heatmap("TROUT"),  config={"displayModeBar": False})),
)

PART1_SENTBARS = two_col(
    card(dcc.Graph(figure=fig_sentiment_bars("FILAH"), config={"displayModeBar": False})),
    card(dcc.Graph(figure=fig_sentiment_bars("TROUT"), config={"displayModeBar": False})),
)

PART1_ZONEBARS = two_col(
    card(dcc.Graph(figure=fig_zone_bars("FILAH"), config={"displayModeBar": False})),
    card(dcc.Graph(figure=fig_zone_bars("TROUT"), config={"displayModeBar": False})),
)

PART1_VERDICTS = html.Div([
    verdict_box(
        "FILAH's accusation — pro-tourism bias",
        [
            html.Strong("Partially supported by their own data. ", style={"color": C["tourism"]}),
            "Simone Kat (tourism avg +0.88) and Carol Limpet (+0.70) show clear pro-tourism sentiment. "
            "However, FILAH's travel records are limited to commercial/tourism zones only — "
            "the full picture shows government-zone travel dominates, "
            "which FILAH did not capture at all.",
        ],
        supported=True,
    ),
    verdict_box(
        "TROUT's accusation — pro-fishing bias",
        [
            html.Strong("Partially supported by their own data. ", style={"color": C["fishing"]}),
            "Teddy Goldstein scores fishing avg +0.93 vs tourism −0.50 — "
            "a strong pro-fishing signal. Ed Helpsford is uniformly positive across all industries. "
            "But TROUT captures almost exclusively government-zone travel, "
            "omitting commercial trips that show a more nuanced picture.",
        ],
        supported=True,
    ),
], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"})

# ─── Part 3 ──────────────────────────────────────────────────────────────────

PART3_INTRO = card(
    html.P([
        "Important finding: the journalist graph is the ",
        html.Strong("union of FILAH + TROUT + extra records"),
        ". Sentiment values for shared members are ",
        html.Strong("identical"),
        " — the journalist adds no new meeting records for existing members. "
        "What it ",
        html.Em("does"),
        " add: (1) ",
        html.Strong("Tante Titan"),
        " — entirely absent from both subsets with 54 participation records and strong pro-tourism sentiment; "
        "(2) roughly ",
        html.Strong("10× more trip records"),
        " — dominated by government-zone official travel that neither FILAH nor TROUT captured, "
        "completely reframing how each member spends their time. "
        "Part 3 therefore focuses on these real differences: member exclusion, trip volume, and zone framing.",
    ], style={"margin": 0, "fontSize": "0.88rem", "color": C["muted"], "lineHeight": "1.7"}),
)

# ── ⑤ Member coverage ────────────────────────────────────────────────────────
PART3_COVERAGE = card(
    dcc.Graph(figure=fig_member_coverage(), config={"displayModeBar": False})
)

PART3_COVERAGE_INSIGHT = insight_box([
    html.Strong("Who each dataset chose to include: ", style={"color": C["accent"]}),
    "FILAH recorded only Simone Kat, Carol Limpet and Seal — the three members with the highest "
    "pro-tourism sentiment scores. TROUT recorded only Teddy Goldstein, Ed Helpsford and Seal — "
    "the three most pro-fishing. ",
    html.Strong("Tante Titan", style={"color": C["journalist"]}),
    " (54 records, tourism avg +0.51) is invisible to both groups. "
    "This selection bias alone is enough to manufacture the appearance of a biased committee.",
])

# ── ⑥ Trip volume gap ────────────────────────────────────────────────────────
PART3_TRIP_VOLUME = card(
    dcc.Graph(figure=fig_trip_volume_gap(), config={"displayModeBar": False})
)

PART3_TRIP_INSIGHT = insight_box([
    html.Strong("The trip volume gap: ", style={"color": C["accent"]}),
    "FILAH recorded 34–99 waypoints per member; TROUT recorded 6–28. "
    "The journalist graph reveals 171–362 waypoints per member — almost entirely ",
    html.Strong("government-zone"),
    " official travel that both groups completely omitted. "
    "This makes members look industry-focused when in reality most of their travel is routine official business.",
])

# ── ⑦ Zone bias heatmaps ─────────────────────────────────────────────────────
PART3_ZONE_HEATMAPS = three_col(
    card(dcc.Graph(figure=fig_zone_bias_heatmap("FILAH"),      config={"displayModeBar": False})),
    card(dcc.Graph(figure=fig_zone_bias_heatmap("TROUT"),      config={"displayModeBar": False})),
    card(dcc.Graph(figure=fig_zone_bias_heatmap("journalist"), config={"displayModeBar": False})),
)

PART3_ZONE_INSIGHT = insight_box([
    html.Strong("Zone framing bias: ", style={"color": C["accent"]}),
    "FILAH's records show 60–85% of trips going to commercial zones, making members look commercially focused. "
    "TROUT's records show 65–85% going to government zones, making members look purely administrative. "
    "In the journalist data, government zones account for 60–80% of travel for all members — "
    "the dominant pattern that both subsets strategically omitted.",
])

# ── ⑧ Tante Titan ────────────────────────────────────────────────────────────
PART3_TANTE = card(
    dcc.Graph(figure=fig_tante_titan(), config={"displayModeBar": False})
)

# ── ⑨ Interactive: per-person zone % breakdown ───────────────────────────────
PERSON_SELECTOR = card(
    html.Div([
        html.Label("Select member:",
                   style={"color": C["muted"], "fontSize": "0.8rem",
                          "marginRight": "14px", "fontWeight": "500"}),
        dcc.Dropdown(
            id="person-selector",
            options=[{"label": p, "value": p} for p in ALL_PERSONS],
            value="Simone Kat",
            clearable=False,
            style={"width": "260px"},
        ),
    ], style={"display": "flex", "alignItems": "center"}),
    style={"padding": "14px 20px"},
)

PART3_PERSON_ZONE = card(dcc.Graph(id="person-zone-pct", config={"displayModeBar": False}))

PART3_VERDICTS = html.Div([
    verdict_box(
        "FILAH accusation — in the full dataset",
        [
            html.Strong("Sentiment unchanged, but travel framing exposed. ", style={"color": C["tourism"]}),
            "Simone Kat and Carol Limpet's pro-tourism sentiment is confirmed — but it was already fully "
            "visible in FILAH's own data. The journalist graph reveals FILAH ",
            html.Em("only recorded commercial/tourism trips"),
            ", hiding that 75–85% of actual travel is government-zone official business. "
            "The bias accusation is real in meetings, but FILAH overstated it through selective trip recording.",
        ],
        supported=True,
    ),
    verdict_box(
        "TROUT accusation — in the full dataset",
        [
            html.Strong("Weakened by Tante Titan and trip context. ", style={"color": C["positive"]}),
            "Teddy Goldstein's anti-tourism sentiment is confirmed. But TROUT hid Tante Titan "
            "(54 records, tourism +0.51) who balances the committee toward tourism. "
            "TROUT also ",
            html.Em("only recorded government-zone trips"),
            ", hiding the commercial travel that shows members do engage with both industries.",
        ],
        supported=False,
    ),
], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"})

PART3_KEY_INSIGHT = insight_box([
    html.Strong("Key missing evidence from TROUT: ", style={"color": C["accent"]}),
    "Tante Titan's 54 participation records (tourism avg +0.51, small vessel +0.75) are the single most "
    "important omission — she alone shifts the committee's overall tourism sentiment from neutral to "
    "moderately positive. Combined with the hidden commercial trip records, TROUT's picture of a "
    "fishing-biased committee does not survive contact with the full dataset.",
])

# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="COOTEFOO Bias Analysis")
app.index_string = """<!DOCTYPE html>
<html>
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      background: """ + C["bg"] + """ !important;
      color: """ + C["text"] + """;
      font-family: 'DM Mono', monospace;
      margin: 0;
      -webkit-font-smoothing: antialiased;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: """ + C["bg"] + """; }
    ::-webkit-scrollbar-thumb { background: """ + C["border"] + """; border-radius: 3px; }
    .Select-control, .Select-menu-outer {
      background: """ + C["card"] + """ !important;
      border-color: """ + C["border"] + """ !important;
      color: """ + C["text"] + """ !important;
    }
    .Select-value-label, .Select-placeholder { color: """ + C["text"] + """ !important; }
    .Select-option { background: """ + C["card"] + """ !important; color: """ + C["text"] + """ !important; }
    .Select-option:hover { background: """ + C["border"] + """ !important; }
    .VirtualizedSelectOption { background: """ + C["card"] + """ !important; }
  </style>
</head>
<body>
  {%app_entry%}
  <footer>
    {%config%}
    {%scripts%}
    {%renderer%}
  </footer>
</body>
</html>"""

CONTENT = html.Div([
    # ── Part 1 ──────────────────────────────────────────────────────────────
    section_title("Part 1 — Are the accusations supported by each group's own data?", C["fishing"]),
    PART1_INTRO,

    html.Div("① Sentiment heatmaps", style={"color": C["muted"], "fontSize": "0.78rem",
             "letterSpacing": "0.06em", "marginBottom": "8px", "marginTop": "4px"}),
    PART1_HEATMAPS,

    html.Div("② Per-member avg sentiment by industry", style={"color": C["muted"],
             "fontSize": "0.78rem", "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART1_SENTBARS,

    html.Div("③ Travel zones captured in each dataset",  style={"color": C["muted"],
             "fontSize": "0.78rem", "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART1_ZONEBARS,

    html.Div("④ Verdicts", style={"color": C["muted"], "fontSize": "0.78rem",
             "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART1_VERDICTS,

    # ── Part 3 ──────────────────────────────────────────────────────────────
    section_title("Part 3 — Accusations in context of the full journalist dataset", C["tourism"]),
    PART3_INTRO,

    html.Div("⑤ Member coverage — who did each dataset choose to record?",
             style={"color": C["muted"], "fontSize": "0.78rem",
                    "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART3_COVERAGE,
    PART3_COVERAGE_INSIGHT,

    html.Div("⑥ Trip volume gap — how many trips were recorded vs reality?",
             style={"color": C["muted"], "fontSize": "0.78rem",
                    "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART3_TRIP_VOLUME,
    PART3_TRIP_INSIGHT,

    html.Div("⑦ Zone bias heatmaps — % of trips per zone in each dataset",
             style={"color": C["muted"], "fontSize": "0.78rem",
                    "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART3_ZONE_HEATMAPS,
    PART3_ZONE_INSIGHT,

    html.Div("⑧ Tante Titan — the hidden member, only visible in journalist data",
             style={"color": C["muted"], "fontSize": "0.78rem",
                    "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART3_TANTE,

    html.Div("⑨ Per-member zone framing — select a member to see how each dataset frames their travel",
             style={"color": C["muted"], "fontSize": "0.78rem",
                    "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PERSON_SELECTOR,
    PART3_PERSON_ZONE,

    html.Div("⑩ Verdicts in full context", style={"color": C["muted"], "fontSize": "0.78rem",
             "letterSpacing": "0.06em", "marginBottom": "8px"}),
    PART3_VERDICTS,
    PART3_KEY_INSIGHT,

    html.Div(style={"height": "60px"}),  # bottom padding
], style={"maxWidth": "1280px", "margin": "0 auto", "padding": "0 32px"})

app.layout = html.Div([HERO, CONTENT])

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("person-zone-pct", "figure"),
    Input("person-selector",  "value"),
)
def update_person_zone(person):
    return fig_zone_bias_per_person(person)


if __name__ == "__main__":
    app.run(debug=True)