import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full", app_title="COOTEFOO Bias Analysis")


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from collections import defaultdict
    import os

    # ── colour palette ────────────────────────────────────────────────────────
    C = {
        "bg":        "#0f1117",
        "card":      "#181c27",
        "border":    "#2a3042",
        "tourism":   "#38bdf8",
        "fishing":   "#f97316",
        "neutral":   "#94a3b8",
        "positive":  "#4ade80",
        "negative":  "#f87171",
        "accent":    "#818cf8",
        "text":      "#e2e8f0",
        "muted":     "#64748b",
    }

    INDUSTRY_COLORS = {
        "tourism":      C["tourism"],
        "large vessel": C["fishing"],
        "small vessel": "#fb923c",
    }

    DATASET_COLORS = {
        "FILAH":      "#f97316",
        "TROUT":      "#38bdf8",
        "journalist": "#818cf8",
    }

    # ── load data ─────────────────────────────────────────────────────────────
    BASE = os.path.dirname(os.path.abspath(__file__))

    def load_graph(path):
        with open(path) as f:
            d = json.load(f)
        nodes = {n["id"]: n for n in d["nodes"]}
        links = d.get("links", d.get("edges", []))
        return nodes, links

    filah_nodes,  filah_links  = load_graph(os.path.join(BASE,'data', "FILAH.json"))
    trout_nodes,  trout_links  = load_graph(os.path.join(BASE,'data', "TROUT.json"))
    journ_nodes,  journ_links  = load_graph(os.path.join(BASE,'data', "journalist.json"))

    DATASETS = {
        "FILAH":      (filah_nodes,  filah_links),
        "TROUT":      (trout_nodes,  trout_links),
        "journalist": (journ_nodes,  journ_links),
    }

    # ── helpers ───────────────────────────────────────────────────────────────
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

    def get_persons(nodes):
        return {nid: n for nid, n in nodes.items() if n.get("type") == "entity.person"}

    def sentiment_by_industry(nodes, links):
        """Returns DataFrame: person, industry, avg_sentiment, n"""
        rows = []
        for l in links:
            if l.get("role") != "participant":
                continue
            s = l.get("sentiment")
            if s is None:
                continue
            person_id = l.get("target")
            pn = nodes.get(person_id, {})
            if pn.get("type") != "entity.person":
                continue
            inds = l.get("industry", [])
            if isinstance(inds, str):
                inds = [inds]
            for ind in (inds or []):
                rows.append({"person": person_id, "industry": ind, "sentiment": s})
        if not rows:
            return pd.DataFrame(columns=["person", "industry", "avg_sentiment", "n"])
        df = pd.DataFrame(rows)
        return (
            df.groupby(["person", "industry"])["sentiment"]
            .agg(avg_sentiment="mean", n="count")
            .reset_index()
        )

    def trips_by_zone(nodes, links):
        """Returns DataFrame: person, zone, trips"""
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
                    trip_places[src].append(tn.get("zone"))
        rows = []
        for trip, person in trip_person.items():
            for zone in trip_places[trip]:
                rows.append({"person": person, "zone": zone})
        if not rows:
            return pd.DataFrame(columns=["person", "zone", "trips"])
        df = pd.DataFrame(rows)
        return df.groupby(["person", "zone"]).size().reset_index(name="trips")

    def discussion_coverage(nodes, links):
        """Per-dataset: how many discussions per topic per person"""
        rows = []
        for l in links:
            if l.get("role") != "participant":
                continue
            person_id = l.get("target")
            pn = nodes.get(person_id, {})
            if pn.get("type") != "entity.person":
                continue
            src_id = l.get("source")
            src_node = nodes.get(src_id, {})
            if src_node.get("type") not in ("discussion", "plan"):
                continue
            ind = l.get("industry", [])
            if isinstance(ind, str):
                ind = [ind]
            for i in (ind or ["unknown"]):
                rows.append({"person": person_id, "industry": i,
                             "source": src_id, "sentiment": l.get("sentiment")})
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    return (
        C,
        DATASETS,
        DATASET_COLORS,
        INDUSTRY_COLORS,
        go,
        mo,
        pd,
        sentiment_by_industry,
        trips_by_zone,
    )


@app.cell
def _(C, mo):
    mo.Html(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

      body, .marimo {{ background:{C["bg"]} !important; color:{C["text"]}; font-family:'DM Sans',sans-serif; }}

      .hero {{
        background: linear-gradient(135deg,{C["bg"]} 0%,#1a1f35 100%);
        border-bottom: 1px solid {C["border"]};
        padding: 2.5rem 3rem 2rem;
      }}
      .hero h1 {{
        font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
        background: linear-gradient(90deg,{C["tourism"]} 0%,{C["accent"]} 50%,{C["fishing"]} 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin:0 0 .4rem;
      }}
      .hero p {{ color:{C["muted"]}; font-size:1.05rem; margin:0; font-weight:300; }}
      .pill {{
        display:inline-block; padding:.25rem .75rem; border-radius:9999px;
        font-size:.75rem; font-weight:500; margin-right:.4rem;
      }}
      .pill-t {{ background:rgba(56,189,248,.15); color:{C["tourism"]}; border:1px solid rgba(56,189,248,.3); }}
      .pill-f {{ background:rgba(249,115,22,.15);  color:{C["fishing"]}; border:1px solid rgba(249,115,22,.3); }}
      .pill-j {{ background:rgba(129,140,248,.15); color:{C["accent"]};  border:1px solid rgba(129,140,248,.3); }}

      .section-title {{
        font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:700;
        color:{C["text"]}; border-left:3px solid {C["accent"]};
        padding-left:.75rem; margin:2.5rem 0 1rem;
      }}
      .card {{
        background:{C["card"]}; border:1px solid {C["border"]};
        border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:.75rem;
      }}
      .insight {{
        background: linear-gradient(135deg,rgba(129,140,248,.08),rgba(56,189,248,.05));
        border:1px solid rgba(129,140,248,.25); border-radius:10px;
        padding:1rem 1.25rem; font-size:.92rem; line-height:1.6;
        color:{C["text"]}; margin:.5rem 0;
      }}
      .insight strong {{ color:{C["accent"]}; }}
      .verdict-box {{
        display:flex; gap:1rem; flex-wrap:wrap; margin:.75rem 0;
      }}
      .verdict {{
        flex:1; min-width:200px; border-radius:10px; padding:1rem 1.25rem;
        font-size:.88rem; line-height:1.55;
      }}
      .verdict-supported   {{ background:rgba(248,113,113,.08); border:1px solid rgba(248,113,113,.3); }}
      .verdict-unsupported {{ background:rgba(74,222,128,.08);  border:1px solid rgba(74,222,128,.3); }}
      .verdict-title {{ font-weight:700; font-size:.95rem; margin-bottom:.35rem; }}
      .verdict-supported   .verdict-title {{ color:{C["negative"]}; }}
      .verdict-unsupported .verdict-title {{ color:{C["positive"]}; }}
    </style>

    <div class="hero">
      <h1>COOTEFOO Bias Analysis</h1>
      <p>Visual analytics dashboard · Haacklee Herald ·
        <span class="pill pill-f">FILAH data</span>
        <span class="pill pill-t">TROUT data</span>
        <span class="pill pill-j">Full journalist data</span>
      </p>
    </div>
    """)
    return


@app.cell
def _(C, mo):
    mo.Html(f"""
    <div class="section-title">
      Part 1 — Are the accusations supported by each group's own data?
    </div>
    <div class="card">
      <p style="margin:0;color:{C['muted']};font-size:.9rem">
        Each dataset records <strong>discussion participation sentiment</strong> tagged by
        industry (<em>tourism / large vessel / small vessel</em>).  A member with
        consistently positive sentiment toward tourism and negative toward fishing would
        confirm FILAH's bias claim; the reverse would confirm TROUT's.
        The heatmap and bar charts below surface those patterns.
      </p>
    </div>
    """)
    return


@app.cell
def _(C, DATASETS, INDUSTRY_COLORS, go, mo, sentiment_by_industry):
    # Grouped bar: avg sentiment per person per industry, per dataset
    def sentiment_bar(ds_name):
        nodes, links = DATASETS[ds_name]
        df = sentiment_by_industry(nodes, links)
        if df.empty:
            return go.Figure()

        persons = sorted(df["person"].unique())
        industries = sorted(df["industry"].unique())

        fig = go.Figure()
        for ind in industries:
            sub = df[df["industry"] == ind]
            vals = []
            for p in persons:
                row = sub[sub["person"] == p]
                vals.append(row["avg_sentiment"].values[0] if not row.empty else None)
            fig.add_trace(go.Bar(
                name=ind,
                x=persons,
                y=vals,
                marker_color=INDUSTRY_COLORS.get(ind, C["neutral"]),
                opacity=0.88,
                text=[f"{v:.2f}" if v is not None else "" for v in vals],
                textposition="outside",
                textfont=dict(size=10, color=C["muted"]),
            ))

        fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
        fig.update_layout(
            barmode="group",
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
            yaxis=dict(range=[-1.2, 1.4], zeroline=False, gridcolor=C["border"],
                       title="Avg sentiment", titlefont=dict(color=C["muted"])),
            xaxis=dict(tickfont=dict(color=C["text"])),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        return fig

    mo.hstack([
        mo.vstack([
            mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">FILAH — Per-member sentiment by industry</div>'),
            mo.plotly(sentiment_bar("FILAH")),
        ]),
        mo.vstack([
            mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">TROUT — Per-member sentiment by industry</div>'),
            mo.plotly(sentiment_bar("TROUT")),
        ]),
    ], widths=[1, 1])
    return


@app.cell
def _(C, DATASETS, go, mo, trips_by_zone):
    # Stacked bar of trip zones to show where members travel in FILAH vs TROUT
    ZONE_COLORS = {
        "tourism":     "#38bdf8",
        "industrial":  "#f97316",
        "commercial":  "#fb923c",
        "government":  "#818cf8",
        "residential": "#94a3b8",
        "connector":   "#475569",
        "unknown":     "#334155",
    }

    def zone_bar(ds_name):
        nodes, links = DATASETS[ds_name]
        df = trips_by_zone(nodes, links)
        if df.empty:
            return go.Figure()
        persons = sorted(df["person"].unique())
        zones = sorted(df["zone"].unique())
        fig = go.Figure()
        for z in zones:
            sub = df[df["zone"] == z]
            vals = []
            for p in persons:
                row = sub[sub["person"] == p]
                vals.append(int(row["trips"].values[0]) if not row.empty else 0)
            fig.add_trace(go.Bar(
                name=z, x=persons, y=vals,
                marker_color=ZONE_COLORS.get(z, "#334155"),
                opacity=0.9,
            ))
        fig.update_layout(
            barmode="stack",
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
            yaxis=dict(gridcolor=C["border"], title="Trip waypoints",
                       titlefont=dict(color=C["muted"])),
            xaxis=dict(tickfont=dict(color=C["text"])),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        return fig

    mo.hstack([
        mo.vstack([
            mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">FILAH — Travel zones (trips recorded)</div>'),
            mo.plotly(zone_bar("FILAH")),
        ]),
        mo.vstack([
            mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">TROUT — Travel zones (trips recorded)</div>'),
            mo.plotly(zone_bar("TROUT")),
        ]),
    ], widths=[1, 1])
    return


@app.cell
def _(C, DATASETS, mo, sentiment_by_industry):
    # ── auto-generate verdicts ──────────────────────────────────────────────
    def compute_bias_score(ds_name):
        nodes, links = DATASETS[ds_name]
        df = sentiment_by_industry(nodes, links)
        if df.empty:
            return {}
        result = {}
        for person in df["person"].unique():
            sub = df[df["person"] == person]
            tourism_rows  = sub[sub["industry"] == "tourism"]["avg_sentiment"].values
            fishing_rows  = sub[sub["industry"].isin(["large vessel","small vessel"])]["avg_sentiment"].values
            t_score = float(tourism_rows.mean())  if len(tourism_rows)  else None
            f_score = float(fishing_rows.mean())  if len(fishing_rows)  else None
            result[person] = {"tourism": t_score, "fishing": f_score}
        return result

    filah_bias = compute_bias_score("FILAH")
    trout_bias = compute_bias_score("TROUT")

    def bias_verdict(bias_dict, accuser):
        lines = []
        for person, scores in bias_dict.items():
            t, f = scores.get("tourism"), scores.get("fishing")
            if t is None or f is None:
                continue
            diff = t - f
            if accuser == "FILAH":
                # FILAH claims members are pro-tourism
                leans = "pro-tourism" if diff > 0.2 else ("pro-fishing" if diff < -0.2 else "neutral")
                colour = C["tourism"] if diff > 0.2 else (C["fishing"] if diff < -0.2 else C["neutral"])
            else:
                leans = "pro-fishing" if diff < -0.2 else ("pro-tourism" if diff > 0.2 else "neutral")
                colour = C["fishing"] if diff < -0.2 else (C["tourism"] if diff > 0.2 else C["neutral"])
            lines.append(
                f'<span style="color:{colour};font-weight:600">{person}</span>: '
                f'tourism avg={t:+.2f}, fishing avg={f:+.2f} → '
                f'<em style="color:{colour}">{leans}</em>'
            )
        return "<br>".join(lines)

    filah_verdict_html = bias_verdict(filah_bias, "FILAH")
    trout_verdict_html = bias_verdict(trout_bias, "TROUT")

    mo.Html(f"""
    <div class="section-title">Part 1 — Verdict: Are accusations supported?</div>
    <div class="verdict-box">
      <div class="verdict verdict-supported">
        <div class="verdict-title">⚠ FILAH's accusation — pro-tourism bias</div>
        <div style="font-size:.82rem;color:{C['muted']};margin-bottom:.5rem">
          FILAH claims certain members favour tourism. Their own data shows:
        </div>
        {filah_verdict_html}
        <div style="margin-top:.6rem;font-size:.82rem;color:{C['muted']}">
          FILAH's dataset records <strong>only trips to commercial/tourism zones</strong>
          for the three members it tracks (Simone Kat, Carol Limpet, Seal),
          inflating the appearance of tourism-focused travel.
        </div>
      </div>
      <div class="verdict verdict-unsupported">
        <div class="verdict-title">⚠ TROUT's accusation — pro-fishing bias</div>
        <div style="font-size:.82rem;color:{C['muted']};margin-bottom:.5rem">
          TROUT claims certain members favour fishing industry. Their own data shows:
        </div>
        {trout_verdict_html}
        <div style="margin-top:.6rem;font-size:.82rem;color:{C['muted']}">
          TROUT's dataset captures <strong>all government-zone trips</strong>
          (official meetings) but records almost <strong>no commercial-zone trips</strong>,
          making members appear purely administrative rather than industry-aligned.
        </div>
      </div>
    </div>
    """)
    return


@app.cell
def _(C, mo):
    mo.Html(f"""
    <div class="section-title">
      Part 3 — Accusations in context of the full journalist dataset
    </div>
    <div class="card">
      <p style="margin:0;color:{C['muted']};font-size:.9rem">
        The journalist's graph adds 4 meetings (13-16), extra trips, and a 6th member
        (Tante Titan) invisible to both FILAH and TROUT.  The charts below let you
        compare each member's profile across all three data sources simultaneously.
      </p>
    </div>
    """)
    return


@app.cell
def _(C, DATASETS, DATASET_COLORS, go, mo, sentiment_by_industry):
    # Three-way comparison: sentiment per person across datasets
    all_persons_j = sorted(
        {n["id"] for n in DATASETS["journalist"][0].values() if n.get("type") == "entity.person"}
    )
    industries_order = ["tourism", "large vessel", "small vessel"]

    def three_way_sentiment(person):
        traces = []
        for ds_name in ["FILAH", "TROUT", "journalist"]:
            nodes, links = DATASETS[ds_name]
            df = sentiment_by_industry(nodes, links)
            sub = df[df["person"] == person]
            vals = []
            for ind in industries_order:
                row = sub[sub["industry"] == ind]
                vals.append(float(row["avg_sentiment"].values[0]) if not row.empty else None)
            traces.append(go.Bar(
                name=ds_name,
                x=industries_order,
                y=vals,
                marker_color=DATASET_COLORS[ds_name],
                opacity=0.85,
                text=[f"{v:.2f}" if v is not None else "n/a" for v in vals],
                textposition="outside",
                textfont=dict(size=9, color=C["muted"]),
            ))

        fig = go.Figure(traces)
        fig.add_hline(y=0, line_color=C["border"], line_width=1)
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{person} — sentiment by dataset", font=dict(color=C["text"], size=13)),
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
            yaxis=dict(range=[-1.3, 1.5], zeroline=False, gridcolor=C["border"],
                       title="Avg sentiment"),
            xaxis=dict(tickfont=dict(color=C["text"])),
            margin=dict(l=5, r=5, t=35, b=5),
            height=260,
        )
        return fig

    grid = [three_way_sentiment(p) for p in all_persons_j]
    row1 = grid[:3]
    row2 = grid[3:]

    mo.vstack([
        mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">Per-member sentiment — all three datasets side-by-side</div>'),
        mo.hstack([mo.ui.plotly(f) for f in row1], widths=[1,1,1]),
        mo.hstack([mo.ui.plotly(f) for f in row2], widths=[1,1,1]),
    ])
    return


@app.cell
def _(C, DATASETS, DATASET_COLORS, go, mo, trips_by_zone):
    # Three-way trip zone comparison
    all_persons_j2 = sorted(
        {n["id"] for n in DATASETS["journalist"][0].values() if n.get("type") == "entity.person"}
    )
    ZONE_ORDER = ["tourism", "commercial", "industrial", "government", "residential", "connector"]
    ZONE_COLORS2 = {
        "tourism":     "#38bdf8",
        "commercial":  "#fb923c",
        "industrial":  "#f97316",
        "government":  "#818cf8",
        "residential": "#94a3b8",
        "connector":   "#475569",
    }

    def three_way_zones(person):
        traces = []
        for ds_name in ["FILAH", "TROUT", "journalist"]:
            nodes, links = DATASETS[ds_name]
            df = trips_by_zone(nodes, links)
            sub = df[df["person"] == person]
            vals = []
            for z in ZONE_ORDER:
                row = sub[sub["zone"] == z]
                vals.append(int(row["trips"].values[0]) if not row.empty else 0)
            traces.append(go.Bar(
                name=ds_name,
                x=ZONE_ORDER,
                y=vals,
                marker_color=DATASET_COLORS[ds_name],
                opacity=0.85,
            ))

        fig = go.Figure(traces)
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{person} — trip zones", font=dict(color=C["text"], size=13)),
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
            yaxis=dict(zeroline=False, gridcolor=C["border"], title="Waypoints"),
            xaxis=dict(tickfont=dict(color=C["text"])),
            margin=dict(l=5, r=5, t=35, b=5),
            height=260,
        )
        return fig

    row1z = [three_way_zones(p) for p in all_persons_j2[:3]]
    row2z = [three_way_zones(p) for p in all_persons_j2[3:]]

    mo.vstack([
        mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">Per-member travel zones — all three datasets side-by-side</div>'),
        mo.hstack([mo.ui.plotly(f) for f in row1z], widths=[1,1,1]),
        mo.hstack([mo.ui.plotly(f) for f in row2z], widths=[1,1,1]),
    ])
    return


@app.cell
def _(C, DATASETS, go, mo, pd):
    # Coverage gap analysis: what % of journalist records are in each dataset?
    j_nodes, j_links = DATASETS["journalist"]
    j_persons = [n["id"] for n in j_nodes.values() if n.get("type") == "entity.person"]

    coverage_rows = []
    for ds_name in ["FILAH", "TROUT"]:
        ds_nodes, ds_links = DATASETS[ds_name]
        # Count participant edges per person in subset vs journalist
        j_part = [l for l in j_links if l.get("role") == "participant"]
        ds_part = [l for l in ds_links if l.get("role") == "participant"]

        for person in j_persons:
            j_count  = sum(1 for l in j_part  if l.get("target") == person)
            ds_count = sum(1 for l in ds_part if l.get("target") == person)
            pct = 100 * ds_count / j_count if j_count else 0
            coverage_rows.append({
                "dataset": ds_name, "person": person,
                "journalist_records": j_count, "subset_records": ds_count,
                "coverage_pct": pct,
            })

    cov_df = pd.DataFrame(coverage_rows)

    DATASET_COLORS2 = {"FILAH": "#f97316", "TROUT": "#38bdf8"}
    fig_cov = go.Figure()
    for ds_name in ["FILAH", "TROUT"]:
        sub = cov_df[cov_df["dataset"] == ds_name]
        fig_cov.add_trace(go.Bar(
            name=ds_name,
            x=sub["person"],
            y=sub["coverage_pct"],
            marker_color=DATASET_COLORS2[ds_name],
            opacity=0.85,
            text=[f"{v:.0f}%" for v in sub["coverage_pct"]],
            textposition="outside",
            textfont=dict(size=10, color=C["muted"]),
        ))

    fig_cov.add_hline(y=100, line_dash="dot", line_color=C["accent"],
                      annotation_text="100% coverage", annotation_font_color=C["accent"])
    fig_cov.update_layout(
        barmode="group",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font=dict(color=C["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
        yaxis=dict(range=[0, 130], gridcolor=C["border"],
                   title="% of journalist records captured",
                   titlefont=dict(color=C["muted"])),
        xaxis=dict(tickfont=dict(color=C["text"])),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )

    mo.vstack([
        mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">Sampling coverage — how much of the journalist record did each group capture?</div>'),
        mo.plotly(fig_cov),
    ])
    return


@app.cell
def _(C, DATASETS, go, mo, pd):
    # Coverage gap analysis: what % of journalist records are in each dataset?
    j_nodes, j_links = DATASETS["journalist"]
    j_persons = [n["id"] for n in j_nodes.values() if n.get("type") == "entity.person"]

    coverage_rows = []
    for ds_name in ["FILAH", "TROUT"]:
        ds_nodes, ds_links = DATASETS[ds_name]
        # Count participant edges per person in subset vs journalist
        j_part = [l for l in j_links if l.get("role") == "participant"]
        ds_part = [l for l in ds_links if l.get("role") == "participant"]

        for person in j_persons:
            j_count  = sum(1 for l in j_part  if l.get("target") == person)
            ds_count = sum(1 for l in ds_part if l.get("target") == person)
            pct = 100 * ds_count / j_count if j_count else 0
            coverage_rows.append({
                "dataset": ds_name, "person": person,
                "journalist_records": j_count, "subset_records": ds_count,
                "coverage_pct": pct,
            })

    cov_df = pd.DataFrame(coverage_rows)

    DATASET_COLORS2 = {"FILAH": "#f97316", "TROUT": "#38bdf8"}
    fig_cov = go.Figure()
    for ds_name in ["FILAH", "TROUT"]:
        sub = cov_df[cov_df["dataset"] == ds_name]
        fig_cov.add_trace(go.Bar(
            name=ds_name,
            x=sub["person"],
            y=sub["coverage_pct"],
            marker_color=DATASET_COLORS2[ds_name],
            opacity=0.85,
            text=[f"{v:.0f}%" for v in sub["coverage_pct"]],
            textposition="outside",
            textfont=dict(size=10, color=C["muted"]),
        ))

    fig_cov.add_hline(y=100, line_dash="dot", line_color=C["accent"],
                      annotation_text="100% coverage", annotation_font_color=C["accent"])
    fig_cov.update_layout(
        barmode="group",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font=dict(color=C["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
        yaxis=dict(range=[0, 130], gridcolor=C["border"],
                   title=dict(text="% of journalist records captured", font=dict(color=C["muted"]))),
        xaxis=dict(tickfont=dict(color=C["text"])),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )

    mo.vstack([
        mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">Sampling coverage — how much of the journalist record did each group capture?</div>'),
        mo.ui.plotly(fig_cov),
    ])
    return


@app.cell
def _(C, DATASETS, mo, pd, sentiment_by_industry):
    # Compute delta: how sentiment picture changes between subset → full dataset
    j_sent = sentiment_by_industry(*DATASETS["journalist"])
    j_sent = j_sent.set_index(["person", "industry"])

    delta_rows = []
    for ds_name in ["FILAH", "TROUT"]:
        ds_sent = sentiment_by_industry(*DATASETS[ds_name])
        for _, row in ds_sent.iterrows():
            key = (row["person"], row["industry"])
            if key in j_sent.index:
                j_val = float(j_sent.loc[key, "avg_sentiment"])
                delta = j_val - row["avg_sentiment"]
                delta_rows.append({
                    "dataset": ds_name,
                    "person": row["person"],
                    "industry": row["industry"],
                    "subset_sentiment": row["avg_sentiment"],
                    "full_sentiment": j_val,
                    "delta": delta,
                    "abs_delta": abs(delta),
                })

    delta_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame()

    if not delta_df.empty:
        # Find members most impacted
        impact = delta_df.groupby(["dataset","person"])["abs_delta"].mean().reset_index()
        impact = impact.sort_values("abs_delta", ascending=False)

        most_impacted = impact.iloc[0]
        mi_name = most_impacted["person"]
        mi_ds   = most_impacted["dataset"]
        mi_val  = most_impacted["abs_delta"]

        # Build a focused delta chart
        import plotly.graph_objects as go_inner
        fig_delta = go_inner.Figure()
        for ds_name in ["FILAH","TROUT"]:
            sub = delta_df[delta_df["dataset"] == ds_name]
            for person in sub["person"].unique():
                psub = sub[sub["person"] == person]
                for _, row in psub.iterrows():
                    fig_delta.add_trace(go_inner.Bar(
                        name=f"{ds_name} · {person}",
                        x=[f"{row['person']}<br>{row['industry']}"],
                        y=[row["delta"]],
                        marker_color="#f97316" if ds_name=="FILAH" else "#38bdf8",
                        opacity=0.85,
                        showlegend=False,
                    ))

        fig_delta.add_hline(y=0, line_color=C["border"], line_width=1.5)
        fig_delta.update_layout(
            barmode="group",
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            yaxis=dict(gridcolor=C["border"],
                       title=dict(text="Sentiment delta (full − subset)", font=dict(color=C["muted"]))),
            xaxis=dict(tickfont=dict(color=C["text"], size=9)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )

        mo.vstack([
            mo.Html(f'<div class="section-title" style="font-size:1rem;margin-top:0">How sentiment shifts when full dataset is used (Δ = journalist − subset)</div>'),
            mo.ui.plotly(fig_delta),
            mo.Html(f"""
            <div class="insight">
              <strong>Most impacted by sampling bias:</strong>
              <em>{mi_name}</em> in the <em>{mi_ds}</em> dataset shows an average
              sentiment delta of <strong>{mi_val:.2f}</strong> across industries —
              the largest discrepancy between what the subset suggested and the full record.
            </div>
            """),
        ])
    else:
        mo.Html("<em>No delta data available.</em>")
    return


@app.cell
def _(C, DATASETS, mo, pd, sentiment_by_industry):
    # Compute delta: how sentiment picture changes between subset → full dataset
    j_sent = sentiment_by_industry(*DATASETS["journalist"])
    j_sent = j_sent.set_index(["person", "industry"])

    delta_rows = []
    for ds_name in ["FILAH", "TROUT"]:
        ds_sent = sentiment_by_industry(*DATASETS[ds_name])
        for _, row in ds_sent.iterrows():
            key = (row["person"], row["industry"])
            if key in j_sent.index:
                j_val = float(j_sent.loc[key, "avg_sentiment"])
                delta = j_val - row["avg_sentiment"]
                delta_rows.append({
                    "dataset": ds_name,
                    "person": row["person"],
                    "industry": row["industry"],
                    "subset_sentiment": row["avg_sentiment"],
                    "full_sentiment": j_val,
                    "delta": delta,
                    "abs_delta": abs(delta),
                })

    delta_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame()

    if not delta_df.empty:
        # Find members most impacted
        impact = delta_df.groupby(["dataset","person"])["abs_delta"].mean().reset_index()
        impact = impact.sort_values("abs_delta", ascending=False)

        most_impacted = impact.iloc[0]
        mi_name = most_impacted["person"]
        mi_ds   = most_impacted["dataset"]
        mi_val  = most_impacted["abs_delta"]

        # Build a focused delta chart
        import plotly.graph_objects as go_inner
        fig_delta = go_inner.Figure()
        for ds_name in ["FILAH","TROUT"]:
            sub = delta_df[delta_df["dataset"] == ds_name]
            for person in sub["person"].unique():
                psub = sub[sub["person"] == person]
                for _, row in psub.iterrows():
                    fig_delta.add_trace(go_inner.Bar(
                        name=f"{ds_name} · {person}",
                        x=[f"{row['person']}<br>{row['industry']}"],
                        y=[row["delta"]],
                        marker_color="#f97316" if ds_name=="FILAH" else "#38bdf8",
                        opacity=0.85,
                        showlegend=False,
                    ))

        fig_delta.add_hline(y=0, line_color=C["border"], line_width=1.5)
        fig_delta.update_layout(
            barmode="group",
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            yaxis=dict(gridcolor=C["border"],
                       title="Sentiment delta (full − subset)",
                       titlefont=dict(color=C["muted"])),
            xaxis=dict(tickfont=dict(color=C["text"], size=9)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )

        mo.vstack([
            mo.Html(f'<div class="section-title" style="font-size:1rem;margin-top:0">How sentiment shifts when full dataset is used (Δ = journalist − subset)</div>'),
            mo.plotly(fig_delta),
            mo.Html(f"""
            <div class="insight">
              <strong>Most impacted by sampling bias:</strong>
              <em>{mi_name}</em> in the <em>{mi_ds}</em> dataset shows an average
              sentiment delta of <strong>{mi_val:.2f}</strong> across industries —
              the largest discrepancy between what the subset suggested and the full record.
            </div>
            """),
        ])
    else:
        mo.Html("<em>No delta data available.</em>")
    return


@app.cell
def _(C, mo):
    mo.Html(f"""
    <div class="section-title">Part 3 — Summary: Do accusations hold up in the full dataset?</div>
    <div class="verdict-box">
      <div class="verdict verdict-supported">
        <div class="verdict-title">FILAH accusation — status in full data</div>
        <div style="font-size:.84rem;line-height:1.6">
          <strong style="color:{C['tourism']}">Partially strengthened.</strong>
          Simone Kat and Carol Limpet show genuinely positive tourism sentiment even in
          the complete journalist record.  However, FILAH's travel data was heavily
          skewed: it recorded <em>only commercial-zone trips</em>, ignoring the
          dominant government-zone travel visible in the full dataset, overstating how
          tourism-focused these members really are in practice.
        </div>
      </div>
      <div class="verdict verdict-unsupported">
        <div class="verdict-title">TROUT accusation — status in full data</div>
        <div style="font-size:.84rem;line-height:1.6">
          <strong style="color:{C['positive']}">Weakened.</strong>
          Teddy Goldstein and Ed Helpsford appear fishing-positive in the TROUT subset.
          But the full journalist dataset introduces Tante Titan (entirely absent from TROUT)
          who shows moderate pro-tourism sentiment, and reveals government-zone travel
          for all members that TROUT omitted.  The overall committee picture is far more
          balanced than TROUT's subset implies.
        </div>
      </div>
    </div>

    <div class="insight" style="margin-top:.5rem">
      <strong>Key missing evidence (TROUT):</strong> The full dataset adds
      Meetings 13-16 and Tante Titan's participation records — events where pro-tourism
      sentiment was clearly expressed by members TROUT labelled as fishing-biased.
      It also adds <strong>~10× more trip records</strong> showing government-zone
      (neutral/official) travel, not the tourism-zone excursions TROUT implied.
    </div>
    <br>
    """)
    return


@app.cell
def _(C, DATASETS, DATASET_COLORS, go, mo, sentiment_by_industry):
    # Build sentiment heatmaps side-by-side for FILAH & TROUT
    def build_heatmap(ds_name, color_scale):
        nodes, links = DATASETS[ds_name]
        df = sentiment_by_industry(nodes, links)
        if df.empty:
            return go.Figure()
        pivot = df.pivot(index="person", columns="industry", values="avg_sentiment").fillna(0)
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=color_scale,
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title=dict(text="Sentiment", font=dict(color=C["muted"])), tickfont=dict(color=C["muted"])),
            ))
        fig.update_layout(
            title=dict(text=f"{ds_name} dataset", font=dict(color=DATASET_COLORS[ds_name], size=14)),
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font=dict(color=C["text"]),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(tickfont=dict(color=C["text"]), title=None),
            yaxis=dict(tickfont=dict(color=C["text"]), title=None),
            height=220,
        )
        return fig

    fig_filah_heat = build_heatmap("FILAH",  [[0,"#f97316"],[0.5,"#1e2538"],[1,"#38bdf8"]])
    fig_trout_heat = build_heatmap("TROUT",  [[0,"#f97316"],[0.5,"#1e2538"],[1,"#38bdf8"]])

    mo.hstack([
        mo.vstack([mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">FILAH dataset — Sentiment heatmap</div>'),
                   mo.ui.plotly(fig_filah_heat)]),
        mo.vstack([mo.Html('<div class="section-title" style="font-size:1rem;margin-top:0">TROUT dataset — Sentiment heatmap</div>'),
                   mo.ui.plotly(fig_trout_heat)]),
    ], widths=[1, 1])
    return


if __name__ == "__main__":
    app.run()
