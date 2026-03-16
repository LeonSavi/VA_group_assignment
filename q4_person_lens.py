import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 1 — imports & constants
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json, os
    from collections import defaultdict

    import pandas as pd
    import plotly.graph_objects as go

    C = {
        "bg":         "#080c14",
        "surface":    "#0e1420",
        "card":       "#131929",
        "border":     "#1e2d45",
        "tourism":    "#22d3ee",
        "fishing":    "#fb923c",
        "neutral":    "#94a3b8",
        "positive":   "#4ade80",
        "negative":   "#f87171",
        "accent":     "#7c6af7",
        "text":       "#e2e8f0",
        "muted":      "#64748b",
        "FILAH":      "#fb923c",
        "TROUT":      "#22d3ee",
        "journalist": "#7c6af7",
    }

    ZONE_COLORS = {
        "tourism":     C["tourism"],
        "commercial":  "#fb923c",
        "industrial":  "#f97316",
        "government":  C["accent"],
        "residential": "#94a3b8",
        "connector":   "#334155",
    }
    # government first — it's the key suppressed zone in FILAH
    ZONE_ORDER = ["government", "commercial", "tourism", "industrial", "residential", "connector"]

    LAYOUT_BASE = dict(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(color=C["text"], family="'DM Mono', monospace, sans-serif"),
        margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=10)),
    )

    def make_title(txt):
        return dict(text=txt, font=dict(color=C["muted"], size=11), x=0.01)

    def make_xax(label=""):
        return dict(
            tickfont=dict(color=C["text"]),
            gridcolor=C["border"],
            title=dict(text=label, font=dict(color=C["muted"], size=10)),
        )

    def make_yax(label=""):
        return dict(
            tickfont=dict(color=C["text"]),
            gridcolor=C["border"],
            zeroline=False,
            title=dict(text=label, font=dict(color=C["muted"], size=10)),
        )

    return (
        C, LAYOUT_BASE, ZONE_COLORS, ZONE_ORDER,
        make_title, make_xax, make_yax,
        defaultdict, go, json, mo, os, pd,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 2 — data loading
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(json, os):
    BASE = os.path.dirname(os.path.abspath(__file__))

    def load_graph(filename):
        with open(os.path.join(BASE, "data", filename)) as f:
            d = json.load(f)
        nodes = {n["id"]: n for n in d["nodes"]}
        links = d.get("links", d.get("edges", []))
        return nodes, links

    DATASETS = {
        "FILAH":      load_graph("FILAH.json"),
        "TROUT":      load_graph("TROUT.json"),
        "journalist": load_graph("journalist.json"),
    }

    ALL_PERSONS = sorted(
        nid for nid, n in DATASETS["journalist"][0].items()
        if n.get("type") == "entity.person"
    )

    return ALL_PERSONS, BASE, DATASETS, load_graph


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 3 — pre-compute all metrics
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(ALL_PERSONS, DATASETS, defaultdict, pd):

    # ── A) Participation record counts per person ─────────────────────────────
    def activity_df(nodes, links):
        rows = []
        for lnk in links:
            if lnk.get("role") != "participant":
                continue
            pid = lnk.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person":
                continue
            rows.append({"person": pid})
        if not rows:
            return pd.DataFrame(columns=["person", "n"])
        return pd.DataFrame(rows).groupby("person").size().reset_index(name="n")

    ACTIVITY = {k: activity_df(*v) for k, v in DATASETS.items()}

    # ── B) Sentiment per person x industry ───────────────────────────────────
    def sentiment_df(nodes, links):
        rows = []
        for lnk in links:
            if lnk.get("role") != "participant":
                continue
            s = lnk.get("sentiment")
            if s is None:
                continue
            pid = lnk.get("target")
            if nodes.get(pid, {}).get("type") != "entity.person":
                continue
            inds = lnk.get("industry", [])
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

    SENT = {k: sentiment_df(*v) for k, v in DATASETS.items()}

    # ── C) Zone travel waypoints per person ───────────────────────────────────
    def zone_df(nodes, links):
        trip_person = {}
        trip_places = defaultdict(list)
        for lnk in links:
            src, tgt = lnk["source"], lnk["target"]
            sn, tn = nodes.get(src, {}), nodes.get(tgt, {})
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
        return pd.DataFrame(rows).groupby(["person", "zone"]).size().reset_index(name="trips")

    ZONES = {k: zone_df(*v) for k, v in DATASETS.items()}

    # ── D) Coverage bias: % of journalist activity records captured ───────────
    def coverage_bias_df():
        journo_act = ACTIVITY["journalist"].set_index("person")["n"].to_dict()
        rows = []
        for ds in ["FILAH", "TROUT"]:
            df = ACTIVITY[ds]
            for person in ALL_PERSONS:
                n_journo = journo_act.get(person, 0)
                sub = df[df["person"] == person]
                n_ds = int(sub["n"].values[0]) if not sub.empty else 0
                coverage = round(100 * n_ds / n_journo, 1) if n_journo > 0 else 0.0
                rows.append({
                    "person":   person,
                    "dataset":  ds,
                    "n_ds":     n_ds,
                    "n_journo": n_journo,
                    "coverage": coverage,
                    "missing":  n_journo - n_ds,
                })
        return pd.DataFrame(rows)

    BIAS_DF = coverage_bias_df()

    ALL_BIAS = {
        p: {
            ds: int(BIAS_DF.loc[
                (BIAS_DF["person"] == p) & (BIAS_DF["dataset"] == ds), "missing"
            ].values[0])
            for ds in ["FILAH", "TROUT"]
        }
        for p in ALL_PERSONS
    }

    return ACTIVITY, ALL_BIAS, BIAS_DF, SENT, ZONES, activity_df, sentiment_df, zone_df


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 4 — figure builders
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(
    ACTIVITY, BIAS_DF, C, LAYOUT_BASE, ZONE_COLORS, ZONE_ORDER, ZONES,
    make_title, make_xax, make_yax,
    go, pd,
):
    DS_ORDER  = ["FILAH", "TROUT", "journalist"]
    DS_COLORS = {k: C[k] for k in DS_ORDER}

    def _base(fig, h=300, title=None):
        kw = dict(**LAYOUT_BASE, height=h)
        if title:
            kw["title"] = make_title(title)
        fig.update_layout(**kw)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # FIG A: Participation Record Coverage
    # Sub-questions: (a) presence/absence per dataset, (b) scale of omission
    # ─────────────────────────────────────────────────────────────────────────
    def fig_activity_profile(person):
        rows = []
        for ds in DS_ORDER:
            df = ACTIVITY[ds]
            sub = df[df["person"] == person]
            n = int(sub["n"].values[0]) if not sub.empty else 0
            rows.append({"dataset": ds, "n": n})
        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for _, row in df_plot.iterrows():
            fig.add_trace(go.Bar(
                name=row["dataset"],
                x=[row["dataset"]],
                y=[row["n"]],
                marker_color=DS_COLORS[row["dataset"]],
                opacity=0.88,
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['dataset']}</b><br>"
                    f"participation records: {row['n']}<extra></extra>"
                ),
            ))

        journo_n = int(df_plot.loc[df_plot["dataset"] == "journalist", "n"].values[0])
        fig.add_hline(
            y=journo_n, line_dash="dot", line_color=C["journalist"], line_width=1.5,
            annotation_text="journalist total",
            annotation_font=dict(color=C["muted"], size=9),
        )
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("participation records"))
        return _base(fig, h=260, title=f"A · Participation record coverage — {person}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIG B: Zone Travel Volume Gap
    # Replaces broken industry-share chart. Shows absolute waypoint counts per
    # zone per dataset. The government-zone gap is the key missing evidence (b).
    # ─────────────────────────────────────────────────────────────────────────
    def fig_zone_volume_gap(person):
        rows = []
        for ds in DS_ORDER:
            df = ZONES[ds]
            sub = df[df["person"] == person]
            for z in ZONE_ORDER:
                n = int(sub.loc[sub["zone"] == z, "trips"].values[0]) \
                    if z in sub["zone"].values else 0
                rows.append({"dataset": ds, "zone": z, "trips": n})
        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for ds in DS_ORDER:
            sub = df_plot[df_plot["dataset"] == ds]
            fig.add_trace(go.Bar(
                name=ds,
                x=sub["zone"],
                y=sub["trips"],
                marker_color=DS_COLORS[ds],
                opacity=0.85,
                hovertemplate=(
                    f"<b>{ds}</b><br>"
                    "zone: %{x}<br>"
                    "waypoints: %{y}<extra></extra>"
                ),
            ))

        fig.update_layout(barmode="group")
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("trip waypoints (absolute)"))
        return _base(
            fig, h=300,
            title=f"B · Zone travel volume — {person}  "
                  f"·  FILAH/TROUT vs journalist ground truth",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FIG C: Zone Framing (100% stacked horizontal)
    # Proportional view — reveals FILAH's commercial narrative. Sub-question (d).
    # ─────────────────────────────────────────────────────────────────────────
    def fig_zone_framing(person):
        rows = []
        for ds in DS_ORDER:
            df = ZONES[ds]
            sub = df[df["person"] == person]
            if sub.empty:
                rows.append({"dataset": ds, "zone": "no travel recorded",
                             "pct": 100.0, "raw": 0})
                continue
            tot = sub["trips"].sum()
            for _, r in sub.iterrows():
                rows.append({
                    "dataset": ds,
                    "zone":    r["zone"],
                    "pct":     round(100 * r["trips"] / tot, 1),
                    "raw":     int(r["trips"]),
                })
        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for z in ZONE_ORDER + ["no travel recorded"]:
            sub = df_plot[df_plot["zone"] == z]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                name=z,
                x=sub["pct"],
                y=sub["dataset"],
                orientation="h",
                marker_color=ZONE_COLORS.get(z, C["neutral"]),
                opacity=0.88,
                customdata=sub["raw"],
                hovertemplate=(
                    f"zone: {z}<br>"
                    "%{x:.1f}%  (%{customdata} waypoints)<extra></extra>"
                ),
            ))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(**make_xax(), range=[0, 101], ticksuffix="%")
        fig.update_yaxes(tickfont=dict(color=C["text"]))
        return _base(
            fig, h=240,
            title=f"C · Zone framing — {person}  ·  proportional breakdown per dataset",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FIG D: Coverage Scorecard — all members ranked
    # % of journalist records captured by FILAH/TROUT. Sub-question (c).
    # ─────────────────────────────────────────────────────────────────────────
    def fig_bias_scorecard(selected_person):
        if BIAS_DF.empty:
            return go.Figure()

        order = (
            BIAS_DF[BIAS_DF["dataset"] == "FILAH"]
            .sort_values("coverage")["person"]
            .tolist()
        )
        for p in BIAS_DF["person"].unique():
            if p not in order:
                order.append(p)

        fig = go.Figure()
        for ds in ["FILAH", "TROUT"]:
            sub = BIAS_DF[BIAS_DF["dataset"] == ds].copy()
            sub["person"] = pd.Categorical(sub["person"], categories=order, ordered=True)
            sub = sub.sort_values("person")

            mask = sub["person"].astype(str) == selected_person
            rest = sub[~mask]
            sel  = sub[mask]

            hover = (
                "<b>%{x}</b>  ·  " + ds + "<br>"
                "coverage: %{y:.1f}%<br>"
                "recorded: %{customdata[0]} / %{customdata[1]} records<br>"
                "missing: %{customdata[2]}<extra></extra>"
            )

            if not rest.empty:
                fig.add_trace(go.Bar(
                    name=ds,
                    x=rest["person"].astype(str),
                    y=rest["coverage"],
                    marker_color=DS_COLORS[ds],
                    opacity=0.6,
                    customdata=rest[["n_ds", "n_journo", "missing"]].values,
                    hovertemplate=hover,
                ))
            if not sel.empty:
                fig.add_trace(go.Bar(
                    name=f"{ds} (selected)",
                    x=sel["person"].astype(str),
                    y=sel["coverage"],
                    marker_color="#ffffff",
                    marker_line_color=DS_COLORS[ds],
                    marker_line_width=2,
                    opacity=1.0,
                    showlegend=False,
                    customdata=sel[["n_ds", "n_journo", "missing"]].values,
                    hovertemplate=hover,
                ))

        fig.add_hline(
            y=100, line_dash="dot", line_color=C["journalist"], line_width=1.5,
            annotation_text="full coverage",
            annotation_font=dict(color=C["muted"], size=9),
        )
        fig.update_layout(barmode="group")
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(
            **make_yax("% of journalist records captured"),
            range=[0, 115], ticksuffix="%",
        )
        return _base(
            fig, h=280,
            title="D · Coverage scorecard — % of journalist records captured "
                  "(white = selected member)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FIG E: Missing Evidence by Zone
    # Replaces broken sentiment radar. Shows journalist waypoints absent from
    # each dataset per zone. Directly answers sub-question (b).
    # ─────────────────────────────────────────────────────────────────────────
    def fig_missing_evidence(person):
        journo_sub = ZONES["journalist"]
        journo_sub = journo_sub[journo_sub["person"] == person]

        rows = []
        for ds in ["FILAH", "TROUT"]:
            ds_sub = ZONES[ds]
            ds_sub = ds_sub[ds_sub["person"] == person]
            for z in ZONE_ORDER:
                j_n = int(journo_sub.loc[journo_sub["zone"] == z, "trips"].values[0]) \
                      if z in journo_sub["zone"].values else 0
                d_n = int(ds_sub.loc[ds_sub["zone"] == z, "trips"].values[0]) \
                      if z in ds_sub["zone"].values else 0
                rows.append({
                    "dataset":   ds,
                    "zone":      z,
                    "missing":   max(0, j_n - d_n),
                    "journo":    j_n,
                    "recorded":  d_n,
                })
        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for ds in ["FILAH", "TROUT"]:
            sub = df_plot[df_plot["dataset"] == ds]
            fig.add_trace(go.Bar(
                name=f"{ds} gap",
                x=sub["zone"],
                y=sub["missing"],
                marker_color=DS_COLORS[ds],
                opacity=0.85,
                customdata=sub[["journo", "recorded"]].values,
                hovertemplate=(
                    f"<b>{ds}</b> — missing waypoints<br>"
                    "zone: %{x}<br>"
                    "missing: %{y}  "
                    "(journalist: %{customdata[0]}, "
                    "recorded: %{customdata[1]})<extra></extra>"
                ),
            ))

        fig.update_layout(barmode="group")
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("waypoints absent from dataset"))
        return _base(
            fig, h=300,
            title=f"E · Missing evidence by zone — {person}  "
                  f"·  journalist waypoints not in FILAH / TROUT",
        )

    return (
        DS_COLORS,
        DS_ORDER,
        fig_activity_profile,
        fig_bias_scorecard,
        fig_missing_evidence,
        fig_zone_framing,
        fig_zone_volume_gap,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 5 — person selector
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(ALL_PERSONS, mo):
    person_selector = mo.ui.dropdown(
        options=ALL_PERSONS,
        value="Teddy Goldstein",
        label="Select COOTEFOO member",
    )
    return (person_selector,)


# ─────────────────────────────────────────────────────────────────────────────
#  CELL 6 — reactive layout
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(
    ALL_BIAS, BIAS_DF, SENT, ZONES,
    fig_activity_profile,
    fig_bias_scorecard,
    fig_missing_evidence,
    fig_zone_framing,
    fig_zone_volume_gap,
    mo,
    person_selector,
    pd,
):
    _person = person_selector.value

    # Which datasets contain this person's meeting records?
    _in_datasets = [
        ds for ds in ["FILAH", "TROUT", "journalist"]
        if not SENT[ds].empty and _person in SENT[ds]["person"].values
    ]
    _coverage_str = " · ".join(_in_datasets) if _in_datasets else "not recorded"

    # Government-zone suppression summary
    _journo_z = ZONES["journalist"]
    _filah_z  = ZONES["FILAH"]
    _journo_gov = int(_journo_z.loc[
        (_journo_z["person"] == _person) & (_journo_z["zone"] == "government"), "trips"
    ].sum())
    _filah_gov = int(_filah_z.loc[
        (_filah_z["person"] == _person) & (_filah_z["zone"] == "government"), "trips"
    ].sum())
    _gov_gap = _journo_gov - _filah_gov

    # Per-dataset record coverage formatter
    def _fmt_cov(person, ds):
        if BIAS_DF.empty:
            return "**no data**"
        row = BIAS_DF[(BIAS_DF["person"] == person) & (BIAS_DF["dataset"] == ds)]
        if row.empty:
            return "**no data**"
        cov  = float(row["coverage"].values[0])
        n_ds = int(row["n_ds"].values[0])
        n_j  = int(row["n_journo"].values[0])
        miss = int(row["missing"].values[0])
        label = ("fully excluded" if cov == 0
                 else "partially covered" if cov < 100
                 else "fully covered")
        return f"**{cov:.0f}%** ({n_ds}/{n_j} records, {miss} missing) — *{label}*"

    # Most FILAH-excluded member (sub-question c answer)
    if not BIAS_DF.empty:
        _filah_rank        = BIAS_DF[BIAS_DF["dataset"] == "FILAH"].sort_values("coverage")
        _most_excl         = _filah_rank.iloc[0]["person"]
        _most_excl_cov     = _filah_rank.iloc[0]["coverage"]
        _most_excl_missing = int(_filah_rank.iloc[0]["missing"])
    else:
        _most_excl         = "—"
        _most_excl_cov     = 100.0
        _most_excl_missing = 0

    # ── Callout cards ─────────────────────────────────────────────────────────
    _card_presence = mo.callout(
        mo.md(
            f"**{_person}** appears in: {_coverage_str}\n\n"
            f"Government-zone travel — journalist: **{_journo_gov}** waypoints, "
            f"FILAH: **{_filah_gov}** — **{_gov_gap} suppressed**"
        ),
        kind="info",
    )

    _card_coverage = mo.callout(
        mo.md(
            f"Meeting record coverage vs journalist:\n\n"
            f"- FILAH: {_fmt_cov(_person, 'FILAH')}\n"
            f"- TROUT: {_fmt_cov(_person, 'TROUT')}"
        ),
        kind="warn",
    )

    _card_most_excl = mo.callout(
        mo.md(
            f"Most FILAH-excluded member *(sub-question c)*:\n\n"
            f"**{_most_excl}** — "
            f"**{_most_excl_cov:.0f}%** coverage "
            f"({_most_excl_missing} records missing)"
        ),
        kind="danger",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md("# Q4 · Person Lens — Dataset Comparison"),
        mo.md(
            "Each chart reacts to the selected member. "
            "The key finding is that the bias between datasets is "
            "**not** in how events were scored, but in **who was included** "
            "and **which travel records were kept**.\n\n"
            "- **(a)** TROUT-accused members: select Teddy Goldstein or Ed Helpsford "
            "and observe how FILAH excluded both entirely  \n"
            "- **(b)** Missing evidence: Figs B and E quantify suppressed government-zone "
            "travel — the main omission  \n"
            "- **(c)** Most FILAH-distorted: Fig D ranks all members by coverage; "
            "the red card above names the answer  \n"
            "- **(d)** FILAH bias: Fig C shows the commercial-only narrative FILAH "
            "constructs by omitting government travel"
        ),

        mo.hstack([person_selector], gap="1rem"),

        mo.hstack(
            [_card_presence, _card_coverage, _card_most_excl],
            widths="equal", gap="0.5rem",
        ),

        mo.md("---"),

        # Row 1: A + B
        mo.hstack([
            mo.vstack([
                mo.md("#### A · Participation Record Coverage"),
                mo.md(
                    "Participation records per dataset. "
                    "Zero bar = person fully excluded from meeting record."
                ),
                mo.ui.plotly(fig_activity_profile(_person)),
            ]),
            mo.vstack([
                mo.md("#### B · Zone Travel Volume *(sub-question b)*"),
                mo.md(
                    "Absolute waypoints per zone. "
                    "FILAH records almost zero government-zone travel; "
                    "journalist shows hundreds — the suppressed evidence."
                ),
                mo.ui.plotly(fig_zone_volume_gap(_person)),
            ]),
        ], widths="equal", gap="1rem"),

        mo.md("---"),

        # Row 2: C + E
        mo.hstack([
            mo.vstack([
                mo.md("#### C · Zone Framing *(sub-question d)*"),
                mo.md(
                    "Proportional view. FILAH bars are almost entirely "
                    "commercial, hiding the government-zone majority that "
                    "the journalist dataset reveals."
                ),
                mo.ui.plotly(fig_zone_framing(_person)),
            ]),
            mo.vstack([
                mo.md("#### E · Missing Evidence by Zone *(sub-question b)*"),
                mo.md(
                    "Journalist waypoints absent from FILAH and TROUT. "
                    "The government bar is the tallest — the largest gap "
                    "between what was reported and the full record."
                ),
                mo.ui.plotly(fig_missing_evidence(_person)),
            ]),
        ], widths="equal", gap="1rem"),

        mo.md("---"),

        # Row 3: D (full width — shows all members at once)
        mo.vstack([
            mo.md("#### D · Coverage Scorecard — All Members *(sub-question c)*"),
            mo.md(
                "% of journalist participation records captured per member. "
                "0% = fully excluded. White bar = currently selected member. "
                "Members absent from a dataset appear at 0% for that dataset."
            ),
            mo.ui.plotly(fig_bias_scorecard(_person)),
        ]),

    ], gap="0.75rem")


if __name__ == "__main__":
    app.run()
