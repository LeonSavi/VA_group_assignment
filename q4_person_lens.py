import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json, os
    from collections import defaultdict

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    # ── colour palette (shared with the rest of the dashboard) ───────────────
    C = {
        "bg":         "#080c14",
        "surface":    "#0e1420",
        "card":       "#131929",
        "border":     "#1e2d45",
        "tourism":    "#22d3ee",   # cyan
        "fishing":    "#fb923c",   # amber
        "neutral":    "#94a3b8",
        "positive":   "#4ade80",
        "negative":   "#f87171",
        "accent":     "#7c6af7",   # violet
        "text":       "#e2e8f0",
        "muted":      "#64748b",
        # dataset colours
        "FILAH":      "#fb923c",
        "TROUT":      "#22d3ee",
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
    ZONE_ORDER = ["tourism", "commercial", "industrial", "government", "residential", "connector"]

    LAYOUT_BASE = dict(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(color=C["text"], family="'DM Mono', monospace, sans-serif"),
        margin=dict(l=16, r=16, t=42, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=10)),
    )

    def make_title(txt):
        return dict(text=txt, font=dict(color=C["muted"], size=11), x=0.01)

    def make_xax():
        return dict(tickfont=dict(color=C["text"]), gridcolor=C["border"])

    def make_yax(label=""):
        return dict(tickfont=dict(color=C["text"]), gridcolor=C["border"], zeroline=False,
                    title=dict(text=label, font=dict(color=C["muted"], size=10)))

    return (
        C,
        INDUSTRY_COLORS,
        LAYOUT_BASE,
        ZONE_COLORS,
        ZONE_ORDER,
        defaultdict,
        go,
        json,
        make_title,
        make_xax,
        make_yax,
        mo,
        os,
        pd,
    )


@app.cell
def _(json, os):
    BASE = os.path.dirname(os.path.abspath(__file__))

    def load_graph(filename):
        path = os.path.join(BASE, "data", filename)
        with open(path) as f:
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
    return ALL_PERSONS, DATASETS


@app.cell
def _(ALL_PERSONS, DATASETS, defaultdict, pd):

    # ── A) Sentiment per person × industry ───────────────────────────────────
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
                rows.append({"person": pid, "industry": ind, "sentiment": float(s),
                             "reason": lnk.get("reason", "")})
        if not rows:
            return pd.DataFrame(columns=["person", "industry", "avg_sentiment", "n"])
        return (pd.DataFrame(rows)
                .groupby(["person", "industry"])["sentiment"]
                .agg(avg_sentiment="mean", n="count")
                .reset_index())

    SENT = {k: sentiment_df(*v) for k, v in DATASETS.items()}

    # ── B) Activity record counts per person (all participant links) ──────────
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

    # ── C) Zone travel per person ─────────────────────────────────────────────
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
        rows = [{"person": p, "zone": z}
                for trip, p in trip_person.items()
                for z in trip_places[trip]]
        if not rows:
            return pd.DataFrame(columns=["person", "zone", "trips"])
        return pd.DataFrame(rows).groupby(["person", "zone"]).size().reset_index(name="trips")

    ZONES = {k: zone_df(*v) for k, v in DATASETS.items()}

    # ── D) Industry share per person (% of sentimentised events) ─────────────
    def industry_share_df(nodes, links):
        """Returns person | industry | share (%) | n  using raw event counts."""
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
                rows.append({"person": pid, "industry": ind})
        if not rows:
            return pd.DataFrame(columns=["person", "industry", "n", "share"])
        df = pd.DataFrame(rows).groupby(["person", "industry"]).size().reset_index(name="n")
        totals = df.groupby("person")["n"].transform("sum")
        df["share"] = (100 * df["n"] / totals).round(1)
        return df

    SHARES = {k: industry_share_df(*v) for k, v in DATASETS.items()}

    # ── E) Bias score: KL-like divergence from journalist baseline ────────────
    def bias_scores(person):
        """
        For each dataset, compute the absolute deviation in tourism-share
        versus the journalist (ground truth) share for that person.
        Returns dict: {dataset_name: delta_pct or None}
        """
        ref = SHARES["journalist"]
        ref_person = ref[ref["person"] == person]
        if ref_person.empty:
            return {}
        ref_tourism = ref_person.loc[ref_person["industry"] == "tourism", "share"]
        ref_val = float(ref_tourism.values[0]) if len(ref_tourism) else 0.0

        scores = {}
        for ds in ["FILAH", "TROUT"]:
            sub = SHARES[ds]
            sub_p = sub[sub["person"] == person]
            if sub_p.empty:
                scores[ds] = None
                continue
            t = sub_p.loc[sub_p["industry"] == "tourism", "share"]
            ds_val = float(t.values[0]) if len(t) else 0.0
            scores[ds] = round(ds_val - ref_val, 1)   # positive = over-represents tourism
        return scores

    ALL_BIAS = {p: bias_scores(p) for p in ALL_PERSONS}

    # ── F) Global bias summary (for score-card chart) ─────────────────────────
    bias_rows = []
    for _p in ALL_PERSONS:
        for _ds, _delta in ALL_BIAS[_p].items():
            if _delta is not None:
                bias_rows.append({"person": _p, "dataset": _ds, "delta": _delta})
    BIAS_DF = pd.DataFrame(bias_rows) if bias_rows else pd.DataFrame(columns=["person","dataset","delta"])
    return ACTIVITY, ALL_BIAS, BIAS_DF, SENT, SHARES, ZONES


@app.cell
def _(
    ACTIVITY,
    BIAS_DF,
    C,
    INDUSTRY_COLORS,
    LAYOUT_BASE,
    SENT,
    SHARES,
    ZONES,
    ZONE_COLORS,
    ZONE_ORDER,
    go,
    make_title,
    make_xax,
    make_yax,
    pd,
):

    # ── shared helpers ────────────────────────────────────────────────────────
    def _base(fig, h=300, title=None):
        kw = dict(**LAYOUT_BASE, height=h)
        if title:
            kw["title"] = make_title(title)
        fig.update_layout(**kw)
        return fig

    DS_ORDER = ["FILAH", "TROUT", "journalist"]
    DS_COLORS = {k: C[k] for k in DS_ORDER}

    # ─────────────────────────────────────────────────────────────────────────
    # FIG A: Activity Profile — grouped bar, 3 datasets, activity count
    # ─────────────────────────────────────────────────────────────────────────
    def fig_activity_profile(person):
        """
        Grouped bars: how many participation records each dataset holds for
        this person vs. the journalist baseline. Missing = dataset excluded them.
        """
        rows = []
        for ds in DS_ORDER:
            df = ACTIVITY[ds]
            sub = df[df["person"] == person]
            n = int(sub["n"].values[0]) if not sub.empty else 0
            rows.append({"dataset": ds, "n": n})
        df = pd.DataFrame(rows)

        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                name=row["dataset"],
                x=[row["dataset"]],
                y=[row["n"]],
                marker_color=DS_COLORS[row["dataset"]],
                opacity=0.88,
                showlegend=False,
                hovertemplate=f"<b>{row['dataset']}</b><br>activity records: {row['n']}<extra></extra>",
            ))

        # reference line = journalist value
        journo_n = int(df[df["dataset"] == "journalist"]["n"].values[0])
        fig.add_hline(y=journo_n, line_dash="dot", line_color=C["journalist"],
                      line_width=1.5, annotation_text="full record",
                      annotation_font=dict(color=C["muted"], size=9))

        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("participation records"))
        return _base(fig, h=260, title=f"A · Activity coverage — {person}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIG B: Industry Bias Strip — diverging bars + reference line
    # ─────────────────────────────────────────────────────────────────────────
    def fig_industry_bias(person):
        """
        For each dataset: % of this person's sentimentised events in each industry.
        Journalist values are shown as reference lines on each sub-bar.
        """
        industries = ["tourism", "large vessel", "small vessel"]
        ref_row = SHARES["journalist"]
        ref_p   = ref_row[ref_row["person"] == person]

        def _get_share(ds, ind):
            sub = SHARES[ds]
            s = sub[(sub["person"] == person) & (sub["industry"] == ind)]
            return float(s["share"].values[0]) if not s.empty else 0.0

        fig = go.Figure()
        for ind in industries:
            ref_val = _get_share("journalist", ind)
            for ds in ["FILAH", "TROUT"]:
                val = _get_share(ds, ind)
                fig.add_trace(go.Bar(
                    name=f"{ds} · {ind}",
                    x=[f"{ds}"],
                    y=[val],
                    offsetgroup=ind,
                    legendgroup=ind,
                    showlegend=(ds == "FILAH"),
                    marker_color=INDUSTRY_COLORS.get(ind, C["neutral"]),
                    opacity=0.85,
                    hovertemplate=(
                        f"<b>{ds}</b> · {ind}<br>"
                        f"share: {val:.1f}%<br>"
                        f"journalist baseline: {ref_val:.1f}%<extra></extra>"
                    ),
                ))

        # Draw reference lines for journalist per industry
        for i, ind in enumerate(industries):
            ref_val = _get_share("journalist", ind)
            # annotate as shapes (one per industry column — approximate x positions)
            # We add invisible scatter to carry annotation
            fig.add_trace(go.Scatter(
                x=["FILAH", "TROUT"],
                y=[ref_val, ref_val],
                mode="lines",
                line=dict(color=C["journalist"], width=1.5, dash="dot"),
                name=f"journalist · {ind}" if i == 0 else "",
                legendgroup="journalist",
                showlegend=(i == 0),
                hovertemplate=f"journalist baseline ({ind}): {ref_val:.1f}%<extra></extra>",
            ))

        fig.update_layout(barmode="group")
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("% of events"), range=[0, 105], ticksuffix="%")
        return _base(fig, h=300, title=f"B · Industry focus — {person}  ·  vs journalist baseline (dotted)")

    # ─────────────────────────────────────────────────────────────────────────
    # FIG C: Zone Framing — 100% stacked bar across 3 datasets
    # ─────────────────────────────────────────────────────────────────────────
    def fig_zone_framing(person):
        """
        100% stacked horizontal bar: zone distribution per dataset.
        Shows how FILAH/TROUT frame travel story vs full record.
        """
        rows = []
        for ds in DS_ORDER:
            df = ZONES[ds]
            sub = df[df["person"] == person]
            if sub.empty:
                rows.append({"dataset": ds, "zone": "—no travel data—", "pct": 100, "raw": 0})
                continue
            tot = sub["trips"].sum()
            for _, r in sub.iterrows():
                rows.append({"dataset": ds, "zone": r["zone"],
                             "pct": round(100 * r["trips"] / tot, 1),
                             "raw": int(r["trips"])})
        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for z in ZONE_ORDER + ["—no travel data—"]:
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
        return _base(fig, h=240, title=f"C · Travel zone framing — {person}  ·  each dataset's story")

    # ─────────────────────────────────────────────────────────────────────────
    # FIG D: Sampling Bias Scorecard — all persons ranked
    # ─────────────────────────────────────────────────────────────────────────
    def fig_bias_scorecard(selected_person):
        """
        For every person × dataset, shows tourism-share delta from journalist.
        Selected person is highlighted. Answers sub-question (c).
        """
        if BIAS_DF.empty:
            return go.Figure()

        fig = go.Figure()
        for ds in ["FILAH", "TROUT"]:
            sub = BIAS_DF[BIAS_DF["dataset"] == ds].copy()
            # persons not in this dataset show as NaN already (excluded by bias_scores)
            sub = sub.sort_values("delta", key=abs, ascending=False)

            mask = sub["person"] == selected_person
            rest = sub[~mask]
            sel  = sub[mask]

            if not rest.empty:
                fig.add_trace(go.Bar(
                    name=ds,
                    x=rest["person"],
                    y=rest["delta"],
                    marker_color=DS_COLORS[ds],
                    opacity=0.55,
                    showlegend=(ds == "FILAH"),
                    hovertemplate=(
                        "<b>%{x}</b>  ·  " + ds + "<br>"
                        "tourism share Δ from journalist: %{y:+.1f}pp<extra></extra>"
                    ),
                ))
            if not sel.empty:
                fig.add_trace(go.Bar(
                    name=f"{ds} (selected)",
                    x=sel["person"],
                    y=sel["delta"],
                    marker_color="#ffffff",
                    marker_line_color=DS_COLORS[ds],
                    marker_line_width=2,
                    opacity=1.0,
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{x}</b>  ·  " + ds + "<br>"
                        "tourism share Δ from journalist: %{y:+.1f}pp<extra></extra>"
                    ),
                ))

        fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
        fig.update_layout(barmode="group")
        fig.update_xaxes(**make_xax())
        fig.update_yaxes(**make_yax("Δ tourism share vs journalist (pp)"))
        return _base(fig, h=280,
                     title="D · Sampling bias scorecard — Δ tourism share vs journalist baseline (highlighted = selected member)")

    # ─────────────────────────────────────────────────────────────────────────
    # FIG E: Sentiment divergence — person across datasets (spider-like bars)
    # ─────────────────────────────────────────────────────────────────────────
    def fig_sentiment_divergence(person):
        """
        For each industry, show this person's avg sentiment in each dataset.
        Journalist line overlaid as reference.
        """
        industries = ["tourism", "large vessel", "small vessel"]

        def _get_sent(ds, ind):
            sub = SENT[ds]
            s = sub[(sub["person"] == person) & (sub["industry"] == ind)]
            return float(s["avg_sentiment"].values[0]) if not s.empty else None

        fig = go.Figure()
        for ds in ["FILAH", "TROUT", "journalist"]:
            vals = [_get_sent(ds, ind) for ind in industries]
            fig.add_trace(go.Scatterpolar(
                r=[v if v is not None else 0 for v in vals],
                theta=industries,
                fill="toself",
                name=ds,
                line=dict(color=DS_COLORS[ds], width=2),
                fillcolor=DS_COLORS[ds],
                opacity=0.25 if ds != "journalist" else 0.15,
                hovertemplate="<b>%{theta}</b><br>" + ds + ": %{r:.2f}<extra></extra>",
            ))

        fig.update_layout(
            polar=dict(
                bgcolor=C["card"],
                radialaxis=dict(
                    visible=True, range=[-1.2, 1.2],
                    tickfont=dict(color=C["muted"], size=8),
                    gridcolor=C["border"],
                    linecolor=C["border"],
                ),
                angularaxis=dict(
                    tickfont=dict(color=C["text"], size=10),
                    gridcolor=C["border"],
                    linecolor=C["border"],
                ),
            ),
            showlegend=True,
        )
        return _base(fig, h=300, title=f"E · Sentiment profile — {person}  ·  FILAH vs TROUT vs journalist")

    return (
        fig_activity_profile,
        fig_bias_scorecard,
        fig_industry_bias,
        fig_sentiment_divergence,
        fig_zone_framing,
    )


@app.cell
def _(ALL_PERSONS, mo):
    person_selector = mo.ui.dropdown(
        options=ALL_PERSONS,
        value="Teddy Goldstein",
        label="🔍  Select COOTEFOO member",
    )
    return (person_selector,)


@app.cell
def _(
    ALL_BIAS,
    BIAS_DF,
    SENT,
    fig_activity_profile,
    fig_bias_scorecard,
    fig_industry_bias,
    fig_sentiment_divergence,
    fig_zone_framing,
    mo,
    person_selector,
):
    _person = person_selector.value

    # ── derive narrative callouts ─────────────────────────────────────────────

    # Coverage: which datasets include this person?
    _in_datasets = []
    for _ds in ["FILAH", "TROUT", "journalist"]:
        _sub = SENT[_ds]
        if not _sub.empty and _person in _sub["person"].values:
            _in_datasets.append(_ds)
    _coverage_str = " · ".join(_in_datasets) if _in_datasets else "not recorded"

    # Bias delta
    _bias = ALL_BIAS.get(_person, {})
    _filah_delta = _bias.get("FILAH")
    _trout_delta = _bias.get("TROUT")

    def _fmt_delta(d):
        if d is None:
            return "**not in dataset**"
        sign = "+" if d >= 0 else ""
        direction = "over-represents tourism" if d > 2 else ("under-represents tourism" if d < -2 else "close to baseline")
        return f"**{sign}{d:.1f} pp** ({direction})"

    # Missing events (in journalist but not TROUT)
    _journo_acts = SENT["journalist"]
    _trout_acts  = SENT["TROUT"]
    _journo_n = int(_journo_acts.loc[_journo_acts["person"] == _person, "n"].sum()) if _person in _journo_acts["person"].values else 0
    _trout_n  = int(_trout_acts.loc[_trout_acts["person"]  == _person, "n"].sum()) if _person in _trout_acts["person"].values else 0
    _missing_n = _journo_n - _trout_n

    # Highest-bias person in FILAH
    if not BIAS_DF.empty:
        _filah_bias = BIAS_DF[BIAS_DF["dataset"] == "FILAH"].copy()
        _filah_bias["abs_delta"] = _filah_bias["delta"].abs()
        _filah_bias = _filah_bias.sort_values("abs_delta", ascending=False)
        _most_biased = _filah_bias.iloc[0]["person"] if not _filah_bias.empty else "—"
        _most_biased_delta = _filah_bias.iloc[0]["delta"] if not _filah_bias.empty else 0
    else:
        _most_biased = "—"
        _most_biased_delta = 0

    # ── callout cards ─────────────────────────────────────────────────────────
    _card_coverage = mo.callout(
        mo.md(
            f"**{_person}** appears in: {_coverage_str}\n\n"
            f"Journalist vs TROUT: **{_journo_n}** records vs **{_trout_n}** — "
            f"**{_missing_n} events missing** from TROUT"
        ),
        kind="info",
    )

    _card_bias = mo.callout(
        mo.md(
            f"Tourism-share deviation from journalist baseline:\n\n"
            f"- FILAH: {_fmt_delta(_filah_delta)}\n"
            f"- TROUT: {_fmt_delta(_trout_delta)}"
        ),
        kind="warn",
    )

    _card_filah_most_biased = mo.callout(
        mo.md(
            f"Most FILAH-distorted member (sub-question c):\n\n"
            f"**{_most_biased}** — tourism share Δ = **{_most_biased_delta:+.1f} pp** vs journalist"
        ),
        kind="danger",
    )

    # ── assemble layout ───────────────────────────────────────────────────────
    mo.vstack([
        mo.md("# 🔬 Q4 · Person Lens — Dataset Comparison"),
        mo.md(
            "Explore how each dataset tells a **different story** about the same COOTEFOO member. "
            "Select a member to see their activity profile, industry bias, travel framing, "
            "and how much of the journalist's complete record was captured by FILAH and TROUT.\n\n"
            "Sub-questions: **(a)** TROUT-accused members · **(b)** missing evidence · "
            "**(c)** most FILAH-biased member · **(d)** FILAH bias illustrated"
        ),

        # controls row
        mo.hstack([person_selector], gap="1rem"),

        # callout info strip
        mo.hstack([_card_coverage, _card_bias, _card_filah_most_biased],
                  widths="equal", gap="0.5rem"),

        mo.md("---"),

        # Row 1: Activity profile + Sentiment spider
        mo.hstack([
            mo.vstack([
                mo.md("#### A · Activity Coverage"),
                mo.md("How many participation records does each dataset hold for this member? "
                      "Missing bars = the dataset excluded this person entirely."),
                mo.ui.plotly(fig_activity_profile(_person)),
            ]),
            mo.vstack([
                mo.md("#### E · Sentiment Profile (radar)"),
                mo.md("How does this member's sentiment across industries compare between "
                      "FILAH, TROUT, and the full journalist record?"),
                mo.ui.plotly(fig_sentiment_divergence(_person)),
            ]),
        ], widths="equal", gap="1rem"),

        mo.md("---"),

        # Row 2: Industry bias strip
        mo.vstack([
            mo.md("#### B · Industry Focus — Sub-question (a) & (b)"),
            mo.md("What share of this member's recorded events touch each industry? "
                  "The dotted line shows the journalist baseline — the gap is the "
                  "**missing or distorted evidence** (sub-question b)."),
            mo.ui.plotly(fig_industry_bias(_person)),
        ]),

        mo.md("---"),

        # Row 3: Zone framing + scorecard
        mo.hstack([
            mo.vstack([
                mo.md("#### C · Travel Zone Framing — Sub-question (d)"),
                mo.md("100% stacked bar of travel zones per dataset. "
                      "Reveals how FILAH channels this member toward commercial zones "
                      "and TROUT toward government/tourism zones, vs the full picture."),
                mo.ui.plotly(fig_zone_framing(_person)),
            ]),
            mo.vstack([
                mo.md("#### D · Bias Scorecard — Sub-question (c)"),
                mo.md("Tourism-share deviation (Δ percentage points) from journalist baseline "
                      "for every member. The **highlighted bar** is the selected member. "
                      "Tallest bar = most sampling-biased."),
                mo.ui.plotly(fig_bias_scorecard(_person)),
            ]),
        ], widths="equal", gap="1rem"),

    ], gap="0.75rem")
    return


if __name__ == "__main__":
    app.run()
