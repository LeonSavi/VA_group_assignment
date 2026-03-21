# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import numpy as np
    import json
    import os
    return alt, json, mo, os, pd


@app.cell
def _(json, os, pd):
    from datetime import datetime as _dt

    _DATA_DIR = "data"

    def _load(name):
        with open(os.path.join(_DATA_DIR, name)) as f:
            return json.load(f)

    _graphs = {
        "FILAH": _load("FILAH.json"),
        "TROUT": _load("TROUT.json"),
        "Journalist": _load("journalist.json"),
    }

    def _classify_zone(zone, zone_detail):
        zone = (zone or "").lower()
        detail = (zone_detail or "").lower()
        if zone == "industrial":
            return "fishing"
        if zone == "government" and detail in ("inspections", "customs"):
            return "fishing"
        if zone == "tourism":
            return "tourism"
        return "neutral"

    _TOPIC_SIDE = {
        "fish_vacuum": "fishing", "low_volume_crane": "fishing",
        "new_crane_lomark": "fishing", "deep_fishing_dock": "fishing",
        "heritage_walking_tour": "tourism", "waterfront_market": "tourism",
        "concert": "tourism", "expanding_tourist_wharf": "tourism",
        "marine_life_deck": "tourism",
        "affordable_housing": "neutral", "seafood_festival": "neutral",
        "statue_john_smoth": "neutral", "name_inspection_office": "neutral",
        "renaming_park_himark": "neutral", "name_harbor_area": "neutral",
    }

    _sent_rows = []
    _trip_rows = []
    _hours_rows = []
    _status_rows = []

    for _ds_name, _g in _graphs.items():
        _nmap = {n["id"]: n for n in _g["nodes"]}
        _trips = {n["id"]: n for n in _g["nodes"] if n.get("type") == "trip"}

        _trip_to_person = {}
        _trip_to_places = {}
        for _e in _g["links"]:
            _st = _nmap.get(_e["source"], {}).get("type", "")
            _tt = _nmap.get(_e["target"], {}).get("type", "")
            if _st == "trip" and _tt == "entity.person":
                _trip_to_person[_e["source"]] = _e["target"]
            if _st == "trip" and _tt == "place":
                _trip_to_places.setdefault(_e["source"], []).append(_e["target"])

        for _e in _g["links"]:
            if _e.get("role") != "participant" or _e.get("sentiment") is None:
                continue
            _tgt_node = _nmap.get(_e["target"], {})
            if _tgt_node.get("type") != "entity.person":
                continue
            _industries = _e.get("industry", [])
            if isinstance(_industries, str):
                _industries = [_industries]
            _src_id = _e["source"]
            _topic = "unknown"
            for _e2 in _g["links"]:
                if _e2["source"] == _src_id and _e2.get("role") in ("about", "plan"):
                    _t_node = _nmap.get(_e2["target"], {})
                    if _t_node.get("type") == "topic":
                        _topic = _t_node.get("short_topic", _e2["target"])
                        break
            for _ind in (_industries if _industries else ["unspecified"]):
                _sent_rows.append({
                    "dataset": _ds_name, "person": _e["target"],
                    "sentiment": _e["sentiment"], "industry": _ind, "topic": _topic,
                })

        for _trip_id, _person in _trip_to_person.items():
            for _place_id in _trip_to_places.get(_trip_id, []):
                _pn = _nmap.get(_place_id, {})
                _trip_rows.append({
                    "dataset": _ds_name, "person": _person,
                    "zone": _pn.get("zone", "unknown"),
                })

        for _tid, _tdata in _trips.items():
            _person = _trip_to_person.get(_tid)
            if not _person:
                continue
            try:
                _s = _dt.strptime(_tdata["start"], "%H:%M:%S")
                _e_time = _dt.strptime(_tdata["end"], "%H:%M:%S")
                _hours = (_e_time - _s).total_seconds() / 3600
                if _hours < 0:
                    _hours += 24
            except Exception:
                continue
            _places = _trip_to_places.get(_tid, [])
            if not _places:
                continue
            _per_place = _hours / len(_places)
            for _pid in _places:
                _pn = _nmap.get(_pid, {})
                _hours_rows.append({
                    "dataset": _ds_name, "person": _person,
                    "zone": _pn.get("zone", "unknown"),
                    "zone_detail": _pn.get("zone_detail", "") or "",
                    "industry_side": _classify_zone(_pn.get("zone", ""), _pn.get("zone_detail", "")),
                    "hours": _per_place,
                })

        for _e in _g["links"]:
            if _e.get("role") != "about" or not _e.get("status"):
                continue
            _src_node = _nmap.get(_e["source"], {})
            _tgt_node = _nmap.get(_e["target"], {})
            if _src_node.get("type") != "discussion" or _tgt_node.get("type") != "plan":
                continue
            _plan_id = _e["target"]
            _status = _e["status"].lower()
            for _e2 in _g["links"]:
                if _e2["source"] == _plan_id and _e2.get("role") == "plan":
                    _t2 = _nmap.get(_e2["target"], {})
                    if _t2.get("type") == "topic":
                        _tn = _t2.get("short_topic", _e2["target"])
                        _status_rows.append({
                            "dataset": _ds_name, "topic": _tn,
                            "topic_side": _TOPIC_SIDE.get(_tn, "neutral"),
                            "plan_id": _plan_id, "status": _status,
                        })
                        break

    df_sent = pd.DataFrame(_sent_rows)
    df_trips = pd.DataFrame(_trip_rows)
    df_hours = pd.DataFrame(_hours_rows)
    df_status = pd.DataFrame(_status_rows)
    TOPIC_SIDE = _TOPIC_SIDE

    _summary = {}
    for _ds_name, _g in _graphs.items():
        _persons = [n["id"] for n in _g["nodes"] if n.get("type") == "entity.person"]
        _n_trips = len([n for n in _g["nodes"] if n.get("type") == "trip"])
        _n_disc = len([n for n in _g["nodes"] if n.get("type") == "discussion"])
        _topics = sorted({n.get("short_topic", n["id"]) for n in _g["nodes"] if n.get("type") == "topic"})
        _summary[_ds_name] = {"persons": _persons, "n_trips": _n_trips, "n_disc": _n_disc, "topics": _topics}
    ds_summary = _summary
    return TOPIC_SIDE, df_hours, df_sent, df_status, df_trips


@app.cell
def _(alt):
    PERSON_ORDER = [
        "Seal", "Ed Helpsford", "Teddy Goldstein",
        "Simone Kat", "Tante Titan", "Carol Limpet",
    ]
    sent_scale = alt.Scale(domain=[-1, 0, 1], range=["#d73027", "#f7f7f7", "#1a9850"])
    ZONE_ORDER = ["government", "commercial", "tourism", "residential", "industrial", "connector"]
    IND_ORDER = ["large vessel", "small vessel", "tourism"]
    return IND_ORDER, PERSON_ORDER, ZONE_ORDER, sent_scale


@app.cell
def _(mo):
    ctl_ds_a = mo.ui.dropdown(
        options=["FILAH", "TROUT", "Journalist"],
        value="FILAH",
        label="**Dataset A**",
    )
    ctl_ds_b = mo.ui.dropdown(
        options=["FILAH", "TROUT", "Journalist"],
        value="Journalist",
        label="**Dataset B**",
    )
    ctl_person = mo.ui.dropdown(
        options=["All members", "Seal", "Ed Helpsford", "Teddy Goldstein",
                 "Simone Kat", "Tante Titan", "Carol Limpet"],
        value="All members",
        label="**Member**",
    )
    return ctl_ds_a, ctl_ds_b, ctl_person


@app.cell
def _(
    PERSON_ORDER,
    TOPIC_SIDE,
    alt,
    ctl_ds_a,
    ctl_ds_b,
    ctl_person,
    df_hours,
    df_sent,
    df_status,
    pd,
):
    _ds_a = ctl_ds_a.value
    _ds_b = ctl_ds_b.value
    _person = ctl_person.value
    _PERSONS = [_person] if _person != "All members" else PERSON_ORDER

    def _compute_scores(ds_name):
        _df = df_sent[(df_sent["dataset"] == ds_name) & (df_sent["topic"] != "unknown")]
        if _person != "All members":
            _df = _df[_df["person"] == _person]
        _df = _df.copy()
        _df["topic_side"] = _df["topic"].map(TOPIC_SIDE).fillna("neutral")
        _fish_sent = _df[_df["topic_side"] == "fishing"]["sentiment"]
        _tour_sent = _df[_df["topic_side"] == "tourism"]["sentiment"]
        _fish_mean = _fish_sent.mean() if len(_fish_sent) > 0 else 0
        _tour_mean = _tour_sent.mean() if len(_tour_sent) > 0 else 0
        _sent_score = max(-2, min(2, _tour_mean - _fish_mean)) / 2

        _hdf = df_hours[df_hours["dataset"] == ds_name]
        if _person != "All members":
            _hdf = _hdf[_hdf["person"] == _person]
        _fish_h = _hdf[_hdf["industry_side"] == "fishing"]["hours"].sum()
        _tour_h = _hdf[_hdf["industry_side"] == "tourism"]["hours"].sum()
        _total_h = _fish_h + _tour_h
        _travel_score = (_tour_h - _fish_h) / _total_h if _total_h > 0 else 0

        _sdf = df_status[df_status["dataset"] == ds_name]
        _sdf_c = _sdf[_sdf["status"] == "completed"]
        _fp = len(_sdf_c[_sdf_c["topic_side"] == "fishing"])
        _tp = len(_sdf_c[_sdf_c["topic_side"] == "tourism"])
        _total_p = _fp + _tp
        _plans_score = (_tp - _fp) / _total_p if _total_p > 0 else 0

        return {"sentiment": round(_sent_score, 3), "travel": round(_travel_score, 3), "plans": round(_plans_score, 3)}

    _scores_a = _compute_scores(_ds_a)
    _scores_b = _compute_scores(_ds_b)

    _gauge_rows = []
    _metrics = [
        ("Sentiment Bias", "sentiment"),
        ("Travel Time Bias", "travel"),
        ("Action Plan Bias", "plans"),
    ]
    for _label, _key in _metrics:
        _gauge_rows.append({"metric": _label, "dataset": _ds_a, "score": _scores_a[_key]})
        _gauge_rows.append({"metric": _label, "dataset": _ds_b, "score": _scores_b[_key]})

    _gdf = pd.DataFrame(_gauge_rows)
    _metric_order = [m[0] for m in _metrics]

    _needles = alt.Chart(_gdf).mark_point(
        size=200, filled=True, stroke="black", strokeWidth=1.5,
    ).encode(
        x=alt.X("score:Q", title=None, scale=alt.Scale(domain=[-1, 1]),
                 axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1],
                               labelExpr="datum.value == -1 ? 'Fishing' : datum.value == 1 ? 'Tourism' : datum.value == 0 ? 'Neutral' : ''",
                               labelFontSize=10, labelFontWeight="bold", grid=False)),
        color=alt.Color("dataset:N",
                         scale=alt.Scale(domain=[_ds_a, _ds_b], range=["#d73027", "#1a9850"]),
                         legend=alt.Legend(title="Dataset", orient="right")),
        shape=alt.Shape("dataset:N",
                          scale=alt.Scale(domain=[_ds_a, _ds_b], range=["triangle-left", "triangle-right"]),
                          legend=None),
        tooltip=[alt.Tooltip("dataset:N"), alt.Tooltip("metric:N"), alt.Tooltip("score:Q", format=".2f")],
    )

    _labels_a = alt.Chart(_gdf).transform_filter(
        alt.datum.dataset == _ds_a
    ).mark_text(fontSize=10, fontWeight="bold", dy=-18).encode(
        x=alt.X("score:Q"), text=alt.Text("score:Q", format=".2f"), color=alt.value("#d73027"),
    )
    _labels_b = alt.Chart(_gdf).transform_filter(
        alt.datum.dataset == _ds_b
    ).mark_text(fontSize=10, fontWeight="bold", dy=22).encode(
        x=alt.X("score:Q"), text=alt.Text("score:Q", format=".2f"), color=alt.value("#1a9850"),
    )

    _center = alt.Chart(_gdf).mark_rule(
        color="#999", strokeWidth=1.5, strokeDash=[4, 4],
    ).encode(x=alt.datum(0))

    vis_meter = (_center + _needles + _labels_a + _labels_b).properties(
        width=400, height=50,
    ).facet(
        row=alt.Row("metric:N", title=None, sort=_metric_order,
                     header=alt.Header(labelFontSize=11, labelFontWeight="bold",
                                       labelAngle=0, labelAlign="left", labelPadding=5)),
    ).properties(
        title=alt.TitleParams(
            text="Committee Bias-O-Meter",
            subtitle=[f"Comparing {_ds_a} (red) vs {_ds_b} (green)",
                      "-1 = fishing bias · 0 = neutral · +1 = tourism bias"],
            fontSize=15, subtitleFontSize=10,
        ),
    )
    return (vis_meter,)


@app.cell
def _(
    IND_ORDER,
    PERSON_ORDER,
    alt,
    ctl_ds_a,
    ctl_ds_b,
    ctl_person,
    df_sent,
    pd,
    sent_scale,
):
    _ds_a = ctl_ds_a.value
    _ds_b = ctl_ds_b.value
    _person = ctl_person.value
    _DS_LIST = [_ds_a, _ds_b]
    _PERSONS = [_person] if _person != "All members" else PERSON_ORDER

    _agg = (
        df_sent[df_sent["industry"].isin(IND_ORDER)]
        .groupby(["dataset", "person", "industry"])
        .agg(mean_sent=("sentiment", "mean"), n=("sentiment", "count"))
        .reset_index()
    )

    _h = max(40, len(_PERSONS) * 20)
    _rows = []

    for _i, _ds in enumerate(_DS_LIST):
        _is_last = (_i == len(_DS_LIST) - 1)
        _ds_persons = df_sent[df_sent["dataset"] == _ds]["person"].unique()

        _combos = []
        for _p in _PERSONS:
            for _ind in IND_ORDER:
                _combos.append({"person": _p, "industry": _ind,
                                "in_dataset": _p in _ds_persons})
        _grid = pd.DataFrame(_combos)
        _ds_data = _agg[_agg["dataset"] == _ds].drop(columns=["dataset"])
        _merged = _grid.merge(_ds_data, on=["person", "industry"], how="left")
        _merged["has_data"] = _merged["mean_sent"].notna()

        _base = alt.Chart(_merged).encode(
            x=alt.X("industry:N", title=None, sort=IND_ORDER,
                     scale=alt.Scale(domain=IND_ORDER),
                     axis=alt.Axis(labelAngle=-30, labelFontSize=8, labels=_is_last)),
            y=alt.Y("person:N", title=_ds, sort=_PERSONS,
                     scale=alt.Scale(domain=_PERSONS),
                     axis=alt.Axis(labelFontSize=9, titleFontSize=11,
                                   titleFontWeight="bold", titleAngle=0,
                                   titleAlign="right", titleX=-5, titleY=-5)),
        )
        _missing = _base.transform_filter(
            alt.datum.has_data == False
        ).mark_rect(stroke="white", strokeWidth=1, cornerRadius=1).encode(
            color=alt.condition(alt.datum.in_dataset == True,
                                alt.value("#e8e8e8"), alt.value("#f5f5f5")),
        )
        _filled = _base.transform_filter(
            alt.datum.has_data == True
        ).mark_rect(stroke="white", strokeWidth=1, cornerRadius=1).encode(
            color=alt.Color("mean_sent:Q", scale=sent_scale,
                            legend=alt.Legend(title="Sentiment", gradientLength=80)
                            if _i == 0 else None),
            tooltip=[alt.Tooltip("person:N"), alt.Tooltip("industry:N"),
                     alt.Tooltip("mean_sent:Q", format=".2f"), alt.Tooltip("n:Q", title="Records")],
        )
        _text = _base.transform_filter(
            alt.datum.has_data == True
        ).mark_text(fontSize=9, fontWeight="bold").encode(
            text=alt.Text("mean_sent:Q", format=".2f"),
            color=alt.condition(
                (alt.datum.mean_sent > 0.55) | (alt.datum.mean_sent < -0.35),
                alt.value("white"), alt.value("black")),
        )
        _rows.append((_missing + _filled + _text).properties(width=160, height=_h))

    vis_ind_sent = alt.vconcat(*_rows, spacing=5).properties(
        title=alt.TitleParams(
            text="Industry Sentiment",
            subtitle="Mean sentiment per member per industry",
            fontSize=13, subtitleFontSize=10,
        ),
    )
    return (vis_ind_sent,)


@app.cell
def _(PERSON_ORDER, alt, ctl_ds_a, ctl_ds_b, ctl_person, df_hours):
    _ds_a = ctl_ds_a.value
    _ds_b = ctl_ds_b.value
    _person = ctl_person.value
    _PERSONS = [_person] if _person != "All members" else PERSON_ORDER

    _df = df_hours[df_hours["person"].isin(_PERSONS) & df_hours["dataset"].isin([_ds_a, _ds_b])]
    _agg = _df.groupby(["dataset", "person", "industry_side"]).agg(hours=("hours", "sum")).reset_index()
    _sides = ["fishing", "tourism", "neutral"]
    _h = max(40, len(_PERSONS) * 20)

    _rows = []
    for _i, _ds in enumerate([_ds_a, _ds_b]):
        _is_last = (_i == 1)
        _ds_data = _agg[_agg["dataset"] == _ds]

        _chart = (
            alt.Chart(_ds_data)
            .mark_bar(cornerRadius=2)
            .encode(
                x=alt.X("hours:Q", title="Hours" if _is_last else None, stack="zero"),
                y=alt.Y("person:N", title=_ds, sort=_PERSONS,
                         scale=alt.Scale(domain=_PERSONS),
                         axis=alt.Axis(labelFontSize=9, titleFontSize=11,
                                       titleFontWeight="bold", titleAngle=0,
                                       titleAlign="right", titleX=-5, titleY=-5)),
                color=alt.Color("industry_side:N",
                                 scale=alt.Scale(domain=_sides,
                                                 range=["#4e79a7", "#f28e2b", "#bdbdbd"]),
                                 legend=alt.Legend(title="Side", orient="right",
                                                   direction="vertical")
                                 if _i == 0 else None),
                tooltip=[alt.Tooltip("person:N"), alt.Tooltip("industry_side:N", title="Side"),
                         alt.Tooltip("hours:Q", title="Hours", format=".0f")],
            )
            .properties(width=200, height=_h)
        )
        _rows.append(_chart)

    vis_hours_bar = alt.vconcat(*_rows, spacing=5).properties(
        title=alt.TitleParams(
            text="Trip Hours per Member",
            subtitle="Stacked by industry side",
            fontSize=13, subtitleFontSize=10,
        ),
    )
    return (vis_hours_bar,)


@app.cell
def _(
    PERSON_ORDER,
    ZONE_ORDER,
    alt,
    ctl_ds_a,
    ctl_ds_b,
    ctl_person,
    df_sent,
    df_trips,
    pd,
    sent_scale,
):
    _ds_a = ctl_ds_a.value
    _ds_b = ctl_ds_b.value
    _person = ctl_person.value
    _DS_LIST = [_ds_a, _ds_b]
    _PERSONS = [_person] if _person != "All members" else PERSON_ORDER

    _s_agg = (
        df_sent[df_sent["topic"] != "unknown"]
        .groupby(["dataset", "person", "topic"])
        .agg(mean_sent=("sentiment", "mean"), n=("sentiment", "count"))
        .reset_index()
    )
    _all_topics = sorted(df_sent[df_sent["topic"] != "unknown"]["topic"].unique())
    _topic_order = (
        _s_agg.groupby("topic")["mean_sent"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    _t_agg = df_trips.groupby(["dataset", "person", "zone"]).agg(n=("zone", "count")).reset_index()
    _h = max(40, len(_PERSONS) * 20)

    _sent_rows = []
    _trip_rows = []

    for _i, _ds in enumerate(_DS_LIST):
        _is_last = (_i == len(_DS_LIST) - 1)

        _ds_persons = df_sent[df_sent["dataset"] == _ds]["person"].unique()
        _s_combos = [{"person": _p, "topic": _t, "in_dataset": _p in _ds_persons}
                     for _p in _PERSONS for _t in _all_topics]
        _s_grid = pd.DataFrame(_s_combos)
        _s_ds = _s_agg[_s_agg["dataset"] == _ds].drop(columns=["dataset"])
        _s_merged = _s_grid.merge(_s_ds, on=["person", "topic"], how="left")
        _s_merged["has_data"] = _s_merged["mean_sent"].notna()

        _s_base = alt.Chart(_s_merged).encode(
            x=alt.X("topic:N", title=None, sort=_topic_order, scale=alt.Scale(domain=_topic_order),
                     axis=alt.Axis(labelAngle=-45, labelFontSize=8, labels=_is_last)),
            y=alt.Y("person:N", title=_ds, sort=_PERSONS, scale=alt.Scale(domain=_PERSONS),
                     axis=alt.Axis(labelFontSize=9, titleFontSize=11, titleFontWeight="bold",
                                   titleAngle=0, titleAlign="right", titleX=-5, titleY=-5)),
        )
        _s_miss = _s_base.transform_filter(alt.datum.has_data == False).mark_rect(
            stroke="white", strokeWidth=1, cornerRadius=1
        ).encode(color=alt.condition(alt.datum.in_dataset == True, alt.value("#e8e8e8"), alt.value("#f5f5f5")))
        _s_fill = _s_base.transform_filter(alt.datum.has_data == True).mark_rect(
            stroke="white", strokeWidth=1, cornerRadius=1
        ).encode(
            color=alt.Color("mean_sent:Q", scale=sent_scale,
                            legend=alt.Legend(title="Sentiment", gradientLength=80, orient="left")
                            if _i == 0 else None),
            tooltip=[alt.Tooltip("person:N"), alt.Tooltip("topic:N"),
                     alt.Tooltip("mean_sent:Q", format=".2f"), alt.Tooltip("n:Q")],
        )
        _s_txt = _s_base.transform_filter(alt.datum.has_data == True).mark_text(
            fontSize=7, fontWeight="bold"
        ).encode(
            text=alt.Text("mean_sent:Q", format=".1f"),
            color=alt.condition((alt.datum.mean_sent > 0.55) | (alt.datum.mean_sent < -0.35),
                                alt.value("white"), alt.value("black")),
        )
        _sent_rows.append((_s_miss + _s_fill + _s_txt).properties(width=500, height=_h))

        _t_ds_persons = df_trips[df_trips["dataset"] == _ds]["person"].unique()
        _t_combos = [{"person": _p, "zone": _z, "in_dataset": _p in _t_ds_persons}
                     for _p in _PERSONS for _z in ZONE_ORDER]
        _t_grid = pd.DataFrame(_t_combos)
        _t_ds = _t_agg[_t_agg["dataset"] == _ds].drop(columns=["dataset"])
        _t_merged = _t_grid.merge(_t_ds, on=["person", "zone"], how="left")
        _t_merged["n"] = _t_merged["n"].fillna(0).astype(int)

        _t_base = alt.Chart(_t_merged).encode(
            x=alt.X("zone:N", title=None, sort=ZONE_ORDER, scale=alt.Scale(domain=ZONE_ORDER),
                     axis=alt.Axis(labelAngle=-35, labelFontSize=8, labels=_is_last)),
            y=alt.Y("person:N", title=_ds, sort=_PERSONS, scale=alt.Scale(domain=_PERSONS),
                     axis=alt.Axis(labels=False, ticks=False, titleFontSize=11, titleFontWeight="bold",
                                   titleAngle=0, titleAlign="right", titleX=-5, titleY=-5, titleColor="white")),
        )
        _t_rect = _t_base.mark_rect(stroke="white", strokeWidth=1, cornerRadius=1).encode(
            color=alt.condition(alt.datum.n > 0,
                                alt.Color("n:Q", scale=alt.Scale(scheme="blues", type="sqrt"),
                                          legend=alt.Legend(title="Trips", gradientLength=80) if _i == 0 else None),
                                alt.value("#f5f5f5")),
            tooltip=[alt.Tooltip("person:N"), alt.Tooltip("zone:N"), alt.Tooltip("n:Q", title="Trips")],
        )
        _t_txt = _t_base.transform_filter(alt.datum.n > 0).mark_text(fontSize=7, fontWeight="bold").encode(
            text=alt.Text("n:Q"),
            color=alt.condition(alt.datum.n > 40, alt.value("white"), alt.value("black")),
        )
        _trip_rows.append((_t_rect + _t_txt).properties(width=280, height=_h))

    _left = alt.vconcat(*_sent_rows, spacing=5).properties(
        title=alt.TitleParams(text="Person × Topic Sentiment",
                              subtitle="Gray = missing from dataset (sampling bias)",
                              fontSize=13, subtitleFontSize=10))
    _right = alt.vconcat(*_trip_rows, spacing=5).properties(
        title=alt.TitleParams(text="Trip Destinations by Zone",
                              subtitle="Compare stop counts between selected datasets",
                              fontSize=13, subtitleFontSize=10))

    vis_detail = alt.hconcat(_left, _right, spacing=15).resolve_scale(color="independent")
    return (vis_detail,)


@app.cell
def _(PERSON_ORDER, alt, ctl_ds_a, ctl_ds_b, ctl_person, df_hours, pd):
    _ds_a = ctl_ds_a.value
    _ds_b = ctl_ds_b.value
    _person = ctl_person.value
    _PERSONS = [_person] if _person != "All members" else PERSON_ORDER

    _agg = df_hours.groupby(["dataset", "person", "industry_side"]).agg(hours=("hours", "sum")).reset_index()
    _sides = ["fishing", "tourism", "neutral"]
    _a = _agg[_agg["dataset"] == _ds_a].rename(columns={"hours": "hours_a"})
    _b = _agg[_agg["dataset"] == _ds_b].rename(columns={"hours": "hours_b"})

    _combos = [{"person": _p, "industry_side": _s} for _p in _PERSONS for _s in _sides]
    _grid = pd.DataFrame(_combos)
    _merged = (_grid
               .merge(_a[["person", "industry_side", "hours_a"]], on=["person", "industry_side"], how="left")
               .merge(_b[["person", "industry_side", "hours_b"]], on=["person", "industry_side"], how="left"))
    _merged["hours_a"] = _merged["hours_a"].fillna(0)
    _merged["hours_b"] = _merged["hours_b"].fillna(0)
    _merged["diff"] = _merged["hours_b"] - _merged["hours_a"]
    _merged["direction"] = _merged["diff"].apply(
        lambda x: f"More in {_ds_b}" if x > 0 else (f"More in {_ds_a}" if x < 0 else "Equal"))
    _merged["label"] = _merged["diff"].apply(
        lambda x: f"+{x:.0f}" if x > 0 else (f"{x:.0f}" if x != 0 else ""))

    _h = max(30, len(_PERSONS) * 18)
    _base = alt.Chart(_merged).encode(
        x=alt.X("diff:Q", title=f"Hours ({_ds_b} − {_ds_a})", scale=alt.Scale(zero=True)),
        y=alt.Y("person:N", title=None, sort=_PERSONS, scale=alt.Scale(domain=_PERSONS),
                 axis=alt.Axis(labelFontSize=9)),
    )
    _bars = _base.mark_bar(cornerRadius=2).encode(
        color=alt.Color("direction:N",
                         scale=alt.Scale(domain=[f"More in {_ds_a}", "Equal", f"More in {_ds_b}"],
                                         range=["#d73027", "#bdbdbd", "#1a9850"]),
                         legend=alt.Legend(title="Direction", orient="right", direction="vertical")),
        tooltip=[alt.Tooltip("person:N"), alt.Tooltip("industry_side:N", title="Side"),
                 alt.Tooltip("hours_a:Q", title=f"{_ds_a} hours", format=".1f"),
                 alt.Tooltip("hours_b:Q", title=f"{_ds_b} hours", format=".1f"),
                 alt.Tooltip("diff:Q", title="Difference", format=".1f")],
    )
    _text = _base.transform_filter(alt.datum.diff != 0).mark_text(
        fontSize=8, fontWeight="bold", align="left", dx=3
    ).encode(text="label:N", color=alt.value("#333"))

    vis_diff = (_bars + _text).properties(width=250, height=_h).facet(
        row=alt.Row("industry_side:N", title=None, sort=_sides,
                    header=alt.Header(labelFontSize=10, labelFontWeight="bold",
                                      labelAngle=0, labelAlign="left")),
    ).properties(
        title=alt.TitleParams(text=f"Trip Hours: {_ds_b} − {_ds_a}",
                              subtitle=["Red = more in A, Green = more in B",
                                         "Differences reflect missing trips AND missing stops"],
                              fontSize=13, subtitleFontSize=9))
    return (vis_diff,)


@app.cell
def _(
    ctl_ds_a,
    ctl_ds_b,
    ctl_person,
    mo,
    vis_detail,
    vis_diff,
    vis_hours_bar,
    vis_ind_sent,
    vis_meter,
):
    _header = mo.md(r"""
    # COOTEFOO Bias Investigation Dashboard
    """)

    _controls = mo.hstack([ctl_ds_a, ctl_ds_b, ctl_person], justify="start", gap=2)
    _top = mo.hstack([vis_meter, vis_ind_sent, vis_hours_bar], justify="start", gap=2, wrap=False)
    _bottom = mo.hstack([vis_detail, vis_diff], justify="start", gap=1, wrap=False)

    mo.vstack([
        _header,
        _controls,
        _top,
        _bottom,
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
