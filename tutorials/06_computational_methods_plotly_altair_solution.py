# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair>=5.0.0",
#     "plotly>=5.18.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "vega_datasets",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **VABD 2026**
    ## 📊 **Week 06 – Computational Methods with Altair & Plotly in a Single Dashboard**

    This week we bring together **two powerful visualization ecosystems**: Altair (declarative, grammar-of-graphics) and Plotly (imperative, highly interactive). Our aim is to build rich and multi-perspective dashboards. Each library has unique strengths: Altair excels at concise, composable specifications while Plotly offers chart types and interactions not available elsewhere.

    We continue with the **Gapminder** dataset, now exploring it through more advanced chart types: hierarchical views (treemap, sunburst), multivariate comparisons (radar charts), temporal flows (streamgraphs), and combining these with Altair's linked selections.

    **Topics covered:**
    - Hierarchical part-to-whole visualizations: Treemap & Sunburst
    - Temporal flow of categories: Streamgraph & Stacked Area
    - Multivariate country profiles: Radar/Spider Chart
    - Maps: Choroplet & Bubble maps
    - Animated visualizations
    - **Combining Altair + Plotly**: Using marimo to link both in a unified dashboard
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Index**
    - [Part I: Setup & Data Loading](#part-i-setup-data-loading)
    - [Part II: Hierarchical Visualizations](#part-ii-hierarchical-visualizations)
    - [Part III: Temporal Flow Visualizations](#part-iii-temporal-flow-visualizations)
    - [Part IV: Multivariate Profiles](#part-iv-multivariate-profiles)
    - [Part V: Maps](#part-v-maps)
    - [Part VI: Animated Visualizations](#part-vi-animated-visualizations)
    - [Part VII: Lasso Selection](#part-vii-lasso-selection)
    - [Part VIII: Integrated Dashboard](#part-viii-integrated-dashboard)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Setup & Data Loading**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import libraries

    We import both Altair and Plotly, allowing us to choose the best tool for each visualization task.
    """)
    return


@app.cell
def _():
    # Altair
    import altair as alt

    # Plotly ecosystem
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Data manipulation
    import pandas as pd
    import numpy as np

    # Dataset
    from vega_datasets import data

    # ML
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    return KMeans, StandardScaler, alt, data, go, np, pd, px, silhouette_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Quick introduction to Plotly

    [**Plotly**](https://plotly.com/python/) is one of the most widely used libraries for interactive visualizations in Python. While **Altair** follows a declarative grammar of graphics (you specify *what* should be shown), Plotly takes a more **imperative** approach, building figures step by step.

    In practice, we mostly use **Plotly Express (`px`)**, a high-level API that creates complete interactive charts with a single function call, similar in spirit to Altair. For more customized charts, Plotly also provides a lower-level interface called **Graph Objects (`go`)**, which we will use occasionally (e.g., for radar charts).

    **Why Plotly?** While Altair is ideal for declarative visualizations and linked interactions, Plotly makes it easier to create certain chart types and other types of interactions (such as treemaps, radar charts, or lasso selection). In this notebook we will first focus on Plotly, and then combine both libraries in an integrated dashboard.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [Plotly Python Documentation](https://plotly.com/python/) — Comprehensive guide to Plotly's Python API

        2 [Plotly Express](https://plotly.com/python/plotly-express/) — High-level interface for quick, beautiful charts

        3 [Marimo + Plotly Integration](https://docs.marimo.io/guides/working_with_data/plotting/) — How Plotly works within Marimo notebooks

        4 [Declarative vs imperative plotting approaches](https://towardsdatascience.com/declarative-vs-imperative-plotting-3ee9952d6bf3/) – An overview of key differences

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load the Gapminder dataset

    We load the same Gapminder dataset from the previous week, adding our region mapping and population buckets for consistency across all notebooks.
    """)
    return


@app.cell
def _(data, pd):
    # Load gapminder data
    df = data.gapminder()

    # Map cluster IDs to human-readable region names
    region_map = {
        0: "South Asia",
        1: "Europe & Central Asia",
        2: "Sub-Saharan Africa",
        3: "Americas",
        4: "East Asia & Pacific",
        5: "Middle East & North Africa",
    }
    df["region"] = df["cluster"].map(region_map)

    # Population bins
    pop_bins = [0, 5e6, 2e7, 1e8, 5e8, 2e9]
    pop_labels = ["<5M", "5-20M", "20-100M", "100-500M", "500M+"]
    df["pop_bucket"] = pd.cut(df["pop"], bins=pop_bins, labels=pop_labels, include_lowest=True)
    df["pop_bucket"] = pd.Categorical(df["pop_bucket"], categories=pop_labels, ordered=True)

    # Quick overview
    print(f"Shape: {df.shape}")
    print(f"Years: {sorted(df['year'].astype(str).unique())}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    df.head(10)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Global color settings

    We define consistent colors for regions across **both** Altair and Plotly visualizations. This ensures visual coherence when combining charts from different libraries.
    """)
    return


@app.cell
def _(alt, df):
    # GLOBAL COLOR PALETTE - used by both Altair and Plotly
    region_colors = {
        "South Asia": "#FAC723",
        "East Asia & Pacific": "#48cae4",
        "Europe & Central Asia": "#aad576",
        "Americas": "#936FAC",
        "Sub-Saharan Africa": "#F29222",
        "Middle East & North Africa": "#E95E50",
    }

    # For Plotly: color_discrete_map format
    color_discrete_map = region_colors.copy()

    # For Altair: Scale object
    region_scale = alt.Scale(
        domain=list(region_colors.keys()),
        range=list(region_colors.values())
    )

    # Fixed axis domains for consistent comparisons
    life_domain = [0, float(df["life_expect"].max()) + 5]
    fert_domain = [0, float(df["fertility"].max()) + 1]
    return (
        color_discrete_map,
        fert_domain,
        life_domain,
        region_colors,
        region_scale,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Hierarchical Visualizations**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Treemap
    🔗 [Treemap in Plotly](https://plotly.com/python/treemaps/)

    A [**treemap**](https://www.data-to-viz.com/graph/treemap.html) displays hierarchical data as nested rectangles. The size of each rectangle represents a quantitative value (e.g., population), while the hierarchy shows containment relationships (e.g., region → country).

    **Why Plotly for treemaps?** Altair does not natively support treemaps. Plotly Express provides `px.treemap()` with built-in drill-down interaction — click a region to zoom in.

    **When to use a treemap:**
    - You have hierarchical categorical data with a meaningful size metric
    - You want to show how parts contribute to a whole at different levels
    - Space efficiency is important (treemaps pack information densely)

    **Limitations:**
    - Hard to compare rectangles of similar sizes precisely
    - Deep hierarchies can make small items difficult to see
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    # Year selector for treemap
    treemap_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=int(df["year"].max()),
        label="Select year:",
        show_value=True,
    )
    treemap_year_slider
    return (treemap_year_slider,)


@app.cell
def _(color_discrete_map, df, mo, px, treemap_year_slider):
    # Filter data for selected year
    df_treemap = df[df["year"] == treemap_year_slider.value].copy()

    # Create treemap with Plotly Express
    fig_treemap = px.treemap(
        df_treemap,
        path=["region", "country"], # Hierarchy: region -> country
        values="pop", # Size by population
        color="region", # Color by region
        color_discrete_map=color_discrete_map,
        title=f"World Population by Region and Country ({treemap_year_slider.value})",
        hover_data={"life_expect": ":.1f", "fertility": ":.2f"},
    )

    fig_treemap.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        height=500,
    )

    fig_treemap.update_layout(clickmode="event", height=500)
    # fig_treemap

    # Note: we can wrap a Plotly figure with mo.ui.plotly(...) in marimo to make it interactive.
    # This exposes user interactions (e.g., clicked/selected points) via chart.points,
    # allowing us to link the visualization to other cells or extract the selected data.

    chart = mo.ui.plotly(fig_treemap)
    chart
    return (chart,)


@app.cell
def _(chart):
    chart.points
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    /// note | **Plotly `clickmode` options**

    In Plotly, the `clickmode` setting controls what happens when the user clicks on a point in the chart.

    | clickmode value | Behavior |
    |---|---|
    | `"event"` | Fires a click event but does not visually select the point |
    | `"select"` | Enables point selection (e.g., box/lasso) but does not emit click events |
    | `"event+select"` | Both fires an event and selects the point (most useful for dashboards) |
    | `False` | Disables click interactions |
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        Modify the treemap to show a **three-level hierarchy**: `region → pop_bucket → country`. This adds an intermediate level showing population size categories within each region.

        Hint: change the `path` parameter in `px.treemap()`.

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    return


@app.cell
def _(df_treemap_ex):
    df_treemap_ex
    return


@app.cell
def _(color_discrete_map, df, px, treemap_year_slider):
    # 💡 Solution

    df_treemap_ex = df[df["year"] == treemap_year_slider.value].copy()

    fig_treemap_3l = px.treemap(
        df_treemap_ex,
        path=["region", "pop_bucket", "country"], # Three-level hierarchy
        values="pop",
        color="region",
        color_discrete_map=color_discrete_map,
        title=f"Population Hierarchy: Region → Size Bucket → Country ({treemap_year_slider.value})",
        hover_data={"life_expect": ":.1f", "fertility": ":.2f"},
    )

    fig_treemap_3l.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        height=500,
    )

    fig_treemap_3l
    return (df_treemap_ex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Sunburst
    🔗 [Sunburst Charts in Plotly](https://plotly.com/python/sunburst-charts/)

    A [**sunburst diagram**](https://www.data-to-viz.com/graph/sunburst.html) is a radial version of a treemap. It displays hierarchical data in concentric rings, where the innermost ring represents the top level and outer rings represent deeper levels. **Interaction:** click on a segment to zoom into that subtree; click the center to zoom back out.
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    ## 1. Population sunburst with drill-down

    sunburst_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=int(df["year"].max()),
        label="Select year:",
        show_value=True,
    )
    sunburst_year_slider
    return (sunburst_year_slider,)


@app.cell
def _(color_discrete_map, df, mo, px, sunburst_year_slider):
    df_sunburst = df[df["year"] == sunburst_year_slider.value].copy()

    fig_sunburst = px.sunburst(
        df_sunburst,
        path=["region", "pop_bucket", "country"],
        values="pop",
        color="region",
        color_discrete_map=color_discrete_map,
        title=f"Population Hierarchy ({sunburst_year_slider.value})",
        hover_data={"life_expect": ":.1f", "fertility": ":.2f"},
    )

    fig_sunburst.update_layout(
        height=550,
        margin=dict(t=50, l=10, r=10, b=10),
    )

    fig_sunburst.update_layout(clickmode="event", height=500)
    # fig_sunburst
    chart_sun = mo.ui.plotly(fig_sunburst)
    chart_sun
    return (chart_sun,)


@app.cell
def _(chart_sun):
    chart_sun.value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Treemap vs Sunburst

    You have now seen the same hierarchical data (region → population bucket → country) displayed as both a treemap and a sunburst. Take a moment to reflect on the differences:

    - **Treemaps** use rectangular areas, which our visual system is better at comparing. It is easier to judge whether China's rectangle is larger than India's than to compare two arc segments. When the main question is *"how big is X relative to Y?"*, treemaps win.
    - **Sunbursts** make the hierarchy more explicit through concentric rings: inner ring = top level, outer ring = leaves. The radial structure invites **drill-down exploration** — click a region to zoom in. When the main question is *"how is this hierarchy structured?"*, sunbursts win.
    - Both struggle with very small categories, but treemaps handle this slightly better because narrow rectangles are still easier to label than thin arc slices.

    There is **no universally correct choice: the best chart depends on the question you want your reader to answer. Effective visualization means matching the representation to the task.**

    ///
    """)


    # - Sunburst: Better for drill-down exploration, radial aesthetics, centered narrative
    # - Treemap: Better for comparing sizes at the same level, denser layout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        Color the sunburst by **life expectancy** using a continuous scale instead of region colors. Use `color="life_expect"` and `color_continuous_scale="RdYlGn"`. What patterns emerge? Which regions show the most internal variation?

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    return


@app.cell
def _(df, px, sunburst_year_slider):
    # 💡 Solution

    df_sun_ex = df[df["year"] == sunburst_year_slider.value].copy()

    fig_sunburst_life = px.sunburst(
        df_sun_ex,
        path=["region", "pop_bucket", "country"],
        values="pop",
        color="life_expect",
        color_continuous_scale="PuBu",
        title=f"Population Hierarchy colored by Life Expectancy ({sunburst_year_slider.value})",
        hover_data={"life_expect": ":.1f", "fertility": ":.2f"},
    )

    fig_sunburst_life.update_layout(
        height=550,
        margin=dict(t=50, l=10, r=10, b=10),
    )

    fig_sunburst_life

    # The green-red gradient reveals that Sub-Saharan Africa shows consistently lower life expectancy (red), while Europe & East Asia are predominantly green. Within regions, you can spot outliers (i.e. countries that deviate from their regional pattern).
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part III: Temporal Flow Visualizations**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Stacked area chart

    🔗 [Stacked Area Chart (Altair Gallery)](https://altair-viz.github.io/gallery/normalized_stacked_area_chart.html)

    A [**stacked area chart**](https://www.data-to-viz.com/graph/stackedarea.html) shows how the total of a quantity evolves over time, broken down by category. Each colored band represents one category, and the total height at any point is the sum of all categories.

    Altair handles this naturally with `mark_area()` and a stacked y-encoding. We show two variants side by side: **absolute** (see totals and individual contributions) and **normalized** (see proportions, regardless of absolute values).
    """)
    return


@app.cell
def _(df):
    # Aggregate population by region and year
    df_temporal = df.groupby(["year", "region"], as_index=False)["pop"].sum()
    df_temporal
    return (df_temporal,)


@app.cell
def _(alt, df_temporal, mo, region_scale):
    # Absolute stacked area
    _area_abs = (
        alt.Chart(df_temporal)
        .mark_area()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("pop:Q", title="Population", stack="zero"),
            color=alt.Color("region:N", scale=region_scale, legend=None),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("pop:Q", title="Population", format=",.0f"),
            ],
        )
        .properties(width=380, height=300, title="Stacked Area | Absolute values").properties(padding={"left": 50})
    )

    # Normalized stacked area (100%)
    _area_norm = (
        alt.Chart(df_temporal)
        .mark_area()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("pop:Q", title="Percentage", stack="normalize"),
            color=alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("pop:Q", title="Population", format=",.0f"),
            ],
        )
        .properties(width=380, height=300, title="Stacked Area | Normalized values (100%)")
    )

    # Note: the stacking order in Altair follows the order of the domain in the color scale 
    # Since region_scale has a fixed domain list, that's the order used
    # It's not random, but it's also not sorted by any meaningful metric

    mo.hstack([_area_abs, _area_norm], justify="center", gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The absolute view reveals that **global population roughly doubled** between 1955 and 2005, with South Asia and Sub-Saharan Africa driving most of the growth. The normalized view tells a complementary story: Europe & Central Asia shrank from ~20% to ~12% of world population, while Sub-Saharan Africa's share grew steadily.

    Both views use a **zero baseline**, meaning that each band starts where the previous one ends. This makes it easy to read the total (top edge) but harder to judge the shape of individual categories, because their baseline keeps shifting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Stream graph

    🔗 [Stream graph in Plotly –filled area plots](https://plotly.com/python/filled-area-plots/)

    A [**stream graph**](https://www.data-to-viz.com/graph/streamgraph.html) is a stacked area chart with a **centered baseline**: instead of stacking from zero, the streams are arranged symmetrically around a central axis. This makes each individual stream's shape easier to read, at the cost of losing the ability to read the total.

    **Why Plotly here?** Altair does not support centered baselines. Plotly's `go.Scatter` with manual baseline computation gives us full control over the stream positioning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note |When to use which baseline:

     - **Zero baseline** (Altair above): best when the **total** matters and you want to read cumulative values
     - **Normalized (100%)** (Altair above): best when you care about **proportions** over time, regardless of absolute values
     - **Centered baseline** (streamgraph below): best when you care about the **shape** of individual categories over time, organic flow aesthetic

    ///
    """)
    return


@app.cell
def _(mo):
    stream_type_dropdown = mo.ui.dropdown(
        options={
            "Centered (streamgraph)": "centered",
            "Stacked (zero baseline)": "zero",
            "Normalized (100%)": "normalize",
        },
        value="Centered (streamgraph)",
        label="Baseline type:",
    )
    stream_type_dropdown
    return (stream_type_dropdown,)


@app.cell
def _(color_discrete_map, df_temporal, go, np, stream_type_dropdown):
    # Pivot to wide format: one column per region
    _pivot = df_temporal.pivot(index="year", columns="region", values="pop").fillna(0)
    _years = _pivot.index.tolist()
    _regions = _pivot.columns.tolist()

    # Sort regions by total population (largest at center for visual balance)
    _region_order = _pivot.sum().sort_values(ascending=False).index.tolist()

    # Reorder: alternate from center outward for symmetric look
    _centered_order = []
    for i, r in enumerate(_region_order):
        if i % 2 == 0:
            _centered_order.append(r)
        else:
            _centered_order.insert(0, r)

    _mode = stream_type_dropdown.value

    if _mode == "normalize":
        # Normalize each year to 100%
        _row_sums = _pivot[_centered_order].sum(axis=1)
        _values = {r: (_pivot[r] / _row_sums * 100).values for r in _centered_order}
    else:
        _values = {r: _pivot[r].values for r in _centered_order}

    if _mode == "centered":
        # Compute centered baseline: total/2 offset
        _total = np.sum([_values[r] for r in _centered_order], axis=0)
        _baseline = -_total / 2
    else:
        _baseline = np.zeros(len(_years))

    # Build traces from bottom to top
    fig_stream = go.Figure()
    _cumulative = _baseline.copy()

    for _region in _centered_order:
        _y_bottom = _cumulative.copy()
        _y_top = _cumulative + _values[_region]

        fig_stream.add_trace(go.Scatter(
            x=_years,
            y=_y_top,
            mode="lines",
            line=dict(width=0.5, color=color_discrete_map.get(_region, "#888")),
            name=_region,
            showlegend=True,
            hovertemplate=f"<b>{_region}</b><br>Year: %{{x}}<br>Pop: %{{customdata:,.0f}}<extra></extra>",
            customdata=df_temporal[df_temporal["region"] == _region].sort_values("year")["pop"].values,
        ))

        fig_stream.add_trace(go.Scatter(
            x=_years,
            y=_y_bottom,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=color_discrete_map.get(_region, "#888"),
            opacity=0.8,
            showlegend=False,
            hoverinfo="skip",
        ))

        _cumulative = _y_top

    _y_label = {
        "centered": "Population (centered)",
        "zero": "Population",
        "normalize": "Percentage",
    }

    fig_stream.update_layout(
        height=450, width=800, 
        title="Population Evolution by Region (1955–2005)",
        xaxis_title="Year",
        yaxis_title=_y_label.get(_mode, "Population"),
        legend_title="Region",
        hovermode="x unified",
    )

    if _mode == "centered":
        fig_stream.update_yaxes(showticklabels=False)

    fig_stream
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part IV: Multivariate Profiles**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Radar chart

    🔗 [Radar chart in Plotly](https://plotly.com/python/radar-chart/)

    A [**radar chart**](https://www.data-to-viz.com/caveat/spider.html) (also called spider chart or star plot) displays multivariate data on axes radiating from a center point. Each axis represents a different variable, and values are plotted as points connected by lines, forming a polygon.

    **Why Plotly for radar charts?** Altair does not have native radar chart support. Plotly's `go.Scatterpolar` provides full control over polar coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | When to use a radar chart:

    - Comparing **multiple entities** across **several variables** (3-8 variables works best)
    - Showing **profiles** or **patterns** across dimensions
    - Highlighting **strengths and weaknesses** relative to other entities

    **Important.** All metrics must be normalized to a common scale (typically 0-1) for fair comparison.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Comparing country profiles**: select countries to compare their normalized indicators. We normalize each metric to 0-1 using the global min/max for context.
    """)
    return


@app.cell
def _(df, mo):
    # Select countries to compare
    available_countries = sorted(df["country"].unique().tolist())

    # Default selection: diverse set of countries
    default_countries = ["United States", "China", "Brazil", "Germany", "Nigeria", "India"]
    default_countries = [c for c in default_countries if c in available_countries]

    radar_country_selector = mo.ui.multiselect(
        options=available_countries,
        value=default_countries[:4],
        label="Select countries to compare (2-6 recommended):",
    )

    radar_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=int(df["year"].max()),
        label="Select year:",
        show_value=True,
    )

    mo.hstack([radar_country_selector, radar_year_slider])
    return radar_country_selector, radar_year_slider


@app.cell
def _(
    df,
    go,
    mo,
    np,
    radar_country_selector,
    radar_year_slider,
    region_colors,
):
    _selected = radar_country_selector.value
    _year = radar_year_slider.value

    if len(_selected) >= 2:
        df_radar = df[(df["country"].isin(_selected)) & (df["year"] == _year)].copy()
        df_full_year = df[df["year"] == _year]

        # Metrics to include in radar
        metrics = ["life_expect", "fertility", "pop"]

        # Normalize each metric to 0-1
        for _m in metrics:
            _min = df_full_year[_m].min()
            _max = df_full_year[_m].max()
            if _m == "pop":
                # Use log scale for population (huge range)
                df_radar[f"{_m}_norm"] = (
                    (np.log10(df_radar[_m]) - np.log10(_min))
                    / (np.log10(_max) - np.log10(_min))
                )
            else:
                df_radar[f"{_m}_norm"] = (df_radar[_m] - _min) / (_max - _min)

        # Invert fertility (lower = "better" for development narrative)
        df_radar["fertility_inv_norm"] = 1 - df_radar["fertility_norm"]

        # Build radar chart
        fig_radar = go.Figure()

        for _, _row in df_radar.iterrows():
            # Each iteration of the loop adds one trace for one country:
            _country = _row["country"]
            _region = _row["region"] # for color encoding
            _color = region_colors.get(_region, "#888888")

            _values = [
                _row["life_expect_norm"],
                _row["fertility_inv_norm"],
                _row["pop_norm"],
            ]

            _values.append(_values[0]) # Close polygon
            # A radar (spider) chart draws a polygon by connecting the points in order: A → B → C (this leaves the shape open)
            # To close the shape, the first point must be repeated at the end: A → B → C → A

            _labels = ["Life Expectancy", "Fertility", "Population (log)"]
            _labels.append(_labels[0]) # Same reason here (to close polygon)

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=_values,
                    theta=_labels,
                    fill="toself",
                    name=_country,
                    line_color=_color,
                    opacity=0.6,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Country Profiles Comparison ({_year})",
            height=500,
        )
        print(_values)
        print(_labels)
        _out = fig_radar
    else:
        _out = mo.md("*Please select at least 2 countries.*")

    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        Create a radar chart comparing **regions** (aggregated means) instead of individual countries. Compute the mean life expectancy, mean fertility, and (log) total population for each region, normalize them to 0–1, and plot one polygon per region.

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    return


@app.cell
def _(df_region_agg):
    df_region_agg
    return


@app.cell
def _(df, go, np, radar_year_slider, region_colors):
    # 💡 Solution

    _year = radar_year_slider.value
    df_year = df[df["year"] == _year]

    # Aggregate by region
    df_region_agg = df_year.groupby("region").agg(
        life_expect=("life_expect", "mean"),
        fertility=("fertility", "mean"),
        pop=("pop", "sum"),
    ).reset_index()

    # Normalize each metric to 0-1
    for m in ["life_expect", "fertility", "pop"]:
        _min, _max = df_region_agg[m].min(), df_region_agg[m].max()
        if m == "pop":
            df_region_agg[f"{m}_norm"] = (np.log10(df_region_agg[m]) - np.log10(_min)) / (np.log10(_max) - np.log10(_min))
        else:
            df_region_agg[f"{m}_norm"] = (df_region_agg[m] - _min) / (_max - _min)

    df_region_agg

    # Invert fertility
    df_region_agg["fertility_inv_norm"] = 1 - df_region_agg["fertility_norm"]

    fig_radar_regions = go.Figure()

    for _, row in df_region_agg.iterrows():
        _vals = [row["life_expect_norm"], row["fertility_inv_norm"], row["pop_norm"]]
        _vals.append(_vals[0])
        _labs = ["Life Expectancy", "Fertility (inverted)", "Population (log)"]
        _labs.append(_labs[0])

        fig_radar_regions.add_trace(go.Scatterpolar(
            r=_vals, theta=_labs, fill="toself",
            name=row["region"],
            line_color=region_colors.get(row["region"], "#888"),
            opacity=0.6,
        ))

    fig_radar_regions.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Regional Profiles Comparison ({_year})",
        height=500,
    )

    fig_radar_regions
    return (df_region_agg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part V: Maps**
    🔗 [Maps in Plotly](https://plotly.com/python/maps/)

    Geographic maps allow us to see **spatial patterns** in our data. Plotly offers two main approaches for mapping country-level data:

    1. **Choropleth map**: countries are filled with colors representing a value
    2. **Bubble map**: points (circles) are placed on countries, sized by a value

    Both have their place, but they communicate very differently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choropleth map

    A **choropleth** colors each geographic region according to a data value. It's intuitive and familiar, but has a critical flaw: **the visual prominence of a country depends on its geographic area, not its data value**.

    Countries such as Russia and Canada may dominate the map visually, while densely populated but small countries (Bangladesh, Netherlands) may be barely visible. This creates a **misleading visual hierarchy**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Preliminary data preprocessing ~ country codes for geographic mapping

    Plotly’s geographic functions must match dataset entries to map polygons. Countries can be identified in several ways:

    - **Country names (`locationmode="country names"`)** – Easy to read but fragile. Variations like "USA", "United States", or "United States of America" can cause mismatches.
    - **ISO-3 codes (`locationmode="ISO-3"`)** – Three-letter standardized codes such as `USA`, `BRA`, `DEU`. These are robust and unambiguous.
    - **ISO-2 codes** – Two-letter codes like `US`, `BR`, `DE`. They are common in many datasets but less frequently used directly in Plotly.

    The `vega_datasets` Gapminder dataset stores countries as full names. Some names do not match Plotly’s expected format (e.g., `"Slovak Republic"` vs `"Slovakia"`). To ensure reliable mapping, we convert country names to **ISO-3 codes** using a lookup dictionary.
    ///
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    import pycountry

    def country_to_iso3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    df["iso3"] = df["country"].apply(country_to_iso3)

    df[df["iso3"].isna()] # check countries that have not be correctly mapped
    return (pycountry,)


@app.cell
def _(pycountry):
    # Inspect why:
    country_tur = pycountry.countries.get(alpha_3="TUR")
    country_tur
    return


@app.cell
def _(df):
    # Correct
    df.loc[df["country"] == "Turkey", "iso3"] = "TUR"

    # check
    # df[df["iso3"].isna()]
    return


@app.cell(hide_code=True)
def _(df, mo):
    map_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=int(df["year"].max()),
        label="Select year:",
        show_value=True,
    )
    map_year_slider
    return (map_year_slider,)


@app.cell
def _(df, map_year_slider, px):
    df_map = df[df["year"] == map_year_slider.value].copy()

    # Choropleth map
    fig_choropleth = px.choropleth(
        df_map,
        locations="iso3", # "country"
        locationmode="ISO-3", # "country names" = Plotly tries to match the string in country with its internal map dataset
        color="pop",
        color_continuous_scale="Viridis",
        #range_color=[30, 85],
        title=f"Choropleth map: Population by Country ({map_year_slider.value})",
        labels={"life_expect": "Life Expectancy", "pop": "Population", "fertility": "Fertility", "country": "Country"},
        hover_name="country",
        hover_data={"fertility": ":.2f", "pop": ":,.0f"},
    )

    fig_choropleth.update_layout(
        height=450,
        margin=dict(t=50, l=0, r=0, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
    )

    fig_choropleth
    return (df_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Map projections

    The `projection_type` parameter controls how the 3D globe is flattened onto a 2D surface. Each projection introduces different distortions. Try changing `projection_type` in the code above to explore:

    | Projection | Preserves | Distorts | Best for |
    |------------|-----------|----------|----------|
    | `"natural earth"` | Overall shape (compromise) | Moderate at poles | General-purpose world maps |
    | `"equirectangular"` | Latitude/longitude grid | Shape at poles | Simple reference maps |
    | `"mercator"` | Angles and shapes | Area (poles appear huge) | Navigation, local maps |
    | `"robinson"` | Overall aesthetics (compromise) | Moderate everywhere | Thematic world maps |
    | `"orthographic"` | Perspective (globe look) | Only shows one hemisphere | Focused regional views |

    🔗 [Full list of Plotly map projections](https://plotly.com/python/map-projections/)

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bubble map

    A **bubble map** places circles on geographic locations (country centroids). Here we encode the same variable as the choropleth (population) but now using **overlapping circles with varying size** instead of filling country polygons.

    This approach avoids the area-distortion problem: a small but populous country like Bangladesh gets the same visual weight as a large but sparse country like Mongolia. **The visual prominence is equal for all countries**, letting the bubble size speaks for itself without geographic bias.

    However, bubble maps also introduce their own limitation: **circles may overlap**, especially in dense regions like Europe or Southeast Asia. This can partially hide smaller countries or make it harder to clearly associate each bubble with its corresponding country.
    """)
    return


@app.cell
def _(df_map, px):
    fig_bubble_map = px.scatter_geo(
        df_map,
        locations="iso3",
        locationmode="ISO-3",
        size="pop",
        size_max=40, 
        title=f"Bubble Map: Population by Country ({df_map['year'].iloc[0]})",
        labels={"life_expect": "Life Expectancy", "pop": "Population", "fertility": "Fertility", "country": "Country"},
        hover_name="country",
        hover_data={"fertility": ":.2f", "region": True, "iso3": False},
    )

    fig_bubble_map.update_traces(
        marker=dict(
            color="steelblue", 
            line=dict(width=1, color="white")
        )
    )

    fig_bubble_map.update_layout(
        height=450,
        margin=dict(t=50, l=0, r=0, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor="black",
            countrywidth=0.5,
            projection_type="natural earth",
            showland=True,
            landcolor="#F2F0EF",
        ),
    )

    fig_bubble_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Extended bubble map with two variables

    The bubble map unlocks an additional benefit over choropleths: **we can encode a second variable using bubble color**. This is impossible with choropleth maps, where the only visual channel is fill color, and makes bubble maps more informationally dense.

    Now we encode:
    - **Size** → Population (same as before)
    - **Color** → Life expectancy
    """)
    return


@app.cell
def _(df_map, px):
    fig_bubble_extended = px.scatter_geo(
        df_map,
        locations="iso3",
        locationmode="ISO-3",
        size="pop",
        color="life_expect",
        color_continuous_scale="viridis",
        range_color=[30, 85],
        size_max=50,
        title=f"Extended Bubble Map: Life Expectancy (color) & Population (size) ({df_map['year'].iloc[0]})",
        labels={"life_expect": "Life Expectancy", "pop": "Population", "country": "Country"},
        hover_name="country",
        hover_data={"fertility": ":.2f", "region": True, "pop": ":,.0f", "iso3": False},
    )

    fig_bubble_extended.update_layout(
        height=450,
        margin=dict(t=50, l=0, r=0, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor="black",
            countrywidth=0.5,
            projection_type="natural earth",
            showland=True,
            landcolor="#F2F0EF",
        ),
    )

    fig_bubble_extended
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This dual encoding reveals patterns invisible in the choropleth: we can now see that India and China (large bubbles) have moderate life expectancy, while small European countries (small bubbles) have high life expectancy. The visualization answers two questions at once: *"How healthy is this country?"* and *"How many people live there?"*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Summary: Choropleth vs Bubble Maps

    | Aspect | Choropleth | Bubble (basic) | Bubble (extended) |
    |--------|------------|----------------|-------------------|
    | **Visual channel** | Fill color | Point size/color | Point color + size |
    | **Variables encoded** | 1 | 1 | 2 |
    | **Geographic bias** | ⚠️ Large countries dominate | ✅ Equal weight | ✅ Data-driven weight |
    | **Best for** | Rates, densities (values already normalized by area) | Fair comparison | Multi-dimensional analysis |
    | **Main drawback** | Area can mislead visual importance | Bubbles may overlap and be hard to match to countries | Same as basic bubble map, plus multiple encodings can make interpretation harder |
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [How election maps lie](https://www.washingtonpost.com/graphics/politics/2016-election/how-election-maps-lie/) — Why choropleth maps may distort political reality (area ≠ votes), by Washington Post

        2 [Try to Impeach This](https://try-to-impeach-this.jetpack.ai/) — Interactive cartogram showing how map projections shape narrative

        3 [Human Terrain](https://pudding.cool/2018/10/city_3d/) — 3D population density spikes reveal where people actually live, by The Pudding

        4 [Mercator Puzzle](https://bramus.github.io/mercator-puzzle-redux/) — Drag countries to the equator and see their true size vs. Mercator distortion

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VI: Animated Visualizations**
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    n_countries = df["country"].nunique()
    n_years = df["year"].nunique()
    n_features = 3  # fertility, life_expect, pop (the encoded variables)

    mo.md(f"""
    Plotly Express has [built-in animation support](https://plotly.com/python/animations/) via `animation_frame`. This recreates the famous **Hans Rosling Gapminder animation** showing global development over time.

    **Data density of the animation.** The Gapminder dataset contains **{n_countries} countries × {n_years} years = {n_countries * n_years} data points**, each with {n_features} visual features (fertility, life expectancy, population). This means the animation encodes **{n_countries * n_years * n_features:,} individual values** — far more than any single static chart could (nicely) convey. Animation lets us perceive *temporal patterns* (convergence, divergence, outlier trajectories) that would be invisible in a single snapshot.

    **Key parameters:**
    - `animation_frame`: column to animate over (here, `year`)
    - `animation_group`: ensures smooth transitions for each entity (here, `country`)
    - `range_x`, `range_y`: fixed axis ranges prevent distracting rescaling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Animated scatter plot
    """)
    return


@app.cell
def _(color_discrete_map, df, px):
    fig_animated = px.scatter(
        df,
        x="fertility",
        y="life_expect",
        size="pop",
        color="region",
        color_discrete_map=color_discrete_map,
        hover_name="country",
        animation_frame="year",
        animation_group="country",
        size_max=60,
        range_x=[0, 9],
        range_y=[20, 85],
        title="The Gapminder Story: Fertility vs Life Expectancy (1955-2005)",
        labels={
            "fertility": "Fertility (children per woman)",
            "life_expect": "Life Expectancy (years)"
        },
    )

    fig_animated.update_layout(
        height=550,
        legend_title="Region",
    )

    # Slow down the animation for better storytelling
    fig_animated.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
    fig_animated.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

    fig_animated
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        Explore Plotly's [**faceted charts**](https://plotly.com/python/facet-plots/) by adding `facet_col="region"` to the animated scatter plot above. This creates **small multiples** of the animation, one per region.

        What regional stories become clearer when each region has its own panel?

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    return


@app.cell
def _(color_discrete_map, df, px):
    fig_animated_facet = px.scatter(
        df,
        x="fertility",
        y="life_expect",
        size="pop",
        color="region",
        color_discrete_map=color_discrete_map,
        hover_name="country",
        animation_frame="year",
        animation_group="country",
        facet_col="region",
        facet_col_wrap=3,
        size_max=45,
        range_x=[0, 9],
        range_y=[20, 85],
        title="Small Multiples by Region",
        labels={
            "fertility": "Fertility",
            "life_expect": "Life Expectancy",
        }
    )

    fig_animated_facet.update_layout(
        height=600,
        showlegend=False,
    )

    fig_animated_facet.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 400
    fig_animated_facet.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

    fig_animated_facet


    # Small multiples reveal that Sub-Saharan Africa's transition started much later than other regions, while East Asia & Pacific shows the most dramatic shift (driven by China's one-child policy). Europe was already in the low-fertility, high-life-expectancy corner by the 1960s.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VII: Lasso Selection**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Throughout the previous tutorials, we've used Altair's powerful selection mechanisms: `selection_point` for clicking, `selection_interval` for rectangular brushing. These cover most use cases, but there's one interaction pattern Altair **cannot** do: **freeform lasso selection**.

    Lasso selection is useful because real data often forms **irregular shapes**, and drawing freely around points makes it easier to explore and select clusters than trying to fit them inside a rectangle.

    Plotly provides lasso selection natively. Combined with `mo.ui.plotly()`, we can capture the selected points and use them reactively in marimo — just like we do with Altair selections. This is yet another reason to use Plotly alongside Altair: each library has interaction patterns the other lacks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Interactive scatter plot with lasso selection

    We recreate the Gapminder scatter plot (fertility vs life expectancy) with Plotly's lasso tool enabled.

    **How to use:**
    1. Click the **lasso icon** in the toolbar (or it's already active by default)
    2. **Draw a freeform shape** around any group of countries
    3. The selected countries appear in the table below
    """)
    return


@app.cell
def _(df, mo):
    lasso_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=1970,
        label="Select year:",
        show_value=True,
    )
    lasso_year_slider
    return (lasso_year_slider,)


@app.cell
def _(color_discrete_map, df, lasso_year_slider, mo, px):
    df_lasso = df[df["year"] == lasso_year_slider.value].copy()

    fig_lasso = px.scatter(
        df_lasso,
        x="fertility",
        y="life_expect",
        size="pop",
        color="region",
        color_discrete_map=color_discrete_map,
        hover_name="country",
        hover_data={
            "life_expect": ":.1f",
            "fertility": ":.2f",
            "pop": ":,.0f",
            "region": True,
        },
        size_max=50,
        title=f"Global Development Overview ({lasso_year_slider.value})",
        labels={
            "fertility": "Fertility",
            "life_expect": "Life Expectancy",
            "pop": "Population (bubble size)",
            "region": "Region",
        },
    )

    fig_lasso.update_layout(
        dragmode="lasso", # Enable lasso selection by default
        height=600,
        xaxis=dict(range=[0, 9]),
        yaxis=dict(range=[20, 85]),
        legend=dict(
            title="Region",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
    )

    # Wrap in mo.ui.plotly to capture selection
    lasso_plot = mo.ui.plotly(fig_lasso)
    lasso_plot
    return df_lasso, lasso_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Extracting selected data

    The selected points are available via `lasso_plot.value` (df indices via `lasso_plot.indices`). We can use this reactively to show details, compute statistics, or filter other charts.
    """)
    return


@app.cell
def _():
    # lasso_plot.value # is a list of point dicts (empty list if nothing selected)
    # lasso_plot.indices # is a list of indeces
    return


@app.cell
def _(df_lasso, lasso_plot, mo):
    # Extract selected point indices from the plotly selection
    selected_indices = lasso_plot.indices

    if len(selected_indices) > 0:
        # Get indices of selected points
        df_selected = df_lasso.iloc[selected_indices][["country", "region", "life_expect", "fertility", "pop"]].copy()
        df_selected = df_selected.sort_values("life_expect", ascending=False)
        df_selected.columns = ["Country", "Region", "Life Expectancy", "Fertility", "Population"]

        n_selected = len(df_selected)
        avg_life = df_selected["Life Expectancy"].mean()
        avg_fert = df_selected["Fertility"].mean()

        summary = mo.md(f"""
        **Selected: {n_selected} countries** | 
        Mean Life Expectancy: **{avg_life:.1f}** years | 
        Mean Fertility: **{avg_fert:.2f}** children/woman
        """)

        _out = mo.vstack([
            summary,
            mo.ui.table(df_selected, selection=None)
        ])
    else:
        _out = mo.md("*Use the lasso tool to draw around a group of countries. Selected data will appear here.*")

    _out
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Altair vs Plotly: selection capabilities

    | Selection Type | Altair | Plotly |
    |----------------|--------|--------|
    | **Click (single point)** | ✅ `selection_point` | ✅ Click events |
    | **Rectangular brush** | ✅ `selection_interval` | ✅ Box select |
    | **Lasso (freeform)** | ❌ Not available | ✅ Native support |
    | **Legend-based filter** | ✅ `selection_point(fields=...)` | ✅ Click legend |
    | **Linked selections across charts** | ✅ Excellent (declarative) | ⚠️ Requires callbacks |

    **Takeaway:** Use Altair for linked views and declarative selections. Use Plotly when you need lasso selection or specialized chart types. Marimo lets you combine both seamlessly.
    ///
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VIII: Integrated Dashboard**

    In the previous tutorials/sections we explored **Altair** and **Plotly** separately, using each library to build different types of visualizations. Now we bring everything together into a **single interactive dashboard**, connected through **Marimo’s reactive execution model**. The main goal is to demonstrate how **multiple visualization libraries can coexist and interact within the same analytical workflow**. _User interactions in one chart propagate to the others, enabling coordinated exploration of the data._

    In addition to visualization, this section also incorporates a **computational step** using **K-Means clustering**, a method introduced in earlier tutorials. Instead of only visualizing existing variables, we derive a **new analytical dimension** by grouping countries according to their demographic characteristics.

    Specifically, we cluster countries based on three standardized indicators:

    - **Life expectancy**
    - **Fertility rate**
    - **Population (log-transformed)**

    To assess how meaningful these clusters are, we compute and display the **silhouette score**, a common metric that measures how well separated the clusters are.

    The dashboard therefore combines:

    - **Computation** (K-Means clustering and evaluation)
    - **Interactive visualization**
    - **Cross-library coordination** between Altair and Plotly

    A key interaction in the dashboard is **cluster selection**. Clicking on a cluster in the **Altair scatter plot** highlights the corresponding countries in the **Plotly radar chart** and **bubble map**, allowing us to examine both their **demographic profiles** and their **geographic distribution**.

    ---

    ### Dashboard layout

    | | Left | Right |
    |---|---|---|
    | **Top** | Altair scatter (click selects cluster) | Cluster detail card (text statistics) |
    | **Bottom** | Plotly radar (cluster profiles) | Plotly bubble map (geographic footprint) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Shared controls

    Three widgets drive the entire dashboard: a **year slider** to select the snapshot, a **k slider** to control the number of clusters, and a **region filter** to focus on subsets of countries. All four panels react to every change.
    """)
    return


@app.cell
def _(df, mo):
    # Year slider — selects the Gapminder snapshot
    dash_year_slider = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=2000,
        label="**Year**",
        show_value=True,
    )

    # Region filter — limits which countries enter the clustering
    dash_region_filter = mo.ui.multiselect(
        options=sorted(df["region"].unique().tolist()),
        value=sorted(df["region"].unique().tolist()),
        label="**Regions**",
    )

    # Cluster count — controls K-Means granularity
    dash_k_slider = mo.ui.slider(
        start=2,
        stop=6,
        step=1,
        value=3,
        label="**Clusters**",
        show_value=True,
    )
    return dash_k_slider, dash_region_filter, dash_year_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. K-Means clustering

    We cluster on **three variables**: fertility, life expectancy, and $log_{10}(population)$. Population is log-transformed before standardization because it spans several orders of magnitude (from ~100K to ~1B). Without the log, a single country like China would dominate the distance calculations.

    `StandardScaler` centers each feature at mean 0, std 1 so all three contribute equally.

    We also compute the **Silhouette score** (−1 to 1): higher values mean clusters are well-separated and internally cohesive. Try sliding k and watch how the score changes, there is often a sweet spot.
    """)
    return


@app.cell
def _(
    KMeans,
    StandardScaler,
    dash_k_slider,
    dash_region_filter,
    dash_year_slider,
    df,
    np,
    silhouette_score,
):
    # Filter data based on shared controls
    df_dash = df[
        (df["year"] == dash_year_slider.value)
        & (df["region"].isin(dash_region_filter.value))
    ].copy()

    # Log-transform population (huge skew: ~100K to ~1B)
    df_dash["log_pop"] = np.log10(df_dash["pop"])

    # Standardize: center at 0, scale to unit variance
    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(
        df_dash[["fertility", "life_expect", "log_pop"]].values
    )

    # K-Means clustering
    _kmeans = KMeans(n_clusters=dash_k_slider.value, random_state=42, n_init=10)
    cluster_labels = _kmeans.fit_predict(features_scaled)
    df_dash["cluster"] = cluster_labels.astype(str)

    # Silhouette score: how well-separated are the clusters?
    sil_score = silhouette_score(features_scaled, cluster_labels)

    # Cluster profiles (aggregated means) — used by the radar chart
    cluster_profiles = df_dash.groupby("cluster").agg(
        mean_fertility=("fertility", "mean"),
        mean_life_expect=("life_expect", "mean"),
        mean_log_pop=("log_pop", "mean"),
        total_pop=("pop", "sum"),
        n_countries=("country", "count"),
    ).reset_index()

    # Normalize profiles to 0–1 for radar chart axes
    for _col in ["mean_fertility", "mean_life_expect", "mean_log_pop"]:
        _min, _max = cluster_profiles[_col].min(), cluster_profiles[_col].max()
        if _max > _min:
            cluster_profiles[f"{_col}_norm"] = (
                (cluster_profiles[_col] - _min) / (_max - _min)
            )
        else:
            cluster_profiles[f"{_col}_norm"] = 0.5

    # Invert fertility for radar: lower fertility → higher value → "more developed"
    cluster_profiles["mean_fertility_inv_norm"] = 1 - cluster_profiles["mean_fertility_norm"]
    return cluster_profiles, df_dash, sil_score


@app.cell
def _():
    # cluster_profiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Altair scatter plot with click selection

    The scatter plot encodes fertility (x), life expectancy (y), and population (size), colored by K-Means cluster. We add `alt.selection_point(fields=["cluster"])` so that **clicking any point selects its entire cluster** — all points in that cluster light up, while the others fade.

    We wrap the chart in `mo.ui.altair_chart()` to capture the selection. The selected cluster ID then flows to the radar, map, and detail card via Marimo's reactivity.
    """)
    return


@app.cell
def _(
    alt,
    dash_k_slider,
    dash_year_slider,
    df_dash,
    fert_domain,
    life_domain,
    mo,
):
    # Consistent cluster color palette
    cluster_colors_list = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    k_val = dash_k_slider.value
    cluster_scale = alt.Scale(
        domain=[str(i) for i in range(k_val)],
        range=cluster_colors_list[:k_val],
    )

    # Click selection: clicking any point selects its cluster field
    click_sel = alt.selection_point(fields=["cluster"], on="click", toggle=False)

    scatter_chart = (
        alt.Chart(df_dash)
        .mark_circle(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("fertility:Q", title="Fertility", scale=alt.Scale(domain=fert_domain)),
            y=alt.Y("life_expect:Q", title="Life Expectancy", scale=alt.Scale(domain=life_domain)),
            size=alt.Size("pop:Q", scale=alt.Scale(range=[100, 1500]), legend=alt.Legend(title="Population")),
            # Selected cluster shows its color; unselected points turn gray
            color=alt.condition(
                click_sel,
                alt.Color("cluster:N", scale=cluster_scale,
                          legend=alt.Legend(title="Cluster")),
                alt.value("#d0d0d0"),
            ),
            opacity=alt.condition(click_sel, alt.value(0.8), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("life_expect:Q", title="Life Exp.", format=".1f"),
                alt.Tooltip("fertility:Q", title="Fertility", format=".2f"),
                alt.Tooltip("pop:Q", title="Population", format=",.0f"),
            ],
        )
        .add_params(click_sel)
        .properties(
            width="container",
            height=340,
            title=alt.TitleParams(
                text=f"K-Means Clusters ({dash_year_slider.value})",
                subtitle="Click a point to select its cluster",
                fontSize=20,
                subtitleFontSize=13,
            )
        )
    )

    # Wrap in mo.ui.altair_chart to make the selection reactive
    scatter_widget = mo.ui.altair_chart(scatter_chart)

    scatter_chart
    return cluster_colors_list, k_val, scatter_widget


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Extract the selected cluster

    `scatter_widget.value` returns a DataFrame of selected points. We extract the cluster ID from the first row (since all selected points share the same cluster). If nothing is selected, `selected_cluster` is `None` and all panels show the full picture.
    """)
    return


@app.cell
def _(scatter_widget):
    # Read the selection from the Altair chart widget
    _sel = scatter_widget.value
    if _sel is not None and len(_sel) > 0 and "cluster" in _sel.columns:
        selected_cluster = str(_sel["cluster"].iloc[0])
    else:
        selected_cluster = None  # nothing selected → show all clusters
    selected_cluster
    return (selected_cluster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cluster detail card

    This cell builds a summary panel for the selected cluster using `mo.callout()`: average indicators, region breakdown, and the list of countries with key statistics. When no cluster is selected, it shows a placeholder prompt inviting the user to click.
    """)
    return


@app.cell
def _(cluster_colors_list, cluster_profiles, df_dash, mo, selected_cluster):
    if selected_cluster is not None:
        _prof = cluster_profiles[cluster_profiles["cluster"] == selected_cluster].iloc[0]
        _countries = df_dash[df_dash["cluster"] == selected_cluster].sort_values("pop", ascending=False)
        _color = cluster_colors_list[int(selected_cluster)]

        # Region breakdown
        _region_counts = _countries.groupby("region")["country"].count().sort_values(ascending=False)
        _region_parts = [f"{reg} ({cnt})" for reg, cnt in _region_counts.items()]

        _card = mo.md(
            f"### Selected cluster: **id. {selected_cluster}** (**{int(_prof['n_countries'])}** countries)\n\n"
            f"**Average profile**\n\n"
            f"Life expectancy: **{_prof['mean_life_expect']:.1f} yrs**, "
            f"Fertility: **{_prof['mean_fertility']:.1f}**, "
            f"Population (log₁₀): **{_prof['mean_log_pop']:.2f}**\n\n"
            f"Regions: {', '.join(_region_parts)}\n\n"
        )

        cluster_detail = mo.vstack([
            # mo.md("&nbsp;"),  # spacer to push it down
            mo.callout(_card, kind="neutral"),
        ])
    else:
        cluster_detail = mo.vstack([
            mo.md("&nbsp;"),
            mo.callout(
                mo.md("👈 **Click a point** in the scatter plot to see cluster details"),
                kind="neutral",
            ),
        ])
    return (cluster_detail,)


@app.cell
def _(cluster_colors_list, cluster_profiles, go, k_val, selected_cluster):
    # Color map for cluster IDs → hex colors
    cluster_color_map = {str(i): cluster_colors_list[i] for i in range(k_val)}

    fig_radar_dash = go.Figure()

    radar_labels = ["Life Expectancy", "Fertility (inv.)", "Population (log)"]
    radar_labels_closed = radar_labels + [radar_labels[0]]

    for _, _row in cluster_profiles.iterrows():
        _cid = _row["cluster"]
        _is_sel = (selected_cluster is None) or (_cid == selected_cluster)

        _vals = [
            _row["mean_life_expect_norm"],
            _row["mean_fertility_inv_norm"],
            _row["mean_log_pop_norm"],
        ]
        _vals.append(_vals[0]) # close the polygon

        fig_radar_dash.add_trace(go.Scatterpolar(
            r=_vals,
            theta=radar_labels_closed,
            fill="toself",
            name=f"Cluster {_cid} ({int(_row['n_countries'])})",
            line=dict(
                color=cluster_color_map.get(_cid, "#888"),
                width=2.5 if _is_sel else 1, # bold if selected
            ),
            opacity=0.7 if _is_sel else 0.12, # fade if not selected
        ))

    fig_radar_dash.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=dict(text="Cluster profiles", font=dict(size=16, color="black"), x=0.5),
        autosize=True,
        height=400,
        margin=dict(t=50, b=40, l=50, r=50),
        showlegend=False
        # legend=dict(font=dict(size=10), orientation="h", y=-0.15),
    )
    return cluster_color_map, fig_radar_dash


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Plotly bubble map

    The bubble map shows the **geographic footprint** of each cluster (a perspective that neither the scatter nor the radar provides). Bubble size encodes population, color encodes cluster. When a cluster is selected, its bubbles stay opaque while the rest fade, revealing spatial patterns (e.g., "most of Sub-Saharan Africa falls in cluster 1").
    """)
    return


@app.cell
def _(cluster_color_map, df_dash, px, selected_cluster):
    fig_map = px.scatter_geo(
        df_dash,
        locations="iso3",
        locationmode="ISO-3",
        size="pop",
        color="cluster",
        color_discrete_map=cluster_color_map,
        size_max=40,
        hover_name="country",
        hover_data={
            "life_expect": ":.1f",
            "fertility": ":.2f",
            "region": True,
            "iso3": False,
            "log_pop": False,
        }  
    )

    # Highlight selected cluster, fade the rest
    for _trace in fig_map.data:
        _cid = _trace.name
        if selected_cluster is not None:
            _trace.marker.opacity = 0.85 if _cid == selected_cluster else 0.1
        else:
            _trace.marker.opacity = 0.8

    fig_map.update_traces(marker=dict(line=dict(width=0.5, color="white")))

    fig_map.update_layout(
        height=400,
        margin=dict(t=40, l=0, r=0, b=0),
        title=dict(text="Geographic footprint of clusters", font=dict(size=16, color="black"), x=0.5),
        autosize=True,
        showlegend=False,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor="#ccc",
            countrywidth=0.3,
            projection_type="natural earth",
            showland=True,
            landcolor="#F2F0EF",
        ),
    )
    return (fig_map,)


@app.cell(column=2, hide_code=True)
def _(
    cluster_detail,
    dash_k_slider,
    dash_region_filter,
    dash_year_slider,
    df_dash,
    fig_map,
    fig_radar_dash,
    mo,
    scatter_widget,
    sil_score,
):
    # ── Compact info line ────────────────────────────────────────────────
    _info = mo.md(
        f"### **Selection**\n\n"
        f"Year **{dash_year_slider.value}** · "
        f"Countries **{len(df_dash)}** · "
        f"Clusters **{dash_k_slider.value}** · "
        f"Silhouette **{sil_score:.2f}**"
    )

    # ── Controls strip ───────────────────────────────────────────────────
    _controls = mo.hstack(
        [dash_year_slider, dash_region_filter, dash_k_slider],
        justify="start",
        gap=1.5,
    )

    # ── 2×2 grid ─────────────────────────────────────────────────────────
    _top_row = mo.hstack(
        [scatter_widget, cluster_detail],
        widths="equal",
        align="stretch",
        gap=0.5,
    )
    _bottom_row = mo.hstack(
        [fig_radar_dash, fig_map],
        widths="equal",
        gap=0.5,
    )

    # ── Full layout ──────────────────────────────────────────────────────
    mo.vstack([
        mo.md("# 🌍 **Exploring Global Demographic Clusters**"), #  with Gapminder data & K-Means
        mo.md(
            "-----\n\n"
            "Countries are clustered by **fertility**, **life expectancy**, and **log(population)** "
            "using K-Means. They update with the selected **year and region**. "
            "**Click any point** in the scatter plot to highlight its cluster "
            "across all panels and see detailed statistics."
        ),
        _controls,
        mo.callout(_info, kind="neutral"),
        _top_row,
        _bottom_row,
    ], gap=0.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
