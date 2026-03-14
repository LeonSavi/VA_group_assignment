# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "networkx>=3.0",
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
    ## 📆 **Week 05 – Time, Text, Networks in a Single View**

    So far we have explored static and interactive charts for tabular data and dimensionality reduction. This week we bring together **three visualization perspectives** — temporal trends, text-based filtering, and network structure — and combine them in a **coordinated, interactive dashboard**.

    We'll work with the classic [**Gapminder**](https://www.gapminder.org/) dataset: countries tracked over decades with indicators like life expectancy, fertility, and population. The dataset is small enough to explore comfortably, yet rich enough to showcase every interaction technique from the previous weeks and introduce new ones.

    **Topics covered:**
    - Temporal visualization: line charts, trails, animated year sliders
    - Text-based interaction: search boxes, dynamic labels, lookup transforms
    - Network visualization: building node-link diagrams from tabular data
    - Coordinated dashboards: linking all three perspectives with shared selections
    - Marimo widgets (`mo.ui`) combined with Altair parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [Altair: Bindings, Selections, Conditions](https://altair-viz.github.io/altair-viz-v4/user_guide/interactions.html) — Comprehensive guide to Altair's interaction model

        2 [Marimo UI Components](https://docs.marimo.io/guides/interactivity/) — Full list of marimo's interactive widgets
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Index**
    - [Part I: Setup & Data Exploration](#part-i-setup-data-exploration)
    - [Part II: Temporal Visualization](#part-ii-temporal-visualization)
    - [Part III: Text-Based Interaction](#part-iii-text-based-interaction)
    - [Part IV: Network Visualization](#part-iv-network-visualization)
    - [Part V: Building the Dashboard](#part-v-building-the-dashboard)
    - [Part VI: Your Turn — Extend the Dashboard](#part-vi-your-turn--extend-the-dashboard)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Setup & Data Exploration**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import libraries
    """)
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    import numpy as np

    import os, certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()

    import networkx as nx
    from pyvis.network import Network

    from vega_datasets import data

    from sklearn.metrics.pairwise import euclidean_distances
    return Network, alt, data, euclidean_distances, np, nx, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Global setup: _defining shared parameters_

    At the beginning of a dashboard development, it is good practice to define a set of **global parameters** that remain consistent across all views and interactions.

    These parameters may include:
    - **color domains and scales** (so that countries and regions always have the same color),
    - **axis domains** (so that axes do not change when filters or selections change),
    - any other shared configuration used across multiple charts.

    Defining them **once** at the start ensures **visual consistency**, **comparability across views**, and a smoother user experience when interacting with the dashboard.
    """)
    return


@app.cell
def _(alt, df):
    # GLOBAL SETTING

    region_colors = {
        "South Asia": "#FAC723",
        "East Asia & Pacific": "#48cae4",
        "Europe & Central Asia": "#aad576",
        "Americas": "#936FAC",
        "Sub-Saharan Africa": "#F29222",
        "Middle East & North Africa": "#E95E50",
    }

    region_scale = alt.Scale(
        domain=list(region_colors.keys()),
        range=list(region_colors.values())
    )

    #region_domain = sorted(df["region"].unique())
    #region_scale = alt.Scale(domain=region_domain, scheme="dark2")

    country_domain = sorted(df["country"].unique())
    country_scale = alt.Scale(domain=country_domain, scheme="tableau20")

    life_domain = [0, float(df["life_expect"].max()) + 5]
    fert_domain = [0, float(df["fertility"].max()) + 1]
    return fert_domain, life_domain, region_colors, region_scale


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load the `Gapminder` dataset

    The `Gapminder` dataset from `vega_datasets` contains **demographic indicators** for countries across multiple decades. Let's load it and understand its structure.
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
    df["region"] = df["cluster"].map(region_map) # created from cluster

    # population bins
    pop_bins = [0, 5e6, 2e7, 1e8, 5e8, 2e9]
    pop_labels = ["<5M", "5–20M", "20–100M", "100–500M", "500M+"]
    df["pop_bucket"] = pd.cut(df["pop"], bins=pop_bins, labels=pop_labels, include_lowest=True)
    df["pop_bucket"] = pd.Categorical(df["pop_bucket"], categories=pop_labels, ordered=True)

    # Quick overview
    print(f"Shape: {df.shape}")
    print(f"Years: {sorted(df['year'].astype(str).unique())}")
    print(f"Years (n): {df['year'].astype(str).nunique()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Population size buckets: {df['pop_bucket'].unique()}")
    df.head(10)
    return df, pop_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **About the dataset.** The Gapminder dataset contains **6 columns**: `country`, `year`, `fertility` (babies per woman), `life_expect` (years), `pop` (population), and `cluster` (region ID, 0–5). We've added a `region` column with human-readable names and `pop_bucket` (population size category) with five ordered levels: <5M, 5–20M, 20–100M, 100–500M, 500M+. The data spans from **1955 to 2005** in 5-year intervals, covering about 60 countries grouped into 6 world regions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [Gapminder Official Website](https://www.gapminder.org/) — Data, tools, and educational material from the Gapminder Foundation

        2 [Hans Rosling: 200 years in 4 minutes (BBC)](https://www.youtube.com/watch?v=Z8t4k0Q8e8Y) — A concise, powerful storytelling of 200 years of global development, all in a single interactive visualization

        3 [Gapminder World Health Chart (interactive tool)](https://www.gapminder.org/fw/world-health-chart/) — Explore the classic income vs life expectancy bubble chart interactively

        4 [Gapminder Tools: Explore Data](https://www.gapminder.org/tools/) — Access multiple interactive charts and datasets from Gapminder

        5 [Animated 3D Data Visualization (Superdot Studio)](https://www.superdot.studio/project/animated-3d-data-visualization) — A 3D animated reinterpretation of Hans Rosling’s famous Gapminder visualization

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Quick data exploration with marimo

    Before building visualizations, let's use marimo's UI to explore the data interactively. This is a good habit: **always explore before you visualize**.
    """)
    return


@app.cell
def _(df):
    # Some summary statiscs
    df.describe()
    return


@app.cell(hide_code=True)
def _(df, mo):
    year_selector = mo.ui.slider(
        start=int(df["year"].min()),
        stop=int(df["year"].max()),
        step=5,
        value=2000,
        label="Select year",
    )
    year_selector
    return (year_selector,)


@app.cell(hide_code=True)
def _(df, mo, year_selector):
    df_year = df[df["year"] == year_selector.value]

    mo.md(
        f"**Year {year_selector.value}:** {len(df_year)} countries, "
        f"mean life expectancy = **{df_year['life_expect'].mean():.1f}** years, "
        f"mean fertility = **{df_year['fertility'].mean():.2f}**"
    )
    return (df_year,)


@app.cell
def _(df_year, mo):
    mo.ui.table(df_year.sort_values("life_expect", ascending=False).head(15))
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Temporal Visualization**

    Time is a first-class citizen in the Gapminder dataset. **Every row is a snapshot of a country at a specific year**. The questions we want to answer are inherently temporal: _How did life expectancy evolve? When did fertility rates drop? Which countries changed the most?_

    In this first section we'll build increasingly interactive temporal views.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Line charts: country trajectories

    The most natural temporal encoding: **time on x, indicator on y, one line per country**. But with ~180 countries, a naive line chart is unreadable. We need interaction to manage the complexity.
    """)
    return


@app.cell
def _(alt, df, region_colors):
    # A naive attempt: all countries at once
    alt.Chart(df).mark_line(point=True, opacity=0.7, strokeWidth=1.5).encode(
        x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("life_expect:Q", title="Life Expectancy"),
        detail="country:N", # draw a separate line for each country
        color=alt.Color(
            "region:N",
            scale=alt.Scale(
                domain=list(region_colors.keys()),
                range=list(region_colors.values())
            ),
            legend=alt.Legend(title="Region")
        )
    ).properties(width=650, height=300, title="Life Expectancy Over Time (all countries)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Why `detail="country:N"` instead of `color="country:N"`?

        With ~60 countries, mapping each to a unique color would create an illegible legend and exhaust the palette. Using `detail` tells Altair to draw separate lines per country **without** assigning a color to each one. Then, we can control color through a different encoding, or with a conditional encoding tied to the selection.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The chart above is also named **spaghetti plot**, because of the abundance of overlapping lines which make it very hard to read. Let's make it useful with interaction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Highlight on hover

    We'll use a **point selection on hover** bound to the `country` field. When the mouse hovers on a line, that country's trajectory lights up while others fade. This pattern is commonly refered to as [**focus + context**](https://infovis-wiki.net/wiki/Focus-plus-Context).
    """)
    return


@app.cell
def _(alt, df, region_colors):
    # Hover-based highlight: focus + context
    hover_country = alt.selection_point(
        fields=["country"],
        on="mouseover",
        clear="mouseout",
        nearest=True # update the highlight to the nearest point as we move the mouse
    )

    _base = alt.Chart(df).encode(
        x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("life_expect:Q", title="Life Expectancy"),
        detail="country:N",
    )

    _lines = _base.mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=1.5).encode(
        color=alt.condition(
            hover_country,
            alt.Color(
                "region:N",
                scale=alt.Scale(
                    domain=list(region_colors.keys()),
                    range=list(region_colors.values())
                ),
                legend=alt.Legend(title="Region")
            ),
            alt.value("lightgray")
        ),
        opacity=alt.condition(hover_country, alt.value(1), alt.value(0.12)),
    )

    # Invisible points used as hover targets for tooltip
    # lines are hard to hover on, but points are easy to hover on
    _points = _base.mark_point(opacity=0, size=10).encode(
        tooltip=["country:N", "region:N", "year:O", "life_expect:Q"],
    ).add_params(hover_country)

    (_lines + _points).properties(
        width=650,
        height=350,
        title="Life Expectancy Over Time (all countries)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Year slider (with Altair parameters)

    Another powerful temporal interaction: let users **scrub through time** with a slider. Combined with a scatter plot, this recreates the famous [Gapminder Trendalyzer](https://www.gapminder.org/tools/).
    """)
    return


@app.cell
def _(alt, df, region_colors):
    # Year slider bound to an Altair selection parameter
    _year_slider = alt.binding_range(min=1955, max=2005, step=5, name="Year ")
    _year_select = alt.selection_point(fields=["year"], bind=_year_slider, value=1980)

    scatter_year = (
        alt.Chart(df)
        .mark_circle(size=80, stroke='black', strokeWidth=1)
        .encode(
            x=alt.X(
                "fertility:Q",
                scale=alt.Scale(zero=False),
                title="Fertility",
            ),
            y=alt.Y(
                "life_expect:Q",
                scale=alt.Scale(zero=False),
                title="Life Expectancy",
            ),
            color=alt.Color(
                "region:N",
                legend=alt.Legend(title="Region"),
                scale=alt.Scale(
                    domain=list(region_colors.keys()),
                    range=list(region_colors.values())
                ),
            ),
            size=alt.Size("pop:Q", scale=alt.Scale(range=[100, 1500]), legend=alt.Legend(title="Population")),
            # try sort=pop_labels: what is it still missing?
            tooltip=["country:N", "region:N", "year:O", "fertility:Q", "life_expect:Q", "pop:Q"],
        )
        .transform_filter(_year_select)
        .add_params(_year_select)
        .properties(width=650, height=400, title="Fertility vs Life Expectancy")
    )

    scatter_year
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Trails: showing history as a visual trace

    Altair's [`mark_trail`](https://altair-viz.github.io/user_guide/marks/trail.html) draws a line whose **width can vary along its path**, making it ideal for encoding **change over time**. Combined with a hover selection, we use it here to reveal a country's entire fertility–life expectancy **trajectory** across all available years the moment a user hovers over it. The _key design insight_ is that the trail layer is not filtered by year. It rather encodes every year for every country in the background and uses conditional opacity to remain invisible until a country is selected, at which point its full historical path appears.
    """)
    return


@app.cell
def _(alt, df, region_scale):
    # Hover to reveal a country's trail across years
    hover_trail = alt.selection_point(
        on="mouseover", fields=["country"], empty=False # nothing highlighted when no hover
    )
    hover_trail_points = alt.selection_point(
        on="mouseover", fields=["country"] # empty=True by default, all points visible when no hover
    )

    # selection bound to the slider, filters which year's points and labels are shown
    year_slider_trail = alt.binding_range(min=1955, max=2005, step=5, name="Year ") # slider
    # alt.bind.. connects the parameter value to a UI widget so the user can change the value
    # selection tied to a data field:
    year_select_trail = alt.selection_point(fields=["year"], bind=year_slider_trail, value=2000)

    base_trail = (
        alt.Chart(df)
        .encode(
            x=alt.X("fertility:Q", scale=alt.Scale(zero=False), title="Fertility"),
            y=alt.Y("life_expect:Q", scale=alt.Scale(zero=False), title="Life Expectancy"),
            color=alt.Color(
                "region:N",
                scale=region_scale,
                legend=alt.Legend(orient="bottom-left", title="Region")
            ),
            detail="country:N",
        )
    )

    # Points for the year selected via the slider
    points_trail = (
        base_trail.mark_circle(size=200, stroke='black', strokeWidth=1)
        .encode(
            opacity=alt.condition(hover_trail_points, alt.value(0.8), alt.value(0.15)),
            tooltip=["country:N", "region:N", "year:O", "fertility:Q", "life_expect:Q"],
        )
        .transform_filter(year_select_trail) # year filter
        .add_params(hover_trail, hover_trail_points)
    )

    # Trail layer: visible only for hovered country
    trail_layer = base_trail.mark_trail(size=3).encode(
        opacity=alt.condition(hover_trail, alt.value(0.7), alt.value(0)),
        order="year:O",
        # the line we're drawing connects points ordered by year:O (:O ensures the trail connects points chronologically)
        # no year filter
    )

    trail_dots = (
        base_trail.mark_circle(size=20)
        .encode(
            opacity=alt.condition(hover_trail, alt.value(0.8), alt.value(0)),
            color=alt.value("black"),
        )
    )

    # Label for hovered country
    label_trail = (
        base_trail.mark_text(align="left", dx=8, fontSize=13, fontWeight="bold") 
        # dx=8 nudges the text 8 pixels to the right, with left-alignment
        .encode(
            text="country:N",
            opacity=alt.condition(hover_trail, alt.value(1), alt.value(0)),
        )
        .transform_filter(year_select_trail) # year filter
    )

    # Label for the first year of the trail
    trail_start_label = (
        base_trail.mark_text(align="right", dx=-8, fontSize=11)
        .encode(
            text="year:O",
            opacity=alt.condition(hover_trail, alt.value(1), alt.value(0)),
        )
        .transform_filter("datum.year === 1955") 
        # keeps only the 1955 row per country, so the label sits exactly where that country was in 1955
    )

    # Label for the last year of the trail
    trail_end_label = (
        base_trail.mark_text(align="left", dx=8, fontSize=11)
        .encode(
            text="year:O",
            opacity=alt.condition(hover_trail, alt.value(1), alt.value(0)),
        )
        .transform_filter("datum.year === 2005")
    )

    (points_trail + trail_layer + label_trail + trail_dots + trail_start_label + trail_end_label).add_params(
        year_select_trail
    ).properties(
        width=700,
        height=450,
        title={
            "text": "Fertility vs Life Expectancy",
            "subtitle": "Hover a country to highlight its trajectory across years"
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | How the trail chart works

        Three layers are stacked:

        1. **Points**: one circle per country, filtered to the slider year. Opacity is conditional on hover.
        2. **Trail**: the full trajectory across all years, visible **only** for the hovered country.
        3. **Label**: the country name, shown next to the hovered point, and the years, shown next to the first and last points in the trajectory.

        The year slider (`year_select_trail`) filters the points and label, while the trail shows the full history regardless of the slider position.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

    Two line charts sit side by side: **life expectancy** (left) and **fertility** (right), both showing all countries over time, colored by region.

    - Clicking a line in either chart highlights the **same country** in both
    - All other lines fade to gray
    - A tooltip shows country, region, and value on hover

    _Hints:_
    - Use a **single** `alt.selection_point(fields=["country"])` shared across both charts
    - Use `detail="country:N"` to draw one line per country without exhausting the color palette
    - Use `alt.condition` for opacity and color
    - Combine with `alt.hconcat` and `.resolve_scale(color="shared")`
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: create a shared click selection on "country"
    # country_click = alt.selection_point(fields=["country"])

    # Step 2: build a base Chart(df) with x=year, detail="country:N"
    # and conditional color + opacity driven by country_click

    # Step 3: build _life and _fert from the same base, different y encodings

    # Step 4: hconcat and add_params on the combined chart
    return


@app.cell
def _(alt, df, region_scale):
    # 💡 Solution

    # Step 1: create a shared click selection on "country"
    country_click = alt.selection_point(fields=["country"])

    # Step 2: build a base Chart(df) with x=year, detail="country:N"
    _base = (
        alt.Chart(df)
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            detail="country:N",
            color=alt.condition(
                country_click,
                alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(country_click, alt.value(1.0), alt.value(0.08)),
            strokeWidth=alt.condition(country_click, alt.value(2.5), alt.value(0.8)),
        )
    ) # shared visualization

    # Step 3: build _life and _fert from the same base, different y encodings
    _life = (
        _base.mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(
            y=alt.Y("life_expect:Q", title="Life Expectancy", scale=alt.Scale(zero=False)),
            tooltip=["country:N", "region:N", "year:O", alt.Tooltip("life_expect:Q", format=".1f")],
        )
        .properties(width=350, height=300, title="Life Expectancy Over Time")
    )

    _fert = (
        _base.mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(
            y=alt.Y("fertility:Q", title="Fertility", scale=alt.Scale(zero=False)),
            tooltip=["country:N", "region:N", "year:O", alt.Tooltip("fertility:Q", format=".2f")],
        )
        .properties(width=350, height=300, title="Fertility Over Time")
    )

    # Step 4: hconcat and add_params on the combined chart
    (
        alt.hconcat(_life, _fert)
        .add_params(country_click)
        .resolve_scale(color="shared")
        .properties(title="Click a country line to highlight it across both charts")
    )
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part III: Text-Based Interaction**

    Country names are **text data**. We can use them as an interactive dimension: searching, filtering, and labeling based on text input. Altair supports this natively through search parameters and regex-based conditions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Search box: find countries by name

    Altair's `alt.param` with an `alt.binding(input='search')` creates a text input that filters by `regex` (regular expression — a pattern-matching language for text, e.g. "land" matches any country containing that substring). This is extremely powerful for large categorical dimensions like country names. For more on regex, see regexone.com, a beginner-friendly interactive tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [RegexOne](https://regexone.com/) — A beginner-friendly interactive tutorial to learn regular expressions step by step

        2 [Altair: Bindings & Widgets](https://altair-viz.github.io/user_guide/interactions/bindings_widgets.html) — Official Altair documentation on binding parameters to input widgets like sliders, dropdowns, and search boxes

    ///
    """)
    return


@app.cell
def _(alt, df, region_scale):
    # Search box parameter
    # alt.param creates a variable that can change based on user interaction
    # Think of it as a container that holds a value and updates when something happens
    search_input = alt.param(
        value="",
        bind=alt.binding(input="search", placeholder="Type a country...", name="Search "),
    )
    # a text input box bound to a parameter
    # whatever the user types gets stored in search_input, it starts as an empty string (value="")

    # Regex test: does the country name match the search string?
    search_test = alt.expr.test(
        alt.expr.regexp(search_input, "i"), alt.datum.country
    )
    # this creates a regular expression from the user's typed text, with the "i" flag for case-insensitive matching
    # then tests whether each country name (alt.datum.country) matches that regex

    # type to search – try "Neth", "Arg", or "land"
    # ..if you type "land", alt.expr.regexp builds the regex /land/i, alt.expr.test tests every country name against it — matching Ireland, Finland, Poland, etc.

    gm_2000_search = df[df["year"] == 2000]

    search_chart = (
        alt.Chart(gm_2000_search)
        .mark_circle(size=150, stroke='black', strokeWidth=1)
        .encode(
            x=alt.X("fertility:Q", scale=alt.Scale(zero=False), title="Fertility"),
            y=alt.Y("life_expect:Q", scale=alt.Scale(zero=False), title="Life Expectancy"),
            color=alt.Color("region:N", scale=region_scale),
            opacity=alt.condition(search_test, alt.value(1), alt.value(0.08)),
            tooltip=["country:N", "region:N", "fertility:Q", "life_expect:Q"],
        )
        .add_params(search_input)
        .properties(width=650, height=400, title="Fertility vs Life Expectancy (2000)")
    )

    # Labels for matched countries
    search_labels = (
        alt.Chart(gm_2000_search)
        .mark_text(align="left", dx=7, fontSize=11)
        .encode(
            x=alt.X("fertility:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("life_expect:Q", scale=alt.Scale(zero=False)),
            text="country:N",
            opacity=alt.condition(search_test, alt.value(0.8), alt.value(0.1)),
        )
        .add_params(search_input)
    )

    search_chart + search_labels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | How the search works

        `alt.expr.regexp(search_input, "i")` creates a **case-insensitive regex** from whatever the user types. `alt.expr.test(...)` returns `true` when a country name matches. We use this boolean in `alt.condition` to control opacity (and labels).

        This means partial matches work too: typing `"land"` highlights Finland, Iceland, Ireland, Netherlands, New Zealand, Poland, Swaziland, Switzerland, Thailand...
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | `alt.selection_point` vs `alt.param`: what's the difference?

        - `alt.selection_point` is designed for selecting **data points** by field values (e.g. picking a year or a country from the dataset)
        - `alt.param` is more general-purpose — it can hold **any value** (a string, a number, etc.) and be used freely in expressions. That's why the search box here uses `alt.param`: the typed text isn't a data field, it's a pattern we pass into a regex
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Combining search with year slider

    The real power emerges when we combine text search with the temporal slider. Now the user can search for a country **and** scrub through time to see how it moves.
    """)
    return


@app.cell
def _(alt, df, region_scale):
    # country search
    search_box = alt.param(
        value="",
        bind=alt.binding(input="search", placeholder="Country", name="Search "),
    )
    _search_match = alt.expr.test(
        alt.expr.regexp(search_box, "i"), alt.datum.country
    )

    # year selection
    _year_slider = alt.binding_range(min=1955, max=2005, step=5, name="Year ")
    _year_select = alt.selection_point(fields=["year"], bind=_year_slider, value=2000)

    _hover = alt.selection_point(on="mouseover", fields=["country"], empty=False) # for names
    _hover_opacity = alt.selection_point(on="mouseover", fields=["region"]) # less opacity for countries of the sel region
    # fields=["country"] / ["region"] means the selection tracks which country/region the mouse is over
    # when you hover a point, Altair records the country/region value of that point and uses it to filter/condition other marks. Otherwise, it would just activate a true/false global selection

    _base = (
        alt.Chart(df)
        .encode(
            x=alt.X(
                "fertility:Q", 
                scale=alt.Scale(zero=True, domain=[0, float(df["fertility"].max()) + 1]),
                title="Fertility"
            ),
            y=alt.Y(
                "life_expect:Q", 
                scale=alt.Scale(zero=True, domain=[0, float(df["life_expect"].max()) + 5]), 
                title="Life Expectancy"
            ),
            color=alt.Color("region:N", scale=region_scale, legend=alt.Legend(orient="bottom-left")),
        )
    )

    # alt.when().then().otherwise()
    _opacity_expr = (
        alt.when(_hover_opacity, _search_match) # OR=if either is true then ...
        .then(alt.value(0.9))
        .otherwise(alt.value(0.08))
    )

    _points = (
        _base.mark_circle(size=100, stroke='black', strokeWidth=1)
        .encode(opacity=_opacity_expr, tooltip=["country:N", "region:N", "year:O", "fertility:Q", "life_expect:Q"])
        .transform_filter(_year_select)
        .add_params(_hover, _hover_opacity)
    )
    _labels = (
        _base.mark_text(align="left", dx=8, fontSize=13, fontWeight="bold")
        .encode(
            text="country:N",
            opacity=alt.condition(_hover, alt.value(1), alt.value(0)),
        ) # simpler syntax for a single condition, alt.condition is more readable when there's only one condition
        .transform_filter(_year_select)
    )

    (_points + _labels).add_params(
        _year_select, search_box
    ).properties(
        width=700,
        height=450,
        title="Fertility vs Life Expectancy",
    ) #.interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | `alt.condition` vs `alt.when`: what's the difference?

        Both control how a visual property (like opacity) changes based on a selection or expression — but they differ in power and syntax.

        `alt.condition(test, if_true, if_false)` is the classic, simple form: one condition, two outcomes. Great for binary cases like "show label if hovered, hide otherwise":
    ```python
        opacity=alt.condition(_hover, alt.value(1), alt.value(0))
    ```

        `alt.when().then().otherwise()` is the newer chainable form that supports **multiple conditions combined**. Use it when you need OR/AND logic across selections and expressions:
    ```python
        # show if hovered region OR search matches — two conditions at once
        _opacity_expr = (
            alt.when(_hover_opacity, _search_match)
            .then(alt.value(0.9))
            .otherwise(alt.value(0.08))
        )
    ```

        To make the labels consistent with the same syntax, you could rewrite:
    ```python
        # before
        opacity=alt.condition(_hover, alt.value(1), alt.value(0))
        # after
        opacity=alt.when(_hover).then(alt.value(1)).otherwise(alt.value(0))
    ```

        Both are valid — `alt.condition` is more concise for simple cases, `alt.when` scales better when logic grows. See the [Altair conditions documentation](https://altair-viz.github.io/user_guide/interactions/conditions.html) for more.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Marimo dropdown: filter by region & metric

    We can also use **marimo UI widgets** to control which data enters the chart. Unlike Altair's built-in widgets (which operate client-side inside the Vega spec), marimo widgets trigger Python re-execution, so we can use them for heavier transformations.
    """)
    return


@app.cell
def _(df, mo):
    region_dropdown = mo.ui.dropdown(
        options=["All regions"] + sorted(df["region"].dropna().unique().tolist()),
        value="All regions",
        label="**Region**",
    )

    metric_buttons = mo.ui.radio(
        options=["Life Expectancy", "Fertility"],
        value="Life Expectancy",
        label="**Metric**",
    )

    mo.vstack([metric_buttons, region_dropdown])
    return metric_buttons, region_dropdown


@app.cell
def _(alt, df, metric_buttons, region_dropdown, region_scale):
    df_filtered_region = (
        df
        if region_dropdown.value == "All regions"
        else df[df["region"] == region_dropdown.value]
    )

    metric_labels = {
        "Life Expectancy": "life_expect",
        "Fertility": "fertility",
    }
    y_field = metric_buttons.value

    alt.Chart(df_filtered_region).mark_line(opacity=0.5, strokeWidth=3).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y(f"{metric_labels[y_field]}:Q", title=y_field),
        color=alt.Color("region:N", scale=region_scale),
        tooltip=["country:N", "year:O", f"{metric_labels[y_field]}:Q"],
    ).properties(
        width=650,
        height=300,
        title={
            "text": y_field,
            "subtitle": f"{region_dropdown.value}"
        }
    ) #.interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Marimo widgets vs Altair widgets

        **Altair widgets** (`alt.binding_range`, `alt.binding_select`, etc.) run inside the Vega-Lite spec in the browser. They're fast and self-contained but limited to what Vega-Lite supports.

        **Marimo widgets** (`mo.ui.slider`, `mo.ui.dropdown`, etc.) trigger Python re-execution. This means you can use them for arbitrary data transformations (filtering, aggregation, even ML) before passing the result to Altair.

        **Rule of thumb:** use Altair widgets when the interaction is purely visual (highlight, filter, scrub). Use marimo widgets when you need Python-side logic.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

    This is an extension of the previous click-based exercise. Instead of clicking a country line, use a **search box** to highlight matching countries across both charts simultaneously.

    - A single search box highlights matching countries in **both** the life expectancy and fertility charts
    - Matched lines are colored by region and bold; unmatched lines fade to gray
    - Partial matches work: typing `"land"` highlights Finland, Iceland, Ireland, Poland...

    _Hints:_
    - Replace `alt.selection_point` with `alt.param` bound to `alt.binding(input="search")`
    - Use `alt.expr.regexp` and `alt.expr.test` to build the match condition (see Part III, section 1)
    - Move `add_params` to the **combined** `hconcat` chart — the search box should appear once, not twice
    - The `_base` encoding stays the same, just swap `country_click` for `_search_test`

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: define a search param and bind it to a search input widget
    # _search_input = alt.param(value="", bind=alt.binding(input="search", ...))

    # Step 2: build the match condition using alt.expr.regexp and alt.expr.test
    # _search_test = alt.expr.test(alt.expr.regexp(_search_input, "i"), alt.datum.country)

    # Step 3: build _base with conditional color, opacity, strokeWidth driven by _search_test
    # (same structure as the click solution, just replace country_click with _search_test)

    # Step 4: build _life and _fert from _base (same as before)

    # Step 5: hconcat and call .add_params(_search_input) on the combined chart
    return


@app.cell
def _(alt, df, region_scale):
    # 💡 Solution

    # Step 1: define a search param and bind it to a search input widget
    _search_input = alt.param(
        value="",
        bind=alt.binding(input="search", placeholder="Type a country...", name="Search "),
    )

    # Step 2: build the match condition using alt.expr.regexp and alt.expr.test
    _search_test = alt.expr.test(
        alt.expr.regexp(_search_input, "i"), alt.datum.country
    )

    # Step 3: build _base with conditional color, opacity, strokeWidth driven by _search_test
    _base = (
        alt.Chart(df)
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            detail="country:N",
            color=alt.condition(
                _search_test,
                alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(_search_test, alt.value(1.0), alt.value(0.08)),
            strokeWidth=alt.condition(_search_test, alt.value(2.5), alt.value(0.8)),
        )
    )

    # Step 4: build _life and _fert from _base (same as before)
    _life = (
        _base.mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(
            y=alt.Y("life_expect:Q", title="Life Expectancy", scale=alt.Scale(zero=False)),
            tooltip=["country:N", "region:N", "year:O", alt.Tooltip("life_expect:Q", format=".1f")],
        )
        .properties(width=350, height=300, title="Life Expectancy Over Time")
    )

    _fert = (
        _base.mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(
            y=alt.Y("fertility:Q", title="Fertility", scale=alt.Scale(zero=False)),
            tooltip=["country:N", "region:N", "year:O", alt.Tooltip("fertility:Q", format=".2f")],
        )
        .properties(width=350, height=300, title="Fertility Over Time")
    )

    # Step 5: hconcat and call .add_params(_search_input) on the combined chart
    (
        alt.hconcat(_life, _fert)
        .add_params(_search_input)
        .resolve_scale(color="shared")
        .properties(title="Search a country to highlight it across both charts")
    )
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part IV: Network Visualization**

    The Gapminder data is tabular, not a graph. But we can **derive** a network by defining relationships between countries. For example: _"two countries are connected if they have similar life expectancy."_ This transforms our data into a node-link diagram that reveals clusters and outliers from a completely different angle.

    💡 Think of Gapminder as a **table of attributes per country-year**. Network analysis becomes possible when you define **relationships between countries** based on those attributes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Building a similarity network

    We'll construct a graph where:
    - **Nodes** = countries (in a given year)
    - **Edges** = countries connected if their life expectancy is within a threshold

    This network is built from an _abstract_ similarity relationship (countries within a threshold on life expectancy). There are no inherent geographic or ordered coordinates for the nodes, so **we need a layout algorithm** to position them. In such context, **force-directed layout** is the default go-to because it tries to place connected nodes close together and push unconnected ones apart, which visually reveals cluster structure. We use [**NetworkX**](https://networkx.org/en/) to compute a force-directed layout, then visualize with Altair.
    """)
    return


@app.cell
def _(nx, pd):
    def build_similarity_network(df, metric="life_expect", threshold=3, random_seed=5):
        """
        Create a graph connecting countries with similar values on a given metric.
        Returns node and edge DataFrames ready for Altair.
        """
        G = nx.Graph()

        # Add nodes with attributes
        for _, row in df.iterrows():
            G.add_node(
                row["country"],
                region=row["region"],
                life_expect=row["life_expect"],
                fertility=row["fertility"],
                pop=row["pop"],
            )

        # Add edges based on similarity
        countries = df["country"].tolist()
        values = df.set_index("country")[metric]
        for i, c1 in enumerate(countries):
            for c2 in countries[i + 1 :]:
                if abs(values[c1] - values[c2]) < threshold:
                    G.add_edge(c1, c2, weight=1 / (1 + abs(values[c1] - values[c2])))

        # Force-directed layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=random_seed)

        # Convert to DataFrames
        nodes = []
        for node in G.nodes():
            attrs = G.nodes[node]
            nodes.append({
                "country": node,
                "x": pos[node][0],
                "y": pos[node][1],
                "region": attrs["region"],
                "life_expect": attrs["life_expect"],
                "fertility": attrs["fertility"],
                "pop": attrs["pop"],
                "degree": G.degree(node), # The node degree is the number of edges adjacent to the node
            })

        edges = []
        for src, tgt in G.edges():
            edges.append({
                "source": src,
                "target": tgt,
                "x": pos[src][0],
                "y": pos[src][1],
                "x2": pos[tgt][0],
                "y2": pos[tgt][1],
            })

        return pd.DataFrame(nodes), pd.DataFrame(edges), G
    return (build_similarity_network,)


@app.cell
def _(build_similarity_network, df, np):
    # Build for year 2000
    gm_2000_net = df[df["year"] == 2000].copy()
    nodes_df, edges_df, G_net = build_similarity_network(gm_2000_net, threshold=3)

    print(f"Network: {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"Average degree: {np.mean([d for _, d in G_net.degree()]):.1f}")
    return G_net, edges_df, nodes_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Static node-link diagram

    Let's start with a basic visualization: edges as rules, nodes as circles colored by region and sized by population.
    """)
    return


@app.cell
def _(alt, edges_df, nodes_df, region_scale):
    # Edge layer
    edge_chart = alt.Chart(edges_df).mark_rule(
        strokeWidth=0.4, opacity=0.15
    ).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        x2="x2:Q",
        y2="y2:Q",
    )

    # Node layer
    node_chart = alt.Chart(nodes_df).mark_circle(stroke='black', strokeWidth=0.8).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        size=alt.Size("pop:Q", scale=alt.Scale(range=[100, 2000]), legend=alt.Legend(title="Population")),
        color=alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
        tooltip=["country:N", "region:N", "life_expect:Q", "fertility:Q", "degree:Q", "pop:Q"],
    )

    (edge_chart + node_chart).properties(
        width=600, height=600,
        title={
            "text": "Country Similarity Network",
            "subtitle": "Life expectancy ±3 years – year 2000",
            "subtitleFontSize": 14,
            "fontSize": 16
        }
    ).configure_view(strokeWidth=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interactive network: click to highlight

    Let's add a **point selection** so that clicking a node highlights it and shows its label. This makes the dense network explorable.
    """)
    return


@app.cell
def _(alt, edges_df, nodes_df, region_scale):
    # Click a node (country)
    click_country = alt.selection_point(fields=["country"], on="click", empty=True)

    # Edge layer (static)
    int_edges = alt.Chart(edges_df).mark_rule(
        strokeWidth=0.4, color="lightgray", opacity=0.8
    ).encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        x2="x2:Q",
        y2="y2:Q",
    )

    # Node layer (interactive)
    int_nodes = (
        alt.Chart(nodes_df)
        .mark_circle(stroke='black', strokeWidth=0.8, opacity=0.8)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            size=alt.Size("pop:Q", scale=alt.Scale(range=[100, 2000]), legend=alt.Legend(title="Population")),
            color=alt.condition(
                click_country,
                alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(click_country, alt.value(0.8), alt.value(0.3)),
            tooltip=["country:N", "region:N", "life_expect:Q", "fertility:Q", "degree:Q"],
        )
        .add_params(click_country)
    )

    # All other labels disappear on click
    int_labels = (
        alt.Chart(nodes_df)
        .mark_text(dy=-12, fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text="country:N",
            opacity=alt.condition(click_country, alt.value(1), alt.value(0)),
        )
    )

    (int_edges + int_nodes + int_labels).properties(
        width=600, height=600,
        title={
            "text": "Country Similarity Network",
            "subtitle": [
                "Life expectancy ±3 years – year 2000",
                "(Click a country to highlight it)"
            ],
            "subtitleFontSize": 14,
            "fontSize": 16
        }
    ).configure_view(strokeWidth=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Dynamic network

    The network structure changes over time. Let's use a **marimo slider** to rebuild the network for different years and observe how clusters reorganize.
    """)
    return


@app.cell
def _(mo):
    net_year_slider = mo.ui.slider(
        start=1955, stop=2005, step=5, value=2000, label="Network year"
    )
    net_year_slider
    return (net_year_slider,)


@app.cell
def _(alt, build_similarity_network, df, net_year_slider, region_scale):
    # Rebuild network for the selected year
    _gm_year = df[df["year"] == net_year_slider.value].copy()
    _nodes, _edges, _G = build_similarity_network(_gm_year, threshold=3)

    _click = alt.selection_point(fields=["country"], on="click", empty=True)

    _edges_chart = alt.Chart(_edges).mark_rule(strokeWidth=0.4, color="lightgray", opacity=0.8).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q",
    )

    _nodes_chart = (
        alt.Chart(_nodes).mark_circle(stroke='black', strokeWidth=0.8, opacity=0.8).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            size=alt.Size("pop:Q", scale=alt.Scale(range=[30, 800]), legend=None),
            color=alt.condition(
                _click,
                alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(_click, alt.value(1), alt.value(0.3)),
            tooltip=["country:N", "region:N", "life_expect:Q", "fertility:Q", "degree:Q"],
        ).add_params(_click)
    )

    _labels = alt.Chart(_nodes).mark_text(dy=-12, fontSize=10, fontWeight="bold").encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        text="country:N",
        opacity=alt.condition(_click, alt.value(1), alt.value(0)),
    )

    (_edges_chart + _nodes_chart + _labels).properties(
        width=600, height=600,
        title={
            "text": "Country Similarity Network",
            "subtitle": [
                f"Life expectancy ±3 years – year {net_year_slider.value}",
                f"({len(_nodes)} nodes, {len(_edges)} edges)"
            ],
            "subtitleFontSize": 14,
            "fontSize": 16
        }
    ).configure_view(strokeWidth=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | What to observe

        - In **1955**, life expectancy varies hugely — the network has distinct clusters (high-LE Europe vs low-LE Africa)
        - By **2005**, many countries have converged, creating a denser, more connected network
        - Some countries remain outliers across all years
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Graph properties: what does the structure tell us?

    A force-directed layout is just one way to explore a network. We can also extract **structural properties** that quantify how countries relate to each other. Some useful measures:

    - **Degree** — the number of connections a country has. High-degree nodes are "typical" (similar to many others); low-degree nodes are outliers.
    - **Connected components** — groups of countries that are all reachable from each other. Separate components indicate fundamentally different demographic profiles.
    - **Clustering coefficient** — how much a node's neighbors are also connected to each other. High clustering means the node sits in a tight, cohesive group.

    These measures can be mapped to visual channels (size, color, position) to add analytical depth to the network visualization.
    """)
    return


@app.cell
def _(G_multi, alt, nodes_multi, nx, region_scale):
    # Add clustering coefficient to nodes
    clustering_coeff = nx.clustering(G_multi)
    _nodes_props = nodes_multi.copy()
    _nodes_props["clustering"] = _nodes_props["country"].map(clustering_coeff)

    # Degree vs clustering coefficient scatter
    _hover_props = alt.selection_point(fields=["country"], on="mouseover", empty=True)

    alt.Chart(_nodes_props).mark_circle(stroke="black", strokeWidth=0.5).encode(
        x=alt.X("degree:Q", title="Degree (number of connections)"),
        y=alt.Y("clustering:Q", title="Clustering Coefficient",
                 scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("pop:Q", scale=alt.Scale(range=[30, 600]), legend=alt.Legend(title="Population")),
        color=alt.condition(
            _hover_props,
            alt.Color("region:N", scale=region_scale, legend=alt.Legend(title="Region")),
            alt.value("lightgray"),
        ),
        opacity=alt.condition(_hover_props, alt.value(1), alt.value(0.3)),
        tooltip=[
            alt.Tooltip("country:N", title="Country"),
            alt.Tooltip("region:N", title="Region"),
            alt.Tooltip("degree:Q", title="Degree"),
            alt.Tooltip("clustering:Q", title="Clustering Coeff.", format=".2f"),
        ],
    ).add_params(_hover_props).properties(
        width=550, height=350,
        title="Network Structure: Degree vs Clustering Coefficient"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Reading the `degree` vs `clustering coef.` plot

        - **Top-left** (low degree, high clustering): outlier countries connected only to a few very similar neighbors — a small, tight clique isolated from the rest.
        - **Top-right** (high degree, high clustering): countries at the center of a large, cohesive cluster where most neighbors are also connected to each other.
        - **Bottom-left** (low degree, low clustering): isolated outliers with few connections, and those few neighbors aren't connected to each other either — truly peripheral nodes.
        - **Bottom-right** (high degree, low clustering): "hub" countries similar to many others, but those others span different groups and aren't necessarily similar to each other — bridges between clusters.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. **PyVis**: adding native interactivity

    The Altair network above works well for static exploration and brushing, but it has a key limitation: **Altair cannot natively highlight a node's neighbors when you click it**. Altair sees nodes and edges as independent marks, it has no built-in concept of graph connectivity, so highlighting neighbors on click would require **_cross-data lookups_** that Vega-Lite doesn't handle elegantly.

    [**PyVis**](https://pyvis.readthedocs.io/en/latest/) is a Python library that wraps [**vis.js**](https://visjs.github.io/vis-network/docs/network/), a battle-tested JavaScript network library. It renders to self-contained HTML, which Marimo can display via `mo.iframe()`. Out of the box you get:

    - **Drag nodes** to rearrange the layout
    - **Zoom and pan** with the mouse wheel
    - **Click a node** to select it; hover for tooltip
    - **Physics simulation** that runs live in the browser

    The tradeoff: PyVis lives outside Altair's selection system, so it cannot be linked to your other charts. Use it when **rich within-network interaction** matters more than cross-chart coordination.
    """)
    return


@app.cell
def _(G_net, Network, mo, region_colors):
    net = Network(height="500px", width="100%", bgcolor="white", font_color="black")

    net.from_nx(G_net)

    for node in net.nodes:
        name = node["id"]
        attrs = G_net.nodes[name]
        node["label"] = name
        node["color"] = region_colors.get(attrs["region"], "#cccccc")
        node["size"] = 10 + G_net.degree(name) * 2
        node["title"] = f"{name} - Region: {attrs['region']} - Life Exp: {attrs['life_expect']:.1f} - Degree: {G_net.degree(name)}"

    net.set_options("""{"physics": {"solver": "forceAtlas2Based"}}""")

    mo.iframe(net.generate_html())
    return attrs, name, node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note

        Note that the graph moves because **physics simulation is enabled by default**. PyVis/vis.js continuously calculates forces (repulsion, attraction, gravity) and updates node positions in real-time until it stabilizes. To stop the movement, disable physics after the layout settles:

        ```python
        net.set_options("\"\"
            {
              "physics": {
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 200},
                "enabled": false
              }
            }
        "\"\")
        ```
        This runs 200 iterations to compute the layout, then freezes it.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. PyVis: enabling the physics UI panel

    PyVis can expose a **built-in configuration panel** that lets users tune the physics simulation live — spring length, gravity, damping — without touching any code. Only one extra line enables it: `net.show_buttons(filter_=["physics"])`
    """)
    return


@app.cell
def _(G_net, Network, attrs, mo, name, node, region_colors):
    net2 = Network(height="700px", width="100%", bgcolor="white", font_color="black")
    net2.from_nx(G_net)

    for _node in net2.nodes:
        _name = node["id"]
        _attrs = G_net.nodes[name]
        node["label"] = name
        node["color"] = region_colors.get(attrs["region"], "#cccccc")
        node["size"] = 10 + G_net.degree(name) * 2
        node["title"] = f"{_name}<br>Region: {_attrs['region']}<br>Life Exp: {_attrs['life_expect']:.1f}<br>Degree: {G_net.degree(_name)}"

    net2.show_buttons(filter_=["physics"]) # ← adds the live physics panel

    # Move the physics panel to the right side instead of top
    html = net2.generate_html()
    html = html.replace(
        "#config {",
        "#config { position: fixed; right: 10px; top: 10px; max-height: 90vh; overflow-y: auto; z-index: 999;"
    )

    mo.iframe(html, height=700)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | What `show_buttons` gives you

    The physics panel lets you feel what each parameter does: e.g., increase `springLength` and nodes spread apart; crank up `gravitationalConstant` and clusters collapse inward. Exploring the parameter space interactively is often more informative than comparing static layouts side by side.

    ///
    """)
    return


@app.cell
def _():
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part V: Building the Dashboard**

    Now we bring everything together. The goal: **a coordinated dashboard** where temporal, textual, and network perspectives are linked through shared selections. In this dashboard, we combine a **bubble chart** (overview), a **line chart** (temporal detail), and more. The bubble chart acts as the controller: brushing updates the line chart, among other visualizations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Interaction design: Shneiderman's mantra

    If we follow [**Shneiderman's mantra**](https://www.cs.umd.edu/~ben/papers/Shneiderman1996eyes.pdf), we can organize interactions into **three complementary stages**:

    - **Overview first** with the starting chart, to get a global sense of the data
    - **Zoom and pan** to explore subsets of the data more closely; then **filter** the data using selections and conditions to focus on relevant observations
    - **Details on demand** using brushing & linking or tooltips to inspect specific values

    These interaction types are **meant to work together**. We combine them **carefully**, so that they are *complementary*, and not redundant. For example, it is usually not a good idea to have two interactions that both control **opacity** of points. Instead, we can design interactions so that e.g., one controls **opacity** (e.g., highlight a subset), another controls **color** or **size** (e.g., encode a second condition). This way, each interaction has a **clear role** and the visualization remains readable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        For more dashboard inspiration in Marimo, explore the [Marimo Gallery](https://marimo.io/gallery?tag=dashboard). Two particularly relevant examples:

        1 [Movies Dashboard](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/movies.py/wasm) — interactive exploration of film data

        2 [Lego Prices Dashboard](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/dashboard/lego/notebook.py/wasm) — price analysis with linked views

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -----
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **🌍 Global Development Overview**

    This interactive dashboard lets you explore relationships between **fertility**, **life expectancy**, and **regional distribution** across countries. Start by choosing a **year** using the slider below, then select (brush or click) a few **countries** to begin your exploration. The following charts will update automatically based on your selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    year_slider = mo.ui.slider(
        start=1955, stop=2005, step=5, value=2000, label="Select a reference year for the analysis:"
    )
    year_slider
    return (year_slider,)


@app.cell(hide_code=True)
def _(
    alt,
    df,
    fert_domain,
    life_domain,
    mo,
    pop_labels,
    region_scale,
    year_slider,
):
    _df_year = df[df["year"] == year_slider.value].copy()

    _brush = alt.selection_interval()
    _click = alt.selection_point(fields=["country"], toggle=True)
    _highlight = _brush | _click

    _scatter = (
        alt.Chart(_df_year)
        .mark_circle(size=60, opacity=0.8, stroke='black', strokeWidth=1)
        .encode(
            x=alt.X(
                "fertility:Q", 
                scale=alt.Scale(zero=False, domain=fert_domain), 
                title="Fertility",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)
            ),
            y=alt.Y(
                "life_expect:Q", 
                scale=alt.Scale(zero=False, domain=life_domain), 
                title="Life Expectancy",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)
            ),
            color=alt.condition(
                _highlight,
                alt.Color(
                    "region:N",
                    scale=region_scale,
                    title="Region",
                    legend=alt.Legend(
                        #orient="bottom",
                        #direction="horizontal",
                        labelFontSize=12,
                        titleFontSize=13
                    ),
                ),
                alt.value("lightgray"),
            ),
            size=alt.Size(
                "pop_bucket:O",
                scale=alt.Scale(range=[50, 1500], domain=pop_labels),
                sort=['<5M', '5-20M', '20-100M', '100-500M', '500M+'],
                title="Population",
                legend=alt.Legend(
                    orient="right",
                    titleFontSize=13,
                    labelFontSize=12,
                ),
            ),
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("life_expect:Q", title="Life Expectancy", format=".1f"),
                alt.Tooltip("fertility:Q", title="Fertility", format=".2f"),
                alt.Tooltip("pop:Q", title="Population", format="~s"),
            ]
        )
        .add_params(_brush, _click)
        .properties(
            padding={"top": 20, "bottom": 10, "left": 10, "right": 10},
            width=450,
            height=380,
            title=alt.TitleParams(
                text=f"Global Development Overview ({year_slider.value})",
                subtitle=[
                    "Compare fertility and life expectancy across countries.",
                    "Interact with the chart (drag, click, legend) to explore.",
                    "Selections update the insights shown below."
                ],
                fontSize=18, 
                subtitleFontSize=13,
                offset=20,
                # anchor="start",
            )
        )
    )

    dash_scatter = mo.ui.altair_chart(_scatter)
    dash_scatter
    return (dash_scatter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The scatter plot above is the **main controller** of this dashboard. Brush a region or click individual countries to select them — all visualizations below will update to reflect your selection.

    ⚠️ **Note**: legend filtering is visual only; summary statistics below reflect brushed or clicked countries.
    """)
    return


@app.cell(hide_code=True)
def _(dash_scatter, df, mo, region_colors, year_slider):
    _sel = dash_scatter.value
    selected_countries = _sel.get("country", [])

    _df_year = df[df["year"] == year_slider.value].copy()

    # Count total selected
    n_selected = len(selected_countries)

    # Counts by region
    counts_by_region = (
        _df_year[_df_year["country"].isin(selected_countries)]
        .groupby("region")["country"]
        .count()
        .reindex(list(region_colors.keys()), fill_value=0)
    )

    lines = ''.join(
        f"- {r}: {c} countries\n"
        for r, c in counts_by_region.items()
        if c > 0
    ) or "_select at least one country to continue the analysis._"

    mo.md(f"""
    ### Selection summary

    You have selected **{n_selected} countries**:
    {lines}
    """)
    return (selected_countries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regional breakdown

    How does your selection compare to the **overall regional distribution**? The bar chart below shows total countries per region (gray) overlaid with your selected countries (colored). Toggle normalization to see percentages instead of counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    normalize_toggle = mo.ui.checkbox(label="Normalize by region", value=False)
    normalize_toggle
    return (normalize_toggle,)


@app.cell(hide_code=True)
def _(
    alt,
    dash_scatter,
    df,
    mo,
    normalize_toggle,
    region_colors,
    region_scale,
    selected_countries,
    year_slider,
):
    _sel = dash_scatter.value

    _df_year = df[df["year"] == year_slider.value].copy()

    _sort_regions = list(region_colors.keys())

    # selected per region:
    selected = (
        _df_year[_df_year["country"].isin(selected_countries)]
        .groupby("region")["country"]
        .count()
        .reindex(list(region_colors.keys()), fill_value=0)
        .rename("selected")
        .reset_index()
    )

    # totals per region:
    _df_year = df[df["year"] == year_slider.value].copy()

    totals = (
        _df_year.groupby("region")["country"]
        .count()
        .reindex(list(region_colors.keys()), fill_value=0)
        .rename("total")
        .reset_index()
        .sort_values(by='total', ascending=False)
    )

    region_order = totals["region"].tolist()

    # build labels like: Europe (15)
    region_label_order = [
        f"{r} ({int(totals.loc[totals['region'] == r, 'total'].values[0])})"
        for r in region_order
    ]

    totals['region_count'] = region_label_order

    # merge:
    df_bar = totals.merge(selected, on="region")

    # compute percentage:
    df_bar["share"] = df_bar["selected"] / df_bar["total"]
    df_bar["share"] = df_bar["share"].fillna(0)

    # choose which variable to plot:
    normalize = normalize_toggle.value
    x_field = "share" if normalize else "total"
    x_title = "Share of countries (%)" if normalize else "Number of countries"

    _tooltip = [
        alt.Tooltip("region:N", title="Region"),
        alt.Tooltip("total:Q", title="Total countries"),
        alt.Tooltip("selected:Q", title="Selected countries"),
        alt.Tooltip("share:Q", title="Selected share", format=".1%")
    ]

    # background bars (total or 100%):
    if normalize:
        background = alt.Chart(df_bar).mark_bar(
            color="lightgray", opacity=0.35
        ).encode(
            y=alt.Y("region_count:N", sort=region_order, title=None),
            x=alt.X("one:Q", title="Share of countries (%)", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=_tooltip
        ).transform_calculate(
            one="1"
        )
    else:
        background = alt.Chart(df_bar).mark_bar(
            color="lightgray", opacity=0.35
        ).encode(
            y=alt.Y("region_count:N", sort=region_order, title=None),
            x=alt.X("total:Q", title="Number of countries", axis=alt.Axis()),
            tooltip=_tooltip
        )

    # overlay bars (selected or share):
    overlay = alt.Chart(df_bar).mark_bar().encode(
        y=alt.Y("region_count:N", sort=region_order, title=None),
        x=alt.X(
            "share" if normalize else "selected",
            axis=alt.Axis(format="%") if normalize else alt.Axis()
        ),
        color=alt.Color("region:N", scale=region_scale, legend=None),
        tooltip=_tooltip
    )

    bar_chart = (background + overlay).properties(
        width=250,
        height=250,
        title="Selected countries vs total (by region)"
    )

    dash_bar = mo.ui.altair_chart(bar_chart)
    dash_bar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Similarity network

        Which of your selected countries are _**demographically similar**_? The network below connects countries with comparable life expectancy and fertility, with your selection highlighted.
    """)
    return


@app.cell(hide_code=True)
def _(df, np, nx, pd):
    from sklearn.preprocessing import StandardScaler

    def build_multimetric_network(df, metrics=["life_expect", "fertility"], threshold_percentile=25, random_seed=5):
        """
        Build a similarity network using multiple standardized metrics.
        Countries are connected if their Euclidean distance (on standardized features)
        falls below the given percentile threshold of all pairwise distances.
        """
        G = nx.Graph()
        features = df[metrics].values
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)

        # Pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(features_std)

        # Threshold: connect pairs below the Nth percentile
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        threshold = np.percentile(upper_tri, threshold_percentile)

        countries = df["country"].tolist()
        for _, row in df.iterrows():
            G.add_node(row["country"], region=row["region"],
                       life_expect=row["life_expect"], fertility=row["fertility"],
                       pop=row["pop"])

        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                if dist_matrix[i, j] < threshold:
                    G.add_edge(countries[i], countries[j],
                               weight=1 / (1 + dist_matrix[i, j]))

        pos = nx.spring_layout(G, k=2, iterations=80, seed=random_seed)

        nodes = []
        for node in G.nodes():
            attrs = G.nodes[node]
            nodes.append({
                "country": node, "x": pos[node][0], "y": pos[node][1],
                "region": attrs["region"], "life_expect": attrs["life_expect"],
                "fertility": attrs["fertility"], "pop": attrs["pop"],
                "degree": G.degree(node),
            })
        edges = []
        for src, tgt in G.edges():
            edges.append({
                "source": src, "target": tgt,
                "x": pos[src][0], "y": pos[src][1],
                "x2": pos[tgt][0], "y2": pos[tgt][1],
            })

        return pd.DataFrame(nodes), pd.DataFrame(edges), G

    gm_2000_multi = df[df["year"] == 2000].copy()
    nodes_multi, edges_multi, G_multi = build_multimetric_network(
        gm_2000_multi, metrics=["life_expect", "fertility"], threshold_percentile=20
    )
    print(f"Multi-metric network: {len(nodes_multi)} nodes, {len(edges_multi)} edges")
    print(f"Average degree: {np.mean([d for _, d in G_multi.degree()]):.1f}")
    print(f"Connected components: {nx.number_connected_components(G_multi)}")
    return G_multi, StandardScaler, build_multimetric_network, nodes_multi


@app.cell(hide_code=True)
def _(
    alt,
    build_multimetric_network,
    dash_scatter,
    df,
    mo,
    region_scale,
    year_slider,
):
    _sel = dash_scatter.value

    if len(_sel) > 0:
        _df_net = df[df["year"] == year_slider.value].copy()
        _nodes_dash, _edges_dash, _G_dash = build_multimetric_network(
            _df_net, metrics=["life_expect", "fertility"], threshold_percentile=25
        )

        _selected_countries = _sel["country"].unique().tolist()
        _nodes_dash["selected"] = _nodes_dash["country"].isin(_selected_countries).astype(int)

        _hover_net = alt.selection_point(fields=["country"], on="mouseover", empty=True)

        _e_dash = alt.Chart(_edges_dash).mark_rule(
            strokeWidth=0.5, color="gray", opacity=0.2
        ).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            x2="x2:Q", y2="y2:Q",
        )

        _n_dash = (
            alt.Chart(_nodes_dash)
            .mark_circle(stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                size=alt.Size("pop:Q", scale=alt.Scale(range=[40, 1000]), legend=None),
                color=alt.condition(
                    alt.datum.selected == 1,
                    alt.Color("region:N", scale=region_scale, legend=None),
                    alt.value("lightgray"),
                ),
                opacity=alt.condition(
                    alt.datum.selected == 1,
                    alt.value(0.9),
                    alt.value(0.3),
                ),
                tooltip=[
                    alt.Tooltip("country:N", title="Country"),
                    alt.Tooltip("region:N", title="Region"),
                    alt.Tooltip("life_expect:Q", title="Life Exp.", format=".1f"),
                    alt.Tooltip("fertility:Q", title="Fertility", format=".2f"),
                    alt.Tooltip("degree:Q", title="Connections"),
                ],
            ).add_params(_hover_net)
        )

        _lbl_dash = (
            alt.Chart(_nodes_dash[_nodes_dash["selected"] == 1])
            .mark_text(dy=-14, fontSize=10, fontWeight="bold")
            .encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                text="country:N",
                color=alt.Color("region:N", scale=region_scale, legend=None),
                opacity=alt.condition(_hover_net, alt.value(1), alt.value(0.7)),
            )
        )

        _out = (_e_dash + _n_dash + _lbl_dash).properties(
            width=500, height=400,
            title=f"Similarity Network — Selected Countries ({year_slider.value})"
        ).configure_view(strokeWidth=0)
    else:
        _out = mo.md("*Brush or click the scatter plot to highlight countries in the network.*")

    _out
    return


@app.cell(hide_code=True)
def _(dash_scatter, mo):
    _sel = dash_scatter.value

    if len(_sel) > 0:
        _out_text = mo.md(
            f"**{len(_sel)} countries selected** — "
            f"Mean life expectancy: **{_sel['life_expect'].mean():.1f}**, "
            f"mean fertility: **{_sel['fertility'].mean():.2f}**"
        )
        table = mo.ui.table(
            _sel[["country", "region", "life_expect", "fertility", "pop"]]
            .sort_values("pop", ascending=False)
            .reset_index(drop=True)
        )
        _out = mo.vstack([_out_text, table])
    else:
        table = None
        _out = mo.md(
            "*Brush or click the scatter plot above to populate the table.*"
        )

    _out
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temporal trends

    How did your selected countries evolve **over time**? For the sake of readability, only 5 countries can be shown (at most). If you selected more than 5 countries, you can choose which countries to display by refining your selection in the table above.
    """)
    return


@app.cell(hide_code=True)
def _(dash_scatter, df, mo, table):
    MAX_COUNTRIES = 5

    if table is None:
        _out = mo.md("")
    else:
        # rows selected via checkboxes in the table
        sel_tbl = table.value
        if len(dash_scatter.value) <= 5:
            sel_tbl = dash_scatter.value
        # if nothing selected in the table, fall back to ALL rows shown in the table
        note_no_data = mo.md("⚠️ No data selected.")
        df_for_analysis = sel_tbl # if len(sel_tbl) > 0 else note_no_data

        # optional: limit to max 5 countries for readability (only for analysis)
        countries = df_for_analysis["country"].unique().tolist()
        if len(countries) > MAX_COUNTRIES:
            note = mo.md(
                f"⚠️ {len(countries)} countries selected for analysis. "
                f"Showing only the first **{MAX_COUNTRIES}** (select fewer in the table for full detail)."
            )
            countries = countries[:MAX_COUNTRIES]
            df_for_analysis = df_for_analysis[df_for_analysis["country"].isin(countries)]
        else:
            note = mo.md("")

        _df_time = df[df["country"].isin(countries)].copy()


        _out = mo.vstack([note, mo.md(f"**Countries selected:** {', '.join(countries)}")])

    _out
    return (countries,)


@app.cell(hide_code=True)
def _(
    alt,
    countries,
    dash_scatter,
    df,
    fert_domain,
    life_domain,
    mo,
    region_scale,
):
    _sel = dash_scatter.value

    if len(_sel) > 0:
        _df_time = df[df["country"].isin(countries)]

        # --- color encodings ---
        region_color_with_legend = alt.Color(
            "region:N",
            scale=region_scale,
            legend=alt.Legend(title="Region", columns=3),
        )

        region_color_no_legend = alt.Color(
            "region:N",
            scale=region_scale,
            legend=None,
        )

        _base = alt.Chart(_df_time).encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            detail="country:N", # one line per country
        )

        # --- life expectancy lines ---
        life_lines = _base.mark_line(point=True, strokeWidth=2).encode(
            y=alt.Y("life_expect:Q", title="Life Expectancy", scale=alt.Scale(domain=life_domain, zero=False)),
            color=region_color_no_legend,
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("life_expect:Q", title="Life Expectancy", format=".1f"),
            ],
        )

        # labels at the end of each line (last year)
        life_labels = (
            alt.Chart(_df_time)
            .transform_aggregate(
                last_year="max(year)",
                groupby=["country", "region"]
            )
            .transform_lookup(
                lookup="country",
                from_=alt.LookupData(
                    _df_time[_df_time["year"] == _df_time["year"].max()],
                    key="country",
                    fields=["life_expect", "year"]
                ),
            )
            .mark_text(align="left", dx=6, dy=0)
            .encode(
                x=alt.X("year:O"),
                y=alt.Y("life_expect:Q"),
                text="country:N",
                color=region_color_no_legend,
            )
        )

        life_chart = (life_lines + life_labels).properties(width=360, height=220, title="Life Expectancy")

        # --- fertility lines ---
        fert_lines = _base.mark_line(point=True, strokeWidth=2).encode(
            y=alt.Y("fertility:Q", title="Fertility", scale=alt.Scale(domain=fert_domain, zero=False)),
            color=region_color_with_legend,
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("fertility:Q", title="Fertility", format=".2f"),
            ],
        )

        fert_labels = (
            alt.Chart(_df_time)
            .transform_aggregate(
                last_year="max(year)",
                groupby=["country", "region"]
            )
            .transform_lookup(
                lookup="country",
                from_=alt.LookupData(
                    _df_time[_df_time["year"] == _df_time["year"].max()],
                    key="country",
                    fields=["fertility", "year"]
                ),
            )
            .mark_text(align="left", dx=6, dy=0)
            .encode(
                x=alt.X("year:O"),
                y=alt.Y("fertility:Q"),
                text="country:N",
                color=region_color_no_legend,
            )
        )

        fert_chart = (fert_lines + fert_labels).properties(width=360, height=220, title="Fertility")

        # combine side by side with shared color scale
        _out = (life_chart | fert_chart).resolve_scale(color="shared")

    else:
        _out = mo.md("*Brush or click the scatter plot above to see temporal trends for selected countries.*")

    _out
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VI: Your Turn!**

    Now it's your turn to explore. The exercises below are intentionally open-ended. Use everything you've learned (temporal charts, text search, network views, linked selections, marimo widgets) to build your own visualizations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        Build a dashboard with **at least two linked visualizations** that answers a question of your choice about the Gapminder data. Some ideas to get you started:

        - **Population explosion:** which regions grew the fastest? Try a stacked area chart with a region filter.
        - **Convergence story:** did countries become more similar over time? Consider the standard deviation of life expectancy per year.
        - **Country deep-dive:** build a profile for a single country — line charts, rank within region, network neighbors.
        - **Fertility transition:** some countries had dramatic fertility drops. Can you identify and highlight their trajectories?

        Be creative! The goal is to practice combining interaction techniques to tell a data story.

    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Example: overview + detail pattern

    One common approach is to pair an **overview** with a **detail** view:

    - **Bottom chart (overview):** a line chart of _global mean life expectancy per year_, with an `alt.selection_interval(encodings=["x"])` brush
    - **Top chart (detail):** a bar chart showing mean life expectancy per region, filtered by the time range selected in the bottom chart

    The overview controls the detail via `transform_filter`.
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Create a brush on the x-axis only
    # time_brush = alt.selection_interval(encodings=["x"])

    # Step 2: Build the overview line chart (all years, global mean)
    # Add .add_params(time_brush) to this chart

    # Step 3: Build the detail bar chart (mean LE per region)
    # Add .transform_filter(time_brush) to this chart

    # Step 4: Combine with alt.vconcat(detail, overview)
    return


@app.cell
def _(alt, df, region_scale):
    # 💡 Solution

    # Step 1: Create a brush on the x-axis only
    time_brush = alt.selection_interval(encodings=["x"])

    # Step 2: Build the overview line chart (all years, global mean)
    overview_line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("mean(life_expect):Q", title="Mean Life Expectancy"),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("mean(life_expect):Q", format=".1f")],
        )
        .add_params(time_brush)
        .properties(width=600, height=150, title="Overview: drag to select a time range")
    )

    detail_bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mean(life_expect):Q", title="Mean Life Expectancy"),
            y=alt.Y("region:N", title="", sort="-x"),
            color=alt.Color("region:N", scale=region_scale, legend=None),
            tooltip=["region:N", alt.Tooltip("mean(life_expect):Q", format=".1f")],
        )
        .transform_filter(time_brush)
        .properties(width=600, height=250, title="Detail: mean life expectancy by region (filtered)")
    )

    alt.vconcat(detail_bars, overview_line).resolve_scale(color="independent")
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VII: A Unified Visual Analytics Tool**

    Now we bring everything together from Part V into a single window, unified visual analytics tool/interface/dashboard. In this visual analytics tool, we combine a **bubble chart** (overview), a **line chart** (temporal detail), and more. The bubble chart acts as the controller: brushing updates the line chart, among other visualizations.
    """)
    return


@app.cell
def _(mo):
    va_year_slider = mo.ui.slider(
        start=1955, stop=2005, step=5, value=2000,
        label="Year"
    )

    va_normalize_toggle = mo.ui.checkbox(
        label="Normalize by region",
        value=False
    )

    # Interaction mode switch
    va_nav_mode = mo.ui.radio(
        options=["Brush", "Pan/Zoom"],
        value="Brush",
        label="Interaction",
    )

    mo.md("")
    return va_nav_mode, va_normalize_toggle, va_year_slider


@app.cell
def _(
    alt,
    df,
    fert_domain,
    life_domain,
    mo,
    pop_labels,
    region_scale,
    va_nav_mode,
    va_year_slider,
):
    va_year2 = va_year_slider.value
    va_df_year2 = df[df["year"] == va_year2].copy()
    va_mode = va_nav_mode.value  # "Brush" or "Pan/Zoom"

    va_region_legend = alt.selection_point(
        name="va_region_legend",
        fields=["region"],
        bind="legend",
        toggle=True,
        empty=True,
    )

    va_click = alt.selection_point(
        name="va_scatter_click",
        fields=["country"],
        toggle=True,
        empty="none",
    )

    # ---- Mode-dependent interactions ----
    if va_mode == "Brush":
        va_brush = alt.selection_interval(name="va_scatter_brush", empty="none")
        va_focus = va_brush | va_click
        interaction_params = [va_brush, va_click]
    else:
        va_zoom = alt.selection_interval(name="va_zoom", bind="scales")
        va_focus = va_click  # IMPORTANT: no brush exists in this mode
        interaction_params = [va_zoom, va_click]

    # ---- Encodings (IMPORTANT for zoom): don't hard-lock domains in Pan/Zoom mode ----
    x_enc = alt.X(
        "fertility:Q",
        scale=alt.Scale(zero=False, domain=fert_domain) if va_mode == "Brush" else alt.Scale(zero=False),
        title="Fertility",
    )
    y_enc = alt.Y(
        "life_expect:Q",
        scale=alt.Scale(zero=False, domain=life_domain) if va_mode == "Brush" else alt.Scale(zero=False),
        title="Life Expectancy",
    )

    base = alt.Chart(va_df_year2).encode(
        x=x_enc,
        y=y_enc,
        size=alt.Size(
            "pop_bucket:O",
            scale=alt.Scale(range=[50, 1500], domain=pop_labels),
            sort=["<5M", "5-20M", "20-100M", "100-500M", "500M+"],
            title="Population",
        ),
        tooltip=[
            alt.Tooltip("country:N", title="Country"),
            alt.Tooltip("region:N", title="Region"),
            alt.Tooltip("life_expect:Q", format=".1f"),
            alt.Tooltip("fertility:Q", format=".2f"),
            alt.Tooltip("pop:Q", format="~s"),
        ],
    )

    SOFT_STROKE = "#777"

    # Gray background
    layer_gray = (
        base.mark_circle(size=60, color="#d0d0d0", opacity=0.30)
        .add_params(*interaction_params)  # click + (brush OR zoom)
    )

    # Colored (legend) layer with stroke mainly for legend points
    layer_color = (
        base.mark_circle(size=60, stroke="#444", strokeWidth=0.35, strokeOpacity=0.55)
        .encode(
            color=alt.Color(
                "region:N",
                scale=region_scale,
                title="Region",
                legend=alt.Legend(orient="right"),
            ),
            opacity=alt.condition(va_focus, alt.value(0.95), alt.value(0.75), empty=True),
        )
        .add_params(va_region_legend)
        .transform_filter(va_region_legend)
    )

    # Click ring
    layer_click_top = (
        base.mark_circle(size=120, fillOpacity=0, stroke="#111", strokeWidth=2.5)
        .transform_filter(va_click)
    )

    va_scatter = (
        (layer_gray + layer_color + layer_click_top)
        .properties(width=450, height=360, title=f"Global Development Overview ({va_year2})")
        .configure_view(strokeWidth=0)
    )

    va_scatter_view2 = mo.ui.altair_chart(va_scatter)
    mo.md("")
    return (va_scatter_view2,)


@app.cell
def _(
    StandardScaler,
    alt,
    df,
    euclidean_distances,
    fert_domain,
    life_domain,
    mo,
    np,
    nx,
    pd,
    region_colors,
    region_scale,
    va_normalize_toggle,
    va_scatter_view2,
    va_year_slider,
):
    va_year = va_year_slider.value
    va_normalize = va_normalize_toggle.value
    va_df_year = df[df["year"] == va_year].copy()

    # Read selection here (NOT in Cell 2)
    va_sel = va_scatter_view2.value
    va_selected_countries = va_sel["country"].unique().tolist() if va_sel is not None and len(va_sel) else []

    # ----------------------------
    # Panel sizing (MAKE CONSISTENT)
    # ----------------------------
    TOP_H = 360          # scatter + bar height
    BOT_H = 250          # network + trends total height

    W_BAR = 400
    W_NET = 600
    W_TRENDS = 550

    # trends: 2 stacked line charts whose total ~ BOT_H
    # leave a little breathing room between them
    LINE_GAP = 0
    LINE_H = (BOT_H - LINE_GAP) // 2   # e.g. 175 if BOT_H=360


    # ----------------------------
    # Bar: selected vs total by region
    # ----------------------------
    va_selected_by_region = (
        va_df_year[va_df_year["country"].isin(va_selected_countries)]
        .groupby("region")["country"]
        .count()
        .reindex(list(region_colors.keys()), fill_value=0)
        .rename("selected")
        .reset_index()
    )

    va_totals_by_region = (
        va_df_year.groupby("region")["country"]
        .count()
        .reindex(list(region_colors.keys()), fill_value=0)
        .rename("total")
        .reset_index()
        .sort_values("total", ascending=False)
    )

    va_region_order = va_totals_by_region["region"].tolist()
    va_totals_by_region["region_count"] = [
        f"{r} ({int(va_totals_by_region.loc[va_totals_by_region['region'] == r, 'total'].values[0])})"
        for r in va_region_order
    ]

    va_df_bar = va_totals_by_region.merge(va_selected_by_region, on="region")
    va_df_bar["share"] = (va_df_bar["selected"] / va_df_bar["total"]).fillna(0)

    va_bar_tooltip = [
        alt.Tooltip("region:N", title="Region"),
        alt.Tooltip("total:Q", title="Total"),
        alt.Tooltip("selected:Q", title="Selected"),
        alt.Tooltip("share:Q", title="Share", format=".1%"),
    ]

    if va_normalize:
        va_bar_bg = (
            alt.Chart(va_df_bar)
            .mark_bar(color="lightgray", opacity=0.35)
            .encode(
                y=alt.Y("region_count:N", sort=va_region_order, title=None),
                x=alt.X(
                    "one:Q",
                    title="Share (%)",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=va_bar_tooltip,
            )
            .transform_calculate(one="1")
        )
        va_bar_fg = (
            alt.Chart(va_df_bar)
            .mark_bar()
            .encode(
                y=alt.Y("region_count:N", sort=va_region_order, title=None),
                x=alt.X("share:Q", axis=alt.Axis(format="%")),
                color=alt.Color("region:N", scale=region_scale, legend=None),
                tooltip=va_bar_tooltip,
            )
        )
    else:
        va_bar_bg = (
            alt.Chart(va_df_bar)
            .mark_bar(color="lightgray", opacity=0.35)
            .encode(
                y=alt.Y("region_count:N", sort=va_region_order, title=None),
                x=alt.X("total:Q", title="Count"),
                tooltip=va_bar_tooltip,
            )
        )
        va_bar_fg = (
            alt.Chart(va_df_bar)
            .mark_bar()
            .encode(
                y=alt.Y("region_count:N", sort=va_region_order, title=None),
                x=alt.X("selected:Q", title="Selected"),
                color=alt.Color("region:N", scale=region_scale, legend=None),
                tooltip=va_bar_tooltip,
            )
        )

    va_bar_chart = (va_bar_bg + va_bar_fg).properties(
        width=W_BAR,
        height=TOP_H,
        title="Region summary",
    ).configure_view(strokeWidth=0)

    va_bar_view = mo.ui.altair_chart(va_bar_chart)


    # ----------------------------
    # Network
    # ----------------------------
    va_hover = alt.selection_point(
        name="va_network_hover",
        fields=["country"],
        on="mouseover",
        empty=True,
    )

    def va_build_multimetric_network(va_df_in, metrics=("life_expect", "fertility"), threshold_percentile=25, random_seed=5):
        G = nx.Graph()
        X = va_df_in[list(metrics)].values
        Xs = StandardScaler().fit_transform(X)

        dist = euclidean_distances(Xs)
        upper = dist[np.triu_indices_from(dist, k=1)]
        thresh = np.percentile(upper, threshold_percentile)

        countries = va_df_in["country"].tolist()
        for _, r in va_df_in.iterrows():
            G.add_node(
                r["country"],
                region=r["region"],
                life_expect=r["life_expect"],
                fertility=r["fertility"],
                pop=r["pop"],
            )

        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                if dist[i, j] < thresh:
                    G.add_edge(countries[i], countries[j])

        pos = nx.spring_layout(G, k=2, iterations=80, seed=random_seed)

        nodes = pd.DataFrame([{
            "country": n,
            "x": pos[n][0], "y": pos[n][1],
            "region": G.nodes[n]["region"],
            "life_expect": G.nodes[n]["life_expect"],
            "fertility": G.nodes[n]["fertility"],
            "pop": G.nodes[n]["pop"],
            "degree": G.degree(n),
        } for n in G.nodes()])

        edges = pd.DataFrame([{
            "x": pos[s][0], "y": pos[s][1],
            "x2": pos[t][0], "y2": pos[t][1],
        } for s, t in G.edges()])

        return nodes, edges

    if len(va_selected_countries) > 0:
        va_nodes, va_edges = va_build_multimetric_network(va_df_year, threshold_percentile=25)
        va_nodes["selected"] = va_nodes["country"].isin(va_selected_countries).astype(int)

        va_e = alt.Chart(va_edges).mark_rule(
            strokeWidth=0.5, color="gray", opacity=0.2
        ).encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            x2="x2:Q",
            y2="y2:Q",
        )

        va_n = (
            alt.Chart(va_nodes)
            .mark_circle(stroke="black", strokeWidth=0.4)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                size=alt.Size("pop:Q", scale=alt.Scale(range=[30, 800]), legend=None),
                color=alt.condition(
                    alt.datum.selected == 1,
                    alt.Color("region:N", scale=region_scale, legend=None),
                    alt.value("lightgray"),
                ),
                opacity=alt.condition(
                    alt.datum.selected == 1,
                    alt.value(0.9),
                    alt.value(0.25),
                ),
                tooltip=[
                    alt.Tooltip("country:N"),
                    alt.Tooltip("region:N"),
                    alt.Tooltip("life_expect:Q", format=".1f"),
                    alt.Tooltip("fertility:Q", format=".2f"),
                    alt.Tooltip("degree:Q", title="Connections"),
                ],
            )
            .add_params(va_hover)
        )

        # ✅ LABELS (only for selected countries)
        va_lbl = (
            alt.Chart(va_nodes[va_nodes["selected"] == 1])
            .mark_text(dy=-14, fontSize=10, fontWeight="bold")
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                text="country:N",
                color=alt.Color("region:N", scale=region_scale, legend=None),
                opacity=alt.condition(va_hover, alt.value(1), alt.value(0.7)),
            )
        )

        va_network_chart2 = (va_e + va_n + va_lbl).properties(
            width=W_NET,
            height=350,
            title="Similarity network",
        ).configure_view(strokeWidth=0)

        va_network_view = mo.ui.altair_chart(va_network_chart2)
    else:
        va_network_view = mo.md("*Select countries in the scatter to show the network.*")


    # ----------------------------
    # Trends (stacked; match Network height)
    # ----------------------------
    VA_MAX_COUNTRIES = 5
    va_countries_for_trends = va_selected_countries[:VA_MAX_COUNTRIES]

    if len(va_countries_for_trends) > 0:
        va_df_time = df[df["country"].isin(va_countries_for_trends)].copy()

        va_base = alt.Chart(va_df_time).encode(
            x=alt.X("year:O", axis=alt.Axis(labelAngle=0), title="Year"),
            detail="country:N",
            color=alt.Color("region:N", scale=region_scale, legend=None),
        )

        va_life_chart = (
            va_base.mark_line(point=True, strokeWidth=2)
            .encode(
                y=alt.Y("life_expect:Q", scale=alt.Scale(domain=life_domain, zero=False), title="Life"),
                tooltip=[
                    alt.Tooltip("country:N"),
                    alt.Tooltip("region:N"),
                    alt.Tooltip("year:O"),
                    alt.Tooltip("life_expect:Q", format=".1f"),
                ],
            )
            .properties(width=W_TRENDS, height=LINE_H, title="Life Expectancy over time")
            .configure_view(strokeWidth=0)
        )

        va_fert_chart = (
            va_base.mark_line(point=True, strokeWidth=2)
            .encode(
                y=alt.Y("fertility:Q", scale=alt.Scale(domain=fert_domain, zero=False), title="Fertility"),
                tooltip=[
                    alt.Tooltip("country:N"),
                    alt.Tooltip("region:N"),
                    alt.Tooltip("year:O"),
                    alt.Tooltip("fertility:Q", format=".2f"),
                ],
            )
            .properties(width=W_TRENDS, height=LINE_H, title="Fertility over time")
            .configure_view(strokeWidth=0)
        )

        va_trends_life_view = mo.ui.altair_chart(va_life_chart)
        va_trends_fert_view = mo.ui.altair_chart(va_fert_chart)

        # gap=1 gives a small spacing; total will be close to BOT_H
        va_trends_view = mo.vstack([va_trends_life_view, va_trends_fert_view], gap=1)
    else:
        va_trends_view = mo.md("*Select countries to show trends.*")
    return va_bar_view, va_network_view, va_trends_view


@app.cell
def _(
    mo,
    va_bar_view,
    va_nav_mode,
    va_network_view,
    va_normalize_toggle,
    va_scatter_view2,
    va_trends_view,
    va_year_slider,
):
    va_controls = mo.hstack([va_year_slider, va_nav_mode, va_normalize_toggle])

    va_top = mo.hstack([
        va_scatter_view2,
        va_bar_view
    ])

    va_bottom = mo.hstack([
        va_network_view,
        va_trends_view
    ])

    mo.vstack([
        va_controls,
        va_top,
        va_bottom
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
