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

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **VABD 2026**
    ## 📆 **Week 02 – Data Visualization for Exploratory Data Analysis**

    This week we move from example datasets to **real-world data**: the [**Utrecht Housing Dataset**](https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset?resource=download&select=2025-housing-dataset-alldata.csv). You'll learn how to explore, transform, and visualize local housing data to uncover patterns in prices, locations, and property features. Building on last week's foundations, we'll dive deeper into encodings, selections, and layered charts.

    **Topics covered:**
    - How to load and explore a real-world dataset
    - The basics of Altair's declarative syntax
    - Different chart types: scatter plots, bar charts, histograms, line charts
    - Encoding channels: x, y, color, size, shape
    - Data transformations and aggregations
    - Interactivity and selections
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Index**
    - [Part I: Setting Up and Loading Data](#part-i-setting-up-and-loading-data)
    - [Part II: Data Exploration](#part-ii-data-exploration)
    - [Part III: Visualization](#part-iii-visualization)
    - [Part IV: Interactive Selections](#part-iv-interactive-selections)
    - [Appendix: A Framework for Choosing Visualizations](#appendix-a-framework-for-choosing-visualizations)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Setting Up and Loading Data**
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
    return alt, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Loading the Utrecht Housing Dataset

    **About the dataset**. In this notebook, we will explore the `Utrecht Housing Dataset` (van Otterloo & Burda, 2025, freely available on [Kaggle](https://www.kaggle.com/code/dfq6h46jg2ik/exploring-the-utrecht-housing-dataset/notebook)), a collection of **153 houses** sold in and around Utrecht, The Netherlands. This dataset was collected in 2024 and includes **23 features** about:

    - 🏠 Property characteristics (house type, build year, area, rooms)
    - 💰 Market metrics (asking price, retail value) — in thousands of euros
    - ⚡ Energy labels (A++ to G)
    - 📍 Location data (coordinates, district, distance from train station)

    We first load the CSV file and do some basic data cleaning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Column descriptions**

    | Column | Type | Unit | Description |
    |--------|------|------|-------------|
    | `zipcode4` | string | — | 4-digit postal code |
    | `zipcode6` | string | — | 6-character postal code |
    | `zipcode6id` | int | — | Unique identifier for zipcode6 |
    | `housetype` | string | — | Type of property (e.g., apartment, townhouse) |
    | `lot-area` | int | m² | Total lot/plot area |
    | `house-area` | int | m² | Living area of the house |
    | `garden-size` | int | m² | Garden area |
    | `rooms` | int | count | Number of rooms |
    | `bathrooms` | int | count | Number of bathrooms |
    | `x-coor` | float | degrees | Longitude (GPS) |
    | `y-coor` | float | degrees | Latitude (GPS) |
    | `buildyear` | int | year | Year the house was built |
    | `retailvalue` | float | k€ | Official retail/tax value |
    | `askingprice` | float | k€ | Listed asking price |
    | `energylabel` | string | — | Energy efficiency rating (A++ to G) |
    | `energyeff` | string | — | Energy efficiency category (derived) |
    | `valuationdate` | string | date | Date of valuation |
    | `street` | string | — | Street name |
    | `subdistrict` | string | — | Subdistrict/neighborhood |
    | `district` | string | — | District name |
    | `city` | string | — | City name |
    | `dist-from-train` | float | km | Euclidean distance to nearest train station |
    """)
    return


@app.cell
def _(pd):
    # Load the (raw) dataset

    # Update this path to where your CSV file is located
    data_path = "./data/2025-housing-dataset-alldata.csv"

    df_raw = pd.read_csv(data_path)
    df_raw
    return (df_raw,)


@app.cell
def _(df_raw, pd):
    # Create a clean copy with better column names and transformations
    df = df_raw.copy()

    # Convert prices from thousands to actual euros
    df["askingprice"] = df["askingprice"] * 1000
    df["retailvalue"] = df["retailvalue"] * 1000

    # Rename columns for easier use (remove hyphens)
    df.columns = df.columns.str.replace("-", "_")

    # Convert garden_size to numeric (it might have some special values)
    df["garden_size"] = (
        pd.to_numeric(df["garden_size"], errors="coerce").fillna(0).astype(int)
    )

    # Clean energy efficiency
    df["energyeff"] = pd.to_numeric(df["energyeff"], errors="coerce")

    # Translate house types in english
    df["housetype_en"] = df["housetype"].map(
        {"woonhuis": "house", "appartement": "apartment"}
    )

    # Parse valuation date
    df["valuationdate"] = pd.to_datetime(df["valuationdate"])

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Data Exploration**
    """)
    return


@app.cell
def _(df):
    # Basic info about the dataset
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nKey columns and their types:")
    key_cols = [
        "housetype",
        "house_area",
        "rooms",
        "buildyear",
        "askingprice",
        "energylabel",
        "district",
        "dist_from_train",
    ]
    for col in key_cols:
        print(f"  {col}: {df[col].dtype}")
    return


@app.cell
def _(df):
    # Summary statistics for numerical columns
    df[
        [
            "house_area",
            "lot_area",
            "rooms",
            "bathrooms",
            "buildyear",
            "askingprice",
            "retailvalue",
            "dist_from_train",
        ]
    ].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | What is the difference between retail value (`retailvalue`) and asking price (`askingprice`)?

    The **retail value** (official retail/tax value)  is the **government-assessed value** of the property, used for taxation purposes. In the Netherlands, this is known as the **WOZ-waarde** (Waardering Onroerende Zaken). It's determined annually by the municipality based on factors like location, size, and comparable sales. It's typically more conservative and may lag behind actual market conditions.

    The **asking price** (listed asking price) is the **seller's listed price** when putting the property on the market. It's set by the homeowner (often with a real estate agent's guidance) and reflects what they hope to receive. It's influenced by market demand, competition, and seller motivation.
    ///
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    # Check unique values for categorical columns
    print("House types:", df["housetype"].unique())
    print("\nDistricts:", df["district"].unique())
    print("\nEnergy labels:", df["energylabel"].unique())
    print("\nCities:", df["city"].unique())
    return


@app.cell
def _(df):
    # Check missing values
    na_counts = df.isna().sum()
    na_counts[na_counts > 0]

    # df[df.energyeff.isna()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part III: Visualization**

    As introduced in Week 01, Altair uses a **declarative approach** to visualization. Instead of specifying how to draw something step by step (_imperative_), you describe *what* you want to see and let Altair handle the rendering details. In other words, **you don't tell the computer *how* to draw something, rather you describe *what* you want to see.**

    The basic formula follows this structure:

    ```python
    alt.Chart(data).mark_*().encode(
        x='column_name',
        y='another_column'
    )
    ```

    Data types in Altair:

    | Type | Description | Example |
    |------|-------------|---------|
    | `:Q` | Quantitative (continuous numbers) | price, area |
    | `:N` | Nominal (unordered categories) | house type, district |
    | `:O` | Ordinal (ordered categories) | energy label (A > B > C...) |
    | `:T` | Temporal (dates/times) | valuation date |

    Now, let's explore the data through a series of research questions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ 1**: What is the distribution of house prices in Utrecht?

    Why do we need to understand distributions? Before conducting any comparative or predictive analysis, we must understand the **shape** of our _target_ variable. The distribution tells us:

    - Is the data symmetric or skewed?
    - Are there outliers that might affect our analysis?
    - What is a "typical" value, and how much variation exists?

    For initial exploration, a **histogram** is often the best starting point because it shows the actual frequency of values in defined intervals, making patterns immediately visible.
    """)
    return


@app.cell
def _(alt, df):
    # Histogram of asking prices
    alt.Chart(df).mark_bar(
        opacity=0.7,  # Slight transparency (0=invisible, 1=solid)
        color="steelblue",
    ).encode(
        # X-axis: bin the continuous 'askingprice' into 20 buckets
        x=alt.X(
            "askingprice:Q", bin=alt.Bin(maxbins=20), title="Asking Price (€)"
        ),  # bin=True uses defaults
        # Y-axis: count how many houses fall in each bin
        y=alt.Y(
            "count():Q", title="Number of Houses"
        ),  # count():Q is a special aggregation function: counts rows per bin, no field needed
        # Tooltip: what shows on hover
        tooltip=[
            alt.Tooltip(
                "askingprice:Q", bin=alt.Bin(maxbins=20), title="Price Range"
            ),
            alt.Tooltip("count():Q", title="Count"),
        ],
    ).properties(
        width=600,
        height=350,
        title="Distribution of Asking Prices in Utrecht (2024)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Understanding the syntax.** Let's break down what happened:

    1. `alt.Chart(df)` - Creates a chart object with your data source
    2. `.mark_bar()` - Specifies the visual mark type (e.g., `mark_point()`, `mark_bar()`, `mark_line()`)
    3. `.encode(x='askingprice', y='count()')` - Maps data columns to visual channels (position, color, size, etc.)

    This declarative pattern remains consistent across all visualizations in Altair, regardless of complexity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation**. The distribution reveals several insights:

    - The distribution is **roughly unimodal** with a slight right skew that suggests some higher-value properties, but no extreme outliers
    - Most houses are priced between **€300,000 and €700,000** – there are relatively few houses below €250,000 or above €900,000

    This right skew is typical of housing markets. There is a floor (houses cannot cost less than construction value) but no ceiling (luxury properties can be very expensive).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ 2**: Do apartments and houses have different price distributions?

    Why compare distributions across groups? The previous histogram showed us the overall price distribution, but it might be **masking important differences** between property types. If apartments and houses have fundamentally different price ranges, analyzing them together could lead to misleading conclusions.

    We now have two variables: a continuous variable (`askingprice`) and a categorical variable (`housetype`). Let's start with what we know: **histograms**. We can create overlapping histograms using color to distinguish the groups.

    Our options include:

    | Visualization | Best for | Limitation |
    |--------------|----------|------------|
    | **Side-by-side histograms** | Comparing full distribution shapes | Hard to compare when counts differ |
    | **Overlapping density plots** | Direct shape comparison | Can be cluttered with many groups |
    | **Box plots** | Comparing medians, quartiles, outliers | Hides distribution shape |
    | **Stacked histogram** | Showing composition within bins | Hard to compare shapes directly |

    For comparing two groups, **box plots** offer a compact summary, while a **stacked or layered histogram** shows the full distribution.
    """)
    return


@app.cell
def _(alt, df):
    alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X(
            "askingprice:Q", bin=alt.Bin(maxbins=20), title="Asking Price (€)"
        ),
        y=alt.Y("count()", title="Number of Houses"),
        color=alt.Color("housetype_en:N", title="Property Type"),
        tooltip=[
            alt.Tooltip(
                "askingprice:Q", bin=alt.Bin(maxbins=20), title="Price Range"
            ),
            alt.Tooltip("housetype_en:N", title="Property Type"),
            alt.Tooltip("count():Q", title="Count"),
        ],
    ).properties(
        width=500, height=300, title="Price Distribution by Property Type"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The overlapping histogram reveals the difference, but it can be hard to read where the bars overlap. An alternative is to **facet**: create separate histograms side by side. Faceting is usually cleaner and easier to explain.
    """)
    return


@app.cell
def _(alt, df):
    alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X(
            "askingprice:Q", bin=alt.Bin(maxbins=20), title="Asking Price (€)"
        ),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("housetype_en:N", legend=None),
    ).facet(column="housetype_en:N").properties(
        title="Price Distribution: Apartments vs Houses"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can clearly see both distributions! But histograms take up space. When we want a **compact summary** that shows medians, quartiles, and outliers, we use a **box plot**.
    """)
    return


@app.cell
def _(alt, df):
    # Box plot comparing house types
    alt.Chart(df).mark_boxplot(
        # extent="min-max", # Shows full data range, no outliers marked
        size=50, # Width of the box
    ).encode(
        x=alt.X("housetype_en:N", title="Property Type").axis(labelAngle=0),
        y=alt.Y(
            "askingprice:Q", title="Asking Price (€)", scale=alt.Scale(zero=False)
        ),
        color=alt.Color("housetype_en:N", legend=None),
    ).properties(
        width=300, height=300, title="Price Distribution by Property Type"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The meaning of `extent` in box plots.** The `extent` parameter controls how far the **whiskers** extend from the box:

    | Value | Whiskers extend to | Outliers |
    |-------|-------------------|----------|
    | `1.5` (default) | 1.5 × IQR from quartiles | Shown as individual points |
    | `'min-max'` | Minimum and maximum values | None marked |

    **What is IQR?** The Interquartile Range (Q3 − Q1) — the height of the box itself.

    With the default `extent=1.5`:
    - Upper whisker: Q3 + 1.5 × IQR
    - Lower whisker: Q1 − 1.5 × IQR
    - Points beyond these limits appear as **outlier dots**

    We use `extent='min-max'` when we want to show the **full data range**, but in practice the default is often more informative as it highlights unusual values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ 3**: Does energy efficiency affect house prices?

    Why study energy efficiency? With increasing attention to climate change and rising energy costs, energy efficiency has become a relevant factor in housing decisions. The dataset includes:

    - `energylabel`: EU energy efficiency rating (G to A++)
    - `energyeff`: binary indicator (1 = efficient, 0 = not efficient)

    Energy labels are **ordinal categorical** data (they have a natural order from G to A++). We want to see if prices increase with better energy ratings. A **box plot ordered by energy label** allows us to compare price distributions while respecting the natural ordering of efficiency classes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Create a box plot that shows the **distribution of asking prices by energy label**.

    **Requirements:**
    - Filter out rows with missing energy labels
    - Order the x-axis from best (A++) to worst (G)
    - Use a color scheme that intuitively represents efficiency (e.g., `green` = `good`, `red` = `bad`)

    **Hints:**
    - Use `sort=` in `alt.X()` to control the order
    - Use `scale=alt.Scale(scheme='redyellowgreen')` for the color
    - The energy label is **ordinal** (`:O`) because it has a natural order
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Define the order for energy labels (best to worst)
    # energy_order = ['A++', 'A+', ...]

    # Step 2: Filter out missing energy labels
    # df_energy = df[df['energylabel'].notna() & df['energylabel'].isin(...)].copy()

    # Step 3: Create the box plot
    # alt.Chart(df_energy).mark_boxplot(...).encode(
    #     x=alt.X('energylabel:O', sort=energy_order, ...),
    #     y=alt.Y('askingprice:Q', ...),
    #     color=alt.Color('energylabel:O', scale=alt.Scale(scheme='redyellowgreen'), ...)
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** The relationship between energy efficiency and price is **not straightforward**:

    - There is no clear monotonic increase in price with better energy labels
    - Some middle-efficiency categories (C, D) show prices comparable to higher-efficiency homes
    - The most efficient homes (A+, A++) do not consistently command the highest prices

    **Why might this be?**

    - **Confounding variables**: Newer homes tend to be more energy-efficient but may be in less central locations; older homes in prime locations may have poor energy ratings but high prices due to location (and other factors, such as the _property size_)
    - **Sample size**: Some energy categories have few observations
    - **Market maturity**: Energy efficiency may not yet be fully priced into the Utrecht market

    This illustrates an important lesson: **correlation does not imply a simple causal relationship**, and controlling for confounding variables would be necessary for rigorous analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ 4**: How does location affect house prices?

    Why **spatial analysis** matters? Real estate professionals often say "_location, location, location_". The dataset includes geographic coordinates (x_coor is latitude ~52°, y_coor is longitude ~5°) and district information. We want to understand:

    - Do prices vary systematically across Utrecht?
    - Are there high-value and low-value neighborhoods?

    We can use two approaches: a **geographic scatter** for spatial patterns and a **bar chart **for easy comparison.
    """)
    return


@app.cell
def _(alt, df):
    # Geographic scatter plot with price as color
    # Note: x_coor is latitude, y_coor is longitude in this dataset
    alt.Chart(df).mark_circle(
        size=200,
        opacity=0.7,
        stroke='white',
        strokeWidth=1
    ).encode(
        longitude='y_coor:Q', # longitude on x-axis
        latitude='x_coor:Q', # latitude on y-axis
        color=alt.Color('askingprice:Q', 
                       scale=alt.Scale(scheme='viridis'),
                       title='Price (€)'),
        tooltip=[
            alt.Tooltip("district", title="District"),
            alt.Tooltip("house_area:Q", title="Area (sqm)"),
            alt.Tooltip("askingprice:Q", title="Asking Price (€)"),
            alt.Tooltip("housetype:N", title="House Type")
        ]
    ).properties(
        width=500,
        height=450,
        title='Geographic Distribution of House Prices in Utrecht'
    )
    return


@app.cell
def _(alt, df):
    # Bar chart of median price by district
    # Note that here we use a horizontal bar chart to improve the legibility of district labels, which would be harder to read in a vertical layout

    alt.Chart(df).mark_bar().encode(
        y=alt.Y('district:N', sort='-x', title='District'), # 'x' → sort ascending, '-x' → sort descending
        x=alt.X('median(askingprice):Q', title='Median Asking Price (€)'),
        color=alt.Color('median(askingprice):Q', 
                       scale=alt.Scale(scheme='viridis'),
                       legend=None),
        tooltip=[
            alt.Tooltip('district:N', title='District'),
            alt.Tooltip('median(askingprice):Q', title='Median Price', format=',.0f'),
            alt.Tooltip('count()', title='Number of Houses')
        ]
    ).properties(
        width=500,
        height=350,
        title='Median House Price by District'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** The visualizations reveal geographic patterns in pricing:
    - **Noord-Oost** and **Leidsche Rijn** have the highest median prices
    - The range from lowest to highest median is substantial (approximately €200,000–300,000)

    This confirms that location is a significant factor in Utrecht's housing market.

    **_Caution_**: Some districts have few observations, so median values may not be reliable estimates of the true district-level prices. Always check the sample sizes!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Create a (horizontal) **bar chart showing the number of houses per district**, **using the same district order**
    defined by **median asking price**.

    **Requirements:**
    - Derive the district order from the median asking price by district (highest to lowest), and use it to sort the y-axis of the *count* bar chart
    - Plot the number of houses per district

    **Hints:**
    - Use `groupby()` + `median()` to define the district order
    - Store the ordered district names in a list
    - Pass that list to `sort=` inside `alt.Y()`
    - Use `count()` as the x-axis encoding
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Compute the median asking price by district
    # district_median = ...

    # Step 2: Sort districts by median price (highest to lowest)
    # district_order = ...

    # Step 3: Create a bar chart of number of houses per district
    # alt.Chart(df).mark_bar().encode(
    #     y=alt.Y('district:N', sort=district_order, title='District'),
    #     x=alt.X('count():Q', title='Number of Houses'),
    #     ...
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ 5**: How does property size relate to asking price?

    After examining distributions and group differences in previous questions (e.g., housing types and energy levels), we now turn to a fundamental question in housing economics:** does larger living area correspond to higher asking prices?**

    In housing data, we expect that larger homes tend to be more expensive, but the shape and strength of this relationship are worth visualizing. Are there diminishing returns on price as size increases? Do some homes defy the general trend (e.g., very large but comparatively inexpensive)? Do smaller homes cluster at specific price ranges?

    To explore the relationship between living area (`house_area`) and asking price (`askingprice`), a **scatterplot** is ideal: it shows every data point and makes patterns (including outliers) immediately visible.
    """)
    return


@app.cell
def _(alt, df):
    # Scatterplot of house area vs askingprice
    alt.Chart(df).mark_circle(opacity=0.6, color="steelblue", size=60).encode(
        x=alt.X("house_area:Q", title="House Area (sqm)"),
        y=alt.Y("askingprice:Q", title="Asking Price (€)"),
        tooltip=[
            alt.Tooltip("house_area:Q", title="Area (sqm)"),
            alt.Tooltip("askingprice:Q", title="Asking Price (€)"),
            alt.Tooltip("housetype:N", title="House Type"),
        ],
    ).properties(title="House Prices by House Area", width=650, height=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation**. As expected, we observe a **positive correlation**: larger houses tend to have higher asking prices. However, the spread of points also suggests that other factors influence pricing beyond size alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Extra: fast tooltips + interactivity
    When exploring a chart, we are not always interested in polishing titles and axis labels.
    Sometimes the goal is simply to **inspect individual observations** quickly. In Altair, you can pass multiple fields to `tooltip` by providing a list of column names as:

    ```python
    tooltip=[
        "housetype_en",
        "house_area",
        "askingprice",
        "district",
        "energylabel",
    ]
    ```

    This is convenient during early exploration because it lets you hover over a point and immediately see relevant attributes without defining `alt.Tooltip(...)` objects.
    You can also append `.interactive()` at the end of the chart to enable **zooming and panning**:

    ```python
    chart = chart.interactive()
    ```
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Layered charts**. We can combine multiple marks in a single chart using layers. A common example is the scatter plot with a **trend line**, offered derived from a simple linear model: individual points show raw observations, while an additional line summarizes the overall relationship between two variables.
    """)
    return


@app.cell
def _(alt, df):
    # Create layers: points + trend line
    base_chart = alt.Chart(df)

    # Scatter plot
    points_layer = base_chart.mark_circle(size=80, opacity=0.5).encode(
        x=alt.X("house_area:Q", title="House Area (m²)"),
        y=alt.Y("askingprice:Q", title="Asking Price (€)"),
        tooltip=["housetype_en", "house_area", "askingprice", "district", "rooms"]
    )

    # Add a regression line showing the average relationship between house size and asking price
    # The model fit is based on all the points in base_chart
    trend_line = (
        base_chart.transform_regression("house_area", "askingprice") # askingprice ≈ a + b · house_area
        .mark_line(color="red", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="house_area:Q", y="askingprice:Q")
    )

    # Layered chart
    (points_layer + trend_line).properties(
        width=600, height=400, title="House Prices with Trend Line"
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Adding color and more encodings.** One of Altair's superpowers is easily adding more dimensions to your visualization through **encoding channels**. As a first step, we encode the property type (`housetype`), to distinguish apartments from houses within the scatter plot.
    """)
    return


@app.cell
def _(alt, df):
    # Color by house type
    alt.Chart(df).mark_circle(size=80).encode(
        x=alt.X("house_area:Q", title="House Area (m²)"),
        y=alt.Y("askingprice:Q", title="Asking Price (€)"),
        color=alt.Color("housetype_en:N", title="Property Type"),
        tooltip=["housetype_en", "house_area", "askingprice", "district", "rooms"],
    ).properties(
        width=600, height=400, title="House Prices by Property Area and Type"
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Multiple encodings: color, size, and shape.** We can encode multiple variables simultaneously. Notice how apartments tend to be smaller and cheaper, while houses have more variation.
    """)
    return


@app.cell
def _(alt, df, energy_order):
    alt.Chart(df).mark_point(filled=True).encode(
        x=alt.X("house_area:Q", title="House Area (m²)"),
        y=alt.Y("askingprice:Q", title="Asking Price (€)"),

        color=alt.Color(
            "energylabel:N",
            title="Energy Efficiency",
            sort=energy_order,
            scale=alt.Scale(scheme="redyellowgreen", reverse=True),
        ),

        size=alt.Size(
            "house_area:Q",
            title="House Area (m²)",
            scale=alt.Scale(range=[30, 300]),
        ),

        shape=alt.Shape("housetype_en:N", title="Type"),

        tooltip=[
            "housetype_en",
            "house_area",
            "askingprice",
            "district",
            "rooms",
            "energylabel",
        ],
    ).properties(
        width=650,
        height=450,
        title="Utrecht Housing: Multi-dimensional View",
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When too many variables are encoded in a single chart, readability can quickly suffer. There is always a **trade-off between informativeness and comprehension**. One effective solution is **faceting** (also called **small multiples**): instead of encoding everything in one view, we split the data into multiple small charts, one for each category. This makes **comparisons across groups** easier and clearer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

    Create a **faceted scatter plot of Asking Price vs House Area**, with **one panel per house type** (see RQ 1).

    **Requirements:**
    - Plot `house_area` on the x-axis and `askingprice` on the y-axis
    - Encode `energylabel` using color
    - Encode `house_area` using size
    - Use **faceting** to create one small chart per `housetype_en`
    - Keep the same axes scales across facets to allow comparison

    **Hints:**
    - Start from a scatter plot you already built
    - Use `.facet()` with `column=` (or `row=`) for house type
    - Set `resolve_scale()` only if you want different scales (otherwise keep them shared)
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Create a scatter plot of asking price vs house area
    # base_scatter = alt.Chart(df).mark_point(...)

    # Step 2: Facet (small multiples) by house type
    # facet_scatter = base_scatter.facet(...)

    # Step 3: Set sizing + title
    # facet_scatter.properties(...)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part IV: Interactive Selections**

    One of Altair's most powerful features is **interactive selections**. These let users explore data dynamically!

    **Why interactivity matters**? Static charts show one view of the data. Interactive charts let users:
    - Filter to subsets of interest
    - See relationships between multiple views
    - Explore outliers and patterns
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Brush selection

    **Brush selection** is a common interactive technique that lets users **select a subset of data directly on the chart** by clicking and dragging over a region. The selected area (the brush) defines which data points are emphasized, filtered, or linked to other views. Brushing is especially useful for exploring patterns, spotting clusters, and focusing attention without removing context from the rest of the data.
    """)
    return


@app.cell
def _(alt, df):
    # Create an interval selection parameter (a rectangular brush)
    # Users can click and drag to select a region of the chart
    brush = alt.selection_interval() # brush is a parameter object that can be used in chart creation

    # Scatter plot with interactive brush selection
    scatter = (
        alt.Chart(df)
        # Draw one point per house
        .mark_point(filled=True)
        .encode(
            # Encode house size on the x-axis
            x=alt.X("house_area:Q", title="House Area (m²)"),

            # Encode asking price on the y-axis
            y=alt.Y("askingprice:Q", title="Asking Price (€)"),

            # Conditional coloring:
            # - If a point is inside the brushed region → color by house type
            # - Otherwise → fade it out in light gray
            color=alt.condition(
                brush,
                alt.Color("housetype_en:N", title="Property Type"),
                alt.value("lightgray"),
            ),

            # Show detailed information on hover
            tooltip=[
                "housetype_en",
                "house_area",
                "askingprice",
                "district",
                "energylabel",
            ],
        )
        # Chart size and title
        .properties(width=600, height=400, title="Drag to select a region!")

        # Attach the brush selection to the chart
        .add_params(brush)
    )

    scatter
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Linked selections

    The real power of interactivity emerges when multiple charts are linked through a **shared selection**. In linked views, a selection made in one chart automatically affects other charts that reference the same selection parameter. For example, selecting points in a scatter plot can update a bar chart to show only the corresponding subset of data.

    Linked selections enable users to **explore relationships across different representations of the same data**, supporting comparison, contextual filtering, and coordinated exploration. This technique is particularly effective when combining **overview + detail** views or when different charts reveal complementary aspects of the data.
    """)
    return


@app.cell
def _(alt, df):
    # Create the same interval selection (rectangular brush) as before
    brush2 = alt.selection_interval()

    # Scatter plot showing individual houses
    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("house_area:Q", title="House Area (m²)"),
            y=alt.Y("askingprice:Q", title="Asking Price (€)"),
            tooltip=[
                "housetype_en",
                "district",
                "askingprice",
                "energylabel",
            ],
            # Conditional coloring:
            # - Points inside the brush → colored by house type
            # - Points outside the brush → light gray
            color=alt.condition(
                brush2,
                alt.Color("housetype_en:N"), # for a fix color: alt.value("steelblue"),
                alt.value("lightgray"),
            )
        )
        .properties(width=400, height=300)
        # Attach the brush selection to the scatter plot
        .add_params(brush2)
    )

    # Bar chart showing the number of houses per district
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Count"),
            y=alt.Y("district:N", sort="-x", title="District"),
            color=alt.Color("district:N", legend=None),
        )
        .properties(width=200, height=300) # Match height with scatter plot
        # Filter the data based on the brush selection
        # Only houses inside the brushed region are counted
        .transform_filter(brush2)
    )

    # Combine scatter plot and bar chart horizontally (|) - to stack vertically, use & 
    # chart1 | chart2 → columns
    # chart1 & chart2 → rows
    (points | bars).properties(
        title="Brush on the scatter plot to filter the bar chart!"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Keeping colors consistent in interactive charts.** Why did the colors in the first chart change? When using conditional color encodings (`alt.condition`), interactive selections (like a brush), and multiple linked views, Altair **may recompute the default color palette**. As a result, the same category can end up with different colors across charts or examples. This is expected behavior: **Altair guarantees correctness, not semantic consistency**.

    If color carries meaning (for example, `Apartment = blue` and `House = orange`), this meaning must be preserved explicitly. In this case, to preserve meaning (e.g. Apartment = blue, House = orange), the color scale must be **explicitly fixed by defining both the domain and the range**:

    ```python
    alt.Color(
        "housetype_en:N",
        scale=alt.Scale(
            domain=["apartment", "house"],
            range=["#4C78A8", "#F58518"],
        )
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Point selection

    **Point selection** allows users to select data by **clicking directly** on visual elements, such as points, bars, or legend entries. Unlike brushing, which selects a continuous region, point selection targets discrete values or categories.
    """)
    return


@app.cell
def _(alt, df):
    # Create a point selection (single/multi click selection)
    # Here we select categories from the "district" field by clicking bars
    click = alt.selection_point(fields=["district"])

    # Define a shared conditional color encoding
    # - If a district is selected → color by district
    # - If not selected → fade marks to light gray
    # This same encoding will be reused for BOTH charts (linked view)
    color_scale = alt.condition(
        click,
        alt.Color("district:N", legend=None),
        alt.value("lightgray"),
    )


    # Bar chart used as the interactive selector
    bar_selector = (
        alt.Chart(df)
        # Draw one bar per district
        .mark_bar()
        .encode(
            # Districts listed on the y-axis, sorted by descending count
            y=alt.Y("district:N", sort="-x", title=""),
            # Bar length represents the number of listings in that district
            x=alt.X("count():Q", title="Count"),
            # Color responds to the click selection (selected = colored, others = gray)
            color=color_scale,
        )
        # Chart size and title
        .properties(width=150, height=280, title="Click a district")

        # Attach the click selection to the bar chart
        # Clicking a bar updates the selection state
        .add_params(click)
    )


    # Scatter plot that responds to the same selection
    scatter_filtered = (
        alt.Chart(df)
        # Draw one point per house listing
        .mark_circle(size=80)
        .encode(
            x=alt.X("house_area:Q", title="House Area (m²)"),
            y=alt.Y("askingprice:Q", title="Asking Price (€)"),

            # Use the SAME conditional color encoding as in the bar chart
            # This links the scatter plot to the selected district(s)
            color=color_scale,

            # Show details on hover
            tooltip=[
                "housetype_en",
                "district",
                "rooms",
                "askingprice",
                "energylabel",
            ],
        )
        # Chart size
        .properties(width=450, height=300)
    )

    # Combine bar selector and scatter plot side-by-side
    bar_selector | scatter_filtered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part V: Wrap-Up**

    Let's put together what we've learned about charts, layering and selections. We'll build up from a simple bar chart to an interactive visualization with dynamic aggregations, starting with a concrete question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **RQ:** How did asking prices evolve over the course of 2024?

    _Were prices stable, or do we observe significant fluctuations?_ To answer this, we'll visualize the **mean asking price by month** and progressively add layers to enrich our visualization.
    """)
    return


@app.cell
def _(alt, df):
    bars1 = (
        alt.Chart(df)
        .mark_bar() # .mark_line(point=True) is also a common option for temporal data, point=True adds circles at each month
        .encode(
            x=alt.X("yearmonth(valuationdate):T", title="Valuation Month"),
            y=alt.Y("mean(askingprice):Q", title="Mean Asking Price (€)"),
            tooltip=[
                alt.Tooltip("yearmonth(valuationdate):T", title="Month"),
                alt.Tooltip("mean(askingprice):Q", title="Mean Asking Price (€)", format=",.0f"),
                alt.Tooltip("count():Q", title="Listings"),
            ],
        )
    )

    bars1.properties(width=650, height=300, title="Mean Asking Price by Month")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | What is `yearmonth()`?

        `yearmonth()` is a **temporal transformation function** in Altair that extracts the **year and month** from a complete date, discarding the day.

        **Example:**
    ```
        2024-01-05 → 2024-01
        2024-01-20 → 2024-01  ← Same representation!
        2024-02-03 → 2024-02
    ```

        **Why use it?**
        When you have precise dates but want to **aggregate by month**:
    ```python
        # Without yearmonth() → each day treated separately
        x="valuationdate:T"  # Too many points, one per date

        # With yearmonth() → groups all dates from the same month
        x="yearmonth(valuationdate):T"  # One bar per month
    ```

        **Other temporal functions:**
        - `year(date)` → extracts only the year (e.g., `2024`)
        - `month(date)` → extracts only the month (e.g., `3` for March)
        - `quarter(date)` → extracts the quarter (e.g., `Q1`)
        - `date(datetime)` → removes time, keeps only the date
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Your turn!
        type: tip

    Add a **horizontal reference line** showing the **overall mean asking price** across all months.

    Hints:
    - Use `mark_rule()` to create a horizontal line
    - Encode `y` with `mean(askingprice):Q`
    - Layer it with `bars1` using the `+` operator
    - Try customizing with `color="firebrick"` and `size=3`
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE
    # Step 1: Create a rule mark for the overall mean
    # overall_mean_rule = (
    #     alt.Chart(df)
    #     ...
    #     .encode(
    #     ...
    #     )
    # )

    # Step 2: Layer the rule with bars1 and apply properties
    # alt.layer(...).properties(
    #     ...
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the overall mean as a reference, we can now quickly identify:
    - **Above-average months:** Which periods had higher-than-typical listings?
    - **Below-average months:** When were asking prices more modest?

    But this raises a new question: *what if we want to compare a specific subset of months?* The fixed reference line always shows the global mean. Let's make it **dynamic**.
    """)
    return


@app.cell
def _(alt, df):
    # Create an interval selection that only responds to horizontal (x-axis) dragging
    # This allows users to brush/select a range of months
    brush3 = alt.selection_interval(encodings=["x"])

    bars_brush = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(valuationdate):T", title="Valuation month"),
            y=alt.Y("mean(askingprice):Q", title="Mean asking price (€)"),

            # Conditional opacity: selected bars are fully opaque (1),
            # unselected bars fade to 25% opacity
            opacity=alt.condition(brush3, alt.value(1), alt.value(0.25)),

            tooltip=[
                alt.Tooltip("yearmonth(valuationdate):T", title="Month"),
                alt.Tooltip("mean(askingprice):Q", title="Mean asking price (€)", format=",.0f"),
                alt.Tooltip("count():Q", title="Listings"),
            ],
        )
        # Attach the brush selection to this chart — makes bars interactive
        .add_params(brush3)
    )

    rule_filtered = (
        alt.Chart(df)
        .mark_rule(color="firebrick", size=3)
        .encode(
            y=alt.Y("mean(askingprice):Q", title="")
        )
        # KEY: transform_filter makes the rule compute its mean ONLY from
        # the data points inside the brushed selection
        # When nothing is selected, it uses all data (overall mean)
        .transform_filter(brush3)
    )

    # Layer the interactive bars with the dynamic rule
    chart3 = alt.layer(bars_brush, rule_filtered).properties(
        width=650,
        height=300,
        title="Brush months to recompute the mean asking price"
    )

    chart3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Key takeaways**

    - **Real data requires cleaning and exploration first**
    - **Choose visualizations based on data types and questions**
    - **Altair's declarative syntax separates WHAT from HOW**
    - **Layering combines multiple visual elements**
    - **Selections enable interactive exploration**
    - **Transform filters allow dynamic aggregations**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # **Appendix: A Framework for Choosing Visualizations**

    ## 1. Start with the question
    - What do we want to learn?
    - What decision will this inform?

    ## 2. Identify the variable types
    | Variable Type | Examples from this dataset |
    |---------------|---------------------------|
    | Continuous | askingprice, house_area, dist_from_train |
    | Categorical (nominal) | housetype, district |
    | Categorical (ordinal) | energylabel |
    | Temporal | buildyear, valuationdate |

    ## 3. Match visualization to question type

    | Question Type | Recommended Visualization |
    |--------------|---------------------------|
    | Distribution of one variable | Histogram, density plot, box plot |
    | Comparison across groups | Box plots, grouped bar charts |
    | Relationship between two continuous | Scatter plot (with regression line) |
    | Spatial patterns | Geographic scatter with color encoding |
    | Trends over time | Line chart |
    | Multiple factors | Faceting, color + size encoding, linked views |

    ## 4. Interpret critically
    - What does the visualization show?
    - What are its limitations?
    - What confounding factors might exist?
    - What follow-up questions arise?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        1 [From Data to Viz](https://www.data-to-viz.com/) — Interactive decision tree for choosing charts

        2 [Data Viz Project](https://datavizproject.com/) — Searchable gallery of chart types

        3 [Altair Example Gallery](https://altair-viz.github.io/gallery/index.html) — See what's possible with Altair

        4 [Observable Plot Gallery](https://observablehq.com/@observablehq/plot-gallery) — Inspiring examples with JavaScript

        5 [A Collection of Dataviz Caveats](https://www.yan-holtz.com/PDF/Dataviz_Caveats.pdf) — Common mistakes and how to avoid them

        6 [Visualization Analysis and Design Lectures](https://www.youtube.com/playlist?list=PLT4XLHmqHJBeB5LwmRmo6ln-m7K3lGvrk) – Complete video series by Tamara Munzner, covering the [book](https://www.cs.ubc.ca/~tmm/vadbook/) _Visualization Analysis and Design_

        7 [ColorBrewer](https://colorbrewer2.org/) — Color schemes for maps and data visualization

        8 [Viz Palette](https://projects.susielu.com/viz-palette) — Test your color palettes for accessibility
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > In summary, data visualizations tell stories. Relatively subtle choices, such as the range of the axes in a bar chart or line graph, can have a big impact on the story that a figure tells. When you look at data graphics, you want to ask yourself whether the graph has been designed to tell a story that accurately reflects the underlying data, or whether it has been designed to tell a story more closely aligned with what the designer would like you to believe.
    >
    > — [callingbullshit.org](https://www.callingbullshit.org/tools/tools_misleading_axes.html)
    """)
    return


if __name__ == "__main__":
    app.run()
