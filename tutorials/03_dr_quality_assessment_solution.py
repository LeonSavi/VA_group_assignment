# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "umap-learn>=0.5.0",
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
    ## 📆 **Week 03 – Quality Assessment of Dimensionality Reduction**

    In this workshop we explore **Dimensionality Reduction (DR)** techniques – methods that transform high-dimensional (HD) data into lower dimensions (LD) for visualization and analysis. We'll learn both **classical linear methods** (PCA, MDS) and **modern nonlinear approaches** (t-SNE, UMAP), and critically assess their quality using **distortion metrics**.

    **Topics covered:**
    - What is dimensionality reduction and why do we need it?
    - Linear DR: PCA (_variance_) and MDS (_distances_)
    - Nonlinear DR: t-SNE and UMAP
    - Types of distortions: stretching, compression, missing/false neighbors
    - Quality metrics: Trustworthiness & Continuity
    - Visualizing projection quality
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Index**
    - [Part I: Introduction to Dimensionality Reduction](#part-i-introduction-to-dimensionality-reduction)
    - [Part II: Linear DR with PCA](#part-ii-linear-dr-with-pca)
    - [Part III: Linear DR with MDS](#part-iii-linear-dr-with-mds)
    - [Part IV: Nonlinear DR with t-SNE](#part-iv-nonlinear-dr-with-t-sne)
    - [Part V: Nonlinear DR with UMAP](#part-v-nonlinear-dr-with-umap)
    - [Part VI: Quality assessment](#part-vi-quality-assessment)
    - [Summary](#summary)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Introduction to Dimensionality Reduction**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What is Dimensionality Reduction?

    **Dimensionality Reduction (DR)** transforms data from a high-dimensional space to a lower-dimensional space while preserving important structure.

    Given input data $\mathbf{X} \in \mathbb{R}^{n \times p}$ (n samples, p features), DR produces output $\mathbf{Y} \in \mathbb{R}^{n \times q}$ where $q \ll p$ (typically $q = 2$ or $3$ for visualization).

    **Why do we need it?**
    - **Visualization**: we can only see 2D or 3D plots
    - **Curse of dimensionality**: distance metrics become less meaningful in high dimensions
    - **Noise reduction**: lower dimensions can filter out noise
    - **Computational efficiency**: fewer features = faster algorithms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Setup and data loading

    We'll use the **Wine dataset**, which is a classic benchmark containing chemical analysis of wines from three different cultivars (grape varieties) grown in Italy. It is conveniently available directly from [`sklearn.datasets`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) (no downloading
    or cleaning required).

    **Dataset overview:**
    - **178 samples** (wines) from 3 cultivars
    - **13 numerical features** (all continuous, already clean)
    - **2 _target_ features** (wine varieties: `cultivar` is the _model-friendly_ target, `cultivar_name` is the _people-friendly_ label)
    - **Source**: Forina, M. et al. (1988), [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

    💡 One **key advantage** of this dataset is that **we know the ground truth**: each wine is already labeled with its true cultivar. This is important because **it allows us to evaluate how well our exploratory techniques preserve meaningful structure**. When we apply PCA or other DR methods, we can visually check whether wines from the same cultivar tend to cluster together. If they do, it suggests that the reduced representation is capturing real, underlying differences in the data. Without ground truth labels, we could still explore patterns, but we would not be able to assess whether those patterns correspond to meaningful or known groupings.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Visualization
    import altair as alt
    import plotly.express as px

    # Data & preprocessing
    from sklearn.datasets import load_wine
    from sklearn.datasets import make_swiss_roll
    from sklearn.preprocessing import StandardScaler

    # DR
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    import umap

    # Metrics
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    from zadu import zadu as zadu_module
    return (
        MDS,
        PCA,
        StandardScaler,
        TSNE,
        alt,
        load_wine,
        make_swiss_roll,
        np,
        pairwise_distances,
        pd,
        px,
        trustworthiness,
        umap,
        zadu_module,
    )


@app.cell
def _(load_wine, pd):
    # Load the Wine dataset
    wine = load_wine()

    # Create a pandas DataFrame with feature names
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine["cultivar"] = wine.target
    df_wine["cultivar_name"] = df_wine["cultivar"].map(
        {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
    )
    df_wine = df_wine.rename(
        columns={"od280/od315_of_diluted_wines": "protein_content"}
    )

    # Store feature names for later
    all_features = df_wine.columns[:-2]

    print(
        f"Dataset shape: {df_wine.shape[0]} samples × {df_wine.shape[1]} features"
    )
    print(f"Cultivar distribution:\n{df_wine['cultivar_name'].value_counts()}")
    df_wine.head()
    return all_features, df_wine


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Column descriptions**

    | Column | Type | Unit | Description |
    |--------|------|------|-------------|
    | `alcohol` | float | % vol | Alcohol content |
    | `malic_acid` | float | g/L | Malic acid concentration |
    | `ash` | float | g/L | Ash content |
    | `alcalinity_of_ash` | float | mEq/L | Alcalinity of ash |
    | `magnesium` | float | mg/L | Magnesium content |
    | `total_phenols` | float | — | Total phenolic compounds |
    | `flavanoids` | float | — | Flavonoid concentration |
    | `nonflavanoid_phenols` | float | — | Non-flavonoid phenols |
    | `proanthocyanins` | float | — | Proanthocyanidins |
    | `color_intensity` | float | — | Color intensity (absorbance) |
    | `hue` | float | — | Hue, color tone ratio |
    | `protein_content` | float | — | OD280/OD315 ratio (protein content proxy)|
    | `proline` | float | mg/L | Proline amino acid content |
    | `cultivar` | int | – | Integer label for the wine cultivar (grape variety class) |
    | `cultivar_name` | object | – | Human-readable name of the wine cultivar corresponding to `cultivar` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exploring the (high-dimensional) data

    With 13 features, we cannot easily visualize all relationships. Let's look at a few pairs first. Here we can exploit **Marimo's functionality** to add interactivity to the notebook and exploratory power — also outside Altair! Marimo provides **reactive UI widgets** that allow us to dynamically change parameters and immediately see results update across all dependent cells. This is particularly useful for exploring different feature combinations without rewriting code & rapid experimentation.
    """)
    return


@app.cell
def _(all_features, mo):
    # Multiselect widget
    feature_selector = mo.ui.multiselect(
        options=all_features,
        value=["alcohol", "malic_acid", "flavanoids", "color_intensity"],
        label="Select features (recommended ~4)",
    )

    feature_selector
    return (feature_selector,)


@app.cell
def _(alt, df_wine, feature_selector):
    # Here we use Altair's `repeat` to automatically create a scatter plot matrix:
    # each feature is plotted against every other feature across rows and columns
    _scatter = (
        alt.Chart(df_wine)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            # `alt.repeat("column")` and `alt.repeat("row")` tell Altair
            # to cycle through the feature list for x and y axes
            x=alt.X(alt.repeat("column"), type="quantitative"),
            y=alt.Y(alt.repeat("row"), type="quantitative"),
            tooltip=feature_selector.value,
        )
        .properties(width=100, height=100)
        .repeat(row=feature_selector.value, column=feature_selector.value)
    )

    _features_names = feature_selector.value

    _scatter.properties(
        title=f"Scatter plot matrix ({', '.join(_features_names)})"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Enhance the visualization by using **color** to represent the different **cultivars**. Include the cultivar in the tooltip as well.
    ///
    """)
    return


@app.cell
def _(alt, df_wine, feature_selector):
    # 💡 Solution
    _scatter = (
        alt.Chart(df_wine)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            x=alt.X(alt.repeat("column"), type="quantitative"),
            y=alt.Y(alt.repeat("row"), type="quantitative"),
            color=alt.Color("cultivar_name:N", title="Cultivar"),
            tooltip=["cultivar_name"] + feature_selector.value,
        )
        .properties(width=100, height=100)
        .repeat(row=feature_selector.value, column=feature_selector.value)
    )

    _features_names = feature_selector.value

    _scatter.properties(
        title=f"Scatter plot matrix ({', '.join(_features_names)})"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see some separation between cultivars, but it's hard to see the full picture with only pairwise plots.

    When dealing with many (numerical) features, a common approach is also to examine the **correlation matrix**, which helps summarize these relationships in a single view. We can effectively visualize this matrix as a **heatmap** in Altair, making patterns of association easier to spot at a glance.
    """)
    return


@app.cell
def _(df_wine, np):
    # Correlation matrix
    corr = df_wine[df_wine.columns[:-2]].corr()

    # Create a mask for showing only the upper triangle (data are symmetrical)
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )  # matrix that is True only in the upper triangle
    corr_upper = corr.where(mask)  # apply mask (keeps True values only)

    # Convert to long format for Altair
    corr_long = (
        corr_upper.reset_index()  # or corr to see the full matrix
        .melt(
            id_vars="index"
        )  # unpivot a df from wide to long format, all cols are identifier variables (id_vars)
        .dropna()
        .rename(
            columns={
                "index": "feature_1",
                "variable": "feature_2",
                "value": "correlation",
            }
        )
    )

    corr_long.head()
    return (corr_long,)


@app.cell
def _(all_features, alt, corr_long):
    _base_heatmap = (
        alt.Chart(corr_long)
        .mark_rect(stroke="black", strokeWidth=0.4)
        .encode(
            x=alt.X(
                "feature_2:N",
                title=None,
                sort=all_features,
                axis=alt.Axis(labelAngle=-45),
            ),  # sort=features ensures consistent ordering
            y=alt.Y("feature_1:N", title=None, sort=all_features),
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                legend=alt.Legend(title="Correlation"),
            ),
            tooltip=[
                alt.Tooltip("feature_1:N", title="feature 1"),
                alt.Tooltip("feature_2:N", title="feature 2"),
                alt.Tooltip("correlation:Q", format=".2f"),
            ],
        )
    )

    _text = _base_heatmap.mark_text(baseline="middle", fontSize=8).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            "datum.correlation > 0.5 || datum.correlation < -0.5",
            alt.value("white"),  # white text on darker cells
            alt.value("black"),  # black text on lighter cells
        ),
    )

    (_base_heatmap + _text).properties(
        width=400, height=400, title="Feature correlation"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Heatmaps are a powerful way to summarize many relationships at once. However, each relationship is reduced to a single number, meaning we lose important details about distribution shape, non-linear patterns, clusters, and outliers that scatterplots can reveal. Notice that we are also losing information about _how different cultivar species are distributed and separated across pair of features_. We should always **be cautious** about **drawing conclusions based on a single metric alone** (see [Same Stats, Different Graphs](https://www.youtube.com/watch?v=DbJyPELmhJc)).

    **This is why we need dimensionality reduction.** These methods summarize the information contained in many variables into a smaller set of components. Before applying them, we **standardize the features** so that differences in scale do not bias the results.
    """)
    return


@app.cell
def _(StandardScaler, all_features, df_wine):
    # Standardize the features (important for DR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_wine[all_features])

    print(f"Scaled data shape: {X_scaled.shape}")
    print(f"Mean of first feature (should be ~0): {X_scaled[:, 0].mean():.6f}")
    print(f"Std of first feature (should be ~1): {X_scaled[:, 0].std():.6f}")
    return (X_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Why standardize?
        type: warning

    DR methods are sensitive to feature scales. A feature measured in thousands will dominate one measured in decimals. **Always standardize before applying DR** (subtract mean, divide by standard deviation). All variables should contribute based on their patterns, not their units or scale.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Linear DR with PCA**

    **Principal Component Analysis (PCA)** finds orthogonal directions (principal components) that maximize **variance** in the data.

    $$\mathbf{X} \cdot \mathbf{P} = \mathbf{Y}$$

    where $\mathbf{P}$ is the projection matrix containing the principal component directions. Given our data $X$, with $n$ number of rows and $p$ features, we compute $k$ principal components, more precisely:

    | Matrix | Dimensions | Description |
    |--------|------------|-------------|
    | $\mathbf{X}$ | $(n \times p)$ | Original data matrix (n samples, p features), centered |
    | $\mathbf{P}$ | $(p \times k)$ | Loadings matrix — columns are eigenvectors of the covariance matrix |
    | $\mathbf{Y}$ | $(n \times k)$ | Projected data in the reduced space (k components) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explained variance

    How much information (variance) does each principal component capture?
    """)
    return


@app.cell
def _(PCA, X_scaled, df_wine, pd):
    # Fit PCA first with all components to see explained variance
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Create column names PC1, PC2, ... & df
    pc_columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=pc_columns)
    # Add cultivar info back
    df_pca["cultivar"] = df_wine["cultivar"].values
    df_pca["cultivar_name"] = df_wine["cultivar_name"].values

    df_pca
    return X_pca, df_pca, pca


@app.cell
def _(alt, np, pca, pd):
    # Create explained variance plot
    _exp_var = pca.explained_variance_ratio_
    _cum_var = np.cumsum(_exp_var)

    _df_var = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(_exp_var))],
            "PC_num": list(range(1, len(_exp_var) + 1)),
            "explained_var": _exp_var,
            "cumulative_var": _cum_var,
        }
    )

    _bars = (
        alt.Chart(_df_var)
        .mark_bar(color="steelblue", opacity=0.7)
        .encode(
            x=alt.X(
                "PC_num:O",
                title="Principal Component",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("explained_var:Q", title="Explained Variance Ratio"),
            tooltip=["PC", alt.Tooltip("explained_var:Q", format=".2%")],
        )
    )

    _line = (
        alt.Chart(_df_var)
        .mark_line(color="firebrick", strokeWidth=2)
        .encode(x="PC_num:O", y=alt.Y("cumulative_var:Q"))
    )

    _points = (
        alt.Chart(_df_var)
        .mark_circle(color="firebrick", size=50)
        .encode(
            x="PC_num:O",
            y="cumulative_var:Q",
            tooltip=["PC", alt.Tooltip("cumulative_var:Q", format=".2%")],
        )
    )

    (_bars + _line + _points).properties(
        width=500,
        height=300,
        title="PCA explained variance (bars) vs cumulative variance (line)",
    )
    return


@app.cell(hide_code=True)
def _(mo, pca):
    _var1 = pca.explained_variance_ratio_[0]
    _var2 = pca.explained_variance_ratio_[1]
    _var3 = pca.explained_variance_ratio_[2]
    _total2 = _var1 + _var2
    _total3 = _var1 + _var2 + _var3

    mo.md(f"""
    **Interpretation.** The first two PCs capture **{_total2:.1%}** of the total variance, and the first three capture **{_total3:.1%}**:
    - PC1: {_var1:.1%}
    - PC2: {_var2:.1%}
    - PC3: {_var3:.1%}

    This tells us that a large portion of the dataset’s variability can be represented in just two or three dimensions. In other words, the low-dimensional projection retains much of the original information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 2D visualization

    PCA gives us new numerical features (the principal components), but their meaning becomes clearer when we visualize them. We start by plotting the data in the space defined by the **first two components**, using **color to distinguish cultivars**, which helps reveal clusters, trends, and separation between groups.
    """)
    return


@app.cell
def _(alt, df_pca):
    # Plot PCA projection
    alt.Chart(df_pca).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("PC1:Q", title="PC1"),
        y=alt.Y("PC2:Q", title="PC2"),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name", "PC1", "PC2"],
    ).properties(width=500, height=400, title="PCA projection")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _But what does **projection** actually mean from a geometric point of view?_

    PCA finds the direction (PC1) along which data has **maximum variance**, i.e., points are most spread out. Then it finds PC2, the direction of maximum remaining variance that is **orthogonal** (perpendicular) to PC1. The visualization below shows **colored points** for the original data in 2D PCA space, **dashed gray lines** for the "drop" from each point to the PC1 axis, and **points on the axis**, which are the 1D projection (what you'd see if you reduced to just PC1).

    Notice how the **spread along PC1** (horizontal) is much larger than along PC2 (vertical). This is exactly what PCA optimizes for: PC1 captures the direction of maximum variance.
    """)
    return


@app.cell
def _(alt, df_pca, pd):
    # Create projection visualization data
    df_proj = pd.DataFrame(
        {
            "PC1": df_pca["PC1"],
            "PC2": df_pca["PC2"],
            "PC1_proj": df_pca["PC1"],  # projection on PC1 axis
            "PC2_proj": 0,  # y=0 for PC1 axis
            "cultivar_name": df_pca["cultivar_name"],
        }
    )

    # Original points
    points = (
        alt.Chart(df_proj)
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X("PC1:Q", title="PC1 (direction of max variance)"),
            y=alt.Y("PC2:Q", title="PC2"),
            color=alt.Color("cultivar_name:N", title="Cultivar"),
        )
    )

    # Projection lines (vertical drop to PC1 axis)
    proj_lines = (
        alt.Chart(df_proj)
        .mark_rule(strokeDash=[3, 3], opacity=0.4, color="gray")
        .encode(x="PC1:Q", y="PC2:Q", x2="PC1_proj:Q", y2="PC2_proj:Q")
    )

    # Projected points on PC1 axis
    proj_points = (
        alt.Chart(df_proj)
        .mark_circle(size=30, opacity=0.8)
        .encode(
            x="PC1_proj:Q",
            y="PC2_proj:Q",
            color=alt.Color("cultivar_name:N", legend=None),
        )
    )

    # PC1 axis line
    axis_line = (
        alt.Chart(pd.DataFrame({"x": [-4, 5], "y": [0, 0]}))
        .mark_line(color="black", strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )

    (axis_line + proj_lines + points + proj_points).properties(
        width=550,
        height=400,
        title="Projection onto PC1: each point drops to the principal axis",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 3D Visualization (with Plotly)

    If we want to visualize the first 3 PC in a 3D space 🎲 , we need to rely on another data visualization library, such as **Plotly**. Unfortunately, Altair has no 3D support – it is built on Vega-Lite which is strictly 2D.

    We won't be using Plotly in detail in this course, but it's important to know that **sometimes the chosen library is not the best for every kind of visualization** we want to explore, and it's perfectly fine to switch to others when needed.
    """)
    return


@app.cell
def _(df_pca, px):
    fig = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color="cultivar_name",
        title="PCA (3D)",
        hover_data=["cultivar_name"],
        color_discrete_sequence=[
            "#4C78A8",
            "#F58518",
            "#E45756",
        ],  # hex codes used in Altair
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.85),  # small bubble + slightly transparent
    )

    fig.update_layout(
        legend_title_text="Cultivar",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        dragmode="orbit",  # for left-right rotation
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While 3D visualizations can help explore structure, they're often harder to interpret on a 2D screen. A more effective approach is to use **pairwise 2D projections** — showing PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3 side by side. With **linked brushing** in Altair, selecting points in one view highlights them across all others, giving you the "3D intuition" without the complexity.
    """)
    return


@app.cell
def _(alt, df_pca):
    brush = alt.selection_interval(name="selection_param") # single shared selection
    # we're using the same selection parameter is being added to multiple sub-charts and automatically deduplicates it

    def scatter(x, y):
        return (
            alt.Chart(df_pca)
            .mark_circle(size=60)
            .encode(
                x=x,
                y=y,
                color=alt.condition(
                    brush,
                    "cultivar_name:N",
                    alt.value("lightgray"),
                    title="Cultivar",
                ),
                tooltip=["cultivar_name", x, y],
            )
            .add_params(brush)
            .properties(width=220, height=220)
        )


    chart = (
        scatter("PC1", "PC2") | scatter("PC1", "PC3") | scatter("PC2", "PC3")
    ).properties(title=alt.Title("First 3 PCs", anchor="middle"))

    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    /// note | 🔍 Show selected data
    In Marimo it is very easy to add interactive data exploration!

    Try yourself by wrapping the chart before displaying it:
    ```python
    chart = mo.ui.altair_chart(chart)
    chart
    ```

    Then in a **new cell**, simply access the selected points:
    ```python
    chart.value
    ```

    Brush some points in the scatter plots and watch the table update automatically!
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part III: Linear DR with MDS**

    **Multidimensional Scaling (MDS)** takes a different approach from PCA: instead of maximizing variance, it tries to preserve the **pairwise distances** between points. Imagine you have a table of distances between cities (Rome → Milan: 570km, Rome → Naples: 225km, etc.). MDS is like reconstructing a map from *only* this distance table, without knowing the actual coordinates.

    **How MDS works**

    1. **Start** with all pairwise distances in the original high-dimensional space
    2. **Initialize** points randomly in 2D (or your target dimension)
    3. **Compare** the 2D distances to the original HD distances
    4. **Adjust** point positions to make LD distances match HD distances better
    5. **Repeat** until the layout stabilizes

    MDS minimizes a quantity called **stress**, which is the total mismatch between original and embedded distances.

    - **Low stress**: distances are well preserved → good embedding
    - **High stress**: many distances are distorted → poor embedding

    **PCA vs MDS**

    | Aspect | PCA | MDS |
    |--------|-----|-----|
    | **Preserves** | Variance (spread of data) | Pairwise distances |
    | **Input** | Raw data matrix | Distance matrix (or raw data) |
    | **Approach** | Find directions of max variance | Find layout matching distances |

    When using Euclidean distances, PCA and MDS often produce nearly identical results.
    """)
    return


@app.cell
def _(MDS, X_scaled):
    # Fit MDS (first 2 components)
    mds = MDS(
        n_components=2,
        random_state=42,
        normalized_stress="auto",
        init="random",
        n_init=10,  # tries 10 initializations, keeps best (better than n_init=1)
    )
    X_mds = mds.fit_transform(X_scaled)

    print(f"MDS stress: {mds.stress_:.4f}")
    return (X_mds,)


@app.cell
def _(X_mds, df_wine, pd):
    # Create df with MDS results
    df_mds = pd.DataFrame(X_mds, columns=["MDS1", "MDS2"])
    df_mds["cultivar_name"] = df_wine["cultivar_name"].values
    df_mds.head()
    return (df_mds,)


@app.cell
def _(alt, df_mds):
    # Plot MDS projection
    alt.Chart(df_mds).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("MDS1:Q", title="MDS Dimension 1"),
        y=alt.Y("MDS2:Q", title="MDS Dimension 2"),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name", "MDS1", "MDS2"],
    ).properties(width=500, height=400, title="MDS projection")
    return


@app.cell
def _(MDS, X_scaled, alt, pd):
    stresses = []

    for n in range(1, 14):
        _mds = MDS(n_components=n, random_state=42, normalized_stress="auto", n_init=4, init='random')
        _mds.fit(X_scaled)
        stresses.append(_mds.stress_)

    # Plot
    df_stress = pd.DataFrame(
        {"n_components": list(range(1, 14)), "stress": stresses}
    )

    alt.Chart(df_stress).mark_bar().encode(
        x=alt.X("n_components:O", title="Number of components"),
        y=alt.Y("stress:Q", title="Stress"),
    ).properties(width=400, title="MDS stress vs components")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** This plot serves the same purpose as **explained variance** in PCA: it helps us evaluate and choose the right number of components. The key difference is directional: in PCA we look for *high* explained variance, while in MDS we look for *low* stress. The chart shows a clear **elbow at 2–3 components**: stress drops dramatically from ~90,000 (1D) to ~20,000 (2D) to ~9,000 (3D), then decreases more gradually. This suggests that 2–3 dimensions capture most of the distance structure and that additional dimensions offer diminishing returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Shepard diagram

    A **Shepard diagram** is the standard diagnostic tool for evaluating MDS quality. It compares original HD distances (x-axis) against embedded LD distances (y-axis) for all pairs of points. **Points on the diagonal (y=x) indicate perfect preservation**. Scatter away from the diagonal indicates distortion – the further from the line, the more the embedding has compressed or stretched that particular distance.
    """)
    return


@app.cell
def _(X_mds, X_scaled, alt, np, pairwise_distances, pd):
    # Compute pairwise distances
    dist_hd = pairwise_distances(X_scaled).flatten() # by default, metric="euclidean" 
    dist_ld = pairwise_distances(X_mds).flatten()

    # Sample for visualization (too many pairs otherwise)
    n_sample = 5000
    idx = np.random.choice(len(dist_hd), size=n_sample, replace=False)

    df_shepard = pd.DataFrame(
        {"HD_distance": dist_hd[idx], "LD_distance": dist_ld[idx]}
    )

    _scatter = (
        alt.Chart(df_shepard)
        .mark_circle(size=10, opacity=0.3)
        .encode(
            x=alt.X("HD_distance:Q", title="HD distance"),
            y=alt.Y("LD_distance:Q", title="LD distance"),
        )
    )

    # Add diagonal reference line
    _max_val = max(
        df_shepard["HD_distance"].max(), df_shepard["LD_distance"].max()
    )
    _line_data = pd.DataFrame({"x": [0, _max_val], "y": [0, _max_val]})
    _line = (
        alt.Chart(_line_data)
        .mark_line(color="firebrick", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    (_scatter + _line).properties(
        width=350, height=350, title="Shepard Diagram: HD vs LD Distances (MDS)"
    )
    return dist_hd, idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** Points above the diagonal indicate **stretching** (LD distance > HD distance), while points below indicate **compression** (LD distance < HD distance).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    **How many dimensions?** Does adding more dimensions improve distance preservation? Use a slider from Marimo widgets to explore how the Shepard diagram changes with different `n_components`, and observe the **trade-off between compression and fidelity**.

    **Requirements:**
    - Create a slider to select `n_components` from 1 to 13
    - Fit MDS with the selected number of components
    - Display a Shepard diagram that updates **reactively**
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Create a slider for n_components (1 to 13)
    # n_comp_slider = mo.ui.slider(...)
    # n_comp_slider

    # Step 2: In a new cell, fit MDS with the selected number of components
    # _mds = MDS(n_components=..., random_state=42, normalized_stress="auto")
    # _X_mds = _mds.fit_transform(X_scaled)

    # Step 3: Compute embedded distances and sample
    # _dist_ld = pairwise_distances(_X_mds).flatten()[idx]
    # _df = pd.DataFrame({"HD": dist_hd[idx], "LD": _dist_ld})

    # Step 4: Create scatter plot with diagonal reference line
    # alt.Chart(_df).mark_circle(...).encode(
    #     x=alt.X("HD:Q", ...),
    #     y=alt.Y("LD:Q", ...),
    # )
    # ...
    # (_scatter + _line).properties(
    #     width=350, height=350, title="Shepard Diagram: HD vs LD distances (MDS)"
    # )
    return


@app.cell
def _(mo):
    # 💡 Solution (1/1)

    # Step 1: Create a slider for n_components (1 to 13)
    n_comp_slider = mo.ui.slider(1, 13, value=2, label="MDS components")
    n_comp_slider
    return (n_comp_slider,)


@app.cell
def _(MDS, X_scaled, alt, dist_hd, idx, n_comp_slider, pairwise_distances, pd):
    # 💡 Solution (2/2)

    # Step 2: In a new cell, fit MDS with the selected number of components
    _mds = MDS(n_components=n_comp_slider.value, random_state=42, normalized_stress="auto")
    _X_mds = _mds.fit_transform(X_scaled)

    # Step 3: Compute embedded distances and sample
    _dist_ld = pairwise_distances(_X_mds).flatten()[idx]
    _df = pd.DataFrame({"HD": dist_hd[idx], "LD": _dist_ld})

    # Step 4: Create scatter plot with diagonal reference line
    _scatter = alt.Chart(_df).mark_circle(size=10, opacity=0.3).encode(
        y=alt.Y("LD:Q", title="LD distance"),
        x=alt.X("HD:Q", title="HD distance"),
    )
    _line = alt.Chart(
        pd.DataFrame({"x": [0, _df["HD"].max()], "y": [0, _df["HD"].max()]})
    ).mark_line(color="firebrick", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")

    (_scatter + _line).properties(
        width=350, height=350, title="Shepard Diagram: HD vs LD distances (MDS)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part IV: Nonlinear DR with t-SNE**

    Linear methods like PCA and MDS preserve global structure but may miss complex nonlinear patterns. Both **t-SNE** and **UMAP** are nonlinear methods designed to preserve **local neighborhood structure**. Non-linear algorithms adapt to the underlying data, performing different transformations on different regions.

    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** was introduced by [van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) and quickly became one of the most popular techniques for visualizing high-dimensional data, especially in fields like genomics and deep learning. t-SNE asks: *"If I pick a point at random, what's the probability that I would choose each other point as its neighbor?"* It computes these probabilities in both high-dimensional (HD) and low-dimensional (LD) space, then adjusts the LD embedding until the two probability distributions match as closely as possible (in terms of Kullback–Leibler divergence).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Understanding the `perplexity` parameter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`Perplexity`** is a key parameter. It controls how many neighbors each point "pays attention to", determining the focus of attention on local or global structure:

    | Perplexity | Effect |
    |------------|--------|
    | **Low (5-10)** | Focus on very local structure; may fragment clusters |
    | **Medium (30)** | Good default; balances local and some global structure |
    | **High (50+)** | More global view; clusters may merge |

    **Rule of thumb**: perplexity should be much smaller than the number of points (N). If perplexity is too close to N, neighborhoods stop being “local” – the local notion of “neighborhood” collapses into something close to global structure. Typical range is **5–50**.
    """)
    return


@app.cell
def _(TSNE, X_scaled):
    # Fit t-SNE with different perplexity values
    tsne_perp5 = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne_p5 = tsne_perp5.fit_transform(X_scaled)

    tsne_perp25 = TSNE(n_components=2, perplexity=25, random_state=42)
    X_tsne_p25 = tsne_perp25.fit_transform(X_scaled)

    tsne_perp50 = TSNE(n_components=2, perplexity=50, random_state=42)
    X_tsne_p50 = tsne_perp50.fit_transform(X_scaled)

    print("t-SNE computed for perplexity = 5, 25, 50")
    return X_tsne_p25, X_tsne_p5, X_tsne_p50


@app.cell
def _(X_tsne_p25, X_tsne_p5, X_tsne_p50, alt, df_wine, pd):
    # Create comparison df
    _df_p5 = pd.DataFrame(X_tsne_p5, columns=["x", "y"])
    _df_p5["cultivar_name"] = df_wine["cultivar_name"].values
    _df_p5["perplexity"] = "perplexity=5"

    _df_p25 = pd.DataFrame(X_tsne_p25, columns=["x", "y"])
    _df_p25["cultivar_name"] = df_wine["cultivar_name"].values
    _df_p25["perplexity"] = "perplexity=25"

    _df_p50 = pd.DataFrame(X_tsne_p50, columns=["x", "y"])
    _df_p50["cultivar_name"] = df_wine["cultivar_name"].values
    _df_p50["perplexity"] = "perplexity=50"

    df_tsne_compare = pd.concat([_df_p5, _df_p25, _df_p50])

    # visualization
    alt.Chart(df_tsne_compare).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X("x:Q", title="t-SNE 1"),
        y=alt.Y("y:Q", title="t-SNE 2"),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name"],
    ).properties(width=250, height=250).facet(
        column=alt.Column(
            "perplexity:N",
            title="",
            sort=["perplexity=5", "perplexity=25", "perplexity=50"],
        )
    ).properties(title="t-SNE with different perplexity values"
    ).resolve_scale(
        x="independent",
        y="independent"
    ) # to rescale each panel to its own x and y axes -makes local structure easier to see
    # (we lose direct visual comparability of distances across panels, but gain readability inside each one)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Comparing PCA and t-SNE: when linear methods fail

    The [**Swiss Roll**](https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html) is a classic benchmark dataset for dimensionality reduction. It's a 2D surface (a flat sheet) that has been "rolled up" into 3D space — like rolling a carpet. In other words, it is **intrinsically 2D but embedded in 3D.**

    **The key insight:** Euclidean distance in 3D does NOT reflect true closeness on the surface, points that are **close in Euclidean distance** (straight line through the roll) may be **far apart along the manifold** (following the surface). Think of two points on opposite sides of a rolled carpet — a straight line connects them through the roll, but walking along the carpet surface would take much longer.

    **What happens with each method?**

    - **PCA** projects data onto the directions of maximum variance. Since it only considers straight-line distances, it essentially "squashes" the roll from above — points from different layers of the roll end up mixed together. **It preserves global variance, not manifold structure.** But the swiss role is not a linear subspace, it is a curved 2D surface embedded in 3D.

    - **t-SNE** preserves local neighborhoods. It respects the structure of the manifold, effectively "unrolling" it back to 2D. Points that are close along the surface stay close in the embedding. Notice that t-SNE does NOT try to recover the manifold globally, it only tries to **preserve who is close to whom locally**.

    This illustrates a fundamental principle: **linear methods (PCA, MDS) fail when the data lies on a curved manifold**. Nonlinear methods (t-SNE, UMAP) can recover the intrinsic structure. To learn more, see [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html#manifold).
    """)
    return


@app.cell
def _(make_swiss_roll):
    # Generate Swiss Roll dataset (3D data that is intrinsically 2D) from sklearn
    X_swiss, t_swiss = make_swiss_roll(n_samples=1500, noise=0.5, random_state=42)
    # t_swiss is the position along the roll — useful for coloring

    #X_swiss.shape, t_swiss.shape
    return X_swiss, t_swiss


@app.cell
def _(X_swiss, px, t_swiss):
    # First, let’s take a look at our data:
    fig_swiss = px.scatter_3d(
        x=X_swiss[:, 0],
        y=X_swiss[:, 1],
        z=X_swiss[:, 2],
        color=t_swiss,
        color_continuous_scale="Viridis",
        title="Swiss Roll (3D)"
    )

    fig_swiss.update_traces(marker=dict(size=5, opacity=0.85))
    fig_swiss.update_layout(
        scene=dict(
            aspectmode="data", # makes the roll fill the space properly
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig_swiss.show()
    return


@app.cell
def _(PCA, X_swiss, pca):
    # Apply PCA
    pca_swiss = PCA(n_components=2)
    X_pca_swiss = pca.fit_transform(X_swiss)
    return (X_pca_swiss,)


@app.cell
def _(mo):
    # Apply t-SNE
    perplexity_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=1,
        value=25,
        label="t-SNE perplexity",
    )
    perplexity_slider
    return (perplexity_slider,)


@app.cell
def _(TSNE, X_pca_swiss, X_swiss, pd, perplexity_slider, t_swiss):
    # Apply t-SNE
    tsne_swiss = TSNE(n_components=2, perplexity=perplexity_slider.value, random_state=42)
    X_tsne_swiss = tsne_swiss.fit_transform(X_swiss)

    # Create DataFrames for plotting
    df_pca_swiss = pd.DataFrame({'x': X_pca_swiss[:, 0], 'y': X_pca_swiss[:, 1], 'position': t_swiss})
    df_tsne_swiss = pd.DataFrame({'x': X_tsne_swiss[:, 0], 'y': X_tsne_swiss[:, 1], 'position': t_swiss})
    return df_pca_swiss, df_tsne_swiss


@app.cell
def _(alt, df_pca_swiss, df_tsne_swiss):
    # PCA result
    chart_pca = alt.Chart(df_pca_swiss).mark_circle(size=30).encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('position:Q', scale=alt.Scale(scheme='viridis'), title='Position along roll'),
        tooltip=['position']
    ).properties(title='PCA (linear)', width=300, height=300)

    # t-SNE result
    chart_tsne = alt.Chart(df_tsne_swiss).mark_circle(size=30).encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('position:Q', scale=alt.Scale(scheme='viridis'), title='Position along roll'),
        tooltip=['position']
    ).properties(title='t-SNE (nonlinear)', width=300, height=300)

    chart_pca | chart_tsne
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Try exploring how the embedding changes as you move the **perplexity slider**.
    What differences do you notice in the structure, spacing, or clustering of the points?
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While t-SNE is able to preserve the general structure of the data, it poorly represents their _continuous_ nature. Instead, it seems to unnecessarily clump sections of points together. When t-SNE appears to “unroll” a manifold like the Swiss Roll, that is a **side effect of preserving local similarities** — not an accurate recovery of the true global structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part V: Nonlinear DR with UMAP**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **UMAP (Uniform Manifold Approximation and Projection)** is a nonlinear dimensionality reduction method, similar in spirit to t-SNE, but typically **faster** and often better at preserving some **global structure**. Like t-SNE, UMAP is based on **neighborhood relationships**: it builds a **graph** of nearby points in high dimensions and finds a low-dimensional layout that keeps those neighbors close.

    **Key parameters**:
    - `n_components`: dimensionality of reduction
    - `n_neighbors`: similar to `perplexity`, tradeoff between local and global structure (lower values only take into consideration the local structure)
    - `min_dist`: minimum distance between points in the embedding (controls clustering tightness); can prevent or encourage forming of clusters
    - `metric`: what distance measure to use

    Both t-SNE and UMAP share some hyperparameters (`perplexity` in t-SNE, `n_neighbors` in UMAP) to control the scale of locality, which is no suprise as they both are based on neighborhood structure. They rely on **local relationships** rather than global linear projections. Neither method should be interpreted as preserving exact distances or geometry. But while t-SNE mainly preserves local similarity, **UMAP often provides a better sense of how clusters relate to each other globally.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Understanding key hyperparameters

    To understand how UMAP shapes a visualization, we’ll explore how its main hyperparameters, `min_dist` and `n_neighbors`, affect the embedding. UMAP reveals structure at different scales depending on how we define neighborhoods and spacing.

    There is **no single “correct” setting**: we should always try multiple parameter combinations and choose values that best support the question we want to answer and the patterns we care about highlighting.
    """)
    return


@app.cell
def _(X_scaled, df_wine, pd, umap):
    n_neighbors_values = [5, 25, 50]
    min_dist_values = [0.01, 0.1, 0.5]

    # Compute UMAP for all combinations
    dfs = []
    for _n in n_neighbors_values:
        for _d in min_dist_values:
            umap_model = umap.UMAP(n_neighbors=_n, min_dist=_d, random_state=42, n_jobs=1)
            X_proj = umap_model.fit_transform(X_scaled)
            _df = pd.DataFrame(X_proj, columns=["x", "y"])
            _df["cultivar_name"] = df_wine["cultivar_name"].values
            _df["n_neighbors"] = _n
            _df["min_dist"] = _d
            dfs.append(_df)

    df_umap_grid = pd.concat(dfs)
    return (df_umap_grid,)


@app.cell
def _(alt, df_umap_grid):
    alt.Chart(df_umap_grid).mark_circle(size=40, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name", "n_neighbors", "min_dist"],
    ).properties(
        width=180, 
        height=180
    ).facet(
        column=alt.Column("n_neighbors:N", title="n_neighbors", sort=["n=5", "n=25", "n=50"]),
        row=alt.Row("min_dist:N", title="min_dist", sort=["d=0.01", "d=0.1", "d=0.5"])
    ).resolve_scale(
        x="independent",
        y="independent"
    ).properties(
        title="UMAP: effect of n_neighbors (local vs global) and min_dist (tight vs spread)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

    Add **UMAP** to the Swiss Roll comparison. Does it successfully unroll the manifold?
    Use **interactive controls** to explore how UMAP’s parameters influence the embedding. Adjust the sliders and observe how the balance between local detail and global structure changes the result.

    **Requirements:**
    - Add a slider for `n_neighbors` ranging **from 1 to 50** with a **step of 5**
    - Add a slider for `min_dist` ranging **from 0.1 to 1.0** with a **step of 0.1**

    As you experiment, pay attention to whether the manifold appears continuous, fragmented into clusters, or overly compressed.
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Create a slider for n_neighbors
    # n_neighbors_slider = mo.ui.slider(
    #     start=5,
    #     ...
    #     label="UMAP n_neighbors",
    # )
    # n_neighbors_slider

    # Step 2: Create a slider for min_dist
    # min_dist_slider = mo.ui.slider(
    #     start=5,
    #     ...
    #     label="UMAP min_dist",
    # )
    # min_dist_slider

    # Step 3: Visualize the sliders
    # controls = mo.hstack([...], justify="start", gap=1)
    # controls

    # (in a new cell)

    # Step 2: Apply UMAP to X_swiss
    # umap_swiss = umap.UMAP(...)
    # X_umap_swiss = ...

    # Step 3: Create df for plotting
    # df_umap_swiss = pd.DataFrame({'x': ..., 'y': ..., 'position': t_swiss})

    # Step 4: Create the chart
    # chart_umap = alt.Chart(df_umap)...
    return


@app.cell
def _(mo):
    # 💡 Solution

    # Step 1: Create a slider for n_neighbors
    n_neighbors_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=15,
        label="UMAP n_neighbors",
    )

    # Step 2: Create a slider for min_dist
    min_dist_slider = mo.ui.slider(
        start=0.1,
        stop=1,
        step=0.1,
        value=0.1,
        label="UMAP min_dist",
    )

    # Step 3: Visualize the sliders
    controls = mo.hstack([n_neighbors_slider, min_dist_slider], justify="start", gap=1)
    controls
    return min_dist_slider, n_neighbors_slider


@app.cell
def _(X_swiss, alt, min_dist_slider, n_neighbors_slider, pd, t_swiss, umap):
    # 💡 Solution

    # Step 2: Apply UMAP to X_swiss
    umap_swiss = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors_slider.value, 
        min_dist=min_dist_slider.value, 
        random_state=42, 
        n_jobs=1
    )
    X_umap_swiss = umap_swiss.fit_transform(X_swiss)

    # Step 3: Create df for plotting
    df_umap_swiss = pd.DataFrame({'x': X_umap_swiss[:, 0], 'y': X_umap_swiss[:, 1], 'position': t_swiss})

    # Step 4: Create the chart
    alt.Chart(df_umap_swiss).mark_circle(size=30).encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('position:Q', scale=alt.Scale(scheme='viridis'), title='Position along roll'),
        tooltip=['position']
    ).properties(title=f'UMAP (n_neighbors={n_neighbors_slider.value})', width=350, height=350)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part VI: Quality assessment**

    > *"All models are wrong, but some are useful."* — George Box

    DR methods **always introduce distortions** — they cannot perfectly preserve all HD relationships in any LD space. The key question is: **which distortions occurred, and where?**

    **Types of distortions**

    | Distortion Type | Description | Effect |
    |----------------|-------------|--------|
    | **Stretching** | Distances increased in LD | Points appear farther apart than they really are |
    | **Compression** | Distances decreased in LD | Points appear closer than they really are |
    | **Missing Neighbors** | HD neighbors are not LD neighbors | True neighbors got separated |
    | **False Neighbors** | LD neighbors are not HD neighbors | Unrelated points appear close |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Comparing all methods

    Let's put all our projections side by side.
    """)
    return


@app.cell
def _(df_umap_grid):
    # Let's save one of umap df_grid results for comparison (the more balanced one)
    X_umap_n25 = df_umap_grid[(df_umap_grid.n_neighbors==25) & (df_umap_grid.min_dist==0.1)][['x', 'y']]
    return (X_umap_n25,)


@app.cell
def _(X_mds, X_pca, X_tsne_p25, X_umap_n25, alt, df_wine, pd):
    # Create unified comparison DataFrame
    _methods = [
        ("PCA", X_pca[:,:2]),
        ("MDS", X_mds),
        ("t-SNE (p=25)", X_tsne_p25),
        ("UMAP (n=25)", X_umap_n25),
    ]
    _dfs = []
    for _name, _coords in _methods:
        _df = pd.DataFrame(_coords, columns=["x", "y"])
        _df["cultivar_name"] = df_wine["cultivar_name"].values
        _df["method"] = _name
        _dfs.append(_df)
    df_all_methods = pd.concat(_dfs).reset_index(names="point_id")

    # Linked brushing
    # _brush = alt.selection_interval()

    # Use point_id to link selections across panels
    # standard approach for faceted DR plots with independent scales
    _linked = alt.selection_point(fields=["point_id"], on="click", toggle="true")

    alt.Chart(df_all_methods).mark_circle(size=60).encode(
        x=alt.X("x:Q", title="Dimension 1"),
        y=alt.Y("y:Q", title="Dimension 2"),
        color=alt.condition(_linked, "cultivar_name:N", alt.value("lightgray"), title='Cultivar'),
        opacity=alt.condition(_linked, alt.value(0.8), alt.value(0.2)),
        tooltip=["cultivar_name", "method", "point_id"],
    ).add_params(
        _linked
    ).properties(width=220, height=220).facet(
        column=alt.Column(
            "method:N",
            title="",
            sort=["PCA", "MDS", "t-SNE (p=25)", "UMAP (n=25)"],
        ),
        columns=2
    ).properties(
        title="Comparison of DR methods — click points to highlight across all views"
    ).resolve_scale(
        x="independent",
        y="independent"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Trustworthiness & Continuity

    These are the most widely used quality metrics for DR:

    - **Trustworthiness**: do the neighbors we see in the LD embedding correspond to true HD neighbors? (measures **false neighbors**)

    - **Continuity**: are the true HD neighbors still neighbors in the LD embedding? (measures **missing neighbors**)

    Both range from 0 to 1, where **1 is perfect**.
    """)
    return


@app.cell
def _(trustworthiness):
    def compute_continuity(X_hd, X_ld, n_neighbors=5):
        """
        Compute continuity score.
        Continuity measures whether HD neighbors remain neighbors in LD.
        (This is the 'inverse' of trustworthiness)
        """
        # Trustworthiness with swapped arguments gives continuity
        return trustworthiness(X_ld, X_hd, n_neighbors=n_neighbors)
    return (compute_continuity,)


@app.cell
def _(mo):
    # Quality metrics depend on the neighborhood size k. Let's see how:
    k_slider = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Number of neighbors (k)")
    k_slider
    return (k_slider,)


@app.cell
def _(
    X_mds,
    X_pca,
    X_scaled,
    X_tsne_p25,
    X_umap_n25,
    compute_continuity,
    k_slider,
    pd,
    trustworthiness,
):
    # Compute metrics for all methods
    _methods = {
        "PCA": X_pca[:,:2],
        "MDS": X_mds,
        "t-SNE": X_tsne_p25,
        "UMAP": X_umap_n25,
    }

    _results = []
    for _name, _embedding in _methods.items():
        _trust = trustworthiness(X_scaled, _embedding, n_neighbors=k_slider.value)
        _cont = compute_continuity(X_scaled, _embedding, n_neighbors=k_slider.value)
        _results.append(
            {"Method": _name, "Trustworthiness": _trust, "Continuity": _cont}
        )

    df_quality = pd.DataFrame(_results)
    #df_quality
    return (df_quality,)


@app.cell
def _(alt, df_quality, k_slider):
    # Visualize quality metrics
    _df_long = df_quality.melt(
        id_vars=["Method"],
        value_vars=["Trustworthiness", "Continuity"],
        var_name="Metric",
        value_name="Score",
    )

    # Points
    _points = alt.Chart(_df_long).mark_point(size=100, filled=True).encode(
        x=alt.X("Method:N", title="", sort=["PCA", "MDS", "t-SNE", "UMAP"]),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.8, 1])),
        color=alt.Color("Metric:N"),
        xOffset="Metric:N",
        tooltip=["Method", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    )

    # Vertical lines from baseline to points
    _rules = alt.Chart(_df_long).mark_rule(strokeWidth=2).encode(
        x=alt.X("Method:N", title="", sort=["PCA", "MDS", "t-SNE", "UMAP"], axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.8, 1])),
        y2=alt.datum(0.8),  # bottom of chart (height in pixels)
        color=alt.Color("Metric:N"),
        xOffset="Metric:N",
    )

    (_rules + _points).properties(
        width=400, height=300, title=f"Quality metrics comparison (k={k_slider.value} neighbors)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Alternative visualization – all methods on one static chart:**
    """)
    return


@app.cell
def _(
    X_mds,
    X_pca,
    X_scaled,
    X_tsne_p25,
    X_umap_n25,
    alt,
    compute_continuity,
    pd,
    trustworthiness,
):
    # Compute metrics for different k values across all methods
    _k_values = range(5, 51, 5)
    _methods = {
        "PCA": X_pca[:, :2],
        "MDS": X_mds,
        "t-SNE": X_tsne_p25,
        "UMAP": X_umap_n25,
    }

    _results_k = []
    for _method_name, _embedding in _methods.items():
        for _k in _k_values:
            _trust = trustworthiness(X_scaled, _embedding, n_neighbors=_k)
            _cont = compute_continuity(X_scaled, _embedding, n_neighbors=_k)
            _results_k.append({
                "Method": _method_name,
                "k": _k,
                "Trustworthiness": _trust,
                "Continuity": _cont
            })

    df_quality_k = pd.DataFrame(_results_k)

    _df_long = df_quality_k.melt(
        id_vars=["Method", "k"],
        value_vars=["Trustworthiness", "Continuity"],
        var_name="Metric",
        value_name="Score",
    )

    alt.Chart(_df_long).mark_line(point=True).encode(
        x=alt.X("k:Q", title="Neighborhood size (k)"),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.85, 1])),
        color=alt.Color("Metric:N"),
        tooltip=["Method", "k", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(
        width=200,
        height=180,
    ).facet(
        facet=alt.Facet("Method:N", title=None, sort=["PCA", "MDS", "t-SNE", "UMAP"]),
        columns=2
    ).properties(
        title="Quality metrics vs Neighborhood size"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** Quality metrics tend to decrease with larger k, preserving larger neighborhoods in 2D becomes increasingly difficult. However, the rate of decline differs by method:

    - **Linear methods (PCA, MDS)** — metrics *increase* with k. They start lower at small k but improve as k grows. Why? These methods preserve global structure, so larger neighborhoods align better with what they optimize.

    - **Nonlinear methods (t-SNE, UMAP)** — metrics *decrease* with k. They start high at small k but decline as k grows. Why? These methods are optimized for local neighborhood preservation — they excel at small k but struggle with larger neighborhoods.
    This reveals a fundamental trade-off: methods that excel at local preservation often sacrifice global structure, and vice versa.

    At **k ≈ 30–50**, all methods converge to similar scores. This is where local and global perspectives meet.

    **There's no universally "best" method**: choose based on whether local or global structure matters for your analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    The quality metrics _also_ depend on the **parameters you chose** for each method:

    - t-SNE with `perplexity=25` optimizes for ~25 neighbors
    - UMAP with `n_neighbors=25` optimizes for exactly 25 neighbors

    This explains why metrics decline when k exceeds these values — you're asking "_did you preserve 50 neighbors?_" when the algorithm only optimized for 25.

    **Experiment:** recompute **t-SNE** with `perplexity=50` and **UMAP** with `n_neighbors=50`. Do the quality metrics stay higher at larger k values?

    **Hints:**
    - Evaluate the metrics for multiple neighborhood sizes, e.g. `k_values = range(5, 51, 5)`
    - Store results in a tidy df with columns like `method`, `k`, `metric`, `score`
    - Create **two charts side by side** (one per metric), each showing four lines (both UMAP and t-SNE with both `25` and `50`)
    - For each vis, use a **line chart** with `k` on the x-axis and `score` on the y-axis, colored by method & parameter value
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Compute new embeddings with larger neighborhood parameters
    # X_tsne_p50 = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_scaled)
    # X_umap_n50 = UMAP(n_components=2, n_neighbors=50, random_state=42, n_init=1).fit_transform(X_scaled)

    # Step 2: Compute quality metrics for different k values
    # _k_values = range(5, 51, 5)
    # ...

    # Step 3: Store results in a df
    # _results = []
    # ...

    # Step 4: Compare with the original embeddings (perplexity=25, n_neighbors=25)
    # Do the new embeddings maintain higher scores at k=50?
    return


@app.cell
def _(
    X_scaled,
    X_tsne_p25,
    X_tsne_p50,
    X_umap_n25,
    alt,
    compute_continuity,
    pd,
    trustworthiness,
    umap,
):
    # 💡 Solution

    # Original embeddings (perplexity=25, n_neighbors=25)
    # X_tsne_p25 = TSNE(n_components=2, perplexity=25, random_state=42).fit_transform(X_scaled)
    # X_umap_n25 = UMAP(n_components=2, n_neighbors=25, random_state=42).fit_transform(X_scaled)

    # Step 1: Compute new embeddings with larger neighborhood parameters
    # New embeddings (perplexity=50, n_neighbors=50)
    # X_tsne_p50 = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_scaled)
    X_umap_n50 = umap.UMAP(n_components=2, n_neighbors=50, random_state=42, n_jobs=1).fit_transform(X_scaled)

    # Step 2: Compute quality metrics for different k values
    _k_values = range(5, 51, 5)
    _embeddings = {
        "t-SNE (perp=25)": X_tsne_p25,
        "t-SNE (perp=50)": X_tsne_p50,
        "UMAP (n=25)": X_umap_n25,
        "UMAP (n=50)": X_umap_n50,
    }

    # Step 3: Store results in a df
    _results = []
    for _name, _emb in _embeddings.items():
        for _k in _k_values:
            _trust = trustworthiness(X_scaled, _emb, n_neighbors=_k)
            _cont = compute_continuity(X_scaled, _emb, n_neighbors=_k)
            _results.append({
                "Embedding": _name,
                "k": _k,
                "Trustworthiness": _trust,
                "Continuity": _cont
            })

    _df = pd.DataFrame(_results)
    _df_long = _df.melt(
        id_vars=["Embedding", "k"],
        value_vars=["Trustworthiness", "Continuity"],
        var_name="Metric",
        value_name="Score"
    )

    # Step 4: Compare with the original embeddings (perplexity=25, n_neighbors=25)
    alt.Chart(_df_long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("k:Q", title="Neighborhood size (k)"),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.90, 1])),
        color=alt.Color("Embedding:N"),
        strokeDash=alt.StrokeDash("Embedding:N"),
        tooltip=["Embedding", "k", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(
        width=300,
        height=220,
    ).facet(
        column=alt.Column("Metric:N", title=None)
    ).properties(
        title="Effect of algorithm parameters on quality metrics"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Comprehensive quality comparison with `ZADU`

    _Trustworthiness_ and _Continuity_ are the most common metrics, but they only capture **one perspective** of projection quality. To get a fuller picture, we can use the [**ZADU**](https://github.com/hj-n/zadu) library (Jeon et al., IEEE VIS 2023), which provides a unified interface for computing many distortion measures at once — with automatic optimization of shared preprocessing steps (e.g., distance matrices, neighbor rankings).

    We'll compare our four projections across **six complementary metrics**:

    | Metric | Measures | Range | Optimum |
    |--------|----------|-------|---------|
    | **Trustworthiness** | False neighbors (points that appear close in LD but aren't in HD) | [0.5, 1] | 1 |
    | **Continuity** | Missing neighbors (points that are close in HD but not in LD) | [0.5, 1] | 1 |
    | **MRRE – False Neighbors** | Mean Relative Rank Error for false neighbors | [0, 1] | 1 |
    | **MRRE – Missing Neighbors** | Mean Relative Rank Error for missing neighbors | [0, 1] | 1 |
    | **LCMC** | Local Continuity Meta-Criterion (combines T&C into a single local quality score) | [0, 1] | 1 |
    | **Steadiness & Cohesiveness** | Inter-cluster reliability: whether groups stay together (cohesiveness) and don't merge (steadiness) | [0, 1] | 1 |

    💡 **Tip:** When computing **multiple metrics at the same time**, the `ZADU` class works better than standalone functions, as it automatically reuses shared computations like pairwise distances and neighbor rankings.

    **Points for reflection** (after examining the results):
    1. Do all methods perform similarly, or are there some that are clearly superior / inferior?
    2. Is it easy to pick the single best one, given all the multiple criteria that are available?
    3. Are the quality criteria redundant, or do they each offer unique insights into the problem?
    """)
    return


@app.cell
def _(X_mds, X_pca, X_scaled, X_tsne_p25, X_umap_n25, np, pd, zadu_module):
    # Define the metrics to compute via ZADU's optimized class interface
    _spec = [
        {"id": "tnc",  "params": {"k": 20}},
        {"id": "mrre", "params": {"k": 20}},
        {"id": "lcmc", "params": {"k": 20}},
        {"id": "snc",  "params": {"k": 50, "clustering_strategy": "dbscan"}},
    ]

    # All four projections
    _methods = {
        "PCA":   np.asarray(X_pca[:, :2], dtype=np.float64),
        "MDS":   np.asarray(X_mds, dtype=np.float64),
        "t-SNE": np.asarray(X_tsne_p25, dtype=np.float64),
        "UMAP":  np.asarray(X_umap_n25, dtype=np.float64),
    }

    # Compute all metrics for each projection
    _rows = []
    for _name, _ld in _methods.items():
        _zadu_obj = zadu_module.ZADU(_spec, np.asarray(X_scaled, dtype=np.float64))
        _scores = _zadu_obj.measure(_ld)
        # _scores is a list of dicts, one per spec entry
        # tnc returns {"trustworthiness": ..., "continuity": ...}
        # mrre returns {"mrre_false": ..., "mrre_missing": ...}
        # lcmc returns {"lcmc": ...}
        # snc returns {"steadiness": ..., "cohesiveness": ...}
        _rows.append({
            "Method": _name,
            "Trustworthiness":       _scores[0]["trustworthiness"],
            "Continuity":            _scores[0]["continuity"],
            "MRRE – False Neighbors":  _scores[1]["mrre_false"],
            "MRRE – Missing Neighbors": _scores[1]["mrre_missing"],
            "LCMC":                  _scores[2]["lcmc"],
            "Steadiness":            _scores[3]["steadiness"],
            "Cohesiveness":          _scores[3]["cohesiveness"],
        })

    df_zadu = pd.DataFrame(_rows)
    df_zadu
    return (df_zadu,)


@app.cell
def _(alt, df_zadu):
    # Helper: create a bar chart row for comparing methods across multiple metrics
    def bar_chart_row(df, metrics, title="Quality metric comparison", domain_min=0.5):
        """Create a faceted bar chart comparing methods across multiple metrics."""
        y_zoom = alt.selection_interval(bind='scales', encodings=['y'])
        _df_long = df.melt(
            id_vars=["Method"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score",
        )
        _bars = alt.Chart(_df_long).mark_bar(opacity=0.85).encode(
            x=alt.X(
                "Method:N",
                title="",
                sort=["PCA", "MDS", "t-SNE", "UMAP"],
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domainMin=0, domainMax=1)),
            color=alt.Color(
                "Method:N",
                sort=["PCA", "MDS", "t-SNE", "UMAP"],
                title="Method",
            ),
            tooltip=["Method", "Metric", alt.Tooltip("Score:Q", format=".4f")],
        ).properties(
            width=130,
            height=200,
        ).add_params(
            y_zoom
        ).facet(
            column=alt.Column("Metric:N", title=None),
        ).properties(
            title=title,
        )
        return _bars

    # Plot all metrics in a single faceted bar chart
    _all_metrics = [
        "Trustworthiness", "Continuity",
        "MRRE – False Neighbors", "MRRE – Missing Neighbors",
        "LCMC", "Steadiness", "Cohesiveness",
    ]
    bar_chart_row(df_zadu, _all_metrics, title="Comprehensive quality comparison (ZADU, k=20)", domain_min=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.**

    - **Trustworthiness & Continuity** (the first two bars) confirm what we already know: nonlinear methods (t-SNE, UMAP) excel at local neighborhood preservation, while linear methods (PCA, MDS) lag slightly behind at small k.
    - **MRRE** (Mean Relative Rank Errors) offers a more granular view — it separately quantifies how much neighbor rankings shift for false neighbors vs. missing neighbors. Methods with high trustworthiness but low MRRE–false suggest that while few false neighbors exist, the ones that do appear are ranked far from their true position.
    - **LCMC** (Local Continuity Meta-Criterion) combines trustworthiness and continuity into a single local quality score. It acts as a convenient summary when you want a quick ranking without juggling two separate numbers.
    - **Steadiness & Cohesiveness** evaluates quality at the _cluster level_: **steadiness** measures whether points that _don't_ belong together in HD remain separated in LD (no false cluster merging), **cohesiveness** measures whether points that _do_ belong together in HD stay together in LD (no cluster splitting).

    **Key takeaway:** no single method dominates across all criteria. The metrics are **not redundant**: they capture different structural aspects of the projection. Using multiple metrics gives a more honest and complete assessment than relying on any single number.

    There are also other ways to evaluate embeddings beyond the _global_ metrics used here. Some methods analyze quality **point by point** to show where the embedding distorts local structure, while others use **more advanced neighborhood or topology-based measures**. These approaches can provide deeper insight but are more complex and go beyond the goals of this tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # **Summary**

    | Method | Type | Preserves | Pros | Cons |
    |--------|------|-----------|------|------|
    | **PCA** | Linear | Global variance | Fast, interpretable axes | May miss nonlinear structure |
    | **MDS** | Linear | Pairwise distances | Intuitive (distances) | Slow for large data |
    | **t-SNE** | Nonlinear | Local neighborhoods | Great cluster separation | Slow, random init matters |
    | **UMAP** | Nonlinear | Local + some global | Fast, scalable | Less interpretable |

    **Quality Assessment**:
    - Always evaluate DR quality — don't blindly trust the visualization!
    - Use **Trustworthiness** (false neighbors) and **Continuity** (missing neighbors)
    - Visualize per-point quality to identify unreliable regions
    - Choose neighborhood size `k` based on your analysis goals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

        **PCA (Principal Component Analysis)**

        1. [StatQuest: PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — excellent intuitive video explanation
        2. [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100) — Shlens' clear tutorial (arXiv)
        3. Pearson, K. (1901). *"On Lines and Planes of Closest Fit to Systems of Points in Space."* — [original paper](https://zenodo.org/record/1430636)

        **MDS (Multidimensional Scaling)**

        4. [Scikit-learn: Multidimensional Scaling](https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling) — practical guide with code
        5. Kruskal, J.B. (1964). *"Multidimensional Scaling by Optimizing Goodness of Fit to a Nonmetric Hypothesis."* — [original paper](https://doi.org/10.1007/BF02289565)

        **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

        6. [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — **essential reading**, interactive guide on pitfalls
        7. [StatQuest: t-SNE Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) — video tutorial
        8. van der Maaten & Hinton (2008). *"Visualizing Data using t-SNE."* JMLR — [original paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

        **UMAP (Uniform Manifold Approximation and Projection)**

        9. [Understanding UMAP](https://pair-code.github.io/understanding-umap/) — **Google's interactive explainer**, highly recommended
        10. [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — how UMAP works
        11. McInnes et al. (2018). *"UMAP: Uniform Manifold Approximation and Projection."* — [original paper](https://arxiv.org/abs/1802.03426)

        **Comparative**

        12. [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — van der Maaten et al.

    ///
    """)
    return


if __name__ == "__main__":
    app.run()
