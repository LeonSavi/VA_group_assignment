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
    ## 📆 **Week 04 – Interpretation of Dimensionality Reduction**

    In the previous lab, we learned how to apply DR techniques and assess their quality. Now we tackle a crucial question: **what do the patterns in our projections actually mean?** We'll learn systematic approaches to interpret clusters and structures revealed by DR, connecting visual patterns back to the original high-dimensional features.

    **Topics covered:**
    - Interpreting PCA axes using loadings
    - Clustering projected data with DBSCAN and K-Means
    - Comparing clusters using univariate statistics
    - Finding differentiating factors with LDA
    - Visualizing feature contributions to projections
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Index**
    - [Part I: Setup and Recap](#part-i-setup-and-recap)
    - [Part II: Interpreting Linear DR (PCA)](#part-ii-interpreting-linear-dr-pca)
    - [Part III: Clustering in Projections](#part-iii-clustering-in-projections)
    - [Part IV: Comparing Clusters](#part-iv-comparing-clusters)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Setup & Recap**
    """)
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score, silhouette_score, silhouette_samples
    )
    import umap
    from scipy import stats
    return (
        DBSCAN,
        KMeans,
        LinearDiscriminantAnalysis,
        PCA,
        StandardScaler,
        adjusted_rand_score,
        alt,
        load_wine,
        np,
        pd,
        silhouette_samples,
        silhouette_score,
        stats,
        umap,
    )


@app.cell
def _(StandardScaler, load_wine, pd):
    # Load and prepare the Wine dataset
    wine = load_wine()
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

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_wine[all_features])

    print(f"Dataset: {X_scaled.shape[0]} wines × {X_scaled.shape[1]} features")
    df_wine.head()
    return X_scaled, all_features, df_wine


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choosing methods for interpretation

    In the previous workshop, we covered four DR methods: PCA, MDS, t-SNE, UMAP. For interpretation, we now select one representative from each DR family: **PCA** (linear) and **UMAP** (nonlinear).

    | Category | Method | Interpretable? | Why / Why Not |
    |----------|--------|----------------|---------------|
    | **Linear** | **PCA** | Axes via loadings | Each axis is a weighted sum of features: directly interpretable |
    | | MDS | _Axes meaningless_ | Preserves distances only; no information about feature contributions |
    | **Nonlinear** | **UMAP** | Clusters reliable | Balances local/global structure; clusters reflect real groupings |
    | | t-SNE | _Clusters unreliable_ | Can split one group into many clusters; distances between clusters are meaningless (the `crowding problem`) |


    /// note | Rule of thumb
    - **Need to interpret axes?** → PCA
    - **Need to interpret clusters?** → UMAP (and validate with known labels if available)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Comparing PCA vs UMAP projections

    Let’s start by revisiting how the two methods represent the same data, and compare their approaches.
    """)
    return


@app.cell
def _(PCA, X_scaled, umap):
    # Compute projections (recap from previous notebook)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    umap_model = umap.UMAP(n_neighbors=25, min_dist=0.1, n_jobs=1, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)

    print("Projections computed: PCA, UMAP")
    return X_pca, X_umap, pca


@app.cell
def _(X_pca, X_umap, df_wine, pd):
    # Create comparison DataFrame (long format)
    df_pca = pd.DataFrame(X_pca, columns=["x", "y"])
    df_pca["cultivar_name"] = df_wine["cultivar_name"].values
    df_pca["method"] = "PCA"

    df_umap = pd.DataFrame(X_umap, columns=["x", "y"])
    df_umap["cultivar_name"] = df_wine["cultivar_name"].values
    df_umap["method"] = "UMAP"

    df_comparison = pd.concat([df_pca, df_umap]).reset_index(names="point_id")
    df_comparison
    return df_comparison, df_umap


@app.cell
def _(alt, df_comparison):
    # Legend/point selection by cultivar
    _linked = alt.selection_point(
        fields=["cultivar_name"], # selection driven by category
        on="click",
        toggle=True,
        bind="legend" # enables legend interaction
    )

    alt.Chart(df_comparison).mark_circle(size=70).encode(
        x=alt.X("x:Q", title="Dimension 1"),
        y=alt.Y("y:Q", title="Dimension 2"),
        color=alt.condition(
            _linked,
            "cultivar_name:N",
            alt.value("lightgray"),
            title="Cultivar"
        ),
        opacity=alt.condition(_linked, alt.value(0.85), alt.value(0.25)),
        tooltip=["cultivar_name", "method", "point_id"]
    ).add_params(
        _linked,
    ).properties(
        width=280, height=280
    ).facet(
        column=alt.Column("method:N", title="")
    ).resolve_scale(
        x="independent", y="independent"
    ).properties(
        title="PCA vs UMAP"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interactive data exploration

    After performing DR, a natural first step in interpreting the results is to **visually explore the embedding together with the related data**. Thanks to marimo's built-in interactivity, this is straightforward: we wrap an Altair chart in `mo.ui.altair_chart()` and selections automatically sync to a filtered dataframe.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        In the previous section, we visualized the UMAP embedding colored by cultivar. Now make it **interactive** using Marimo's built-in selection support.

        **Task:** Wrap the UMAP scatter plot in `mo.ui.altair_chart()` and add an `alt.selection_interval()` so that brush-selected points are highlighted. Then display the filtered dataframe below the chart — first using the UMAP coordinates, then **linked back to `df_wine`** via the dataframe index.
    ///
    """)
    return


@app.cell
def _(alt, df_umap, mo):
    # 💡 Solution

    # Create selection
    selection = alt.selection_interval()
    # or for point selection:
    # selection = alt.selection_point(name="select", fields=["cultivar_name"])

    # Chart with selection
    _chart = alt.Chart(df_umap).mark_circle(size=70).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.condition(selection, "cultivar_name:N", alt.value("lightgray")),
        tooltip=["cultivar_name"]
    ).add_params(selection).properties(width=400, height=400)

    # Wrap in mo.ui.altair_chart to capture selection
    chart_widget = mo.ui.altair_chart(_chart)

    chart_widget
    return (chart_widget,)


@app.cell
def _(chart_widget, mo):
    # Get selected points as a filtered dataframe
    selected_df = chart_widget.value
    mo.md(f"**Selected points:** {len(selected_df)}")
    return (selected_df,)


@app.cell
def _(mo, selected_df):
    # Show table of selected points
    selected_df if len(selected_df) > 0 else mo.md("*Click on points to select*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the UMAP df only contains coordinates and cluster labels.

    **Is this really the data we're interested in?**
    To understand _why_ certain points cluster together and what their characteristics are, we need to connect back to the original features: alcohol content, malic acid, color intensity, and so on. This becomes even more important in settings where ground-truth labels are unavailable. We can simply do this by linking the selection to `df_wine` using the dataframe index.
    """)
    return


@app.cell
def _(df_wine, mo, selected_df):
    selected_df_wine = df_wine.loc[selected_df.index]
    selected_df_wine if len(selected_df) > 0 else mo.md("*Click on points to select*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Interpreting Linear DR (PCA)**

    Linear DR methods like PCA have a key advantage: **the axes are interpretable** because they're linear combinations of the original features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. PCA loadings

    The **loadings matrix** tells us exactly how each original feature contributes to each principal component:

    $$\text{PC}_j = \sum_{i=1}^{p} w_{ij} \cdot x_i$$

    where the coefficient $w_{ij}$ is the loading (weight) of feature $i$ on component $j$.
    """)
    return


@app.cell
def _(all_features, pca, pd):
    # Create loadings df
    df_loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=all_features)
    df_loadings = df_loadings.reset_index().rename(columns={"index": "feature"})

    df_loadings
    return (df_loadings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        The `df_loadings` DataFrame is in **wide format** (one column per PC). To create a faceted bar chart, we need **long format** (one row per feature-component pair).

        **Task:** Reshape `df_loadings` using `pd.melt()` and add an absolute loading column. The resulting df will have shape 26 x 4, and columns named: ['feature', 'component', 'loading', 'abs_loading'].
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Reshape from wide to long format for plotting
    # df_loadings_long = df_loadings.melt(
    #     id_vars=...,
    #     value_vars=...,
    #     var_name=...,
    #     value_name=...,
    # )

    # Step 2: Add absolute loading column & display the result
    # df_loadings_long["abs_loading"] = ...
    # df_loadings_long
    return


@app.cell
def _(df_loadings):
    # 💡 Solution

    # Step 1: Reshape from wide to long format for plotting
    df_loadings_long = df_loadings.melt(
        id_vars=["feature"],
        value_vars=["PC1", "PC2"],
        var_name="component",
        value_name="loading",
    )

    # Step 2: Add absolute loading column & display the result
    df_loadings_long["abs_loading"] = df_loadings_long["loading"].abs()
    df_loadings_long
    return (df_loadings_long,)


@app.cell
def _(alt, df_loadings_long, mo):
    mo.stop(
        df_loadings_long is None,
        mo.md("⚠️ `df_loadings_long` not found — complete the reshaping step above first.")
    )

    _sel = alt.selection_point(fields=["feature"], on="click", toggle=True)

    alt.Chart(df_loadings_long).mark_bar(stroke="black", strokeWidth=0.5).encode(
        y=alt.Y("feature:N", sort=alt.SortField(field="abs_loading", order="descending"), title=""),
        x=alt.X("loading:Q", title="Loading Value"),
        color=alt.condition(
            _sel,
            alt.Color("loading:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
            alt.value("lightgray")
        ),
        opacity=alt.condition(_sel, alt.value(1.0), alt.value(0.3)),
        tooltip=["feature:N", "loading:Q", "component:N"],
        row=alt.Row("component:N", title="")
    ).add_params(
        _sel
    ).properties(
        width=400,
        height=250,
        title="PCA Loadings: how features contribute to each PC"
    ).resolve_scale(
        y="independent"
    )
    return


@app.cell(hide_code=True)
def _(df_loadings, mo):
    # Compute top/bottom features for each component
    _pc1_top = df_loadings.nlargest(3, "PC1")["feature"].tolist()
    _pc1_bottom = df_loadings.nsmallest(3, "PC1")["feature"].tolist()
    _pc2_top = df_loadings.nlargest(3, "PC2")["feature"].tolist()
    _pc2_bottom = df_loadings.nsmallest(3, "PC2")["feature"].tolist()

    mo.md(f"""
    ## 2. Interpreting the components

    Looking at the loadings, we can give **meaningful names** to our principal components.

    | Component | High Positive Loadings | High Negative Loadings | Interpretation |
    |-----------|------------------------|------------------------|----------------|
    | **PC1** | {", ".join(_pc1_top)} | {", ".join(_pc1_bottom)} | **"Phenolic Compounds"** axis |
    | **PC2** | {", ".join(_pc2_top)} | {", ".join(_pc2_bottom)} | **"Color & Alcohol"** axis |

    **Reading the axes:**
    - Wines with **high PC1** → high in {_pc1_top[0]}, {_pc1_top[1]} → more phenolic/antioxidant content
    - Wines with **high PC2** → high in {_pc2_top[0]}, {_pc2_top[1]} → darker and more alcoholic

    This interpretation helps us understand what the scatter plot actually shows!
    """)
    return


@app.cell
def _(X_pca, alt, df_wine, pd):
    # Visualize PCA with interpretable labels
    df_pca_chart = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca_chart["cultivar_name"] = df_wine["cultivar_name"].values

    alt.Chart(df_pca_chart).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("PC1:Q", title="PC1: Phenolic Compounds →", axis=alt.Axis(titleFontSize=14, labelFontSize=12)),
        y=alt.Y("PC2:Q", title="PC2: Color & Alcohol →", axis=alt.Axis(titleFontSize=14, labelFontSize=12)),
        color=alt.Color("cultivar_name:N", title="Cultivar", legend=alt.Legend(titleFontSize=14, labelFontSize=12)),
        tooltip=["cultivar_name", "PC1", "PC2"]
    ).properties(
        width=500, height=400,
        title=alt.Title(
            "PCA with interpreted axes",
            fontSize=16,
            fontWeight="bold",
            anchor="middle", # "start", "middle", "end"
            subtitle="Axes labeled by top contributing features",
            subtitleFontSize=12
        )
    )
    return (df_pca_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Biplot: showing data and loadings together

    A **biplot** overlays the data points with arrows representing how features contribute to the projection.
    """)
    return


@app.cell
def _(all_features, alt, df_pca_chart, np, pca, pd):
    # Scale loadings for visibility (arrows)
    _scale_factor = 4  # Adjust for visual clarity
    df_biplot_arrows = pd.DataFrame({
        "Feature": all_features,
        "PC1": pca.components_[0] * _scale_factor,
        "PC2": pca.components_[1] * _scale_factor,
        "PC1_origin": 0,
        "PC2_origin": 0
    })

    # Only show top features (by loading magnitude) to avoid clutter
    df_biplot_arrows["magnitude"] = np.sqrt(df_biplot_arrows["PC1"]**2 + df_biplot_arrows["PC2"]**2)
    df_biplot_arrows = df_biplot_arrows.nlargest(6, "magnitude")
    # Magnitude is the length of the arrow, proportional to how each feature contributes to the 2D projection (important for both PC1 and PC2) –or, the Euclidean distance from the origin (0,0) to every point (PC1, PC2)

    # Points
    _points = alt.Chart(df_pca_chart).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("PC1:Q", title="PC1: Phenolic Compounds →", axis=alt.Axis(titleFontSize=14, labelFontSize=12)),
        y=alt.Y("PC2:Q", title="PC2: Color & Alcohol →", axis=alt.Axis(titleFontSize=14, labelFontSize=12)),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name", "PC1", "PC2"]
    )

    # Arrows (using rule mark)
    _arrows = alt.Chart(df_biplot_arrows).mark_rule(color="black", strokeWidth=1.5).encode(
        x="PC1_origin:Q",
        y="PC2_origin:Q",
        x2="PC1:Q",
        y2="PC2:Q"
    )

    # Arrow heads (approximate with points)
    _arrow_heads = alt.Chart(df_biplot_arrows).mark_point(
        shape="triangle", size=100, color="black", filled=True
    ).encode(
        x="PC1:Q",
        y="PC2:Q",
    )

    # Labels
    _labels = alt.Chart(df_biplot_arrows).mark_text(
        align="left", dx=5, dy=-5, fontSize=12
    ).encode(
        x="PC1:Q",
        y="PC2:Q",
        text="Feature:N"
    )

    (_points + _arrows + _arrow_heads + _labels).properties(
        width=500, height=450,
        title="PCA Biplot: Data + Feature directions"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Reading the biplot
    - **Arrow direction**: shows which direction in the plot corresponds to increasing values of that feature
    - **Arrow length**: longer arrows = stronger contribution to the projection
    - **Arrow angle**: features pointing in similar directions are positively correlated

    For example, wines in the upper-right tend to have high alcohol, color_intensity, and flavanoids.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part III: Clustering in Projected Spaces**

    For **nonlinear DR** methods like UMAP and t-SNE, axes don't have direct meaning. Instead, we interpret the (visual) **clusters** that emerge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources (clustering)

        1 [An overview of clustering methods](https://scikit-learn.org/stable/modules/clustering.html#) – Summary of most common clustering algorithms in scikit-learn

        2 [Visualizing K-Means clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) – Interactive visual guide of how K-Means iteratively updates centroids and assignments

        3 [Visualizing DBSCAN clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) – Interactive visual guide to understand how DBSCAN forms clusters based on density

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. K-Means: partition-based clustering

    K-Means is one of the most intuitive clustering algorithms. Given a target number of clusters K, it partitions data by **minimizing within-cluster variance**. The algorithm iteratively places centroids, assigns each point to the nearest one, and updates centroids to the mean of their assigned points until assignments stabilize. Since **we know there are 3 cultivars**, we can directly apply K-Means with `K=3` and compare the results against the true labels.
    """)
    return


@app.cell
def _(KMeans, X_umap, pd):
    # Apply K-Means to UMAP embedding
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels_kmeans = kmeans.fit_predict(X_umap)

    print(f"K-Means cluster sizes: {pd.Series(cluster_labels_kmeans).value_counts().sort_index().tolist()}")
    return (cluster_labels_kmeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        K-Means has been applied to the UMAP embedding and the cluster labels are stored in `cluster_labels_kmeans`.

        **Task:** Create a side-by-side scatter plot on the UMAP embedding that shows (1) the K-Means cluster assignments and (2) the true cultivar labels. Use this comparison to assess how well K-Means recovered the original groups.
    ///
    """)
    return


@app.cell
def _(X_umap, alt, cluster_labels_kmeans, df_wine, pd):
    # 💡 Solution
    # Compare K-Means clusters with true labels
    df_kmeans = pd.DataFrame(X_umap, columns=["x", "y"])
    df_kmeans["KMeans_cluster"] = cluster_labels_kmeans.astype(str)
    df_kmeans["true_cultivar"] = df_wine["cultivar_name"].values

    _chart_km = alt.Chart(df_kmeans).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.Color("KMeans_cluster:N", title="K-Means Cluster"),
        tooltip=["true_cultivar", "KMeans_cluster"]
    ).properties(width=280, height=280, title="K-Means Clusters (k=3)")

    _chart_true = alt.Chart(df_kmeans).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.Color("true_cultivar:N", title="True Cultivar"),
        tooltip=["true_cultivar", "KMeans_cluster"]
    ).properties(width=280, height=280, title="True Cultivars")

    (_chart_km | _chart_true).resolve_scale(color='independent')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. DBSCAN: density-based clustering

    _But what if we don't know the number of clusters in advance?_ Unlike K-Means, **DBSCAN doesn't require specifying K**. Instead, it discovers **clusters as dense regions separated by sparse areas**, and automatically labels outliers as noise. This makes it particularly well-suited for the blob-like structures that UMAP produces.

    **Key parameters:**
    - `eps`: maximum distance between neighbors (the radius of the neighborhood around any given data point)
    - `min_samples`: minimum number of points required to form a dense region (a cluster)
    """)
    return


@app.cell
def _(DBSCAN, X_umap):
    # Apply DBSCAN to UMAP embedding
    dbscan = DBSCAN(eps=0.5, min_samples=5) # default values
    cluster_labels_dbscan = dbscan.fit_predict(X_umap)

    print(f"DBSCAN found {len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)} clusters")
    print(f"Noise points (label=-1): {(cluster_labels_dbscan == -1).sum()}")
    return (cluster_labels_dbscan,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        DBSCAN has been applied to the UMAP embedding and cluster labels are in `cluster_labels_dbscan`. Note that DBSCAN assigns label `-1` to noise points.

        **Task:** Create a side-by-side visualization comparing DBSCAN cluster assignments with the true cultivar labels. Make sure noise points (`cluster == "-1"`) are visually distinct (e.g., gray). How does DBSCAN's result compare to K-Means?
    ///
    """)
    return


@app.cell
def _(X_umap, alt, cluster_labels_dbscan, df_wine, pd):
    # 💡 Solution
    # Visualize DBSCAN clusters vs true labels
    df_dbscan = pd.DataFrame(X_umap, columns=["x", "y"])
    df_dbscan["DBSCAN_cluster"] = cluster_labels_dbscan.astype(str)
    df_dbscan["true_cultivar"] = df_wine["cultivar_name"].values

    _chart_dbscan = alt.Chart(df_dbscan).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.Color("DBSCAN_cluster:N", title="DBSCAN cluster"),
        tooltip=["true_cultivar", "DBSCAN_cluster"]
    ).properties(width=280, height=280, title="DBSCAN Clusters")

    _chart_true = alt.Chart(df_dbscan).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y:Q", title="UMAP 2"),
        color=alt.Color("true_cultivar:N", title="True cultivar"),
        tooltip=["true_cultivar", "DBSCAN_cluster"]
    ).properties(width=280, height=280, title="True cultivars")

    (_chart_dbscan | _chart_true).resolve_scale(color='independent')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1. What is the effect of `eps`?
    Below, we vary `eps` (in $[0.3, 0.5, 0.8, 1.2]$) and visualize the resulting clusters.
    """)
    return


@app.cell
def _(DBSCAN, X_umap, alt, pd):
    eps_values = [0.3, 0.5, 0.8, 1.2]

    charts = []
    for _eps in eps_values:
        dbscan_model = DBSCAN(eps=_eps, min_samples=5)
        labels = dbscan_model.fit_predict(X_umap)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        df_temp = pd.DataFrame(X_umap, columns=["x", "y"])
        df_temp["cluster"] = labels.astype(str)

        chart = alt.Chart(df_temp).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X("x:Q", title="UMAP 1"),
            y=alt.Y("y:Q", title="UMAP 2"),
            color=alt.Color("cluster:N", title="Cluster"),
            tooltip=["cluster"]
        ).properties(
            width=200, 
            height=200, 
            title=f"eps={_eps} (clusters={n_clusters}, noise={n_noise})"
        )
        charts.append(chart)

    # 2x2 grid
    (charts[0] | charts[1]) & (charts[2] | charts[3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** The plot clearly shows how `eps` controls **cluster granularity** in DBSCAN:

    - `eps=0.3` (too small): the algorithm is too strict about what points belong together, resulting in 13 fragmented clusters and 73 noise points (about 41% of the data). Many points that should be grouped are left unassigned.
    - `eps=0.5` (optimal): this produces 3 clusters matching the true number of cultivars, with only 6 noise points. The clustering aligns well with the actual data structure.
    - `eps=0.8-1.2` (too large): the algorithm becomes too permissive, merging the two right-hand clusters into one. At eps=1.2, there's no noise at all, meaning even outliers are forced into clusters.

    `eps` acts as a _neighborhood radius_: too small creates over-segmentation, too large causes under-segmentation. **The optimal value (here ~0.5) captures the natural density structure of the data.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluating clustering results

    **Why cluster when we already have labels?** It might seem pointless to apply clustering when we already know the true cultivars. However, there are (at least) two relevant reasons.

    1. **Validating DR**: if unsupervised clustering aligns well with true labels, it confirms that UMAP (or any DR technique) preserved meaningful structure. Poor alignment suggests the DR lost important information — or that the original features don't separate classes well.

    2. **Simulating real-world workflows**: In practice, labels are often unavailable. This exercise demonstrates the typical pipeline: 1) reduce dimensions, 2) cluster, 3) interpret. Having ground truth lets us validate this approach before applying it to truly unlabeled data.

    **So how do we evaluate clustering quality?** This also depends on whether we have access to ground truth labels.

    - **External metrics**: when labels are available, we can use them as an external benchmark to measure how well clusters match the true groupings (e.g., Adjusted Rand Index, Normalized Mutual Information).

    - **Internal metrics**: when labels are unavailable, we rely on the geometry of the clusters themselves — how tight and well-separated they are (e.g., Silhouette score, Davies-Bouldin index).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1. External validation: Adjusted Rand Index (ARI)

    **ARI** compares cluster assignments to true labels. It ranges from -1 to 1 and measures the similarity between two clusterings, adjusted for _chance_ (i.e., the similarity that would happen just by random assignment):
    - **1.0** = perfect match
    - **0.0** = random assignment
    - **< 0** = worse than random
    """)
    return


@app.cell
def _(
    adjusted_rand_score,
    cluster_labels_dbscan,
    cluster_labels_kmeans,
    df_wine,
):
    # Calculate ARI for K-Means and DBSCAN
    ari_kmeans = adjusted_rand_score(df_wine["cultivar_name"], cluster_labels_kmeans)
    ari_dbscan = adjusted_rand_score(df_wine["cultivar_name"], cluster_labels_dbscan)
    return ari_dbscan, ari_kmeans


@app.cell(hide_code=True)
def _(ari_dbscan, ari_kmeans, mo):
    mo.md(f"""
    **Results:**
    - K-Means ARI: **{ari_kmeans:.3f}**
    - DBSCAN ARI: **{ari_dbscan:.3f}**

    Both values are close to 1, confirming that the clusters align well with true cultivar labels and that UMAP preserved the class structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Using ARI for hyperparameter tuning

    ARI is rarely used in isolation. One very common application is as an **objective function for hyperparameter optimization**: we compute ARI across a range of parameter values and select the one that maximizes agreement with true labels.

    This is particularly useful for algorithms like DBSCAN, where the `eps` hyperparameter significantly affects clustering results but has no obvious "correct" value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        The `eps` parameter in DBSCAN controls the neighborhood radius and strongly affects clustering results. We can use **ARI as an objective function** to find the best value automatically.

        **Task:** Compute ARI for `eps` values in the range `[0.1, 2.0]` (step 0.1), keeping `min_samples=5`. Plot ARI vs `eps` as a line chart and mark the optimal `eps` value. What does the chart reveal about DBSCAN's sensitivity to this hyperparameter?
    ///
    """)
    return


@app.cell
def _(DBSCAN, X_umap, adjusted_rand_score, alt, df_wine, np, pd):
    # 💡 Solution
    # Compute ARI for different eps values
    eps_range = np.arange(0.1, 2.0, 0.1)
    ari_scores = [
        adjusted_rand_score(df_wine["cultivar_name"], DBSCAN(eps=eps, min_samples=5).fit_predict(X_umap))
        for eps in eps_range
    ]

    df_ari = pd.DataFrame({"eps": eps_range, "ARI": ari_scores})
    best_eps = df_ari.loc[df_ari["ARI"].idxmax(), "eps"]

    # Plot
    alt.Chart(df_ari).mark_line(point=True).encode(
        x=alt.X("eps:Q", title="eps"),
        y=alt.Y("ARI:Q", title="ARI score")
    ).properties(
        width=400, 
        height=250, 
        title=f"ARI vs eps (best = {best_eps:.1f}, ARI = {df_ari.loc[df_ari["ARI"].idxmax(), "ARI"]:.3f})"
    )
    return (df_ari,)


@app.cell
def _(df_ari):
    df_ari
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** The maximum ARI is reached around `eps` ~ 0.6, but the default 0.5 lies in the same high-performance region. Outside this range, performance drops because clusters either become too fragmented (small `eps`) or overly merged (large `eps`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.2 Internal validation: Silhouette score

    When true labels are unavailable, we need metrics based on **cluster geometry** alone. The Silhouette score ranges from **-1** (wrong cluster) to **+1** (well-separated clusters) and measures:
    - **Cohesion**: how close points are to others in the same cluster
    - **Separation**: how far points are from the nearest other cluster

    $$\text{Silhouette} = \frac{b - a}{\max(a, b)}$$

    Where *a* = mean intra-cluster distance, *b* = mean nearest-cluster distance.
    """)
    return


@app.cell
def _(X_umap, cluster_labels_dbscan, cluster_labels_kmeans, silhouette_score):
    # Calculate Silhouette for K-Means
    sil_kmeans = silhouette_score(X_umap, cluster_labels_kmeans)

    # For DBSCAN, exclude noise points (-1)
    mask_no_noise = cluster_labels_dbscan != -1
    sil_dbscan = silhouette_score(X_umap[mask_no_noise], cluster_labels_dbscan[mask_no_noise])
    return sil_dbscan, sil_kmeans


@app.cell(hide_code=True)
def _(mo, sil_dbscan, sil_kmeans):
    mo.md(f"""
    **Results:**
    - K-Means Silhouette: **{sil_kmeans:.3f}**
    - DBSCAN Silhouette: **{sil_dbscan:.3f}**

    Both scores are positive and relatively high, indicating well-defined, separated clusters — even without looking at true labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Silhouette plot

    The silhouette plot visualizes how well each point fits within its assigned cluster:
    - **Wide bars** extending to the right → well-clustered points
    - **Bars extending left** (negative values) → points possibly assigned to wrong cluster
    - **Dashed line** = average silhouette score
    """)
    return


@app.cell
def _(X_umap, cluster_labels_kmeans, pd, silhouette_samples):
    # Compute silhouette for each sample
    silhouette_vals = silhouette_samples(X_umap, cluster_labels_kmeans)

    # Create dataframe for plotting
    df_silhouette = pd.DataFrame({
        "silhouette": silhouette_vals,
        "cluster": cluster_labels_kmeans.astype(str)
    })

    # Sort by cluster and silhouette value
    df_silhouette = df_silhouette.sort_values(["cluster", "silhouette"], ascending=[True, False]).reset_index(drop=True)
    df_silhouette["y"] = range(len(df_silhouette))
    return df_silhouette, silhouette_vals


@app.cell
def _(df_silhouette):
    df_silhouette
    return


@app.cell
def _(
    X_umap,
    alt,
    cluster_labels_kmeans,
    pd,
    sil_kmeans,
    silhouette_samples,
    silhouette_vals,
):
    # Compute silhouette for each sample
    _silhouette_vals = silhouette_samples(X_umap, cluster_labels_kmeans)

    # Create dataframe with original index
    _df_silhouette = pd.DataFrame({
        "silhouette": _silhouette_vals,
        "cluster": cluster_labels_kmeans.astype(str),
        "point_id": range(len(silhouette_vals))  # original point index
    })

    # Sort by cluster and silhouette value for display
    _df_silhouette = _df_silhouette.sort_values(["cluster", "silhouette"], ascending=[True, False]).reset_index(drop=True)
    _df_silhouette["y"] = range(len(_df_silhouette))  # y is just for bar positioning
    df_tmp = _df_silhouette
    # Selection linked by point_id (not y!)
    _sel = alt.selection_point(fields=["point_id"], name="silhouette_sel")

    # Silhouette bars
    bars = alt.Chart(_df_silhouette).mark_bar().encode(
        x=alt.X("silhouette:Q", title="Silhouette Coefficient", scale=alt.Scale(domain=[-0.2, 1])),
        y=alt.Y("y:O", axis=None),
        color=alt.condition(_sel, "cluster:N", alt.value("lightgray"), title="Cluster"),
        tooltip=["cluster", "point_id", alt.Tooltip("silhouette:Q", format=".3f")]
    ).add_params(_sel).properties(width=300, height=300)

    # Average line
    avg_line = alt.Chart(pd.DataFrame({"avg": [sil_kmeans]})).mark_rule(
        strokeDash=[5, 5], 
        color="red",
        strokeWidth=2
    ).encode(x="avg:Q")

    silhouette_chart = (bars + avg_line).properties(title=f"Silhouette Plot (avg = {sil_kmeans:.3f})")

    # UMAP chart - also needs point_id
    umap_chart = alt.Chart(_df_silhouette.assign(
        x=X_umap[:, 0][_df_silhouette["point_id"]], 
        y_coord=X_umap[:, 1][_df_silhouette["point_id"]]
    )).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x:Q", title="UMAP 1"),
        y=alt.Y("y_coord:Q", title="UMAP 2"),
        color=alt.condition(_sel, "cluster:N", alt.value("lightgray"), title="Cluster"),
        tooltip=["cluster", "point_id", alt.Tooltip("silhouette:Q", format=".3f")]
    ).add_params(_sel).properties(width=300, height=300, title="K-Means Clusters")

    umap_chart | silhouette_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation.** High `ARI` confirms clusters match true labels. High `Silhouette` confirms clusters are geometrically well-separated (a metric we could rely on even without ground truth).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part IV: Comparing Clusters**

    Now comes the key question: **what makes the clusters different?** We'll use several approaches to find the distinguishing features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Univariate approach: feature distributions per cluster

    The simplest approach: look at how each feature is distributed within each cluster.
    """)
    return


@app.cell
def _(mo):
    # Let user select which clustering to analyze
    cluster_choice = mo.ui.dropdown(
        options=["DBSCAN clusters", "K-Means clusters", "True cultivars"],
        value="DBSCAN clusters",
        label="Analyze clusters from:"
    )
    cluster_choice
    return (cluster_choice,)


@app.cell
def _(
    all_features,
    cluster_choice,
    cluster_labels_dbscan,
    cluster_labels_kmeans,
    df_wine,
):
    # Prepare data based on choice
    if cluster_choice.value == "True cultivars":
        _labels = df_wine["cultivar_name"].values
        _label_col = "cluster"
    elif cluster_choice.value == "K-Means clusters":
        _labels = [f"cluster {c}" for c in cluster_labels_kmeans]
        _label_col = "cluster"
    elif cluster_choice.value == "DBSCAN clusters":
        _labels = [f"cluster {c}" for c in cluster_labels_dbscan]
        _label_col = "cluster"

    df_features = df_wine[all_features].copy()
    df_features["cluster"] = _labels
    df_features
    return (df_features,)


@app.cell
def _(mo):
    # Feature selector
    feature_selector = mo.ui.dropdown(
        options=["alcohol", "malic_acid", "flavanoids", "color_intensity", 
                 "proline", "total_phenols", "hue", "protein_content"],
        value="flavanoids",
        label="Select feature to compare:"
    )
    feature_selector
    return (feature_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        `df_features` contains the original wine features plus a `"cluster"` column. A dropdown widget (`feature_selector`) is already wired up for you.

        **Task:** Build an overlapping histogram that shows the distribution of the selected feature split by cluster. Use `stack(None)` and `opacity` to make the distributions comparable. What features best separate the clusters?
    ///
    """)
    return


@app.cell
def _(alt, df_features, feature_selector):
    # 💡 Solution
    # Histogram comparison
    _feature = feature_selector.value

    alt.Chart(df_features).mark_bar(opacity=0.6).encode(
        x=alt.X(f"{_feature}:Q", bin=alt.Bin(maxbins=20), title=_feature),
        y=alt.Y("count():Q", title="Count").stack(None),
        color=alt.Color("cluster:N"),
        tooltip=["cluster", "count()"]
    ).properties(
        width=500, height=300,
        title=f"Distribution of {_feature} by cluster"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Statistical tests: t-test and ANOVA

    In this section, we systematically test all features to find which ones best distinguish the clusters. We use both t-test and ANOVA.
    - **ANOVA:** tests whether at least one group mean differs across *all* clusters at once — statistically preferable for 3+ groups as it avoids the multiple comparisons problem (running k separate t-tests inflates the false positive rate). It allows to **rank features globally**. However, note that ANOVA tells you that groups differ, but not _which_ ones differ.

    - **t-test (one-vs-rest):** for each cluster $k$, tests whether feature means differ between points *in* $k$ vs. *not in* $k$. It's a binary comparison: it compares one cluster against all others combined (label X vs. not-label X). Gives a **per-cluster, per-feature result**. Conceptually, the issue is that "all others combined" is not a coherent group, especially when clusters differ substantially from each other. When there are 3+ groups, ANOVA is preferred.
    """)
    return


@app.cell
def _(all_features, df_features, pd, stats):
    # ANOVA: one score per feature
    _anova_rows = []
    for _feat in all_features:
        _groups = [
            df_features[df_features["cluster"] == c][_feat].values
                for c in df_features["cluster"].unique()
        ]
        _f, _p = stats.f_oneway(*_groups) # * is for unpacking = f_oneway(_groups[0], _groups[1], _groups[2])
        _anova_rows.append({"feature": _feat, "F-statistic": _f, "p-value": _p})
    df_anova = pd.DataFrame(_anova_rows).sort_values("F-statistic", ascending=False)

    # t-test: one score per (cluster × feature)
    _ttest_rows = []
    for _cluster in df_features["cluster"].unique():
        _mask = df_features["cluster"] == _cluster
        for _feat in all_features:
            _t, _p = stats.ttest_ind(
                df_features.loc[_mask, _feat].values,
                df_features.loc[~_mask, _feat].values
            )
            _ttest_rows.append({
                "cluster": _cluster,
                "feature": _feat,
                "abs_t": abs(_t),
                "p-value": _p,
                "significant": "yes" if _p < 0.05 else "no"
            })
    df_ttest = pd.DataFrame(_ttest_rows)
    return df_anova, df_ttest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        `df_anova` contains one row per feature with its ANOVA F-statistic and p-value. A high F-statistic means the feature is highly discriminative across clusters.

        **Task:** Create a horizontal bar chart sorted by F-statistic (highest at top), colored by F-statistic value using a sequential color scheme. Add tooltips showing the feature name, F-statistic, and p-value.
    ///
    """)
    return


@app.cell
def _(alt, df_anova):
    # 💡 Solution
    # Visualize feature importance
    alt.Chart(df_anova).mark_bar(stroke="black", strokeWidth=0.5).encode(
        y=alt.Y("feature:N", sort="-x", title=""),
        x=alt.X("F-statistic:Q", title="F-statistic (higher = more discriminative)"),
        color=alt.Color("F-statistic:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["feature", alt.Tooltip("F-statistic:Q", format=".1f"), 
                 alt.Tooltip("p-value:Q", format=".2e")]
    ).properties(
        width=450, height=350,
        title="ANOVA – global feature ranking" # Feature importance for cluster separation
    )
    return


@app.cell
def _(alt, df_ttest):
    _feat_order = (
        df_ttest.groupby("feature")["abs_t"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    _base = alt.Chart(df_ttest)

    _heatmap = _base.mark_rect(color="white", stroke="lightgray", strokeWidth=1).encode(
        x=alt.X("feature:N", sort=_feat_order, title="", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("cluster:N", title=""),
    )

    _text = _base.mark_text(fontSize=12).encode(
        x=alt.X("feature:N", sort=_feat_order),
        y=alt.Y("cluster:N"),
        text=alt.Text("p-value:Q", format=".3f"),
        color=alt.condition(
            "datum.significant == 'yes'",
            alt.value("red"),
            alt.value("black")
        )
    )

    (_heatmap + _text).properties(
        width=600, height=120,
        title="Per-cluster t-test p-values  (red = p < 0.05)"
    )

    # The result shows many attributes have p < 0.05
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. LDA: finding the optimal separation

    **Linear Discriminant Analysis (LDA)** finds linear combinations of features that best separate predefined groups — ideal for understanding cluster differences.
    """)
    return


@app.cell
def _(LinearDiscriminantAnalysis, X_scaled, df_wine):
    # Fit LDA using true cultivar labels
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, df_wine["cultivar"].values)

    print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
    return X_lda, lda


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        LDA has been fitted and `X_lda` contains the 2D projection. Unlike UMAP, these axes have a meaning: they are linear combinations of features **optimized to separate the known classes**.

        **Task:** Plot the LDA projection as a scatter plot colored by `cultivar_name`. Compare the class separation visually with what you saw in the previous embeddings — which method produces tighter, more distinct clusters?
    ///
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(X_lda, alt, df_wine, pd):
    # 💡 Solution
    # Visualize LDA projection
    df_lda = pd.DataFrame(X_lda, columns=["LD1", "LD2"])
    df_lda["cultivar_name"] = df_wine["cultivar_name"].values

    alt.Chart(df_lda).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("LD1:Q", title="Linear Discriminant 1"),
        y=alt.Y("LD2:Q", title="Linear Discriminant 2"),
        color=alt.Color("cultivar_name:N", title="Cultivar"),
        tooltip=["cultivar_name", "LD1", "LD2"]
    ).properties(
        width=500, height=400,
        title="LDA Projection (Optimized for Class Separation)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1. LDA coefficients: what separates the groups?

    Unlike t-SNE/UMAP, LDA gives us interpretable coefficients showing how features contribute to group separation.
    """)
    return


@app.cell
def _(all_features, lda, pd):
    # Extract and visualize LDA coefficients
    df_lda_coef = pd.DataFrame({
        "feature": all_features,
        "LD1": lda.scalings_[:, 0],
        "LD2": lda.scalings_[:, 1]
    })

    # Sort by absolute LD1 coefficient
    df_lda_coef["abs_LD1"] = df_lda_coef["LD1"].abs()
    df_lda_coef = df_lda_coef.sort_values("abs_LD1", ascending=True).drop("abs_LD1", axis=1)

    df_lda_long = df_lda_coef.melt(id_vars=["feature"], var_name="component", value_name="coefficient")
    df_lda_long['abs_coefficient'] = df_lda_long.coefficient.abs()
    return (df_lda_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        `df_lda_coef` contains the LDA scaling coefficients (loadings) for each feature on LD1 and LD2. These tell us which features drive the class separation.

        **Task:** Taking inspiration from the PCA loadings visualization you built earlier, create a bar chart of LDA coefficients sorted by absolute values. How do the most important features here compare to the top PCA loadings?
    ///
    """)
    return


@app.cell
def _(alt, df_lda_long):
    # 💡 Solution
    _sel = alt.selection_point(fields=["feature"], on="click", toggle=True)

    alt.Chart(df_lda_long).mark_bar(stroke="black", strokeWidth=0.5).encode(
        y=alt.Y(
            "feature:N",
            sort=alt.SortField(field="abs_coefficient", order="descending"),
            title=""
        ),
        x=alt.X("coefficient:Q", title="LDA coefficient"),
        color=alt.condition(
            _sel,
            alt.Color("coefficient:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
            alt.value("lightgray")
        ),
        opacity=alt.condition(_sel, alt.value(1.0), alt.value(0.3)),
        row=alt.Row("component:N", title="")
    ).add_params(
        _sel
    ).properties(width=400, height=220, title="LDA coefficients: features that separate cultivars")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | PCA vs LDA
    - **PCA** finds directions of maximum **variance** (unsupervised)
    - **LDA** finds directions of maximum **separation between groups** (supervised)

    LDA often reveals different important features than PCA because it optimizes for a different goal!
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Per-cluster characterization: PCA and LDA

    To get a **cluster-specific fingerprint** we now fit a separate model per cluster:

    | Method | Fits on | Finds |
    |--------|---------|-------|
    | **PCA** | `X[cluster == k]` | Direction of max variance *inside* the cluster |
    | **LDA** | All `X`, binary label `cluster == k` | Direction that best separates this cluster *from the rest* |

    For each method we produce two charts:
    1. **Weights**: which features drive the direction found
    2. **Projection histogram**: project *all* points onto that direction and color by cluster. If the target cluster separates clearly from the rest, the method found a useful direction.

    **Contrastive perspective.** PCA is non-contrastive: it only looks inside the target cluster and may highlight features that vary everywhere. LDA is contrastive: it compares the cluster against the background, surfacing features that are *distinctive*.
    """)
    return


@app.cell
def _(
    LinearDiscriminantAnalysis,
    PCA,
    X_scaled,
    all_features,
    cluster_labels_dbscan,
    np,
    pd,
):
    _pca = PCA(n_components=1)
    _lda = LinearDiscriminantAnalysis(n_components=1)
    _weight_rows = []
    _proj_rows = []

    for _label in np.unique(cluster_labels_dbscan):
        _mask = cluster_labels_dbscan == _label
        _name = "noise" if _label == -1 else f"cluster {_label}"

        # PCA: fit only on target cluster, project all points
        _pca.fit(X_scaled[_mask])
        _pca_w = _pca.components_[0]
        _pca_w = _pca_w / (np.abs(_pca_w).max() + 1e-10)
        _pca_proj = _pca.transform(X_scaled)[:, 0]

        # LDA: fit with binary label (cluster k vs rest), project all points
        _lda.fit(X_scaled, _mask.astype(int))
        _lda_w = _lda.scalings_[:, 0]
        _lda_w = _lda_w / (np.abs(_lda_w).max() + 1e-10)
        _lda_proj = _lda.transform(X_scaled)[:, 0]

        for _i, _feat in enumerate(all_features):
            _weight_rows.append({
                "cluster": _name,
                "feature": _feat,
                "PCA weight": float(_pca_w[_i]),
                "LDA weight": float(_lda_w[_i]),
            })

        for _i_pt in range(len(X_scaled)):
            _pt_label = cluster_labels_dbscan[_i_pt]
            _pt_name = "noise" if _pt_label == -1 else f"cluster {_pt_label}"
            _proj_rows.append({
                "fit_cluster": _name,
                "point_cluster": _pt_name,
                "PCA projection": float(_pca_proj[_i_pt]),
                "LDA projection": float(_lda_proj[_i_pt]),
            })

    df_cluster_weights = pd.DataFrame(_weight_rows)
    df_cluster_projections = pd.DataFrame(_proj_rows)
    return df_cluster_projections, df_cluster_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1. PCA — weights and projection

    Each panel shows the PC1 weights when PCA is fitted **only on that cluster**.
    The histogram below shows where all points land when projected onto that direction.
    """)
    return


@app.cell
def _(alt, df_cluster_weights):
    alt.Chart(df_cluster_weights).mark_bar(stroke="black", strokeWidth=0.3).encode(
        y=alt.Y("feature:N", sort=None, title=""),
        x=alt.X("PCA weight:Q", scale=alt.Scale(domain=[-1, 1]), title="Normalized weight"),
        color=alt.Color("PCA weight:Q",
                        scale=alt.Scale(scheme="redblue", domainMid=0), legend=None),
        facet=alt.Facet("cluster:N", columns=4, title=""),
        tooltip=["feature", alt.Tooltip("PCA weight:Q", format=".3f")]
    ).properties(width=170, height=260).resolve_scale(x="shared")
    return


@app.cell
def _(alt, df_cluster_projections):
    alt.Chart(df_cluster_projections).mark_bar(opacity=0.5).encode(
        x=alt.X("PCA projection:Q", bin=alt.Bin(maxbins=20), title="PC1 score"),
        y=alt.Y("count():Q", title="Count").stack(None),
        color=alt.Color("point_cluster:N", title="Cluster"),
        facet=alt.Facet("fit_cluster:N", columns=4, title=""),
        tooltip=["point_cluster", "count()"]
    ).properties(
        width=170, height=130,
        title=alt.Title(
            "PCA projection",
            subtitle="Each panel: PC1 fitted on that cluster"
        )
    ).resolve_scale(x="independent")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2. LDA — weights and projection

    Each panel shows LD1 weights fitted with a binary label **"is this cluster vs. all others"**.
    The histogram shows how well that direction separates the target cluster from the rest.

    /// note | PCA & LDA
    - **PCA high, LDA low** → feature varies within the cluster but is similar across all clusters → not distinctive
    - **LDA high, PCA low** → feature cleanly separates this cluster, even with low internal spread
    - **Both high** → feature is both internally variable *and* externally distinctive
    ///
    """)
    return


@app.cell
def _(alt, df_cluster_weights):
    alt.Chart(df_cluster_weights).mark_bar(stroke="black", strokeWidth=0.3).encode(
        y=alt.Y("feature:N", sort=None, title=""),
        x=alt.X("LDA weight:Q", scale=alt.Scale(domain=[-1, 1]), title="Normalized weight"),
        color=alt.Color("LDA weight:Q",
                        scale=alt.Scale(scheme="redblue", domainMid=0), legend=None),
        facet=alt.Facet("cluster:N", columns=4, title=""),
        tooltip=["feature", alt.Tooltip("LDA weight:Q", format=".3f")]
    ).properties(width=170, height=260).resolve_scale(x="shared")
    return


@app.cell
def _(alt, df_cluster_projections):
    alt.Chart(df_cluster_projections).mark_bar(opacity=0.5).encode(
        x=alt.X("LDA projection:Q", bin=alt.Bin(maxbins=20), title="LD1 score"),
        y=alt.Y("count():Q", title="Count").stack(None),
        color=alt.Color("point_cluster:N", title="Cluster"),
        facet=alt.Facet("fit_cluster:N", columns=4, title=""),
        tooltip=["point_cluster", "count()"]
    ).properties(
        width=170, height=130,
        title=alt.Title("LDA projection",
                        subtitle="Each panel: LD1 fitted one-vs-rest on that cluster")
    ).resolve_scale(x="independent")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Look at the **LDA projection histograms**. For which cluster does the target cluster separate
    most clearly from the others? Which features drive that separation according to the LDA weights?

    Now compare PCA and LDA weights for that same cluster — do they agree or disagree?
    What does a disagreement tell you?
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cluster profiles: parallel coordinates

    Let's create a comprehensive view of how clusters differ across all features (see [Parallel Coordinates in Altair](https://altair-viz.github.io/gallery/parallel_coordinates.html)). We explore two complementary views:
    - **Cluster means**: one line per cluster, useful to compare profiles at a glance
    - **Individual points**: all points from a selected cluster, reveals internal spread and outliers
    """)
    return


@app.cell
def _(all_features, df_features):
    # Individual points data
    _df = df_features[df_features["cluster"] != "cluster -1"].copy()

    _feat_min = _df[all_features].min()
    _feat_max = _df[all_features].max()

    # Means chart data
    _means = _df.groupby("cluster")[all_features].mean()
    _means_norm = (_means - _feat_min) / (_feat_max - _feat_min)
    df_parallel = _means_norm.reset_index().melt(
        id_vars=["cluster"], var_name="feature", value_name="value"
    )

    # Individual points data (keep noise excluded)
    _pts = _df.copy()
    _pts[list(all_features)] = (_pts[list(all_features)] - _feat_min) / (_feat_max - _feat_min)
    _pts = _pts.reset_index(names="point_id")
    df_parallel_points = _pts.melt(
        id_vars=["point_id", "cluster"], var_name="feature", value_name="value"
    )
    return df_parallel, df_parallel_points


@app.cell
def _(alt, df_parallel):
    _sel = alt.selection_point(fields=["cluster"], bind="legend")

    _rules = alt.Chart(df_parallel).mark_rule(
        color="black", strokeOpacity=0.4, strokeWidth=1, strokeDash=[4, 4]
    ).encode(x=alt.X("feature:N", sort=None))

    _lines = alt.Chart(df_parallel).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("feature:N", sort=None, title="",
                 axis=alt.Axis(labelAngle=-45, domain=False, ticks=False)),
        y=alt.Y("value:Q", title="Normalized value (0–1)"),
        color=alt.Color("cluster:N"),
        opacity=alt.condition(_sel, alt.value(1), alt.value(0.1)),
        detail="cluster:N",
        tooltip=["cluster", "feature", alt.Tooltip("value:Q", format=".2f")]
    ).add_params(_sel)

    (_rules + _lines).properties(
        width=700, height=300,
        title=alt.Title(
            "Cluster profiles (normalized means)",
            subtitle="Click legend to highlight"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// note | Reading the parallel coordinates
    - Each line represents a cluster's "profile"
    - Features where lines are far apart = **differentiating features**
    - Features where lines are close = clusters are similar on that dimension

    This gives a quick overview of what makes each cluster unique!
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1. Individual points for a selected cluster
    """)
    return


@app.cell
def _(df_parallel, mo):
    _clusters = sorted(df_parallel["cluster"].unique().tolist())
    cluster_points_selector = mo.ui.dropdown(
        options=_clusters,
        value=_clusters[0],
        label="Show points for cluster:"
    )
    cluster_points_selector
    return (cluster_points_selector,)


@app.cell
def _(alt, cluster_points_selector, df_parallel_points):
    _cluster = cluster_points_selector.value
    _df = df_parallel_points[df_parallel_points["cluster"] == _cluster]
    _mean_df = _df.groupby("feature")["value"].mean().reset_index()

    _rules = alt.Chart(_df).mark_rule(
        color="black", strokeOpacity=0.4, strokeWidth=1, strokeDash=[4, 4]
    ).encode(x=alt.X("feature:N", sort=None))

    _lines = alt.Chart(_df).mark_line(strokeWidth=1, opacity=0.35).encode(
        x=alt.X("feature:N", sort=None, title="",
                 axis=alt.Axis(labelAngle=-45, domain=False, ticks=False)),
        y=alt.Y("value:Q", title="Normalized value (0–1)", scale=alt.Scale(domain=[0, 1])),
        detail="point_id:N",
        tooltip=["point_id", "feature", alt.Tooltip("value:Q", format=".2f")]
    )

    _mean_line = alt.Chart(_mean_df).mark_line(
        strokeWidth=2, color="black", strokeDash=[6, 3]
    ).encode(
        x=alt.X("feature:N", sort=None),
        y=alt.Y("value:Q")
    )

    (_rules + _lines + _mean_line).properties(
        width=700, height=300,
        title=f"Individual points for {_cluster} (dashed black = cluster mean)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Feature attribution

    Another powerful approach is to **color the projection by original feature values** to see how features relate to the visual structure.
    """)
    return


@app.cell
def _(mo):
    # Multi-feature selector
    feature_color_selector = mo.ui.dropdown(
        options=["alcohol", "malic_acid", "flavanoids", "color_intensity", 
                 "proline", "total_phenols", "hue", "protein_content",
                 "ash", "magnesium"],
        value="flavanoids",
        label="Color by feature:"
    )
    feature_color_selector
    return (feature_color_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!

        `df_comparison` holds both PCA and UMAP projections in long format (column `"method"`). A dropdown (`feature_color_selector`) lets you pick any continuous wine feature to use as a color channel.

        **Task:** Create a faceted scatter plot (one panel per DR method) where points are colored by the selected continuous feature using the `"viridis"` color scheme. What patterns become visible that were hidden when coloring by cultivar label?
    ///
    """)
    return


@app.cell
def _(alt, df_comparison, df_wine, feature_color_selector):
    # 💡 Solution
    # Create comparison with feature coloring
    _feature = feature_color_selector.value
    df_comparison_ft = df_comparison.copy()
    df_comparison_ft[_feature] = [el for el in [df_wine[_feature].values, df_wine[_feature].values] for el in el]

    alt.Chart(df_comparison_ft).mark_circle(size=70, opacity=0.8).encode(
        x=alt.X("x:Q", title="Dimension 1"),
        y=alt.Y("y:Q", title="Dimension 2"),
        color=alt.Color(f"{_feature}:Q", scale=alt.Scale(scheme="viridis"), title=_feature),
        tooltip=[alt.Tooltip(f"{_feature}:Q", format=".2f")]
    ).properties(width=300, height=300).facet(
        column=alt.Column("method:N", title="")
    ).properties(title=f"Projections colored by {_feature}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔗 Resources
    /// admonition | 🔗 Resources

        1 [EuroVis 2025 DR Tutorial](https://hyeonword.com/dr-tutorial/) — Comprehensive DR evaluation & interpretation guide

        2 [ccPCA Github repo](https://github.com/takanori-fujiwara/ccpca)
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
