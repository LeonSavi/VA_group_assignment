import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    ROOT_DATA = 'data'
    return (ROOT_DATA,)


@app.cell
def _():
    import pandas as pd
    import json
    import altair as alt


    import pandas as pd 
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from typing import Callable, Union, Literal
    from scipy.stats import entropy
    import desbordante as db
    import itertools
    from tqdm import tqdm #usually this library slow down loops
    return Callable, alt, entropy, json, np, pd


@app.cell
def _(ROOT_DATA, json, pd):
    def load_knowledge_graph(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        nodes_df = pd.DataFrame(data.get('nodes', []))

        edges_df = pd.DataFrame(data.get('links', data.get('edges', [])))

        return nodes_df, edges_df

    filah_nodes, filah_edges = load_knowledge_graph(f"{ROOT_DATA}/FILAH.json")
    trout_nodes, trout_edges = load_knowledge_graph(f"{ROOT_DATA}/TROUT.json")
    journo_nodes, journo_edges = load_knowledge_graph(f"{ROOT_DATA}/journalist.json")
    return (
        filah_edges,
        filah_nodes,
        journo_edges,
        journo_nodes,
        load_knowledge_graph,
        trout_edges,
        trout_nodes,
    )


@app.cell
def _(
    filah_edges,
    filah_nodes,
    journo_edges,
    journo_nodes,
    np,
    pd,
    trout_edges,
    trout_nodes,
):
    def clean_knowledge_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,ref_df:str):
        """
        Applies correct data types and cleans lists for both nodes and edges dataframes.
        Returns cleaned_nodes, cleaned_edges.
        """
        # Create copies so we don't overwrite the original raw data
        n_df = nodes_df.copy()
        e_df = edges_df.copy()

        n_df['dataset'] = ref_df
        e_df['dataset'] = ref_df

        # 1. Handle List Columns (Applies to both nodes and edges)
        list_cols = ['industry', 'Activities', 'fish_species_present']

        # Loop through both dataframes to clean list columns wherever they appear
        for df in [n_df, e_df]:
            for col in list_cols:
                if col in df.columns:
                    # Join lists with a comma, handle empty lists and None
                    df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                    df[col] = df[col].replace('', np.nan)
                    df[col] = df[col].astype(str).replace('nan', np.nan) # Ensure pure strings/NaN

        # 2. Handle Edges specific columns
        if 'time' in e_df.columns:
            # Fix the 0040 typo in TROUT data by replacing it with 2040
            e_df['time'] = e_df['time'].astype(str).str.replace('0040-', '2040-')
            # Convert to true datetime objects
            e_df['time'] = pd.to_datetime(e_df['time'], errors='coerce')

        if 'sentiment' in e_df.columns:
            e_df['sentiment'] = pd.to_numeric(e_df['sentiment'], errors='coerce')

        if 'role' in e_df.columns:
            e_df['role'] = e_df['role'].astype('category')

        # 3. Handle Nodes specific columns
        if 'type' in n_df.columns:
            n_df['type'] = n_df['type'].astype('category')

        if 'longitude' in n_df.columns:
            n_df['longitude'] = pd.to_numeric(n_df['longitude'], errors='coerce')

        if 'latitude' in n_df.columns:
            n_df['latitude'] = pd.to_numeric(n_df['latitude'], errors='coerce')

        return n_df.fillna(np.nan), e_df.fillna(np.nan)

    def enrich_edges(nodes_df: pd.DataFrame,edges_df: pd.DataFrame):
        """Merges node details into the edge dataframe for easier plotting."""

        # Ensure ID columns are strings for safe merging
        nodes_df['id'] = nodes_df['id'].astype(str)
        edges_df['source'] = edges_df['source'].astype(str)
        edges_df['target'] = edges_df['target'].astype(str)

        # Merge Source node details (Using 'label' instead of 'name' based on the JSON structure)
        # If your dataframe specifically created a 'name' column, you can switch this back to 'name'
        target_cols = ['id', 'type', 'label','dataset'] if 'label' in nodes_df.columns else ['id', 'type', 'name','dataset']
        node_name_col = 'label' if 'label' in nodes_df.columns else 'name'

        enriched = edges_df.merge(
            nodes_df[target_cols], 
            left_on='source', 
            right_on='id', 
            how='left'
        ).rename(columns={'type': 'source_type', node_name_col: 'source_name'})

        # Merge Target node details
        enriched = enriched.merge(
            nodes_df[target_cols], 
            left_on='target', 
            right_on='id', 
            how='left',
            suffixes=('', '_target')
        ).rename(columns={'type': 'target_type', node_name_col: 'target_name'})

        return enriched


    clean_filah_nodes, clean_filah_edges = clean_knowledge_graph(filah_nodes, filah_edges, 'FILAH')
    filah_enriched = enrich_edges(clean_filah_nodes,clean_filah_edges)

    clean_trout_nodes, clean_trout_edges = clean_knowledge_graph(trout_nodes, trout_edges,'TROUT')
    trout_enriched = enrich_edges(clean_trout_nodes, clean_trout_edges)

    clean_journo_nodes, clean_journo_edges = clean_knowledge_graph(journo_nodes, journo_edges,'Journalist')
    journo_enriched = enrich_edges(clean_journo_nodes, clean_journo_edges)
    return (
        clean_filah_edges,
        clean_filah_nodes,
        clean_journo_edges,
        clean_journo_nodes,
        clean_trout_edges,
        clean_trout_nodes,
        filah_enriched,
        journo_enriched,
        trout_enriched,
    )


@app.cell
def _(Callable, entropy, pd):
    ## HELPER FUNCTIONS for Data Profiling
    def looping_col(func:Callable, df:pd.DataFrame, chart:bool = False) -> pd.DataFrame:
        """I don t want to see too many for loops later

        Apply a column-wise function over a DataFrame.

        If chart=False (default), calls func(df, col) for each column and
        returns a DataFrame assembled from the results (dict of col -> output).
        If chart=True, simply iterates columns and calls func(df, col) for its side effects
        (e.g., plotting/saving charts), returning nothing.

        https://github.com/LeonSavi/ADS---Assignemnt-2/blob/master/TASK%201%20-%20dataset%20profiling.ipynb

        """
        if not chart:
            output = {col: func(df,col) for col in df.columns}
            return pd.DataFrame(output)
        else:
            for col in df.columns:
                func(df,col)


    def summary_df(df:pd.DataFrame, col:str) -> int:

        '''
        Compute lightweight summary stats for one column.

        Returns count, missing, % missing, '-1' placeholders, distincts,
        uniqueness ratio, dtype, mode and its share, entropy (Shannon),
        and for numeric quartiles, IQR whiskers, and outlier count/% via the IQR rule.

        https://github.com/LeonSavi/ADS---Assignemnt-2/blob/master/TASK%201%20-%20dataset%20profiling.ipynb

        '''

        s = df[col].copy()
        #i explicit dropna=True
        Q25 = Q50 = Q75 = lower = upper = out_count = out_pct = None

        # let s get the wiskers
        if pd.api.types.is_numeric_dtype(s):
            Q25 = s.quantile(0.25)
            Q50 = s.quantile(0.5)
            Q75 = s.quantile(0.75)
            IQR = Q75 - Q25
            lower = Q25 - 1.5*IQR
            upper = Q75 + 1.5*IQR
            outliers = s[(s < lower) | (s > upper)]
            out_count = outliers.count()
            out_pct = outliers.count() / s.count() * 100

        try:
            vlconts = s.value_counts(ascending=False, dropna=True)
        except:
            vlconts = None
            print(f'Error for col {col}')

        stats = {
            "count": s.count(),
            "missing": s.isna().sum(),
            "missing_pct": s.isna().mean()*100,
            "nunique": s.nunique(dropna=True),
            'uniqueness_ratio': round(s.nunique(dropna=True)/len(s),2),
            "dtype": str(s.dtype),
            'is_uniform': (s.apply(type).nunique() == 1), # all rows have the same dtype
            'most_freq': vlconts.index[0] if len(vlconts)>0 else None,
            '%_most_freq': round((vlconts.iloc[0]/s.count())*100,2) if len(vlconts)>0 else None,
            'entropy': entropy(s.dropna().value_counts(normalize=True)) if len(vlconts)>0 else None,
            "Q25": Q25,
            "Q50": Q50,
            "Q75": Q75,
            "upper_wisker": upper,
            "lower_wisker": lower,
            "outlier_count": out_count,
            "outlier_pct": out_pct
        }

        return stats
    return looping_col, summary_df


@app.cell
def _(filah_enriched, looping_col, summary_df):
    looping_col(summary_df,filah_enriched)
    return


@app.cell
def _(looping_col, summary_df, trout_enriched):
    looping_col(summary_df,trout_enriched)
    return


@app.cell
def _(journo_enriched, looping_col, summary_df):
    looping_col(summary_df,journo_enriched)
    return


@app.cell
def _(alt, pd):
    def node_comparison_chart(df:pd.DataFrame):
        node_comparison_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('dataset:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y('count():Q', title='Number of Records'),
            color=alt.Color('dataset:N', title='Dataset Source'),
            column=alt.Column('type:N', title='Node Type')
        ).properties(
            width=80,
            title="Comparison of Collected Entities Across Datasets"
        )

        return node_comparison_chart



    def sentiment_chart(df:pd.DataFrame,dataset='Journalist'):
        sentiment_chart = alt.Chart(df[['industry','sentiment','source_type']].dropna()).mark_boxplot().encode(
            x=alt.X('industry:N', title='Industry Discussed'),
            y=alt.Y('sentiment:Q', title='Sentiment Score'),
            color='source_type:N'
        ).properties(
            width=400,
            title=f"Sentiment Distribution by Industry in the {dataset} Dataset"
        )
        return sentiment_chart
    return node_comparison_chart, sentiment_chart


@app.cell
def _(
    clean_filah_edges,
    clean_filah_nodes,
    clean_journo_edges,
    clean_journo_nodes,
    clean_trout_edges,
    clean_trout_nodes,
    node_comparison_chart,
    pd,
):
    all_nodes_combined = pd.concat(
        [clean_filah_nodes, clean_trout_nodes, clean_journo_nodes], 
        ignore_index=True
    )

    all_edges_combined = pd.concat(
        [clean_filah_edges, clean_trout_edges, clean_journo_edges], 
        ignore_index=True
    )

    node_comparison_chart(all_nodes_combined)
    return (all_edges_combined,)


@app.cell
def _(filah_enriched, journo_enriched, sentiment_chart, trout_enriched):
    a = sentiment_chart(journo_enriched,'Journalist')
    b = sentiment_chart(trout_enriched,'TROUT')
    c = sentiment_chart(filah_enriched,'FILAH')

    a & b & c
    return


@app.cell
def _(all_edges_combined, alt, pd):
    # Assuming you have an all_edges_combined dataframe (similar to the nodes one)
    # with a 'dataset' column containing 'TROUT' or 'FILAH'

    def action_volume_chart(df:pd.DataFrame):
        temp = df[['dataset','industry']].dropna()

        return alt.Chart(temp).mark_bar().encode(
            x=alt.X('dataset:N', title='Dataset'),
            y=alt.Y('count():Q', title='Number of Recorded Actions'),
            color=alt.Color('industry:N', title='Industry Focus'),
            tooltip=['dataset', 'industry', 'count()']
        ).properties(
            width=300,
            title="Volume of Actions by Industry in Biased Datasets"
        )

    action_volume_chart(all_edges_combined)
    return


@app.cell
def _(clean_filah_edges):
    clean_filah_edges.dropna(subset=['sentiment'])
    return


@app.cell
def _(alt, clean_filah_edges, pd):
    # Calculate average sentiment per industry in FILAH

    def sentiment_diverging_chart(df:pd.DataFrame,ref_df:str='FILAH'):
        df = clean_filah_edges[['sentiment','industry']].dropna()
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('mean(sentiment):Q', title='Average Sentiment Score'),
            y=alt.Y('industry:N', title='Industry Discussed', sort='-x'),
            color=alt.condition(
                alt.datum['mean(sentiment)'] > 0,
                alt.value("steelblue"), 
                alt.value("orange")    
            )
        ).properties(
            title=f"{ref_df} Dataset: Average Sentiment by Industry"
        )

    sentiment_diverging_chart(clean_filah_edges)
    return


@app.cell
def _(ROOT_DATA, alt, load_knowledge_graph, trout_enriched):
    import geopandas as gpd
    oceanus_map = gpd.read_file(f'{ROOT_DATA}/oceanus_map.geojson')

    # 1. Load the road map using the same function we used earlier
    road_nodes, _ = load_knowledge_graph(f"{ROOT_DATA}/road_map.json")

    # 2. Ensure IDs are strings for a safe merge
    road_nodes['id'] = road_nodes['id'].astype(str)

    # 3. Merge the coordinates into your enriched TROUT edges
    # We use an 'inner' join so it automatically filters down to ONLY the rows 
    # where the target was a physical location with coordinates.
    trout_travel_with_coords = trout_enriched.merge(
        road_nodes[['id', 'longitude', 'latitude']], 
        left_on='target', 
        right_on='id', 
        how='inner'
    )

    background = alt.Chart(oceanus_map).mark_geoshape(
        fill='lightgray',
        stroke='white'
    )


    ttp = trout_travel_with_coords[['longitude','latitude','source_name', 'target_name', 'time']]
    # 2. Plot the travel destinations using the merged dataframe
    travel_points = alt.Chart(ttp).mark_circle(size=100, opacity=0.7).encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        color=alt.Color('target_name:N', title='Destination'), # Color by the name of the place
        tooltip=['source_name', 'target_name', 'time']
    ).properties(
        title="TROUT Dataset: Official Travel Destinations"
    )

    # Layer them together
    travel_map = background + travel_points
    travel_map
    return (trout_travel_with_coords,)


@app.cell
def _(trout_travel_with_coords):
    trout_travel_with_coords
    return


if __name__ == "__main__":
    app.run()
