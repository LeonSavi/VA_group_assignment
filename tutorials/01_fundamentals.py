# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair>=5.0.0",
#     "pandas>=2.0.0",
#     "vega_datasets",
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
    ## 📆 **Week 01 – Introduction to Marimo & Vega-Altair**

    Welcome to your first hands-on lab! In this tutorial, we'll explore the basics of **Marimo** (a reactive Python notebook) and **Vega-Altair** (a declarative visualization library). You'll learn to create your first charts, understand how data flows through visualizations, and build a foundation for the more advanced techniques we'll cover in the coming weeks.

    **Topics covered:**
    - Marimo's reactive notebook environment
    - Altair fundamentals: marks, encodings, data types (quantitative, nominal, ordinal, temporal)
    - Long vs. wide data formats and why they matter
    - Tooltips, sorting, and basic interactivity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **Index**
    - [Part I: Welcome to marimo!](#part-i-welcome-to-marimo)
    - [Part II: Hello, Altair!](#part-ii-hello-altair)
    - [Appendix: Quick Reference](#appendix-quick-reference)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part I: Welcome to marimo!**

    _This tutorial is largely based on the official marimo tutorial (try it yourself by running _`marimo tutorial intro`_ in your terminal)._
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 22)
    return (slider,)


@app.cell(hide_code=True)
def _(mo, slider):
    mo.md(rf"""
    `marimo` is a **reactive** Python notebook.

    This means that unlike traditional notebooks, marimo notebooks **run
    automatically** when you modify them or
    interact with UI elements, like this slider: 

    {slider}.

    {"🍃" * slider.value}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: disabling automatic execution": mo.md(
                rf"""
            marimo lets you disable automatic execution: in the notebook
            footer, change "On Cell Change" to "lazy".

            When the runtime is lazy, after running a cell, marimo marks its
            descendants as stale instead of automatically running them. The
            lazy runtime puts you in control over when cells are run, while
            still giving guarantees about the notebook state.
            """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **How-to**: You can create your own notebooks by entering `marimo edit` at the command line.
        """
    ).callout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reactive execution

    A marimo notebook is made up of small blocks of Python code called cells.

    marimo reads your cells and models the dependencies among them: whenever a cell that defines a global variable  is run, marimo **automatically runs** all cells that reference that variable.

    Reactivity keeps your program state and outputs in sync with your code, making for a dynamic programming environment that prevents bugs before they happen. When you modify a cell, all dependent cells automatically re-run so that what you see on screen **always reflects** the current code.
    """)
    return


@app.cell(hide_code=True)
def _(changed, mo):
    (
        mo.md(
            f"""
            **✨ Nice!** The value of `changed` is now {changed}.

            When you updated the value of the variable `changed`, marimo
            **reacted** by running this cell automatically, because this cell
            references the global variable `changed`.

            Reactivity ensures that your notebook state is always
            consistent, which is crucial for doing good science; it's also what
            enables marimo notebooks to double as tools and  apps.
            """
        )
        if changed
        else mo.md(
            """
            **🌊 See it in action.** In the next cell, change the value of the
            variable  `changed` to `True`, then click the run button.
            """
        )
    )
    return


@app.cell
def _():
    changed = False
    return (changed,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: execution order": (
                """
                The order of cells on the page has no bearing on
                the order in which cells are executed: marimo knows that a cell
                reading a variable must run after the cell that  defines it. This
                frees you to organize your code in the way that makes the most
                sense for you.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Global names must be unique.** To enable reactivity, marimo imposes a constraint on how names appear in cells: no two cells may define the same variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: encapsulation": (
                """
                By encapsulating logic in functions, classes, or Python modules,
                you can minimize the number of global variables in your notebook.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: private variables": (
                """
                Variables prefixed with an underscore are "private" to a cell, so
                they can be defined by multiple cells.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. UI elements

    Cells can output interactive UI elements. Interacting with a UI element **automatically triggers notebook execution**: when you interact with a UI element, its value is sent back to Python, and every cell that references that element is re-run.

    marimo provides a library of UI elements to choose from under `marimo.ui`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **🌊 Some UI elements.** Try interacting with the below elements.
    """)
    return


@app.cell
def _(mo):
    icon = mo.ui.dropdown(["🍃", "🌊", "✨"], value="🍃")
    return (icon,)


@app.cell
def _(icon, mo):
    repetitions = mo.ui.slider(1, 16, label=f"number of {icon.value}: ")
    return (repetitions,)


@app.cell
def _(icon, repetitions):
    icon, repetitions
    return


@app.cell
def _(icon, mo, repetitions):
    mo.md(icon.value * repetitions.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. marimo is just Python

    marimo cells parse Python (and only Python), and marimo notebooks are
    stored as pure Python files — outputs are _not_ included. There's no
    magical syntax.

    The Python files generated by marimo are:

    - easily versioned with git, yielding minimal diffs
    - legible for both humans and machines
    - formattable using your tool of choice,
    - usable as Python  scripts, with UI  elements taking their default
    values, and
    - importable by other modules (more on that in the future).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Running notebooks as apps

    marimo notebooks can double as apps. Click the app window icon in the
    bottom-right to see this notebook in "app view."

    Serve a notebook as an app with `marimo run` at the command-line.
    Of course, you can use marimo just to level-up your
    notebooking, without ever making apps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The `marimo` command-line tool

    **Creating and editing notebooks.** Use

    ```
    marimo edit
    ```

    in a terminal to start the marimo editor.

    **Running notebooks.** Use

    ```
    marimo run notebook.py
    ```

    in a terminal to run a notebook as an app.

    **Exporting notebooks.** Export to standard HTML or PDF with

    ```
    marimo export html notebook.py > notebook.html
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. marimo supports markdown

    marimo has native markdown support: cells can hold pure markdown, which renders to HTML.

    **Note**: Unlike Jupyter markdown cells, marimo markdown cells are stored just as Python files — markdown included via `mo.md(...)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. The marimo editor

    Here are **some tips** to help you get started with the marimo editor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tips = {
        "Saving": (
            """
            **Saving**

            - _Name_ your app using the box at the top of the screen, or
              with `Ctrl/Cmd+s`. You can also create a named app at the
              command line, e.g., `marimo edit app_name.py`.

            - _Save_ by clicking the save icon on the bottom right, or by
              inputting `Ctrl/Cmd+s`. By default marimo is configured
              to autosave.
            """
        ),
        "Running": (
            """
            1. _Run a cell_ by clicking the play ( ▷ ) button on the top
            right of a cell, or by inputting `Ctrl/Cmd+Enter`.

            2. _Run a stale cell_  by clicking the yellow run button on the
            right of the cell, or by inputting `Ctrl/Cmd+Enter`. A cell is
            stale when its code has been modified but not run.

            3. _Run all stale cells_ by clicking the play ( ▷ ) button on
            the bottom right of the screen, or input `Ctrl/Cmd+Shift+r`.
            """
        ),
        "Console Output": (
            """
            Console output (e.g., `print()` statements) is shown below a
            cell.
            """
        ),
        "Creating, Moving, and Deleting Cells": (
            """
            1. _Create_ a new cell above or below a given one by clicking
                the plus button to the left of the cell, which appears on
                mouse hover.

            2. _Move_ a cell up or down by dragging on the handle to the
                right of the cell, which appears on mouse hover.

            3. _Delete_ a cell by clicking the trash bin icon. Bring it
                back by clicking the undo button on the bottom right of the
                screen, or with `Ctrl/Cmd+Shift+z`.
            """
        ),
        "Disabling Automatic Execution": (
            """
            Via the notebook settings (gear icon) or footer panel, you
            can disable automatic execution. This is helpful when
            working with expensive notebooks or notebooks that have
            side-effects like database transactions.
            """
        ),
        "Disabling Cells": (
            """
            You can disable a cell via the cell context menu.
            marimo will never run a disabled cell or any cells that depend on it.
            This can help prevent accidental execution of expensive computations
            when editing a notebook.
            """
        ),
        "Code Folding": (
            """
            You can collapse or fold the code in a cell by clicking the arrow
            icons in the line number column to the left, or by using keyboard
            shortcuts.

            Use the command palette (`Ctrl/Cmd+k`) or a keyboard shortcut to
            quickly fold or unfold all cells.
            """
        ),
        "Code Formatting": (
            """
            If you have [ruff](https://github.com/astral-sh/ruff) installed,
            you can format a cell with the keyboard shortcut `Ctrl/Cmd+b`.
            """
        ),
        "Command Palette": (
            """
            Use `Ctrl/Cmd+k` to open the command palette.
            """
        ),
        "Keyboard Shortcuts": (
            """
            Open the notebook menu (top-right) or input `Ctrl/Cmd+Shift+h` to
            view a list of all keyboard shortcuts.
            """
        ),
        "Configuration": (
            """
           Configure the editor by clicking the gears icon near the top-right
           of the screen.
           """
        ),
        "Exit & Shutdown": (
            """
           You can leave Marimo & shut down the server by clicking the
           circled X at the top right of the screen and responding
           to the prompt.

           :floppy_disk: _Be sure to save your work first!_
           """
        ),
    }

    mo.accordion(tips)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Shortcuts**

    | Action | Shortcut |
    |--------|----------|
    | Run cell | `Ctrl/Cmd + Enter` |
    | Create cell below | `Ctrl/Cmd + Shift + Enter` |
    | Delete cell | `Ctrl/Cmd + Shift + Backspace` |
    | Command palette | `Ctrl/Cmd + K` |
    | Convert to markdown | `Ctrl/Cmd + Shift + M` |
    | Hide cell code | `Ctrl/Cmd + H` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Fun fact
    The name "marimo" is a reference to a type of algae that, under the right conditions, clumps together to form a small sphere called a "[marimo moss ball](https://www.mossball.com/)". Made of just strands of algae, these beloved assemblages are greater than the sum of their parts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Add a new cell below and try:
    ```python
        name = mo.ui.text(placeholder="Your name")
        name
    ```
        Then in another cell:
    ```python
        mo.md(f"Hello, **{name.value or 'stranger'}**! 👋")
    ```
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources
    1 **[Hello, Python!](https://www.kaggle.com/code/amykzhang/digi118-h25-w0-hello-python)** — A quick refresher on Python basics you'll need for Altair (by [Amy Zhang](https://vis.uib.no/team/amy-zhang/))

    2 **[Marimo Introduction](https://www.youtube.com/watch?v=oOqRctpii6I)** — Why reactive notebooks? A 30-min talk from PyData Amsterdam 2025 (by [Vincent Warmerdam](https://koaning.io/about/))

    3 **[Marimo Getting Started](https://docs.marimo.io/getting_started/)** — Official documentation for installation and first steps
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Part II: Hello, Altair!**

    **What is Altair?**

    [**Altair**](https://altair-viz.github.io/) is a **declarative statistical visualization library** for Python that offers a powerful and concise grammar for building a wide range of **interactive graphics**. It is particularly suited for visualizing **tabular datasets** (think spreadsheets or database tables), including geographic data with latitude/longitude coordinates.

    **Declarative vs. Imperative**

    The term *declarative* means you provide a **high-level specification** of what you want the visualization to include — data, graphical marks, and encoding channels — rather than specifying *how* to draw it with loops and low-level commands (such as Matlpotlib). You **declare links between data fields and visual channels** (x-axis, y-axis, color, size), and Altair handles the rest automatically. Using Altair, you have a friendly Python API for generating visual specifications in interactive environments like notebooks or in a regular Python file that is rendered in the web browser.

    **Origins**

    Altair is built on top of **Vega-Lite**, a high-level JSON specification for interactive visualizations, which itself compiles to **Vega** and ultimately renders via **D3.js** in the browser. Conceptually, this stack is grounded in Leland Wilkinson's *The Grammar of Graphics* (1999) –the same theoretical foundation behind `ggplot2` in R– which describes visualizations as compositions of independent components: data, marks, encodings, scales, and coordinates.

    **The underlying stack**

    | Layer | What it is | Role |
    |-------|-----------|------|
    | **Grammar of Graphics** | Theory (Wilkinson, 1999) | Conceptual foundation |
    | **Vega** | Low-level JSON spec | Fine-grained control |
    | **Vega-Lite** | High-level JSON spec | Concise, practical grammar |
    | **Altair** | Python library | Generates Vega-Lite JSON |
    | **D3.js** | JavaScript library | Renders to SVG/Canvas |

    **How it works**

    ```
    Altair (Python) → Vega-Lite (JSON) → Vega (JSON) → D3.js → Browser
    ```

    When you write `alt.Chart(...).mark_bar().encode(...)`, Altair builds a Vega-Lite JSON spec behind the scenes. The browser's Vega runtime then compiles and renders it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

    1 Explore the [Altair Example Gallery](https://altair-viz.github.io/gallery/index.html) to see the full range of visualizations, from basic charts to advanced interactive graphics

    2 [Visualization Analysis and Design, 2021](https://www.youtube.com/playlist?list=PLT4XLHmqHJBeB5LwmRmo6ln-m7K3lGvrk) — Video lectures on core visualization concepts: marks, channels, encodings, and more (by [Tamara Munzner](https://www.cs.ubc.ca/~tmm/))
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Getting started (libraries)

    **[Pandas](https://pandas.pydata.org/)** is a Python library for data analysis and visualization, frequently used together with Altair. It mostly handles data manipulation (e.g., reading files, filtering, transforming) and provides the DataFrame structure that Altair visualizations are built on. The typical workflow is: **prepare data with Pandas, then visualize with Altair**.

    **[vega_datasets](https://github.com/vega/vega-datasets)** provides example datasets (`cars`, `iris`, `stocks`, etc.) commonly used in visualization tutorials. Data can be loaded as a URL or directly as a DataFrame.
    """)
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd
    from vega_datasets import data as vega_data
    return alt, pd, vega_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 🔗 Resources

    1 [Hello, Pandas!](https://www.kaggle.com/code/amykzhang/digi118-h25-w0-hello-pandas) by [Amy Zhang](https://vis.uib.no/team/amy-zhang/)

    1 [Hello, Altair!](https://www.kaggle.com/code/amykzhang/digi118-h25-w1-hello-altair) by [Amy Zhang](https://vis.uib.no/team/amy-zhang/)
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data sources

    **Altair** visualizations can be built from different data sources:
    - as a **pandas DataFrame**,
    - as a **url string pointing to a json or csv** formatted text file,
    - as a **geopandas GeoDataFrame**, Shapely Geometries, GeoJSON Objects

    We can create a dataframe using pandas `pd.DataFrame`, read in a CSV, TSV, JSON file with e.g., `pd.read_csv`, or import data samples from the [vega datasets repository](https://github.com/vega/vega-datasets/tree/main/data) on GitHub with details about each dataset in their [data package readme](https://github.com/vega/vega-datasets/blob/main/datapackage.md#resources).

    For example:
    """)
    return


@app.cell
def _(pd):
    df_dict = {
        'Country': [
            'Germany', 'France', 'Italy', 'Spain', 'Poland',
            'Romania', 'Netherlands', 'Belgium', 'Czech Republic', 'Greece'
        ],
        'Population (Millions)': [
            83.155, 67.657, 59.236, 47.399, 37.840,
            19.202, 17.475, 11.555, 10.702, 10.679
        ]
    }

    df = pd.DataFrame(df_dict)
    df #.head() # to only preview first five rows of dataframe
    return (df,)


@app.cell
def _(pd):
    # Load data via CSV file path
    df_url = "https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"
    df_penguins = pd.read_csv(df_url)
    df_penguins.head()
    return (df_penguins,)


@app.cell
def _(pd, vega_data):
    df_stocks = pd.read_csv(vega_data.stocks.url)
    df_stocks.head()

    # OR load data with df = data.stocks.url if you don't need to preprocess with pandas
    # vega_data.stocks()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The building blocks of Altair

    Every Altair visualization is built from **four core components**:

    ```
        alt.Chart(data).mark_*().encode(...).properties(...)
    ```

    Let's break this down.

    ### 3.1. Chart: "What data am I visualizing?"

    The `Chart` object is the foundation. It receives your data (typically a pandas DataFrame):

    ```python
        alt.Chart(data)
    ```

    At this point, Altair knows *what* data you have, but not *how* to display it.

    ### 3.2. Mark: "What shape should represent my data?"

    Marks are the **visual elements** that appear on screen. Each row in your data becomes one mark.

        | Method | Mark | Best for |
        |--------|------|----------|
        | `mark_point()` | ● | Scatter plots |
        | `mark_bar()` | ▮ | Bar charts, histograms |
        | `mark_line()` | ─ | Trends over time |
        | `mark_circle()` | ● | Scatter plots (filled) |
        | `mark_area()` | ▤ | Area charts |
        | `mark_rect()` | ■ | Heatmaps |
        | `mark_boxplot()` | ⊏⊐ | Distribution summaries |

    ```python
        alt.Chart(data).mark_bar()  # Now I know to draw bars!
    ```

    ### 3.3. Encode: *"How should data map to visual properties?"*

    Encoding is the **heart of visualization**. It connects columns in your data to visual channels:

        | Channel | Controls | Example |
        |---------|----------|---------|
        | `x` | Horizontal position | `x='Age:Q'` |
        | `y` | Vertical position | `y='Salary:Q'` |
        | `color` | Fill color | `color='Department:N'` |
        | `size` | Mark size | `size='Sales:Q'` |
        | `shape` | Mark shape | `shape='Category:N'` |
        | `opacity` | Transparency | `opacity='Confidence:Q'` |
        | `tooltip` | Hover text | `tooltip=['Name', 'Value']` |

    ```python
        alt.Chart(data).mark_bar().encode(
            x='Category:N',    # Categories on x-axis
            y='Value:Q',       # Values determine bar height
            color='Group:N'    # Different colors per group
        )
    ```

    ### 3.4. Properties: "How should the chart look?"

    Properties configure **chart-level settings** — things about the visualization itself, not the data:

        | Property | Controls | Example |
        |---------|----------|---------|
        | `title` | Chart title | `title='Sales by Region'` |
        | `width` | width in pixels | `width=400` |
        | `height` | Height in pixels | `height=300` |

    ### 3.5. Putting it all together

    Think of it as a sentence:
    > "Take this DATA, draw it as MARKS, and map these columns to visual ENCODINGS."

    ```python
        alt.Chart(data)      # 1. Here's my data
           .mark_bar()       # 2. Draw bars
           .encode(          # 3. Map columns to visuals
               x='country',
               y='population',
               color='region'
           )
           .properties(      # 4. Style the chart
               title='European Population',
               width=500
           )
    ```

    **You describe WHAT you want, not HOW to draw it**. Altair figures out the details (axes, legends, colors) automatically.
    """)
    return


@app.cell
def _(alt, df):
    # Our Chart object with marks and channels

    # Creating the chart, with mark_bar to define a bar chart
    alt.Chart(df).mark_bar().encode(
        alt.X('Country:N'), # Categories in X axis
        alt.Y('Population (Millions):Q'), # Length of the bars in Y axis
    ).properties(title='Population') # Title of the chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that, although we used the command `mark_bar`, what we are encoding is the length parameter. Then, we call the method `encode` that is used to define how the marks appear on screen and what parameters are used to create them. In this case, we are saying that the marks have an **X position** (`alt.X`) determined by the name of the country, and a **Y position** (`alt.Y`) that depends on the population.

    Altair automatically infers that, since we are plotting bar charts, the population needs to be encoded as the length of the bars, and that we need to create several bars, one next to each other (as many as different values we have for the `"Country"` variable"). We can then add a **title** for the chart using the `properties` method with the parameter `title`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Data types
    Altair supports different data types that can be specified explicitly (verbosely or shorthand) in the encode method of the chart.

    *   **Quantitative:** continuous attribute
    *   **Ordinal**: discrete ordered attribute
    *   **Nominal:** discrete unordered attribute
    *   **Temporal:** a time or date attribute
    """)
    return


@app.cell
def _(alt, vega_data):
    # Verbose within method
    cars_url = vega_data.cars.url

    alt.Chart(cars_url).mark_point().encode(
        alt.X('Acceleration', type='quantitative'),
        alt.Y('Miles_per_Gallon', type='quantitative'),
        alt.Color('Origin', type='nominal')
    )
    return (cars_url,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is equivalent to:
    """)
    return


@app.cell
def _(alt, cars_url):
    # Shorthand notation
    alt.Chart(cars_url).mark_point().encode(
        alt.X('Acceleration:Q'),
        alt.Y('Miles_per_Gallon:Q'),
        alt.Color('Origin:N')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    What happens when we change the notation for `Origin` from nominal N to ordinal O in the chart above? Try it and observe the difference in the legend.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Encoding channels

    In the first example, we saw two possible parameters (encoding channels) for the `encode` method: `alt.X` (horizontal position) and `alt.Y` (vertical position). But there are many more parameters available to change the position, mark properities, and chart text ([full list is in the Altair documentation](https://altair-viz.github.io/user_guide/encodings/channels.html#user-guide-encoding-channels)).

    **What if we want each bar to have a different color?** To do so, we can simply add another encoding: `alt.Color('Country')` maps the country name to the bar's fill color.
    """)
    return


@app.cell
def _(alt, df):
    # Bar charts with colored categories

    alt.Chart(df).mark_bar().encode(
        alt.X('Country:N'),
        alt.Y('Population (Millions):Q'),
        alt.Color('Country')  
    ).properties(title = 'Population')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this visualization, we assigned a different color to each country — **but does that add information?** The x-axis already tells us which bar is which. Color is most effective when it encodes something *meaningful*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Do the same visualization where only countries with **population above 50** are colored differently.

    **Hint**: create a new column in the DataFrame that indicates whether the population is above 50, then use that column for the color encoding.
    ///
    """)
    return


@app.cell
def _():
    # YOUR CODE HERE

    # Step 1: Create the data with a new column indicating population > 50
    # df['above_50'] = ...

    # Step 2: Use that column for the color encoding
    # alt.Chart(df).mark_bar().encode(
    # ...
    # ).properties(title='Population (highlighted: >50M)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sorting

    By default, Altair **sorts categorical data alphabetically**. But often we want to sort by a different criterion — for example, sorting bars by their values to create a more readable chart.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1. Sorting by value

    Use the `sort` parameter with `-x` or `-y` to sort by the other axis in descending order:
    """)
    return


@app.cell
def _(alt, df):
    # Default alphabetical sorting
    chart_unsorted = alt.Chart(df).mark_bar().encode(
        x='Country:N',
        y='Population (Millions):Q'
    ).properties(title='Alphabetical order (default)', width=300)

    # Sorted by population (descending)
    chart_sorted = alt.Chart(df).mark_bar().encode(
        x=alt.X('Country:N', sort='-y'), # Sort by y-value descending
        y='Population (Millions):Q'
    ).properties(title='Sorted by population', width=300)

    chart_unsorted | chart_sorted
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2. Horizontal bar charts with sorting

    For **long category names**, horizontal bar charts are often more readable. Use `sort='-x'` when the values are on the x-axis:
    """)
    return


@app.cell
def _(alt, df):
    # Horizontal bar chart sorted by value
    alt.Chart(df).mark_bar().encode(
        x='Population (Millions):Q',
        y=alt.Y('Country:N', sort='-x')  # Sort by x-value descending
    ).properties(
        title='EU Countries by population',
        height=250
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3. Custom sort order

    You can also specify an explicit order using a list:
    """)
    return


@app.cell
def _(alt, df_penguins):
    # Custom sort order for species
    custom_order = ['Chinstrap', 'Adelie', 'Gentoo']

    alt.Chart(df_penguins).mark_bar().encode(
        x=alt.X('species:N', sort=custom_order),
        y='count():Q'
    ).properties(title='Penguin counts (custom order)', width=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Using the `penguins` dataset (loaded earlier as `df_penguins`), create a horizontal bar chart showing the **mean body mass** for each penguin species, sorted from highest to lowest mass.

    **Hints:**
    - Use `mark_bar()` with `x='mean(body_mass_g):Q'`
    - Use `alt.Y('species:N', sort='-x')` to sort by the x-value
    ///
    """)
    return


@app.cell
def _(df_penguins):
    # YOUR CODE HERE

    # alt.Chart(df_penguins)...

    # Preview the data to help you get started
    df_penguins[['species', 'body_mass_g']].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Long vs. wide format

    Understanding the difference between **long** (tidy) and **wide** data formats is essential for effective visualization with Altair. Altair works best with **long-format data**, where each row represents a single observation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.1. Wide format

    In **wide format**, each variable has its own column. This is common in spreadsheets and is often how we initially collect data.
    """)
    return


@app.cell
def _(pd):
    # Example: Stock prices in WIDE format
    df_stocks_wide = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
        'AAPL': [300, 320, 280, 290],
        'GOOG': [1400, 1450, 1350, 1400],
        'MSFT': [160, 170, 150, 165]
    })
    df_stocks_wide
    return (df_stocks_wide,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.2. Long format (tidy)

    In **long format** (also called "tidy" format), each row represents a single observation, and there's a column that identifies the category/group.
    """)
    return


@app.cell
def _(pd):
    # Same data in LONG format
    df_stocks_long = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'] * 3,
        'symbol': ['AAPL'] * 4 + ['GOOG'] * 4 + ['MSFT'] * 4,
        'price': [300, 320, 280, 290, 1400, 1450, 1350, 1400, 160, 170, 150, 165]
    })
    df_stocks_long
    return (df_stocks_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.3. Why does Altair prefer long format?

    In long format, we can easily map the `symbol` column to visual properties like **color**, and the `price` column to the **y-axis**. Let's see the difference:
    """)
    return


@app.cell
def _(alt, df_stocks_long):
    # With LONG format: easy to encode color by symbol
    alt.Chart(df_stocks_long).mark_line(point=True).encode(
        x='date:T',
        y='price:Q',
        color='symbol:N'
    ).properties(
        title='Stock prices (long format)',
        width=400
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.4. Converting wide to long with `pd.melt()`

    If your data is in wide format, you can convert it to long format using pandas' `melt()` function (see the [documentation](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)):
    """)
    return


@app.cell
def _(df_stocks_wide, pd):
    # Convert wide to long
    df_melted = pd.melt(
        df_stocks_wide,
        id_vars=['date'], # Columns to keep as identifiers
        value_vars=['AAPL', 'GOOG', 'MSFT'],  # Columns to "melt"
        var_name='symbol', # Name for the new category column
        value_name='price' # Name for the new value column
    )
    df_melted
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.5. Altair's `fold` transform

    Alternatively, Altair provides a built-in **`fold` transform** that converts wide data to long format on-the-fly, without modifying your original DataFrame:
    """)
    return


@app.cell
def _(alt, df_stocks_wide):
    # Using fold transform directly in Altair
    alt.Chart(df_stocks_wide).transform_fold(
        fold=['AAPL', 'GOOG', 'MSFT'],  # Columns to fold
        as_=['symbol', 'price'] # Names for the new columns
    ).mark_line(point=True).encode(
        x='date:T',
        y='price:Q',
        color='symbol:N'
    ).properties(
        title='Stock prices (using fold transform)',
        width=400
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    The `penguins` dataset below has measurements for `bill_length_mm` and `bill_depth_mm`. Create a chart that shows **the mean of both measurements** for each species using:

    1. First, `pd.melt()` to reshape the data
    2. Then, a bar chart with species on the x-axis, the (mean) measurement value on y, and color by measurement type

    **Hint**: check out the `xOffset` encoding to place bars side by side.
    ///
    """)
    return


@app.cell
def _(df_penguins):
    # YOUR CODE HERE

    # Step 1: Melt the penguins data
    # df_penguins_long = pd.melt(
    #     df_penguins,
    #     id_vars=...,
    #     value_vars=...,
    #     var_name=...,
    #     value_name=...
    # )

    # Step 2: Create the chart
    # alt.Chart(df_penguins_long)...

    # Preview the data to help you get started
    df_penguins[['species', 'bill_length_mm', 'bill_depth_mm']].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Tooltips

    Tooltips are a simple but powerful way to add **interactivity** to your charts. They display additional information when users hover over data points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.1. Basic tooltips

    The simplest way to add tooltips is to include a `tooltip` encoding with a list of fields:
    """)
    return


@app.cell
def _(alt, cars_url):
    # Basic tooltip example

    alt.Chart(cars_url).mark_circle(size=60).encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        tooltip=['Name:N', 'Origin:N', 'Horsepower:Q', 'Miles_per_Gallon:Q']
    ).properties(
        title='Car fuel efficiency vs. horsepower',
        width=500
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8.2. Customizing tooltips

    For more control, use `alt.Tooltip()` to customize the title, format numbers, and more:
    """)
    return


@app.cell
def _(alt, cars_url):
    # Customized tooltips with formatting
    alt.Chart(cars_url).mark_circle(size=60).encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        tooltip=[
            alt.Tooltip('Name:N', title='Car Model'),
            alt.Tooltip('Origin:N', title='Country of Origin'),
            alt.Tooltip('Horsepower:Q', title='Horsepower (hp)', format='.0f'),
            alt.Tooltip('Miles_per_Gallon:Q', title='Fuel Efficiency (mpg)', format='.1f'),
            alt.Tooltip('Year:O', title='Model Year')
        ]
    ).properties(
        title='Car fuel efficiency vs. horsepower',
        width=500
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Your turn!
    Using the `penguins` dataset, create a scatter plot of `flipper_length_mm` vs `body_mass_g`. Add tooltips that show: `species`, `island`, `body mass` (formatted with a thousands separator), `flipper length` (formatted as an integer).

    **Hint**: use `format=','` for thousands separator and `format='.0f'` for integers.
    ///
    """)
    return


@app.cell
def _(df_penguins):
    # YOUR CODE HERE
    # alt.Chart(df_penguins).mark_circle()...

    # Preview the data
    df_penguins.head() 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Zoom
    In Altair, we can zoom and pan along a chart by simply adding the method `.interactive()` to the end of our `Chart` object. With just one line of code, we can immediately give our charts some interactivity. How neat! Interacting with the chart is similar to using a tool like Google Maps. Try these for yourself:

    - To **zoom**, scroll in and out of the chart.
    - To **pan**, click and drag inside the chart.
    """)
    return


@app.cell
def _(alt, cars_url):
    # Draw chart with zoom and pan interactivity
    alt.Chart(cars_url).mark_circle(size=60).encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        tooltip=['Name:N', 'Origin:N', 'Horsepower:Q', 'Miles_per_Gallon:Q']
    ).properties(
        title='Car fuel efficiency vs. horsepower',
        width=500
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Interactive legend

    An interactive legend allows users to **filter data by clicking on legend items**, a simple way to explore subsets of your data. Click on a category to highlight it; click again to reset. Altair supports many forms of interactivity _(brushing, linked views, cross-filtering, and more)_ — we'll explore these in the next notebooks.
    """)
    return


@app.cell
def _(alt, cars_url):
    # Create a selection bound to the legend
    legend_selection = alt.selection_point(fields=['Origin'], bind='legend')

    # The selection parameter allows us to click on the legend, arguments:
    # Selection targets: what data features (items, attributes, etc.) are targeted by the selection
    # Selection binding: if we want to make a selection using an external chart element or widget (e.g., legend, radio button, checkbox, slider, dropdown), then we have to bind the selection to that widget

    alt.Chart(cars_url).mark_circle(size=60).encode(
        x='Horsepower:Q',
        y='Miles_per_Gallon:Q',
        color='Origin:N',
        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.1)),
        tooltip=['Name:N', 'Origin:N', 'Horsepower:Q', 'Miles_per_Gallon:Q']
    ).properties(
        title='Click on the legend to filter',
        width=500
    ).add_params(legend_selection) # connects the selection parameter to the chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Saving and exporting charts
    Once you have visualized your data, perhaps you would like to publish it somewhere on the web or save it for yourself.

    Typically, we assign our Chart object to a new variable so we can save what is in the variable. Then, we generate a stand-alone HTML document for it with help from the `Chart.save()` method:

    ```python
    # Assign Chart object to a variable called mychart
    mychart = alt.Chart(source, title=chart_title).mark_circle().encode(
        alt.X("Country:N"),
        alt.Y("Population:Q")
    )

    # Export my_chart as a html file with a name of your choice (no extra dependencies)
    mychart.save('mychart_name.html')
    ```

    **Additional export formats**

    To export charts as **PNG**, **SVG**, or **PDF**, you need to install the `vl-convert-python` package:
    ```bash
    pip install vl-convert-python
    ```

    Once installed, you can save in multiple formats:
    ```python
    # Static image formats
    mychart.save('mychart.png')
    mychart.save('mychart.svg')
    mychart.save('mychart.pdf')

    # Vega-Lite JSON specification
    mychart.save('mychart.json')
    ```

    You can also adjust the resolution for PNG export:
    ```python
    mychart.save('mychart.png', scale_factor=2)  # Higher resolution
    ```

    **Note:** In Marimo (and most browsers), you can also right-click on a rendered chart and select "Save image as..." for a quick export.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # **Appendix: Quick Reference**
    This reference includes elements (such as aggregations and selections) that we haven't covered yet — we'll explore these in upcoming labs. Keep this as a handy cheat sheet throughout the course!

    ### Mark Types
    ```python
    .mark_point()      # Scatter plot
    .mark_circle()     # Circles (filled points)
    .mark_bar()        # Bar chart
    .mark_line()       # Line chart
    .mark_area()       # Area chart
    .mark_rect()       # Rectangles (heatmaps)
    .mark_text()       # Text labels
    .mark_boxplot()    # Box plot
    ```

    ### Common Encodings
    ```python
    x='column:Q'           # X-axis
    y='column:Q'           # Y-axis
    color='column:N'       # Color
    size='column:Q'        # Size
    shape='column:N'       # Shape
    opacity='column:Q'     # Transparency
    tooltip=[...]          # Hover info
    ```

    ### Aggregations
    ```python
    'count():Q'            # Count rows
    'mean(column):Q'       # Average
    'sum(column):Q'        # Total
    'min(column):Q'        # Minimum
    'max(column):Q'        # Maximum
    'median(column):Q'     # Median
    ```

    ### Selections
    ```python
    alt.selection_interval()   # Brush selection
    alt.selection_point()      # Click selection
    .add_params(selection)     # Add to chart
    .transform_filter(sel)     # Filter by selection
    alt.condition(sel, ...)    # Conditional encoding
    ```

    ### Data Types
    ```python
    :Q  # Quantitative (continuous numbers)
    :N  # Nominal (unordered categories)
    :O  # Ordinal (ordered categories)
    :T  # Temporal (dates/times)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **What's next?**

    You've learned the fundamentals of Altair: creating basic charts, encoding data, and adding some interactivity. In the next notebook, we'll apply these skills to a real-world dataset and explore more advanced techniques like brushing, linked views, and layered charts.

    For extra practice on **chart customization** (colors, axes, scales), check out [Right Chart for the Job](https://www.kaggle.com/code/amykzhang/digi118-h25-w2-right-chart-for-the-job), by Amy Zhang.
    """)
    return


if __name__ == "__main__":
    app.run()
