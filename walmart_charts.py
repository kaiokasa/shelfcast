"""Interactive charts drawn for the SQL agent (make_chart), with Plotly.

The agent only chooses a SELECT query, a chart kind and the columns: this module draws the chart. It follows a
validated palette (categorical order checked for color-vision deficiencies) and simple mark rules: recessive
hairline grid on the value axis only, no frame, 2px lines, bars with a 4px rounded end and air between them,
markers with a ring in the surface color, compact numbers (10K, 1.2M) and direct value labels on short bar charts.
It is interactive: a crosshair and tooltip on line charts, a tooltip on each bar or point, zoom and PNG export.
The chart follows the app theme (light or dark): both modes use the same hues, stepped for their surface.
"""
import math

import pandas as pd
import plotly.graph_objects as go

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7", "tooltip": "#ffffff",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835", "tooltip": "#2c2c2a",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LABELLED_BARS = 12  # direct value labels only on short bar charts (never a number on every point of a long one)
CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2}}


def compact(value):
    """10K, 1.2M, 2.4B: one style per axis, full numbers under 10,000."""
    for limit, suffix in [(1e9, "B"), (1e6, "M"), (1e4, "K")]:
        if abs(value) >= limit:
            number = value / (limit if suffix != "K" else 1e3)
            return f"{number:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:,.0f}" if abs(value) >= 100 or float(value).is_integer() else f"{value:,.2f}"


def readable(name):
    """Total_Sales -> Total Sales, sales_per_sqft -> Sales per sqft, Type_A -> Type A."""
    text = " ".join(str(name).replace("_", " ").split())
    return text[:1].upper() + text[1:]


def nice_ticks(low, high, count=5, include_zero=True):
    """Round tick values (0, 150M, 300M...) covering [low, high], labelled with compact().
    Bars start at zero (include_zero); lines and points can start near their lowest value."""
    if include_zero:
        low, high = min(low, 0), max(high, 0)
    if high == low:
        high = low + 1
    raw_step = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 1.5, 2, 2.5, 3, 5, 10) if m * magnitude >= raw_step)
    start = math.floor(low / step) * step
    ticks = [round(start + i * step, 10) for i in range(int(math.ceil((high - start) / step)) + 1)]
    if max(abs(t) for t in ticks) >= 1e4:
        return ticks, [compact(t) for t in ticks]
    # small values: the same number of decimals on the whole axis (0.0, 2.5, 5.0), never "5" next to "2.50"
    decimals = max(len(f"{t:.2f}".rstrip("0").split(".")[1]) for t in ticks)
    return ticks, [f"{t:,.{decimals}f}" for t in ticks]


def draw_chart(data, kind, x, y, title, theme="dark"):
    """An interactive Plotly figure: a line, bar, barh or scatter chart of the y column(s) by x."""
    t = THEMES[theme]
    y_columns = [y] if isinstance(y, str) else list(y)
    colors = t["series"][:len(y_columns)]  # fixed order, never cycled (at most 8 series)
    plot_data = data.copy()
    fig = go.Figure()

    if kind == "line":
        if not pd.api.types.is_numeric_dtype(plot_data[x]):
            try:
                plot_data[x] = pd.to_datetime(plot_data[x])
            except (ValueError, TypeError):
                pass
        for column, color in zip(y_columns, colors):
            fig.add_trace(go.Scatter(
                x=plot_data[x], y=plot_data[column], mode="lines", name=readable(column),
                line={"color": color, "width": 2},
                hovertemplate=f"{readable(column)}: %{{y:,.2f}}<extra></extra>"))

    elif kind in ("bar", "barh"):
        categories = plot_data[x].astype(str)  # store or department numbers are categories, not values
        labelled = len(plot_data) <= LABELLED_BARS and len(y_columns) == 1
        for column, color in zip(y_columns, colors):
            values = plot_data[column]
            bar = {"x": categories, "y": values} if kind == "bar" else {"x": values, "y": categories,
                                                                         "orientation": "h"}
            fig.add_trace(go.Bar(
                **bar, name=readable(column), marker={"color": color, "line": {"width": 0}},
                text=[compact(v) for v in values] if labelled else None, textposition="outside",
                textfont={"color": t["secondary"], "size": 12}, cliponaxis=False,
                hovertemplate=f"<b>{readable(x)} %{{{'x' if kind == 'bar' else 'y'}}}</b><br>"
                              f"{readable(column)}: %{{{'y' if kind == 'bar' else 'x'}:,.2f}}<extra></extra>"))

    elif kind == "scatter":
        for column, color in zip(y_columns, colors):
            fig.add_trace(go.Scatter(
                x=plot_data[x], y=plot_data[column], mode="markers", name=readable(column),
                marker={"color": color, "size": 9, "opacity": 0.9, "line": {"color": t["surface"], "width": 1}},
                hovertemplate=f"{readable(x)}: %{{x:,.2f}}<br>{readable(column)}: %{{y:,.2f}}<extra></extra>"))

    # value axis: round compact ticks; category / time axis: no grid
    values = pd.concat([plot_data[c] for c in y_columns])
    ticks, labels = nice_ticks(float(values.min()), float(values.max()), include_zero=kind in ("bar", "barh"))
    value_axis = {"tickvals": ticks, "ticktext": labels, "showgrid": True, "gridcolor": t["grid"], "gridwidth": 1,
                  "zeroline": False, "showline": False,
                  "title": {"text": readable(y_columns[0]) if len(y_columns) == 1 else ""}}
    category_axis = {"showgrid": kind == "scatter", "gridcolor": t["grid"], "zeroline": False, "showline": True,
                     "linecolor": t["baseline"], "linewidth": 1, "title": {"text": readable(x)}}
    if kind in ("bar", "barh"):
        category_axis["type"] = "category"
        value_axis["range"] = [ticks[0], ticks[-1]]  # the tallest bar always stays under the last gridline
    if kind == "barh":
        category_axis["autorange"] = "reversed"  # keep the query order: the #1 of a ranking on top
    if kind == "scatter" and pd.api.types.is_numeric_dtype(plot_data[x]):
        x_ticks, x_labels = nice_ticks(float(plot_data[x].min()), float(plot_data[x].max()), include_zero=False)
        category_axis.update(tickvals=x_ticks, ticktext=x_labels)
    if kind == "line":  # a vertical crosshair follows the mouse
        category_axis.update(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor=t["muted"],
                             spikethickness=1, spikedash="solid")
    axis_style = {"tickfont": {"color": t["muted"], "size": 12}, "ticks": ""}
    for axis in (value_axis, category_axis):  # axis titles in muted ink, like the ticks
        axis["title"]["font"] = {"color": t["muted"], "size": 12}
    x_axis, y_axis = (value_axis, category_axis) if kind == "barh" else (category_axis, value_axis)
    bars = len(plot_data) * len(y_columns)
    fig.update_layout(
        title={"text": title, "x": 0, "xref": "paper", "xanchor": "left",  # aligned with the plot area
               "font": {"size": 18, "color": t["ink"]}},
        font={"family": FONT, "color": t["ink"]},
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        xaxis={**axis_style, **x_axis}, yaxis={**axis_style, **y_axis},
        showlegend=len(y_columns) > 1,  # a legend for 2+ series; one series is named by the title
        # the zoom / export toolbar stands vertically on the right, so it never covers the title
        modebar={"orientation": "v", "bgcolor": "rgba(0,0,0,0)", "color": t["muted"], "activecolor": t["ink"]},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom", "font": {"color": t["secondary"]}},
        hovermode="x unified" if kind == "line" else "closest",
        hoverlabel={"bgcolor": t["tooltip"], "bordercolor": t["grid"], "font": {"color": t["ink"], "family": FONT}},
        bargap=0.45, bargroupgap=0.08, barcornerradius=4,
        height=max(320, 140 + 34 * bars) if kind == "barh" else 420,
        margin={"l": 60, "r": 70, "t": 80 if len(y_columns) > 1 else 60, "b": 50},
    )
    return fig
