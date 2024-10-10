import holoviews as hv
import holoviews.operation.datashader as hd
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import HoverTool

from .. import config
from ..plot_helpers import center_plot_title
from ..plot_helpers import customise_ticks
from ..plot_helpers import hover_points
from ..plot_helpers import make_hist_on_axis
from ..plot_helpers import selected_hist
from ..plot_helpers import filter_points


def make_muts_panel(log_y, tsm):
    plot_width = 1000
    muts_df = tsm.mutations_df
    y_dim = "time"
    if log_y:
        muts_df["log_time"] = np.log10(1 + tsm.mutations_df["time"])
        y_dim = "log_time"

    hover_tool = HoverTool(tooltips=[
        ("ID", "@id"),
        ("parents", "@num_parents"),
        ("descendants", "@num_descendants"),
        ("inheritors", "@num_inheritors"),
    ])

    points = muts_df.hvplot.scatter(
        x="position",
        y=y_dim,
        hover_cols=["id", "num_parents", "num_descendants", "num_inheritors"],
    ).opts(
        width=plot_width,
        height=config.PLOT_HEIGHT,
        ## For some reason they are zero so come out transparent!
        # color="num_inheritors",
        # alpha="num_inheritors",
        # cmap="BuGn",
        # colorbar_position="left",
        # clabel="inheritors",
        tools=[hover_tool, "tap"],
    )

    range_stream = hv.streams.RangeXY(source=points)
    streams = [range_stream]
    
    filtered = points.apply(filter_points, streams=streams)
    hover = filtered.apply(hover_points, threshold=config.THRESHOLD)
    shaded = hd.datashade(
        points,
        width=400,
        height=400,
        streams=streams,
        cmap=config.PLOT_COLOURS[1:],
    )

    main = (shaded * hover).opts(
        hv.opts.Points(tools=["hover"], alpha=0.1, hover_alpha=0.2, size=10),
    )

    time_hist = hv.DynamicMap(
        make_hist_on_axis(dimension=y_dim, points=points), streams=streams
    )
    site_hist = hv.DynamicMap(
        make_hist_on_axis(dimension="position", points=points),
        streams=streams,
    )

    breakpoints = tsm.ts.breakpoints(as_array=True)
    bp_df = pd.DataFrame(
        {
            "position": breakpoints,
            "x1": breakpoints,
            "y0": tsm.mutations_df[y_dim].min(),
            "y1": tsm.mutations_df[y_dim].max(),
        }
    )
    trees_hist = hv.DynamicMap(selected_hist(bp_df), streams=[range_stream])
    trees_hist.opts(
        width=config.PLOT_WIDTH,
        height=100,
        hooks=[customise_ticks],
        xlabel="tree density",
    )

    layout = (main << time_hist << site_hist) + trees_hist

    if config.ANNOTATIONS_FILE is not None:
        genes_df = tsm.genes_df(config.ANNOTATIONS_FILE)
        annot_track = make_annotation_plot(tsm, genes_df)
        layout += annot_track

    selection_stream = hv.streams.Selection1D(source=points)

    def get_mut_data(x_range, y_range, index):
        if x_range and y_range and index:
            filtered_data = muts_df[
                (muts_df["position"] >= x_range[0])
                & (muts_df["position"] <= x_range[1])
                & (muts_df[y_dim] >= y_range[0])
                & (muts_df[y_dim] <= y_range[1])
            ]
            filtered_data.reset_index(drop=True, inplace=True)
            mut_data = filtered_data.loc[index[0]]
            return mut_data

    def update_pop_freq_plot(x_range, y_range, index):
        if not index:
            return hv.Bars([], "population", "frequency").opts(
                title="Population frequencies",
                default_tools=[],
                tools=["hover"],
                hooks=[center_plot_title],
            )

        mut_data = get_mut_data(x_range, y_range, index)
        pops = [col for col in mut_data.index if "pop_" in col]

        if pops:
            df = pd.DataFrame(
                {
                    "population": [
                        pop.replace("pop_", "").replace("_freq", "") for pop in pops
                    ],
                    "frequency": [mut_data[col] for col in pops],
                }
            )
            df = df[df["frequency"] > 0]

            bars = hv.Bars(df, "population", "frequency").opts(
                framewise=True,
                title=f"Mutation {mut_data['id']}",
                ylim=(0, max(df["frequency"]) * 1.1),
                xrotation=45,
                tools=["hover"],
                default_tools=[],
                yticks=3,
                yformatter="%.3f",
                hooks=[center_plot_title],
            )
            return bars
        else:
            return hv.Bars([], "population", "frequency").opts(
                title="Population frequencies", 
                default_tools=[],
                tools=["hover"],
                hooks=[center_plot_title],
            )

    def update_mut_info_table(x_range, y_range, index):
        if not index:
            float_panel.visible = False
            return hv.Table([], kdims=["mutation"], vdims=["value"])
        float_panel.visible = True
        mut_data = get_mut_data(x_range, y_range, index)
        pops = [col for col in mut_data.index if "pop_" in col]
        mut_data = mut_data.drop(pops)
        mut_data["time"] = mut_data["time"].round(2)
        if "log_time" in mut_data:
            mut_data["log_time"] = mut_data["log_time"].round(2)
        return hv.Table(mut_data.items(), kdims=["mutation"], vdims=["value"])

    pop_data_dynamic = hv.DynamicMap(
        update_pop_freq_plot, streams=[range_stream, selection_stream]
    )
    pop_data_dynamic.opts(align=("center"))
    mut_info_table_dynamic = hv.DynamicMap(
        update_mut_info_table, streams=[range_stream, selection_stream]
    )
    tap_widgets_layout = (pop_data_dynamic + mut_info_table_dynamic).cols(1)
    float_panel = pn.layout.FloatPanel(
        pn.Column(
            tap_widgets_layout, 
            align="center",
        ),
        name="Mutation information",
        position="left-top",
        config = {
            "contentSize": {"width": 450, "height": 660},
            "headerControls": {"close": "remove", "maximize": "remove", "normalize": "remove", "minimize": "remove"}
        },
        visible=False # Initially not shown
    )
    return pn.Column(
        float_panel,
        layout.opts(shared_axes=True).cols(1),
    )


def make_annotation_plot(tsm, genes_df):
    min_y = tsm.mutations_df["time"].min()
    max_y = tsm.mutations_df["time"].max()
    genes_df["y0"] = min_y + 0.3 * (max_y - min_y)
    genes_df["y1"] = max_y - 0.3 * (max_y - min_y)
    genes_rects = hv.Rectangles(
        genes_df, kdims=["position", "y0", "end", "y1"], vdims=["name", "id", "strand"]
    )
    hover_tool = HoverTool(
        tooltips=[
            ("gene name", "@name"),
            ("ensembl id", "@id"),
            ("strand", "@strand"),
        ]
    )
    genes_rects.opts(
        ylabel=None,
        shared_axes=True,
        hooks=[customise_ticks],
        width=config.PLOT_WIDTH,
        height=100,
        yaxis=None,
        xlabel="genes",
        tools=[hover_tool],
    )

    genes_rects = (
        hv.HLine(min_y + (max_y - min_y) / 2).opts(color="black", line_width=0.7)
        * genes_rects
    )
    return genes_rects


class MutationsPage:
    key = "mutations"
    title = "Mutations"

    def __init__(self, tsm):
        self.tsm = tsm
        log_y_checkbox = pn.widgets.Checkbox(name="Log y-axis", value=False)
        muts_panel = pn.Column(pn.bind(make_muts_panel, log_y=log_y_checkbox, tsm=tsm))
        plot_options = pn.Column(
            pn.pane.Markdown("# Mutations"),
            log_y_checkbox,
        )
        self.content = muts_panel
        self.sidebar = plot_options
