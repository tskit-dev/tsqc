import pathlib
import time
import traceback

import click
import daiquiri
import holoviews as hv
import tskit
import tszip
from holoviews import opts

# Need to import daiquiri and set up logging before importing panel
# and bokeh, so we can set logging up correctly
daiquiri.setup(level="WARN")  # noqa
import panel as pn  # noqa

from . import model  # noqa
from . import pages  # noqa
from . import config  # noqa

logger = daiquiri.getLogger("tsqc")


def load_data(path, calc_population_frequencies):
    logger.info(f"Loading {path}")
    try:
        ts = tskit.load(path)

    except tskit.FileFormatError:
        ts = tszip.decompress(path)

    tsm = model.TSModel(ts, calc_population_frequencies, path.name)
    return tsm


def get_app(tsm):
    pn.extension(sizing_mode="stretch_width")
    pn.extension("tabulator")

    def show(page_name):
        hv.extension("bokeh")
        hv.opts.defaults(
            opts.Scatter(color=config.PLOT_COLOURS[2]),
            opts.Points(color=config.PLOT_COLOURS[2]),
            opts.Histogram(
                fill_color=config.PLOT_COLOURS[0], line_color=config.PLOT_COLOURS[2]
            ),
            opts.Bars(color=config.PLOT_COLOURS[0], line_color=config.PLOT_COLOURS[2]),
            opts.Segments(color=config.PLOT_COLOURS[2]),
            opts.Curve(color=config.PLOT_COLOURS[2]),
            opts.Rectangles(
                fill_color=config.PLOT_COLOURS[0], line_color=config.PLOT_COLOURS[2]
            ),
        )
        logger.info(f"Showing page {page_name}")
        yield pn.indicators.LoadingSpinner(value=True, width=50, height=50)
        try:
            before = time.time()
            content = pages.PAGES_MAP[page_name].page(tsm)
            duration = time.time() - before
            logger.info(f"Loaded page {page_name} in {duration:.2f}s")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            error_traceback = traceback.format_exc().replace("\n", "<br>")
            error_traceback = f"<pre>{error_traceback}</pre>"
            error_panel = pn.pane.Markdown(
                f"## Error\n\n{error_message}\n\n{error_traceback}",
                style={"color": "red"},
            )
            yield error_panel
            return
        yield content

    starting_page = pn.state.session_args.get("page", [b"Overview"])[0].decode()
    page_options = pn.widgets.RadioButtonGroup(
        value=starting_page,
        options=list(pages.PAGES_MAP.keys()),
        name="Page",
        # sizing_mode="fixed",
        button_type="default",
        button_style="outline",
        orientation="vertical",
    )
    ishow = pn.bind(show, page_name=page_options)
    pn.state.location.sync(page_options, {"value": "page"})

    RAW_CSS = """
        .sidenav#sidebar {
            background-color: #15E3AC;
        }
    """
    DEFAULT_PARAMS = {
        "site": "QC dashboard",
        "header_background": "#0D5160",
    }

    return pn.template.FastListTemplate(
        title=tsm.name,
        sidebar=[page_options],
        main=[ishow],
        raw_css=[RAW_CSS],
        **DEFAULT_PARAMS,
    )


def setup_logging(log_level, no_log_filter):
    if no_log_filter:
        logger = daiquiri.getLogger("root")
        logger.setLevel(log_level)
    else:
        # TODO figure out what's useful for users to track here, including
        # bokeh and tornado for now for dev
        loggers = ["tsqc", "cache", "bokeh", "tornado"]
        for logname in loggers:
            logger = daiquiri.getLogger(logname)
            logger.setLevel(log_level)

        # Suppress traceback messages in the log like:
        # bokeh.core.serialization.DeserializationError: can't resolve reference 'p2091'
        # These pop up when the user clicks around between screens quickly,
        # and seems to occur when leaving a screen before it's fully rendered.
        # Some googling indicates that the messages are harmless, so simplest
        # to filter.
        logger = daiquiri.getLogger("bokeh.server.protocol_handler")
        logger.setLevel("CRITICAL")


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--annotations-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8080, help="Port to serve on")
@click.option(
    "--show/--no-show",
    default=True,
    help="Launch a web-browser showing the app",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option(
    "--no-log-filter",
    default=False,
    is_flag=True,
    help="Do not filter the output log (advanced debugging only)",
)
@click.option(
    "--calc-population-frequencies",
    default=False,
    help="Calculate population frequencies for sample nodes",
)
def main(
    path,
    port,
    show,
    log_level,
    no_log_filter,
    annotations_file,
    calc_population_frequencies,
):
    """
    Run the tsqc server.
    """
    setup_logging(log_level, no_log_filter)

    tsm = load_data(pathlib.Path(path), calc_population_frequencies)
    if annotations_file:
        config.ANNOTATIONS_FILE = annotations_file

    # Note: functools.partial doesn't work here
    def app():
        return get_app(tsm)

    logger.info("Starting panel server")
    pn.serve(app, port=port, show=show, verbose=False)


if __name__ == "__main__":
    main()
