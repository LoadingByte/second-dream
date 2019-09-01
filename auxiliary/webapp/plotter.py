from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from bokeh.embed import components
from bokeh.layouts import row, column, Spacer
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Div, Button, Slider, CustomJS, CustomJSTransform
from bokeh.palettes import Pastel1_9
from bokeh.plotting import figure
from bokeh.transform import cumsum, transform
from matplotlib.animation import FuncAnimation

_BASE_TS_COLOR = "#A0A0A0"  # gray
_DREAM_TS_COLOR = "#8B0000"  # dark red


def _pred_prob_colors(num):
    return list(islice(cycle(Pastel1_9), num))


def _octaves(dream_u_timesers):
    octave_lens = np.unique([len(ts) for ts in dream_u_timesers])
    # The first element is the base series and thus has the length of the last octave
    # even though it actually is part of the first octave, so we need a special case.
    return [0] + [np.argwhere(len(ts) == octave_lens)[0, 0] for ts in dream_u_timesers[1:]]


def plot_dream_js(dream_u_timesers, dream_predictions):
    def cumsum_center_transform(vis_limit):
        return CustomJSTransform(v_func="""
            var cumsum = []
            var result = []

            for (var i = 0; i < xs.length; i++) {
                if (i > 0) { cumsum[i] = xs[i] + cumsum[i - 1] }
                else       { cumsum[i] = xs[i] }

                if (xs[i] >= """ + str(vis_limit) + """) {
                    if (i > 0) { result[i] = xs[i] / 2 + cumsum[i - 1] }
                    else       { result[i] = xs[i] / 2 }
                } else {
                    result[i] = -100  // move text off-screen
                }
            }

            return result
        """)

    two_decimal_places_formatter = CustomJSTransform(v_func="""
            var result = []
            for (var i = 0; i < xs.length; i++) {
                result[i] = Number(xs[i]).toFixed(2)
            }
            return result
        """)

    num_steps = len(dream_u_timesers)
    max_step = num_steps - 1
    num_probs = len(dream_predictions[0])
    # We can't use numpy because lengths of dream time series may vary due to octaves.
    min_y = min(min(ts) for ts in dream_u_timesers)
    max_y = max(max(ts) for ts in dream_u_timesers)

    # Create octave list before we resize the dream time series.
    octaves = _octaves(dream_u_timesers)

    # Convert all dream time series to a common length (their lengths may vary due to octaves).
    max_x = max(len(ts) for ts in dream_u_timesers)
    dream_u_timesers = [np.interp(np.arange(max_x + 1), np.linspace(0, max_x, len(ts)), ts) for ts in dream_u_timesers]

    # Define a data source which holds the dream time series from all steps...
    source_available_ts = ColumnDataSource({
        str(i): ts for i, ts in enumerate(dream_u_timesers)
    })
    # ... and a data source which holds the dream time series currently being displayed.
    source_active_ts = ColumnDataSource({
        "y": dream_u_timesers[max_step],
        "x": np.arange(max_x + 1)
    })

    # Same for the prediction probabilities.
    source_available_pred = ColumnDataSource({
        str(i): pred for i, pred in enumerate(dream_predictions)
    })
    source_active_pred = ColumnDataSource({
        "prob": dream_predictions[max_step],
        "class": np.arange(num_probs),
        "color": _pred_prob_colors(num_probs)
    })

    # And finally, a data source which holds the octave of each dream time series
    source_octaves = ColumnDataSource({
        "octave": octaves
    })

    # Create and configure the main figure depicting the base and dream time series.
    p1 = figure(plot_height=300, outline_line_color="black", tools="pan,wheel_zoom,box_zoom,save,reset")
    padding_x = max_x * 0.05
    padding_y = (max_y - min_y) * 0.05
    p1.x_range = Range1d(-padding_x, max_x + padding_x, bounds=(-5 * padding_x, max_x + 5 * padding_x))
    p1.y_range = Range1d(min_y - padding_y, max_y + padding_y, bounds=(min_y - 5 * padding_y, max_y + 5 * padding_y))
    p1.grid.visible = False
    p1.toolbar.logo = None

    # Plot the base time series (static) and the active dream time series (data source reference).
    p1.line(np.arange(max_x + 1), dream_u_timesers[0], color=_BASE_TS_COLOR, line_width=2, legend="Base")
    p1.line(source=source_active_ts, x="x", y="y", color=_DREAM_TS_COLOR, line_width=2, legend="Dream")

    # Create and configure the auxiliary figure depicting the prediction probabilities.
    p2 = figure(plot_height=90, outline_line_color=None, tools="save", title="Dream prediction:")
    p2.axis.visible = False
    p2.grid.visible = False
    p2.x_range.range_padding = 0
    p2.y_range.range_padding = 0
    p2.toolbar.logo = None

    # Plot the active prediction probabilities as well as labels for those (all from a data source reference).
    p2.hbar(source=source_active_pred, y=0, height=1,
            left=cumsum("prob", include_zero=True), right=cumsum("prob"), color="color")
    p2.add_layout(LabelSet(source=source_active_pred,
                           x=transform("prob", cumsum_center_transform(0.02)), y=0.15, text="class",
                           text_baseline="middle", text_align="center", text_font_style="bold"))
    p2.add_layout(LabelSet(source=source_active_pred,
                           x=transform("prob", cumsum_center_transform(0.07)), y=-0.2,
                           text=transform("prob", two_decimal_places_formatter),
                           text_baseline="middle", text_align="center"))

    info = Div(align="center", text=f"Octave: <b>{octaves[-1]}</b>")

    # Add the step slider and add a client-slide JavaScript callback which adjusts the active time series data source.
    slider = Slider(start=0, end=max_step, value=max_step,
                    title="Dreaming step", sizing_mode="stretch_width")
    slider.js_on_change("value", CustomJS(
        args={
            "source_available_ts": source_available_ts,
            "source_active_ts": source_active_ts,
            "source_available_pred": source_available_pred,
            "source_active_pred": source_active_pred,
            "source_octaves": source_octaves,
            "info": info
        },
        code="""
            var selected_step = cb_obj.value;

            // Change active data according to the selected dreaming step
            source_active_ts.data.y = source_available_ts.data[selected_step]
            source_active_pred.data.prob = source_available_pred.data[selected_step]
            // Update the plot
            source_active_ts.change.emit()
            source_active_pred.change.emit()

            info.text = "Octave: <b>" + source_octaves.data.octave[selected_step] + "</b>"
        """))

    # Add the play button and its callback which starts/stops the slider progressing automatically.
    play_button = Button(label="\u25b6", width=50)
    play_step_size = int(np.ceil(num_steps / 200))
    num_play_steps = num_steps / play_step_size
    play_step_interval = 5000 / num_play_steps
    play_button.callback = CustomJS(
        args={
            "slider": slider
        },
        code="""
            function clear() {
                clearInterval(play_interval_id)
                play_interval_id = undefined
                cb_obj.label = "\u25b6"
            }

            if (typeof play_interval_id === "undefined") {
                if (slider.value === slider.end) { slider.value = 0 }
                var play_step_size = """ + str(play_step_size) + """
                play_interval_id = setInterval(function() {
                    if (slider.value + play_step_size >= slider.end) {
                        slider.value = slider.end
                        clear()
                    } else {
                        slider.value += play_step_size
                    }
                }, """ + str(play_step_interval) + """)
                cb_obj.label = "\u275a\u275a"
            } else {
                clear()
            }
        """)

    # Put the parts together and layout them.
    plot = column(row(play_button, Spacer(width=10), slider, Spacer(width=10), info), p1, p2,
                  sizing_mode="stretch_width")

    # Get script and HTML portion of the plot.
    return components(plot)


def plot_dream_video(dream_u_timesers, dream_predictions):
    num_steps = len(dream_u_timesers)
    num_probs = len(dream_predictions[0])
    max_x = len(dream_u_timesers[0]) - 1
    # We can't use numpy because lengths of dream time series may vary due to octaves.
    min_y = min(min(ts) for ts in dream_u_timesers)
    max_y = max(max(ts) for ts in dream_u_timesers)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [4.5, 1]})

    padding_y = 0.1 * (max_y - min_y)
    ax1.set_xlim(-0.05 * max_x, 1.05 * max_x)
    ax1.set_ylim(min_y - padding_y, max_y + padding_y)

    # Setup lines in the main plot.
    ax1.plot(dream_u_timesers[0], color=_BASE_TS_COLOR, label="Base")
    dream_line, = ax1.plot([], [], color=_DREAM_TS_COLOR, label="Dream")
    ax1.legend()

    # Required for tight_layout() to consider these titles (which will be set later in update())
    ax1.set_title("tmp")
    ax2.set_title("tmp")

    fig.tight_layout(h_pad=2)

    octaves = _octaves(dream_u_timesers)

    def update(step):
        ts = dream_u_timesers[step]

        # Update the lines in the main plot.
        dream_line.set_xdata(np.linspace(0, max_x, num=len(ts)))
        dream_line.set_ydata(ts)

        # Clear the previous prediction plot and draw new bars.
        ax2.clear()
        ax2.axis("off")
        ax2.set_title("Dream prediction:")
        ax2.set_xlim(0, 1)

        left = 0
        for cls, (prob, color) in enumerate(zip(dream_predictions[step], _pred_prob_colors(num_probs))):
            ax2.barh(y=0, width=[prob], left=left, color=color)
            if prob >= 0.02:
                ax2.text(left + prob / 2, 0.12, f"{cls}", ha="center", va="center", fontweight="bold")
            if prob >= 0.07:
                ax2.text(left + prob / 2, -0.12, f"{prob:.2f}", ha="center", va="center")
            left += prob

        ax1.set_title(f"Step: {step:04d}      Octave: {octaves[step]}", fontname="monospace")

    anim = FuncAnimation(fig, update, frames=num_steps, interval=5000 / num_steps, repeat_delay=2000)

    video = anim.to_html5_video()
    plt.close(fig)
    return video
