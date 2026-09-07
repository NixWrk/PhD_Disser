"""Interactive chamber-volume plots on an absolute millisecond time axis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_volume_figure_ms(
    subject: str,
    phases: list[dict],
    chambers: Sequence[tuple[str, str, str]],
    profile_builder: Callable[[list[dict], str, int], dict],
) -> go.Figure:
    """Plot volumes in milliseconds and retain the local R–R percentage in labels."""
    cycle_indices = sorted({int(phase["cycle_index"]) for phase in phases})
    split_cycles = subject == "georg"
    cycle_symbols = ("circle", "diamond", "square")
    cycle_dashes = ("solid", "dash", "dot")
    text_positions = (
        ("top left", "bottom left"),
        ("top right", "bottom right"),
        ("bottom left", "top left"),
        ("bottom right", "top right"),
    )

    cycle_data: dict[int, dict] = {}
    for cycle_index in cycle_indices:
        cycle_phases = sorted(
            (phase for phase in phases if int(phase["cycle_index"]) == cycle_index),
            key=lambda phase: phase["phase_percent_within_cycle"],
        )
        rr_ms = float(np.median([phase["rr_interval_ms"] for phase in cycle_phases]))
        heart_rate = float(np.median([
            phase["heart_rate_from_rpeaks_bpm"] for phase in cycle_phases
        ]))
        cycle_start_ms = float(np.median([
            phase["derived_time_from_R0_ms"]
            - phase["phase_percent_within_cycle"] / 100.0 * rr_ms
            for phase in cycle_phases
        ]))
        cycle_data[cycle_index] = {
            "phases": cycle_phases,
            "rr_ms": rr_ms,
            "heart_rate_bpm": heart_rate,
            "cycle_start_ms": cycle_start_ms,
        }

    subplot_titles = []
    if split_cycles:
        for cycle_index in cycle_indices:
            item = cycle_data[cycle_index]
            observed = item["phases"]
            subplot_titles.append((
                f"Интервал {cycle_index + 1}: R–R = {item['rr_ms']:.0f} мс, "
                f"ЧСС = {item['heart_rate_bpm']:.1f} уд/мин; "
                f"наблюдаемые фазы {observed[0]['phase_percent_within_cycle']:.0f}–"
                f"{observed[-1]['phase_percent_within_cycle']:.0f}%"
            ).replace(".", ","))
        figure = make_subplots(
            rows=len(cycle_indices),
            cols=1,
            shared_xaxes=False,
            subplot_titles=subplot_titles,
            vertical_spacing=0.13,
        )
    else:
        figure = make_subplots(rows=1, cols=1)

    tick_values_by_row: dict[int, list[float]] = defaultdict(list)
    tick_text_by_row: dict[int, list[str]] = defaultdict(list)
    observed_x_all: list[float] = []

    for cycle_order, cycle_index in enumerate(cycle_indices):
        item = cycle_data[cycle_index]
        rr_ms = item["rr_ms"]
        heart_rate = item["heart_rate_bpm"]
        cycle_start_ms = item["cycle_start_ms"]
        row = cycle_order + 1 if split_cycles else 1

        for chamber_index, (key, label, color) in enumerate(chambers):
            profile = profile_builder(phases, key, cycle_index)
            dense_phase = np.asarray(profile["phase_percent"], dtype=float)
            dense_after_r_ms = dense_phase / 100.0 * rr_ms
            dense_x = dense_after_r_ms if split_cycles else cycle_start_ms + dense_after_r_ms
            dense_customdata = np.column_stack([
                np.full(len(dense_phase), cycle_index + 1),
                dense_after_r_ms,
                dense_phase,
                np.full(len(dense_phase), rr_ms),
                np.full(len(dense_phase), heart_rate),
            ])
            figure.add_trace(go.Scatter(
                x=dense_x,
                y=profile["volume_ml"],
                customdata=dense_customdata,
                mode="lines",
                name=label,
                legendgroup=key,
                showlegend=cycle_order == 0,
                line={
                    "color": color,
                    "width": 3,
                    "dash": cycle_dashes[cycle_order % len(cycle_dashes)],
                },
                hovertemplate=(
                    "%{fullData.name}<br>Интервал: %{customdata[0]:.0f}"
                    "<br>После R: %{customdata[1]:.1f} мс (%{customdata[2]:.0f}%)"
                    "<br>R–R: %{customdata[3]:.1f} мс"
                    "<br>ЧСС: %{customdata[4]:.2f} уд/мин"
                    "<br>Интерполированный объём: %{y:.1f} мл<extra></extra>"
                ),
            ), row=row, col=1)

            observations = profile["observations"]
            after_r_ms = np.asarray([
                observation["phase_percent"] / 100.0 * rr_ms
                for observation in observations
            ], dtype=float)
            observation_x = after_r_ms if split_cycles else cycle_start_ms + after_r_ms
            customdata = [
                [
                    cycle_index + 1,
                    time_ms,
                    observation["phase_percent"],
                    observation["raw_rr_percent"],
                    rr_ms,
                    heart_rate,
                    observation["phase_id"],
                ]
                for observation, time_ms in zip(observations, after_r_ms)
            ]
            figure.add_trace(go.Scatter(
                x=observation_x,
                y=[observation["volume_ml"] for observation in observations],
                text=[
                    f"{observation['volume_ml']:.1f}".replace(".", ",")
                    for observation in observations
                ],
                textposition=[
                    text_positions[chamber_index][cycle_order % 2]
                ] * len(observations),
                customdata=customdata,
                mode="markers+text",
                name=f"{label}: измерено, интервал {cycle_index + 1}",
                legendgroup=key,
                showlegend=False,
                marker={
                    "color": color,
                    "size": 9,
                    "symbol": cycle_symbols[cycle_order % len(cycle_symbols)],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                textfont={"color": color, "size": 10},
                cliponaxis=False,
                hovertemplate=(
                    "%{fullData.name}<br>Интервал: %{customdata[0]}"
                    "<br>После R: %{customdata[1]:.1f} мс (%{customdata[2]:.0f}%)"
                    "<br>Исходная координата GE: %{customdata[3]:.0f}%"
                    "<br>R–R: %{customdata[4]:.1f} мс"
                    "<br>ЧСС: %{customdata[5]:.2f} уд/мин"
                    "<br>Измеренный объём: %{y:.1f} мл"
                    "<br>%{customdata[6]}<extra></extra>"
                ),
            ), row=row, col=1)

        observed_phases = item["phases"]
        for phase in observed_phases:
            time_after_r_ms = phase["phase_percent_within_cycle"] / 100.0 * rr_ms
            x_value = time_after_r_ms if split_cycles else cycle_start_ms + time_after_r_ms
            tick_values_by_row[row].append(float(x_value))
            tick_text_by_row[row].append(
                f"{x_value:.0f} мс<br>({phase['phase_percent_within_cycle']:.0f}%)"
            )
            observed_x_all.append(float(x_value))

        if split_cycles:
            observed_start_ms = observed_phases[0]["phase_percent_within_cycle"] / 100.0 * rr_ms
            observed_end_ms = observed_phases[-1]["phase_percent_within_cycle"] / 100.0 * rr_ms
            for left, right in ((0.0, observed_start_ms), (observed_end_ms, rr_ms)):
                if right > left:
                    figure.add_vrect(
                        x0=left,
                        x1=right,
                        fillcolor="#E5E7EB",
                        opacity=0.25,
                        line_width=0,
                        row=row,
                        col=1,
                    )
            figure.add_vline(
                x=0.0,
                line={"color": "#B91C1C", "dash": "dash", "width": 1.5},
                annotation_text="R",
                annotation_position="top right",
                row=row,
                col=1,
            )
            figure.add_vline(
                x=rr_ms,
                line={"color": "#B91C1C", "dash": "dash", "width": 1.5},
                annotation_text="следующий R",
                annotation_position="top left",
                row=row,
                col=1,
            )
            figure.update_xaxes(range=[0.0, rr_ms], row=row, col=1)

    if not split_cycles:
        maximum_observed_ms = max(observed_x_all)
        minimum_observed_ms = min(observed_x_all)
        padding_ms = max(25.0, 0.04 * (maximum_observed_ms - minimum_observed_ms))
        figure.update_xaxes(range=[0.0, maximum_observed_ms + padding_ms], row=1, col=1)
        for cycle_index in cycle_indices:
            cycle_start_ms = cycle_data[cycle_index]["cycle_start_ms"]
            if cycle_start_ms <= maximum_observed_ms + padding_ms:
                figure.add_vline(
                    x=cycle_start_ms,
                    line={"color": "#B91C1C", "dash": "dash", "width": 1.5},
                    annotation_text=f"R{cycle_index}",
                    annotation_position="top right",
                    row=1,
                    col=1,
                )

    for row, values in tick_values_by_row.items():
        order = np.argsort(values)
        figure.update_xaxes(
            tickmode="array",
            tickvals=[values[index] for index in order],
            ticktext=[tick_text_by_row[row][index] for index in order],
            tickangle=-35,
            title_text=(
                "Время после R-триггера, мс (процент соответствующего R–R)"
                if split_cycles
                else "Время от первой R-метки, мс (процент соответствующего R–R)"
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(
            title_text="Объём автоматической маски, мл",
            row=row,
            col=1,
        )

    timing_text = "; ".join(
        (
            f"интервал {cycle_index + 1}: R–R {cycle_data[cycle_index]['rr_ms']:.0f} мс, "
            f"ЧСС {cycle_data[cycle_index]['heart_rate_bpm']:.1f} уд/мин"
        ).replace(".", ",")
        for cycle_index in cycle_indices
    )
    if split_cycles:
        title_text = f"{subject.upper()}: два раздельных интервала реконструкции"
        interpretation_note = (
            "Оси двух панелей имеют собственный масштаб времени. Серый фон обозначает участки "
            "R–R без реконструированных фаз; PCHIP ограничен измеренными точками. Одинаковый "
            "процент разных R–R не трактуется как одинаковая механическая фаза."
        )
    else:
        title_text = f"{subject.upper()}: объёмы камер в непрерывном времени"
        interpretation_note = (
            "Положение точки задано временем от первой R-метки; в скобках указан процент "
            "соответствующего R–R. PCHIP строится отдельно внутри каждого наблюдаемого "
            "интервала и не соединяет соседние сокращения."
        )

    figure.update_layout(
        title={"text": title_text, "x": 0.5, "y": 0.99},
        template="plotly_white",
        height=1040 if split_cycles else 720,
        hovermode="closest",
        separators=", ",
        legend={
            "orientation": "h",
            "y": 1.05,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
            "groupclick": "togglegroup",
        },
        margin={"l": 95, "r": 45, "t": 230 if split_cycles else 200, "b": 125},
        uirevision=f"volume-ms-{subject}",
    )
    figure.add_annotation(
        x=0.5,
        y=1.16,
        xref="paper",
        yref="paper",
        text=timing_text,
        showarrow=False,
        font={"size": 12, "color": "#374151"},
        align="center",
    )
    figure.add_annotation(
        x=0.5,
        y=1.10,
        xref="paper",
        yref="paper",
        text=interpretation_note,
        showarrow=False,
        font={"size": 11, "color": "#6B7280"},
        align="center",
    )
    return figure
