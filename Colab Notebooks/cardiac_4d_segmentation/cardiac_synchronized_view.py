"""Synchronized 3D cardiac surfaces, schematic ECG, and chamber-volume curves."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _axis_range(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    margin = max(2.0, 0.03 * (high - low))
    return [low - margin, high + margin]


def _mesh_trace(mesh: dict[str, np.ndarray], style: object, showlegend: bool) -> go.Mesh3d:
    return go.Mesh3d(
        **mesh,
        name=style.label,
        legendgroup=style.key,
        color=style.color,
        opacity=style.opacity,
        flatshading=False,
        showlegend=showlegend,
        lighting={"ambient": 0.6, "diffuse": 0.7, "specular": 0.15, "roughness": 0.7},
        hovertemplate=style.label + "<extra></extra>",
    )


def _cycle_data(phases: list[dict]) -> tuple[list[int], dict[int, dict]]:
    cycle_indices = sorted({int(phase["cycle_index"]) for phase in phases})
    result: dict[int, dict] = {}
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
        result[cycle_index] = {
            "phases": cycle_phases,
            "rr_ms": rr_ms,
            "heart_rate_bpm": heart_rate,
            "cycle_start_ms": cycle_start_ms,
        }
    return cycle_indices, result


def build_synchronized_cardiac_figure(
    subject: str,
    phases: list[dict],
    surfaces: Sequence[object],
    chambers: Sequence[tuple[str, str, str]],
    profile_builder: Callable[[list[dict], str, int], dict],
    ecg_builder: Callable[[np.ndarray], np.ndarray],
) -> go.Figure:
    """Build one phase-controlled figure for anatomy, ECG orientation, and volumes."""
    cycle_indices, cycles = _cycle_data(phases)
    split_cycles = subject == "georg"
    if split_cycles:
        specs = [
            [{"type": "scene", "rowspan": 4}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ]
        titles = ["3D-маски"]
        for cycle_index in cycle_indices:
            item = cycles[cycle_index]
            interval = (
                f"интервал {cycle_index + 1}: R–R {item['rr_ms']:.0f} мс, "
                f"ЧСС {item['heart_rate_bpm']:.1f} уд/мин"
            ).replace(".", ",")
            titles.extend([
                f"Схематическая ЭКГ — {interval}",
                f"Объёмы камер — {interval}",
            ])
        figure = make_subplots(
            rows=4,
            cols=2,
            specs=specs,
            column_widths=[0.48, 0.52],
            row_heights=[0.15, 0.35, 0.15, 0.35],
            horizontal_spacing=0.055,
            vertical_spacing=0.075,
            subplot_titles=titles,
        )
        row_map = {
            cycle_index: (2 * order + 1, 2 * order + 2)
            for order, cycle_index in enumerate(cycle_indices)
        }
    else:
        figure = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                [None, {"type": "xy"}],
            ],
            column_widths=[0.48, 0.52],
            row_heights=[0.27, 0.73],
            horizontal_spacing=0.055,
            vertical_spacing=0.11,
            subplot_titles=[
                "3D-маски",
                "Схематическая ЭКГ в пределах R–R",
                "Объёмы камер",
            ],
        )
        row_map = {cycle_index: (1, 2) for cycle_index in cycle_indices}

    expected_meshes = len(surfaces)
    if any(len(phase["surface_meshes"]) != expected_meshes for phase in phases):
        raise ValueError("Surface count differs between phases")

    mesh_indices: list[int] = []
    for mesh, style in zip(phases[0]["surface_meshes"], surfaces):
        mesh_indices.append(len(figure.data))
        figure.add_trace(_mesh_trace(mesh, style, True), row=1, col=1)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for phase in phases:
        for mesh in phase["surface_meshes"]:
            xs.extend(mesh["x"])
            ys.extend(mesh["y"])
            zs.extend(mesh["z"])

    figure.update_scenes(
        xaxis={"title": "R–L, мм", "range": _axis_range(xs)},
        yaxis={"title": "A–P, мм", "range": _axis_range(ys)},
        zaxis={"title": "I–S, мм", "range": _axis_range(zs)},
        aspectmode="data",
        camera={"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
    )

    observed_ticks: dict[int, list[tuple[float, str]]] = defaultdict(list)
    cycle_symbols = ("circle", "diamond", "square")
    cycle_dashes = ("solid", "dash", "dot")
    text_positions = (
        ("top left", "bottom left"),
        ("top right", "bottom right"),
        ("bottom left", "top left"),
        ("bottom right", "top right"),
    )

    maximum_observed_ms = max(float(phase["derived_time_from_R0_ms"]) for phase in phases)
    minimum_observed_ms = min(float(phase["derived_time_from_R0_ms"]) for phase in phases)
    padding_ms = max(25.0, 0.04 * max(1.0, maximum_observed_ms - minimum_observed_ms))
    continuous_end_ms = maximum_observed_ms + padding_ms

    for cycle_order, cycle_index in enumerate(cycle_indices):
        item = cycles[cycle_index]
        rr_ms = item["rr_ms"]
        heart_rate = item["heart_rate_bpm"]
        cycle_start_ms = item["cycle_start_ms"]
        ecg_row, volume_row = row_map[cycle_index]

        if split_cycles:
            ecg_phase_end = 100.0
        else:
            ecg_phase_end = min(100.0, max(0.0, (continuous_end_ms - cycle_start_ms) / rr_ms * 100.0))
        ecg_percent = np.linspace(0.0, ecg_phase_end, max(2, int(4 * ecg_phase_end) + 1))
        ecg_x = ecg_percent / 100.0 * rr_ms
        if not split_cycles:
            ecg_x = cycle_start_ms + ecg_x
        figure.add_trace(go.Scatter(
            x=ecg_x,
            y=ecg_builder(ecg_percent),
            mode="lines",
            name="Схематическая ЭКГ",
            legendgroup="schematic_ecg",
            showlegend=False,
            line={"color": "#4B5563", "width": 2},
            hovertemplate=(
                f"Интервал {cycle_index + 1}<br>После R: %{{customdata[0]:.1f}} мс "
                f"(%{{customdata[1]:.1f}}%)<br>R–R: {rr_ms:.1f} мс"
                "<br>Сигнал схематический<extra></extra>"
            ),
            customdata=np.column_stack([ecg_percent / 100.0 * rr_ms, ecg_percent]),
        ), row=ecg_row, col=2)

        r_start = 0.0 if split_cycles else cycle_start_ms
        r_end = rr_ms if split_cycles else cycle_start_ms + rr_ms
        for r_position, label, position in (
            (r_start, "R", "top right"),
            (r_end, "R", "top left"),
        ):
            if split_cycles or r_position <= continuous_end_ms:
                figure.add_vline(
                    x=r_position,
                    line={"color": "#B91C1C", "dash": "dash", "width": 1.2},
                    annotation_text=label,
                    annotation_position=position,
                    exclude_empty_subplots=False,
                    row=ecg_row,
                    col=2,
                )

        for chamber_index, (key, label, color) in enumerate(chambers):
            profile = profile_builder(phases, key, cycle_index)
            dense_percent = np.asarray(profile["phase_percent"], dtype=float)
            dense_after_r_ms = dense_percent / 100.0 * rr_ms
            dense_x = dense_after_r_ms if split_cycles else cycle_start_ms + dense_after_r_ms
            dense_customdata = np.column_stack([
                np.full(len(dense_percent), cycle_index + 1),
                dense_after_r_ms,
                dense_percent,
                np.full(len(dense_percent), rr_ms),
                np.full(len(dense_percent), heart_rate),
            ])
            figure.add_trace(go.Scatter(
                x=dense_x,
                y=profile["volume_ml"],
                customdata=dense_customdata,
                mode="lines",
                name=label,
                legendgroup=key,
                showlegend=False,
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
            ), row=volume_row, col=2)

            observations = profile["observations"]
            after_r_ms = np.asarray([
                observation["phase_percent"] / 100.0 * rr_ms
                for observation in observations
            ], dtype=float)
            observation_x = after_r_ms if split_cycles else cycle_start_ms + after_r_ms
            figure.add_trace(go.Scatter(
                x=observation_x,
                y=[observation["volume_ml"] for observation in observations],
                text=[f"{observation['volume_ml']:.1f}".replace(".", ",") for observation in observations],
                textposition=[text_positions[chamber_index][cycle_order % 2]] * len(observations),
                customdata=[
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
                ],
                mode="markers+text",
                name=f"{label}: измерено, интервал {cycle_index + 1}",
                legendgroup=key,
                showlegend=False,
                marker={
                    "color": color,
                    "size": 8,
                    "symbol": cycle_symbols[cycle_order % len(cycle_symbols)],
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                textfont={"color": color, "size": 9},
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
            ), row=volume_row, col=2)

        for phase in item["phases"]:
            after_r_ms = phase["phase_percent_within_cycle"] / 100.0 * rr_ms
            x_value = after_r_ms if split_cycles else cycle_start_ms + after_r_ms
            observed_ticks[volume_row].append((
                float(x_value),
                f"{x_value:.0f} мс<br>({phase['phase_percent_within_cycle']:.0f}%)",
            ))

        if split_cycles:
            observed_start_ms = item["phases"][0]["phase_percent_within_cycle"] / 100.0 * rr_ms
            observed_end_ms = item["phases"][-1]["phase_percent_within_cycle"] / 100.0 * rr_ms
            for left, right in ((0.0, observed_start_ms), (observed_end_ms, rr_ms)):
                if right > left:
                    figure.add_vrect(
                        x0=left,
                        x1=right,
                        fillcolor="#E5E7EB",
                        opacity=0.25,
                        line_width=0,
                        exclude_empty_subplots=False,
                        row=volume_row,
                        col=2,
                    )

    if split_cycles:
        for cycle_index in cycle_indices:
            rr_ms = cycles[cycle_index]["rr_ms"]
            ecg_row, volume_row = row_map[cycle_index]
            figure.update_xaxes(range=[0.0, rr_ms], showticklabels=False, row=ecg_row, col=2)
            figure.update_xaxes(range=[0.0, rr_ms], row=volume_row, col=2)
    else:
        figure.update_xaxes(range=[0.0, continuous_end_ms], showticklabels=False, row=1, col=2)
        figure.update_xaxes(range=[0.0, continuous_end_ms], row=2, col=2)

    for cycle_index in cycle_indices:
        ecg_row, volume_row = row_map[cycle_index]
        figure.update_yaxes(
            title_text="Условная амплитуда",
            range=[-0.42, 1.28],
            showticklabels=False,
            zeroline=True,
            row=ecg_row,
            col=2,
        )
        figure.update_yaxes(title_text="Объём, мл", row=volume_row, col=2)

    for row, tick_pairs in observed_ticks.items():
        unique = {value: label for value, label in tick_pairs}
        tick_values = sorted(unique)
        figure.update_xaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=[unique[value] for value in tick_values],
            tickangle=-35,
            title_text=(
                "Время после R-триггера, мс (процент соответствующего R–R)"
                if split_cycles
                else "Время от первой R-метки, мс (процент соответствующего R–R)"
            ),
            row=row,
            col=2,
        )

    initial_phase = phases[0]
    dynamic_indices: dict[int, dict[str, object]] = {}
    for cycle_index in cycle_indices:
        ecg_row, volume_row = row_map[cycle_index]
        active = int(initial_phase["cycle_index"]) == cycle_index
        local_percent = float(initial_phase["phase_percent_within_cycle"])
        local_ms = local_percent / 100.0 * cycles[cycle_index]["rr_ms"]
        active_x = local_ms if split_cycles else cycles[cycle_index]["cycle_start_ms"] + local_ms

        guide_index = len(figure.data)
        figure.add_trace(go.Scatter(
            x=[active_x, active_x] if active else [],
            y=[-0.38, 1.22] if active else [],
            mode="lines",
            line={"color": "#111827", "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        ), row=ecg_row, col=2)
        ecg_point_index = len(figure.data)
        figure.add_trace(go.Scatter(
            x=[active_x] if active else [],
            y=[float(ecg_builder(np.asarray([local_percent]))[0])] if active else [],
            mode="markers",
            marker={"color": "#111827", "size": 10, "line": {"color": "#FFFFFF", "width": 2}},
            hovertemplate="Текущая реконструированная фаза<extra></extra>",
            showlegend=False,
        ), row=ecg_row, col=2)

        chamber_indices: dict[str, int] = {}
        for key, label, color in chambers:
            chamber_indices[key] = len(figure.data)
            figure.add_trace(go.Scatter(
                x=[active_x] if active else [],
                y=[initial_phase["chambers_ml"][key]] if active else [],
                mode="markers",
                name=f"Текущая фаза: {label}",
                legendgroup=key,
                showlegend=False,
                marker={"color": color, "size": 16, "line": {"color": "#111827", "width": 2.5}},
                hovertemplate=(
                    f"{label}<br>Текущая фаза: {initial_phase['phase_id']}"
                    "<br>Объём: %{y:.1f} мл<extra></extra>"
                ),
            ), row=volume_row, col=2)
        dynamic_indices[cycle_index] = {
            "guide": guide_index,
            "ecg_point": ecg_point_index,
            "chambers": chamber_indices,
        }

    frame_trace_indices = list(mesh_indices)
    for cycle_index in cycle_indices:
        frame_trace_indices.extend([
            dynamic_indices[cycle_index]["guide"],
            dynamic_indices[cycle_index]["ecg_point"],
        ])
        frame_trace_indices.extend(dynamic_indices[cycle_index]["chambers"][key] for key, _, _ in chambers)

    frames: list[go.Frame] = []
    for phase in phases:
        frame_data: list[go.BaseTraceType] = [
            _mesh_trace(mesh, style, False)
            for mesh, style in zip(phase["surface_meshes"], surfaces)
        ]
        for cycle_index in cycle_indices:
            active = int(phase["cycle_index"]) == cycle_index
            local_percent = float(phase["phase_percent_within_cycle"])
            local_ms = local_percent / 100.0 * cycles[cycle_index]["rr_ms"]
            active_x = local_ms if split_cycles else cycles[cycle_index]["cycle_start_ms"] + local_ms
            frame_data.append(go.Scatter(
                x=[active_x, active_x] if active else [],
                y=[-0.38, 1.22] if active else [],
            ))
            frame_data.append(go.Scatter(
                x=[active_x] if active else [],
                y=[float(ecg_builder(np.asarray([local_percent]))[0])] if active else [],
            ))
            for key, label, color in chambers:
                frame_data.append(go.Scatter(
                    x=[active_x] if active else [],
                    y=[phase["chambers_ml"][key]] if active else [],
                    hovertemplate=(
                        f"{label}<br>Текущая фаза: {phase['phase_id']}"
                        f"<br>После R: {local_ms:.1f} мс ({local_percent:.0f}%)"
                        f"<br>Объём: %{{y:.1f}} мл<extra></extra>"
                    ),
                ))
        frames.append(go.Frame(
            name=phase["phase_id"],
            traces=frame_trace_indices,
            data=frame_data,
        ))
    figure.frames = tuple(frames)

    def slider_label(phase: dict) -> str:
        if split_cycles:
            return (
                f"инт. {int(phase['cycle_index']) + 1}: "
                f"{phase['time_after_cycle_r_ms']:.0f} мс "
                f"({phase['phase_percent_within_cycle']:.0f}%)"
            )
        return (
            f"{phase['derived_time_from_R0_ms']:.0f} мс "
            f"({phase['phase_percent_within_cycle']:.0f}%)"
        )

    steps = [{
        "label": slider_label(phase),
        "method": "animate",
        "args": [[phase["phase_id"]], {
            "mode": "immediate",
            "frame": {"duration": 0, "redraw": True},
            "transition": {"duration": 0},
        }],
    } for phase in phases]
    timing_text = "; ".join(
        (
            f"интервал {cycle_index + 1}: R–R {cycles[cycle_index]['rr_ms']:.0f} мс, "
            f"ЧСС {cycles[cycle_index]['heart_rate_bpm']:.1f} уд/мин"
        ).replace(".", ",")
        for cycle_index in cycle_indices
    )
    figure.update_layout(
        title={
            "text": f"{subject.upper()}: синхронная 4D-модель и объёмы камер",
            "x": 0.5,
            "y": 0.985,
        },
        template="plotly_white",
        height=1220 if split_cycles else 900,
        margin={"l": 40, "r": 30, "t": 220, "b": 115},
        hovermode="closest",
        separators=", ",
        legend={
            "orientation": "h",
            "y": 1.015,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
            "groupclick": "togglegroup",
        },
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Текущая реконструированная фаза: "},
            "pad": {"t": 60},
            "steps": steps,
        }],
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0,
            "y": 0,
            "pad": {"t": 78},
            "buttons": [
                {"label": "▶", "method": "animate", "args": [None, {
                    "fromcurrent": True,
                    "frame": {"duration": 650, "redraw": True},
                    "transition": {"duration": 0},
                }]},
                {"label": "Ⅱ", "method": "animate", "args": [[None], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                }]},
            ],
        }],
        uirevision=f"synchronized-cardiac-{subject}",
    )
    figure.add_annotation(
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        text=timing_text,
        showarrow=False,
        font={"size": 12, "color": "#374151"},
        align="center",
    )
    figure.add_annotation(
        x=0.5,
        y=1.075,
        xref="paper",
        yref="paper",
        text=(
            "Кривая P–QRS–T является схемой, а не индивидуальной ЭКГ. "
            "Точное сопоставление КТ-фаз с механическими фазами сердца требует отдельной проверки."
        ),
        showarrow=False,
        font={"size": 11, "color": "#6B7280"},
        align="center",
    )
    return figure
