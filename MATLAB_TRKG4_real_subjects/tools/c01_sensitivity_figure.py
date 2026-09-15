"""Interactive reader figure for the common C01 sensitivity atlas."""
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from c01_baseline_study import OUT
from c01_baseline_analysis import LABELS


def sensitivity_figure():
    atlas = pd.read_csv(OUT / "sensitivity_atlas.csv")
    states = (
        atlas[["case", "rho1", "rho2", "ratio"]]
        .drop_duplicates()
        .sort_values("case")
        .reset_index(drop=True)
    )
    models = (
        atlas[["model_order", "model", "model_label"]]
        .drop_duplicates()
        .sort_values("model_order")
    )
    sizes = np.sort(atlas.L_mm.unique())
    assert len(states) == 65 and len(models) == len(LABELS) and len(sizes) == 9
    assert not atlas.duplicated(["case", "model", "L_mm"]).any()
    model_names = models.model.tolist()
    model_labels = models.model_label.tolist()

    def matrix(frame, column):
        return (
            frame.pivot(index="model", columns="L_mm", values=column)
            .reindex(index=model_names, columns=sizes)
            .to_numpy()
        )

    def traces(case):
        frame = atlas[atlas.case == case]
        result = []
        for tissue in [1, 2]:
            j = matrix(frame, f"dZ_drho{tissue}_per_m")
            k = matrix(frame, f"dlogZ_dlogrho{tissue}")
            error = matrix(frame, f"error_dZ_drho{tissue}_pct")
            reference = matrix(frame, f"reference_dZ_drho{tissue}_per_m")
            norm_error = matrix(frame, f"norm_error_dZ_drho{tissue}_pct")
            condition = matrix(frame, "condition_log_parameters")
            custom = np.stack([k, error, reference, norm_error, condition], axis=-1)
            result.append(
                go.Heatmap(
                    x=sizes,
                    y=model_labels,
                    z=j,
                    customdata=custom,
                    coloraxis=f"coloraxis{tissue}" if tissue > 1 else "coloraxis",
                    hovertemplate=(
                        "<b>%{y}</b><br>L = %{x:.0f} мм<br>"
                        f"J{tissue} = %{{z:.6g}} м⁻¹<br>"
                        f"∂lnZ/∂lnρ{tissue} = %{{customdata[0]:.5f}}<br>"
                        "отклонение J от КТ/FEM = %{customdata[1]:+.2f}%<br>"
                        "J КТ/FEM = %{customdata[2]:.6g} м⁻¹<br>"
                        "расхождение столбца по 9 размерам = %{customdata[3]:.2f}%<br>"
                        "κ логарифмического якобиана = %{customdata[4]:.2f}<extra></extra>"
                    ),
                )
            )
        for tissue in [1, 2]:
            j = matrix(frame, f"dZ_drho{tissue}_per_m")
            k = matrix(frame, f"dlogZ_dlogrho{tissue}")
            error = matrix(frame, f"error_dZ_drho{tissue}_pct")
            reference = matrix(frame, f"reference_dZ_drho{tissue}_per_m")
            norm_error = matrix(frame, f"norm_error_dZ_drho{tissue}_pct")
            condition = matrix(frame, "condition_log_parameters")
            custom = np.stack([j, k, reference, norm_error, condition], axis=-1)
            result.append(
                go.Heatmap(
                    x=sizes,
                    y=model_labels,
                    z=error,
                    customdata=custom,
                    coloraxis="coloraxis3",
                    hovertemplate=(
                        "<b>%{y}</b><br>L = %{x:.0f} мм<br>"
                        f"отклонение J{tissue} = %{{z:+.2f}}%<br>"
                        f"J{tissue} = %{{customdata[0]:.6g}} м⁻¹<br>"
                        f"∂lnZ/∂lnρ{tissue} = %{{customdata[1]:.5f}}<br>"
                        "J КТ/FEM = %{customdata[2]:.6g} м⁻¹<br>"
                        "расхождение столбца по 9 размерам = %{customdata[3]:.2f}%<br>"
                        "κ логарифмического якобиана = %{customdata[4]:.2f}<extra></extra>"
                    ),
                )
            )
        return result

    initial = int(((states.rho1 - 4) ** 2 + (states.rho2 - 16) ** 2).idxmin())
    initial_state = states.iloc[initial]
    subplot_titles = [
        "Абсолютная чувствительность J₁ = ∂Z/∂ρ₁, м⁻¹",
        "Абсолютная чувствительность J₂ = ∂Z/∂ρ₂, м⁻¹",
        "Отклонение J₁ от КТ/FEM, %",
        "Отклонение J₂ от КТ/FEM, %",
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for trace, (row, col) in zip(traces(int(initial_state.case)), positions):
        figure.add_trace(trace, row=row, col=col)

    buttons = []
    for state in states.itertuples():
        state_traces = traces(int(state.case))
        buttons.append(
            dict(
                label=f"{int(state.case):02d}: {state.rho1:.3f} / {state.rho2:.3f}",
                method="update",
                args=[
                    dict(
                        z=[trace.z for trace in state_traces],
                        customdata=[trace.customdata for trace in state_traces],
                    ),
                    {},
                ],
            )
        )

    j1 = atlas.dZ_drho1_per_m.to_numpy()
    j2 = atlas.dZ_drho2_per_m.to_numpy()
    figure.update_layout(
        title="Локальная чувствительность 19 реализаций при выбранной паре ρ₁/ρ₂",
        width=1450,
        height=1180,
        margin=dict(l=390, r=155, t=155, b=90),
        coloraxis=dict(
            colorscale="Viridis",
            cmin=float(j1.min()),
            cmax=float(j1.max()),
            colorbar=dict(title="J₁, м⁻¹", x=0.455, y=0.78, len=0.38),
        ),
        coloraxis2=dict(
            colorscale="RdBu_r",
            cmid=0,
            cmin=float(j2.min()),
            cmax=float(j2.max()),
            colorbar=dict(title="J₂, м⁻¹", x=1.015, y=0.78, len=0.38),
        ),
        coloraxis3=dict(
            colorscale="RdBu_r",
            cmid=0,
            cmin=-100,
            cmax=100,
            colorbar=dict(title="ΔJ, %", x=1.015, y=0.22, len=0.38),
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0,
                y=1.10,
                xanchor="left",
                yanchor="top",
                active=initial,
                showactive=True,
                buttons=buttons,
            )
        ],
    )
    figure.update_xaxes(title_text="Размер сборки L, мм")
    for axis in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        figure.layout[axis].update(autorange="reversed")
    figure.update_yaxes(showticklabels=False, col=2)
    display(
        HTML(
            pio.to_html(
                figure,
                include_plotlyjs=False,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
        )
    )
