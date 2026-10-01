"""Interactive 3D comparison of observed and predicted TCIA respiratory phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import pyvista as pv


def mesh_trace(path: Path, color: str, name: str, opacity: float) -> go.Mesh3d:
    mesh = pv.read(str(path)).extract_surface().triangulate().clean()
    if mesh.n_cells > 15000:
        mesh = mesh.decimate(1 - 15000 / mesh.n_cells)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = mesh.points
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        flatshading=True, visible=False, hoverinfo="skip",
        showlegend=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evaluation_dir / "evaluation.json").read_text(encoding="utf-8"))
    rows = report["metrics"]
    fig = go.Figure()
    descriptions = []
    for row in rows:
        phase = row["phase"]
        stage = row["model_stage"]
        fig.add_trace(mesh_trace(args.evaluation_dir / f"observed_lung_{phase}.vtp",
                                 "#34a3c7", "Лёгкие: реальная КТ", 0.34))
        warped_ct_path = args.evaluation_dir / f"predicted_ct_lung_{phase}.vtp"
        predicted_path = (warped_ct_path if warped_ct_path.is_file()
                          else args.scenario_dir / f"lung_stage_{stage:.2f}.vtp")
        fig.add_trace(mesh_trace(predicted_path, "#e47d2d",
                                 "Лёгкие: КТ + поле сети", 0.46))
        fig.add_trace(mesh_trace(args.evaluation_dir / f"observed_heart_{phase}.vtp",
                                 "#c73859", "Сердце: реальная КТ", 0.69))
        fig.add_trace(mesh_trace(args.scenario_dir / f"heart_stage_{stage:.2f}.vtp",
                                 "#8331a6", "Сердце: перенос поля", 0.64))
        lung_error = row.get("lung_warped_ct_surface_vs_observed",
                             row["lung_neural_prediction_vs_observed"])["symmetric_mean_mm"]
        descriptions.append(
            f"{phase}: объём лёгких {row['lung_volume_ml']/1000:.2f} л; "
            f"ошибка поверхности {lung_error:.1f} мм; "
            f"ошибка центра сердца {row['heart_predicted_centroid_error_mm']:.1f} мм"
        )
    initial = [i < 4 for i in range(len(fig.data))]
    for trace, visible in zip(fig.data, initial):
        trace.visible = visible
    fig.update_layout(
        title="100_HM10395: наблюдаемые и рассчитанные дыхательные фазы",
        height=780,
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(x=0.01, y=0.98),
        uirevision="mp10-tcia-phase-camera",
        scene=dict(
            xaxis_title="L (мм)", yaxis_title="P (мм)", zaxis_title="S (мм)",
            aspectmode="data", uirevision="mp10-tcia-phase-camera",
            bgcolor="#f8fafc",
        ),
    )
    plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False, div_id="phase_plot")
    controls = """<div style='font:16px system-ui;margin:14px;display:flex;align-items:center;gap:12px'>
      <button id='play' style='padding:6px 14px'>▶ Воспроизвести</button>
      <input id='phase' type='range' min='0' max='PHASE_MAX' value='0' step='1' style='width:240px'>
      <strong id='phase_description'></strong>
      </div>""".replace("PHASE_MAX", str(len(rows) - 1))
    script = """<script>
      const descriptions = DESCRIPTIONS;
      const plot = document.getElementById('phase_plot');
      const slider = document.getElementById('phase');
      const label = document.getElementById('phase_description');
      const button = document.getElementById('play');
      let timer = null;
      function showPhase(k) {
        const shown = Array.from({length: TRACE_COUNT}, (_, i) => Math.floor(i / 4) === k);
        Plotly.restyle(plot, {visible: shown});
        label.textContent = descriptions[k];
      }
      slider.addEventListener('input', () => showPhase(Number(slider.value)));
      button.addEventListener('click', () => {
        if (timer) {clearInterval(timer); timer = null; button.textContent = '▶ Воспроизвести'; return;}
        button.textContent = '⏸ Остановить';
        timer = setInterval(() => {
          slider.value = (Number(slider.value) + 1) % descriptions.length;
          showPhase(Number(slider.value));
        }, 900);
      });
      showPhase(0);
      </script>""".replace("DESCRIPTIONS", json.dumps(descriptions, ensure_ascii=False)).replace(
        "TRACE_COUNT", str(len(fig.data))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("<!doctype html><html lang='ru'><meta charset='utf-8'><body>"
                           + controls + plot_html + script + "</body></html>", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
