"""Side-by-side 3D comparison of observed and predicted TCIA respiratory phases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import pyvista as pv


def mesh_trace(path: Path, color: str, opacity: float, scene: str,
               bounds: list[tuple[float, ...]]) -> go.Mesh3d:
    mesh = pv.read(str(path)).extract_surface().triangulate().clean()
    if mesh.n_cells > 15000:
        mesh = mesh.decimate(1 - 15000 / mesh.n_cells)
    bounds.append(tuple(mesh.bounds))
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = mesh.points
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, scene=scene,
        flatshading=True, visible=False, hoverinfo="skip", showlegend=False,
    )


def magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evaluation_dir / "evaluation.json").read_text(encoding="utf-8"))
    rows = report["metrics"]
    source = report["source_phase"]
    source_row = next(row for row in rows if row["phase"] == source)
    source_real_ml = source_row["lung_volume_ml"]
    fig = go.Figure()
    bounds: list[tuple[float, ...]] = []

    source_lung = args.evaluation_dir / f"observed_lung_{source}.vtp"
    for scene in ("scene", "scene2"):
        ghost = mesh_trace(source_lung, "#6b7280", 0.13, scene, bounds)
        ghost.visible = True
        fig.add_trace(ghost)

    descriptions = []
    for row in rows:
        phase = row["phase"]
        stage = row["model_stage"]
        predicted_lung = args.evaluation_dir / f"predicted_ct_lung_{phase}.vtp"
        if not predicted_lung.is_file():
            raise FileNotFoundError(predicted_lung)
        fig.add_trace(mesh_trace(args.evaluation_dir / f"observed_lung_{phase}.vtp",
                                 "#1688ad", 0.83, "scene", bounds))
        fig.add_trace(mesh_trace(args.evaluation_dir / f"observed_heart_{phase}.vtp",
                                 "#cb3659", 0.95, "scene", bounds))
        fig.add_trace(mesh_trace(predicted_lung, "#e07820", 0.83, "scene2", bounds))
        fig.add_trace(mesh_trace(args.scenario_dir / f"heart_stage_{stage:.2f}.vtp",
                                 "#7c3db6", 0.95, "scene2", bounds))
        real_ml = row["lung_volume_ml"]
        predicted_ml = row["lung_predicted_volume_ml"]
        real_delta = real_ml - source_real_ml
        predicted_delta = predicted_ml - source_real_ml
        real_heart_mm = magnitude(row["heart_observed_centroid_shift_lps_mm"])
        predicted_heart_mm = magnitude(row["heart_predicted_centroid_shift_lps_mm"])
        descriptions.append({
            "phase": phase,
            "real": (f"КТ: {real_ml / 1000:.2f} л; от входа {real_delta:+.0f} мл "
                     f"({real_delta / source_real_ml:+.1%}); центр сердца {real_heart_mm:.2f} мм"),
            "predicted": (f"Прогноз: {predicted_ml / 1000:.2f} л; от входа "
                          f"{predicted_delta:+.0f} мл "
                          f"({predicted_delta / source_real_ml:+.1%}); "
                          f"центр сердца {predicted_heart_mm:.2f} мм"),
        })

    ranges = []
    for axis in range(3):
        minimum = min(item[2 * axis] for item in bounds)
        maximum = max(item[2 * axis + 1] for item in bounds)
        ranges.append([minimum - 10, maximum + 10])
    scene_layout = dict(
        xaxis=dict(title="L (мм)", range=ranges[0]),
        yaxis=dict(title="P (мм)", range=ranges[1]),
        zaxis=dict(title="S (мм)", range=ranges[2]),
        aspectmode="data", bgcolor="#f8fafc", uirevision="mp10-tcia-camera",
    )
    initial_index = next((i for i, row in enumerate(rows) if row["phase"] == "g050"), 0)
    for index, trace in enumerate(fig.data[2:]):
        trace.visible = index // 4 == initial_index
    fig.update_layout(
        height=720, width=1500, margin=dict(l=0, r=0, b=0, t=4),
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        uirevision="mp10-tcia-camera", showlegend=False,
        scene=dict(**scene_layout, domain=dict(x=[0, 0.49], y=[0, 1])),
        scene2=dict(**scene_layout, domain=dict(x=[0.51, 1], y=[0, 1])),
    )
    plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False, div_id="phase_plot")
    controls = f"""<style>body {{font:16px system-ui;margin:12px;color:#1f2937}}
      .panels {{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
      .panel {{padding:8px 12px;border-radius:8px;background:#edf4f8}}
      .panel.forecast {{background:#fff2e5}}</style>
      <h2>Дыхательная 4D-КТ: наблюдение и прогноз из одного кадра {source}</h2>
      <p>Открытый случай TCIA 100_HM10395. Это не КТ Ника. Для фазы {source}
      глубина вдоха и задержка дыхания не подтверждены. Серый полупрозрачный
      контур в обеих сценах — один и тот же входной кадр {source}; масштаб общий.</p>
      <div style='display:flex;align-items:center;gap:12px'>
      <button id='play' style='padding:6px 14px'>▶ Воспроизвести</button>
      <input id='phase' type='range' min='0' max='{len(rows) - 1}'
      value='{initial_index}' step='1' style='width:300px'>
      <strong id='phase_name'></strong></div>
      <div class='panels'><div class='panel'><b>Снятая КТ · голубые лёгкие, красное сердце</b><br>
      <span id='observed_description'></span></div>
      <div class='panel forecast'><b>Прогноз · оранжевые лёгкие, фиолетовое сердце</b><br>
      <span id='predicted_description'></span></div></div>"""
    script = """<script>
      const descriptions = DESCRIPTIONS;
      const plot = document.getElementById('phase_plot');
      const slider = document.getElementById('phase');
      const button = document.getElementById('play');
      let timer = null;
      let syncingCamera = false;
      function showPhase(k) {
        const shown = Array.from({length: TRACE_COUNT}, (_, i) =>
          i < 2 || Math.floor((i - 2) / 4) === k);
        Plotly.restyle(plot, {visible: shown});
        document.getElementById('phase_name').textContent = descriptions[k].phase;
        document.getElementById('observed_description').textContent = descriptions[k].real;
        document.getElementById('predicted_description').textContent = descriptions[k].predicted;
      }
      plot.on('plotly_relayout', event => {
        if (syncingCamera) return;
        const left = event['scene.camera'];
        const right = event['scene2.camera'];
        if (left && !right) {
          syncingCamera = true;
          Plotly.relayout(plot, {'scene2.camera': left}).finally(() => syncingCamera = false);
        } else if (right && !left) {
          syncingCamera = true;
          Plotly.relayout(plot, {'scene.camera': right}).finally(() => syncingCamera = false);
        }
      });
      slider.addEventListener('input', () => showPhase(Number(slider.value)));
      button.addEventListener('click', () => {
        if (timer) {clearInterval(timer); timer = null; button.textContent = '▶ Воспроизвести'; return;}
        button.textContent = '⏸ Остановить';
        timer = setInterval(() => {
          slider.value = (Number(slider.value) + 1) % descriptions.length;
          showPhase(Number(slider.value));
        }, 900);
      });
      showPhase(Number(slider.value));
      </script>""".replace("DESCRIPTIONS", json.dumps(descriptions, ensure_ascii=False)).replace(
        "TRACE_COUNT", str(len(fig.data))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("<!doctype html><html lang='ru'><meta charset='utf-8'><body>"
                           + controls + plot_html + script + "</body></html>", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
