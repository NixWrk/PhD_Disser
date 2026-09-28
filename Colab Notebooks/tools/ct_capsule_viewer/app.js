(() => {
  "use strict";
  const meta = JSON.parse(document.getElementById("surface-meta").textContent);
  const plot = document.getElementById("surface-plot");
  const nameInput = document.getElementById("name");
  const noteInput = document.getElementById("note");
  const status = document.getElementById("status");
  const list = document.getElementById("points");
  const count = document.getElementById("count");
  let points = [];
  let pressedAt = null;
  let dragged = false;

  function say(message, error = false) {
    status.textContent = message;
    status.style.background = error ? "#fff0ed" : "#edf6f8";
  }

  function selectedSurfaceXYZ(clientX, clientY) {
    const scene = plot._fullLayout?.scene?._scene;
    const gl = scene?.glplot;
    const selection = gl?.selection;
    const canvas = gl?.canvas;
    if (!selection?.object || !canvas || gl.objects.indexOf(selection.object) !== 0) return null;
    const rect = canvas.getBoundingClientRect();
    if (!selection.mouse || Math.hypot(selection.mouse[0] - (clientX - rect.left),
      selection.mouse[1] - (clientY - rect.top)) > 4) return null;
    const position = selection.dataPosition;
    const scale = scene.dataScale;
    if (!position || !scale || position.length !== 3 || scale.length !== 3) return null;
    const xyz = position.map((value, index) => value / scale[index]);
    return xyz.every(Number.isFinite) ? xyz.map(value => Number(value.toFixed(6))) : null;
  }

  async function redraw() {
    const changes = {
      x: [points.map(p => p.xyz[0])],
      y: [points.map(p => p.xyz[1])],
      z: [points.map(p => p.xyz[2])],
      text: [points.map(p => p.name)]
    };
    await Plotly.restyle(plot, changes, [1]);
    count.textContent = String(points.length);
    list.replaceChildren();
    points.forEach((point, index) => {
      const item = document.createElement("li");
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", "Удалить " + point.name);
      remove.addEventListener("click", () => {
        points.splice(index, 1);
        redraw();
        say("Точка удалена.");
      });
      const title = document.createElement("strong");
      title.textContent = point.name;
      const coords = document.createElement("code");
      coords.textContent = point.xyz.map(n => n.toFixed(3)).join("; ");
      item.append(remove, title, coords);
      if (point.note) {
        const note = document.createElement("div");
        note.textContent = point.note;
        item.append(note);
      }
      list.append(item);
    });
  }

  function add(xyz) {
    const name = nameInput.value.trim() || "Метка " + (points.length + 1);
    const note = noteInput.value.trim();
    points.push({name, xyz, note, selection_method: "manual_plotly_mesh_click",
      rendering_revision: meta.rendering_revision});
    redraw();
    nameInput.value = "";
    noteInput.value = "";
    say(name + ": " + xyz.map(n => n.toFixed(3)).join(", ") + " (XYZ STL)");
  }

  function download(filename, mime, content) {
    const url = URL.createObjectURL(new Blob([content], {type: mime}));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  function payload() {
    return {...meta, annotation_status: "manual_unverified", landmarks: points};
  }

  function csvCell(value) {
    return '"' + String(value).replaceAll('"', '""') + '"';
  }

  document.getElementById("undo").addEventListener("click", () => {
    if (!points.length) return;
    points.pop();
    redraw();
    say("Последняя точка удалена.");
  });
  document.getElementById("clear").addEventListener("click", () => {
    if (!points.length || !confirm("Удалить все точки текущей разметки?")) return;
    points = [];
    redraw();
    say("Разметка очищена.");
  });
  document.getElementById("save-json").addEventListener("click", () => {
    download(meta.subject + "_capsule_landmarks.json", "application/json;charset=utf-8",
      JSON.stringify(payload(), null, 2) + "\n");
    say("JSON сохранён.");
  });
  document.getElementById("save-csv").addEventListener("click", () => {
    const header = ["subject", "source_stl_basename", "source_stl_sha256", "name", "x", "y", "z", "note", "selection_method", "rendering_revision"];
    const lines = [header.map(csvCell).join(",")];
    points.forEach(p => lines.push([
      meta.subject, meta.source_stl_basename, meta.source_stl_sha256,
      p.name, ...p.xyz, p.note, p.selection_method, p.rendering_revision
    ].map(csvCell).join(",")));
    download(meta.subject + "_capsule_landmarks.csv", "text/csv;charset=utf-8",
      "\ufeff" + lines.join("\r\n") + "\r\n");
    say("CSV сохранён.");
  });
  document.getElementById("import-json").addEventListener("change", async event => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    try {
      const imported = JSON.parse(await file.text());
      if (imported.subject !== meta.subject || imported.source_stl_sha256 !== meta.source_stl_sha256) {
        throw new Error("Разметка относится к другому испытуемому или другой версии STL.");
      }
      if (!Array.isArray(imported.landmarks) || !imported.landmarks.every(p =>
        typeof p.name === "string" && Array.isArray(p.xyz) && p.xyz.length === 3 &&
        p.xyz.every(Number.isFinite))) {
        throw new Error("Некорректный список координат.");
      }
      points = imported.landmarks.map(p => ({
        name: p.name, xyz: p.xyz, note: String(p.note || ""),
        selection_method: p.selection_method || "imported_manual_landmark",
        rendering_revision: p.rendering_revision || imported.rendering_revision || "unshaded_v1"
      }));
      await redraw();
      const older = points.filter(p => p.rendering_revision !== meta.rendering_revision).length;
      say("Загружено точек: " + points.length + (older ? ". Из них " + older +
        " выбраны до исправления освещения; проверьте их по КТ и рельефной модели." : ""), !!older);
    } catch (error) {
      say("Импорт не выполнен: " + error.message, true);
    }
    event.target.value = "";
  });

  const directions = {
    xp: {x: 2, y: 0, z: 0}, xm: {x: -2, y: 0, z: 0},
    yp: {x: 0, y: 2, z: 0}, ym: {x: 0, y: -2, z: 0},
    zp: {x: 0, y: 0, z: 2}, zm: {x: 0, y: 0, z: -2}
  };
  document.querySelectorAll("[data-view]").forEach(button => {
    button.addEventListener("click", () => {
      const key = button.dataset.view;
      const up = key === "zp" || key === "zm" ? {x: 0, y: 1, z: 0} : {x: 0, y: 0, z: 1};
      Plotly.relayout(plot, {"scene.camera": {eye: directions[key], up,
        center: {x: 0, y: 0, z: 0}, projection: {type: "orthographic"}}});
    });
  });

  plot.addEventListener("pointerdown", event => {
    pressedAt = [event.clientX, event.clientY];
    dragged = false;
  }, true);
  plot.addEventListener("pointermove", event => {
    if (pressedAt && Math.hypot(event.clientX - pressedAt[0], event.clientY - pressedAt[1]) > 4) dragged = true;
  }, true);
  window.addEventListener("pointerup", event => {
    if (!pressedAt) return;
    if (Math.hypot(event.clientX - pressedAt[0], event.clientY - pressedAt[1]) > 4) dragged = true;
    const didDrag = dragged;
    pressedAt = null;
    if (didDrag) return;
    requestAnimationFrame(() => {
      const xyz = selectedSurfaceXYZ(event.clientX, event.clientY);
      if (xyz) add(xyz);
      else say("Щёлкните по видимой поверхности тела.", true);
    });
  }, true);

  function attach(attempt = 0) {
    if (typeof plot.on !== "function") {
      if (attempt < 100) setTimeout(() => attach(attempt + 1), 50);
      else say("Не удалось запустить 3D‑график.", true);
      return;
    }
    say("Поверхность готова. Щёлкните по видимой метке капсулы.");
  }
  attach();
})();
