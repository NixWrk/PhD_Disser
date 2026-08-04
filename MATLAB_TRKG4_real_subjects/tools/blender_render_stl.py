"""Render orthographic front/side/top views of lungs with the heart."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def load(path: Path, name: str, color):
    bpy.ops.wm.stl_import(filepath=str(path.resolve()))
    obj = bpy.context.active_object
    obj.name = name
    obj.color = color
    return obj


def render_view(camera, target, location, up_axis, output):
    camera.location = Vector(location)
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", up_axis).to_euler()
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass lungs, heart and output prefix after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 3:
        raise SystemExit("Expected: lungs.stl heart.stl output_prefix")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    lungs = load(Path(args[0]), "lungs", (0.10, 0.55, 0.90, 1.0))
    heart = load(Path(args[1]), "heart", (0.85, 0.08, 0.05, 1.0))

    points = []
    for obj in (lungs, heart):
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    mins = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    maxs = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (mins + maxs) / 2
    span = max(maxs - mins)

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 1.15
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (1.0, 1.0, 1.0)

    prefix = Path(args[2]).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    distance = span * 2
    render_view(camera, center, center + Vector((0, -distance, 0)), "Y", prefix.with_name(prefix.name + "_front.png"))
    render_view(camera, center, center + Vector((distance, 0, 0)), "Z", prefix.with_name(prefix.name + "_side.png"))
    render_view(camera, center, center + Vector((0, 0, distance)), "Y", prefix.with_name(prefix.name + "_top.png"))


if __name__ == "__main__":
    main()
