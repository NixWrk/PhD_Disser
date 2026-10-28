"""Separate 2D control on the frozen Nik whole-heart geometry baseline.

Reuses silhouette/equiarc and the unchanged EqualSphere implementation. No
3D centre depth, sphere volume, electrical response or cardiac function is inferred.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from shapely.geometry import Polygon

import run_tikhomirov_ct_contours as original
from cardio_model import sphere_fit
from run_heart_method_comparison import get_mask
from run_reduced_heart_geometry import digest, serial, write_json

HERE = Path(__file__).resolve().parent
COORDINATES = {'sagittal': ('S', 'A'), 'frontal': ('R', 'S')}
BOUND_TOL_MM = 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def projection_geometry(mask, affine, plane):
    """Native RAS projection cell area/centroid, without replacing silhouette."""
    image = nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), affine))
    matrix = image.affine[:3, :3]
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-8, rtol=0) or np.any(np.diag(matrix) <= 0):
        raise ValueError('Native projection requires axis-aligned positive RAS; oblique grids need a separate protocol')
    volume = np.asanyarray(image.dataobj).astype(bool)
    drop = 0 if plane == 'sagittal' else 1
    projected = volume.any(axis=drop)
    pixels = np.argwhere(projected)
    mean = pixels.mean(axis=0)
    index = np.array([0., mean[0], mean[1]]) if drop == 0 else np.array([mean[0], 0., mean[1]])
    center = original.planar_centroid(nib.affines.apply_affine(image.affine, index), plane)
    area_per_pixel = float(np.prod(np.diag(matrix)[[1, 2] if drop == 0 else [0, 2]]))
    return center, int(len(pixels)), area_per_pixel


def fit_with_diagnostics(points, method):
    """Observe scipy's result inside EqualSphere; delegate arguments unchanged.

    Local sequential process only. The module attribute is restored even on
    failure; source files and optimizer settings are never modified.
    """
    calls = []
    real_minimize = sphere_fit.minimize

    def observed_minimize(*args, **kwargs):
        result = real_minimize(*args, **kwargs)
        bounds = kwargs.get('bounds')
        lower, upper = np.asarray(bounds.lb), np.asarray(bounds.ub)
        x = np.asarray(result.x)
        names = ['center_1_mm', 'center_2_mm', 'radius_mm'] if method == 'Use_Contour' else ['radius_mm']
        calls.append({
            'algorithm': kwargs.get('method'), 'options': kwargs.get('options'),
            'parameter_names': names, 'initial_parameters': np.asarray(args[1]).tolist(),
            'parameters': x.tolist(), 'lower_bounds': lower.tolist(), 'upper_bounds': upper.tolist(),
            'success': bool(result.success), 'status_code': int(result.status),
            'message': str(result.message), 'nfev': int(result.nfev), 'nit': int(result.nit),
            'objective_sum_mm2': float(result.fun),
            'lower_bound_hits': (np.abs(x-lower) <= BOUND_TOL_MM).tolist(),
            'upper_bound_hits': (np.abs(x-upper) <= BOUND_TOL_MM).tolist(),
            'distance_to_lower_bounds_mm': (x-lower).tolist(),
            'distance_to_upper_bounds_mm': (upper-x).tolist(),
            'final_simplex': [np.asarray(a).tolist() for a in result.final_simplex],
        })
        return result

    sphere_fit.minimize = observed_minimize
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            center, radius = original.equal_sphere(points, method)
    finally:
        sphere_fit.minimize = real_minimize
    expected_calls = 0 if method == 'Use_Contour_MassCenter' else 1
    if len(calls) != expected_calls:
        raise ValueError('Unexpected EqualSphere optimizer call count')
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid fitted circle')
    hit = any(any(c['lower_bound_hits']) or any(c['upper_bound_hits']) for c in calls)
    converged = calls[0]['success'] if calls else None
    status = ('analytic' if not calls else 'solver_not_converged' if not converged
              else 'solver_converged_at_bound' if hit else 'solver_converged')
    return np.asarray(center), float(radius), {
        'status': status, 'solver_success': converged, 'any_bound_hit': hit,
        'calls': calls, 'warnings': [str(w.message) for w in caught],
        'observation_only_no_solver_change': True,
    }


def decorate_motion(rows):
    for cycle, plane, method in sorted({(r['cycle_index'], r['plane'], r['method']) for r in rows}):
        group = sorted((r for r in rows if (r['cycle_index'], r['plane'], r['method']) == (cycle, plane, method)),
                       key=lambda r: r['derived_time_from_R0_ms'])
        anchors = [r for r in group if r['is_reference_phase']]
        if len(anchors) != 1 or anchors[0] is not group[0]:
            raise ValueError('Expected exactly one earliest available anchor per R-R')
        anchor = anchors[0]
        for row in group:
            for kind in ('fit', 'projection', 'projected_3d_centroid'):
                delta = np.array([row[f'{kind}_center_{j}_mm']-anchor[f'{kind}_center_{j}_mm'] for j in (1, 2)])
                row[f'{kind}_delta_1_mm'], row[f'{kind}_delta_2_mm'] = delta.tolist()
                row[f'{kind}_displacement_mm'] = float(np.linalg.norm(delta))
            for ref in ('projection', 'projected_3d_centroid'):
                error = np.array([row[f'fit_delta_{j}_mm']-row[f'{ref}_delta_{j}_mm'] for j in (1, 2)])
                row[f'motion_error_vs_{ref}_mm'] = float(np.linalg.norm(error))


def summarize(rows):
    result = []
    for cycle, plane, method in sorted({(r['cycle_index'], r['plane'], r['method']) for r in rows}):
        group = [r for r in rows if (r['cycle_index'], r['plane'], r['method']) == (cycle, plane, method)]
        nonref = [r for r in group if not r['is_reference_phase']]
        def rms(key, source):
            return float(np.sqrt(np.mean([r[key]**2 for r in source])))
        result.append({
            'cycle_index': cycle, 'plane': plane, 'method': method,
            'phases': len(group), 'nonreference_phases': len(nonref),
            'reference_phase_id': next(r['phase_id'] for r in group if r['is_reference_phase']),
            'motion_RMS_vs_projection_nonreference_mm': rms('motion_error_vs_projection_mm', nonref),
            'motion_RMS_vs_projection_all_mm': rms('motion_error_vs_projection_mm', group),
            'motion_max_vs_projection_mm': max(r['motion_error_vs_projection_mm'] for r in group),
            'motion_RMS_vs_projected_3d_centroid_nonreference_mm': rms('motion_error_vs_projected_3d_centroid_mm', nonref),
            'center_offset_RMS_vs_projection_mm': rms('center_offset_vs_projection_mm', group),
            'radius_min_mm': min(r['radius_mm'] for r in group),
            'radius_max_mm': max(r['radius_mm'] for r in group),
            'radius_range_mm': float(np.ptp([r['radius_mm'] for r in group])),
            'circle_area_bias_vs_projection_percent': float(np.mean([r['circle_area_error_vs_projection_percent'] for r in group])),
            'circle_area_RMS_vs_projection_percent': rms('circle_area_error_vs_projection_percent', group),
            'fit_max_excursion_mm': max(r['fit_displacement_mm'] for r in group),
            'projection_max_excursion_mm': max(r['projection_displacement_mm'] for r in group),
            'solver_calls': sum(r['solver_success'] is not None for r in group),
            'solver_nonconverged': sum(r['solver_success'] is False for r in group),
            'bound_hits': sum(r['any_bound_hit'] for r in group),
            'target_boundary_phases': sum(r['target_touches_CT_boundary'] for r in group),
        })
    return result


def write_readme(out, summary, protocol):
    text = [
        '# Отдельный 2D-контроль контурных критериев на WHOLE_HEART Ника\n',
        '28.10.2026. Исследовательский геометрический расчёт на тех же 12 автоматических масках целого сердца, что в полном 3D-прогоне. Два R–R анализируются раздельно. Эти результаты **не входят в 28 пространственных методов** и не задают объём шара, третью координату центра, УО или ФВ.\n',
        'Существующие `silhouette`, `equiarc` и `equal_sphere` использованы без изменения. Получены 96 сочетаний: 12 фаз × 2 плоскости × 4 позиции. Это три критерия и вариант первого с семью точками, а не четыре независимых критерия. Сагиттальные координаты — (S, A), фронтальные — (R, S), физические миллиметры RAS; центр сохраняется в исходной системе координат.\n',
        'Контур построен по выпуклой оболочке бинарной проекции целого сердца. Это не исходные электродно привязанные МРТ-ориентиры Тихомирова. Площадь исходной проекции, площадь растровой выпуклой оболочки, площадь внешнего контура и площадь 7/128-точечного полигона сохранены отдельно. Их различия не скрываются нормировкой. `Area128` точно сохраняет площадь собственного 128-точечного полигона, а не исходной невыпуклой проекции.\n',
        'Оригинальные ограничения оптимизации действуют в локальных координатах относительно проекции 3D-центроида первой доступной фазы каждого R–R. Для `Use_Contour`: центр1 ∈ [−40;40], центр2 ∈ [−60;40], радиус ∈ [20;60] мм; для `Use_Contour_Radius` радиус ∈ [0;60] мм. Они не расширялись и не объявляются анатомическими нормами. `Area128` вычисляется аналитически и не имеет этих ограничений. Перехват вызова `minimize` только сохраняет возвращаемый результат; аргументы, критерий и решатель не изменяются. `success` означает остановку по критериям локального алгоритма, не глобальный минимум и не сходимость по дискретизации.\n',
        'Основной показатель движения — RMS векторной ошибки `[c_fit(t)−c_fit(t0)]−[c_projection(t)−c_projection(t0)]` относительно центроида исходной 2D-проекции. В таблицах опорная фаза исключена из RMS, чтобы её тождественный ноль не улучшал результат. Полный показатель с опорой и сравнение с проекцией 3D-центроида также сохранены. Это различия геометрических определений центра на одной маске, а не независимая точность анатомического движения. Первый кадр не называется ED; размах радиуса не интерпретируется как УО.\n',
        '<details>\n<summary>Таблица 1. Четыре позиции существующей ветки</summary>\n\n',
        '| Позиция | Метод библиотеки | Определение |\n|---|---|---|\n',
        '| LS128 | Use_Contour | МНК расстояний: свободный центр и радиус, 128 равноудалённых по дуге точек |\n',
        '| Area128 | Use_Contour_MassCenter | Центроид и равная площадь 128-точечного полигона |\n',
        '| FixedCenter128 | Use_Contour_Radius | Центроид 128-точечного полигона, подбор радиуса МНК |\n',
        '| LS7_uniform | Use_Contour | Тот же критерий, 7 равномерных точек; не исторические 7 ориентиров |\n\n</details>\n',
    ]
    for cycle in sorted({r['cycle_index'] for r in summary}):
        group = [r for r in summary if r['cycle_index'] == cycle]
        text += [f'\n<details>\n<summary>Таблица {cycle+2}. R–R {cycle}: {group[0]["phases"]} фаз, опора {group[0]["reference_phase_id"]}</summary>\n\n',
                 '| Плоскость | Позиция | RMS движения, мм | Радиус min–max, мм | Смещение площади, % | RMS площади, % | Несходимость / вызовы | Граница / фазы |\n',
                 '|---|---|---:|---:|---:|---:|---:|---:|\n']
        for r in group:
            text.append(f'| {r["plane"]} | {r["method"]} | {r["motion_RMS_vs_projection_nonreference_mm"]:.3f} | {r["radius_min_mm"]:.3f}–{r["radius_max_mm"]:.3f} | {r["circle_area_bias_vs_projection_percent"]:.3f} | {r["circle_area_RMS_vs_projection_percent"]:.3f} | {r["solver_nonconverged"]}/{r["solver_calls"]} | {r["bound_hits"]}/{r["phases"]} |\n')
        text.append('\n</details>\n')
    text += ['\nФайлы: [пофазные показатели](phase_fits.csv), [сводка](summary.csv), [диагностика каждого вызова](solver_diagnostics.json), [контуры в RAS](contours.json), [протокол и происхождение](provenance.local.json), [проверки](checks.json).\n',
             '\nВсе исходные медицинские файлы только читались. Их хэши, affine, источник временной разметки, версии библиотек и исходников записаны в provenance. Проверки кода и привязки исходников не являются независимой анатомической валидацией. Влияние усечения поля зрения и вариабельности автоматической сегментации сохраняется. Отдельная проверка плотности контура сверх исходных 7/128 точек не выполнялась.\n']
    (out/'README.md').write_text(''.join(text), encoding='utf-8')


def run(base):
    base = Path(base).resolve()
    cfg_path, baseline_path = base/'inputs/config.local.json', base/'inputs/geometry_comparison.jsonl'
    cfg = read(cfg_path)
    baseline = [json.loads(line) for line in baseline_path.read_text(encoding='utf-8').splitlines()]
    selected = [r for r in baseline if r['subject'] == 'nix' and r['region'] == 'whole_heart']
    selected.sort(key=lambda r: (r['cycle_index'], r['derived_time_from_R0_ms']))
    if len(selected) != 12 or len({r['phase_id'] for r in selected}) != 12:
        raise ValueError('Expected exactly 12 unique Nik whole-heart phases')
    if dict(Counter(r['cycle_index'] for r in selected)) != {0: 9, 1: 3}:
        raise ValueError('Expected separate 9-phase and 3-phase R-R intervals')
    for cycle in (0, 1):
        group = [r for r in selected if r['cycle_index'] == cycle]
        if sum(r['is_reference_phase'] for r in group) != 1 or not group[0]['is_reference_phase']:
            raise ValueError('Invalid per-cycle reference')
    out = base/'contour_control'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be new or empty; previous evidence is immutable')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'status.json', {'status': 'running', 'expected_fits': 96})
    sources = [Path(__file__), Path(original.__file__), Path(sphere_fit.__file__),
               HERE/'run_heart_method_comparison.py', HERE/'run_reduced_heart_geometry.py',
               HERE/'cardiac_function_analysis.py']
    original_hashes = {str(p): digest(p) for p in sources}
    protocol = {
        'schema': 'nik_whole_heart_contour_control_v1', 'evidence_status': 'exploratory_hypothesis_not_validated',
        'baseline_sha256': digest(baseline_path), 'config_sha256': digest(cfg_path),
        'code_sha256': original_hashes, 'python': sys.version, 'platform': platform.platform(),
        'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'nibabel', 'scikit-image', 'shapely')},
        'source_region': 'whole_heart', 'subject': 'nix', 'planes': COORDINATES,
        'source_time_table_sha256': digest(Path(cfg['time_table'])),
        'methods': original.METHODS, 'bound_tolerance_mm': BOUND_TOL_MM,
        'independent_anatomical_reference': False, '3D_candidate': False,
        'sphere_volume_inferred': False, '3D_center_depth_inferred': False,
        'FEM_executed': False, 'medical_inputs_modified': False,
        'motion_reference': 'unhulled_2D_projection_area_centroid',
        'secondary_motion_reference': 'projection_of_3D_voxel_volume_centroid',
        'local_optimizer_origin': 'projected_3D_centroid_of_earliest_available_phase_per_RR',
        'surface': 'existing_silhouette_convex_hull_and_find_contours_0.5',
        'phases': [],
    }
    rows, diagnostics, contours, origins = [], [], [], {}
    try:
        for source in selected:
            mask, affine = get_mask(Path(cfg['derived_root']), source)
            volume_center = original.voxel_centroid(mask, affine)
            if not np.allclose(volume_center, source['moments']['centroid_mm'], atol=1e-7, rtol=0):
                raise ValueError('Baseline centroid does not match the frozen mask')
            pid, cycle = source['phase_id'], source['cycle_index']
            protocol['phases'].append({k: source[k] for k in ('phase_id', 'cycle_index', 'ct_sha256',
                'provenance', 'reference_phase_id', 'is_reference_phase', 'phase_percent_within_cycle',
                'derived_time_from_R0_ms', 'rr_interval_ms', 'temporal_resolution_ms')})
            for plane in original.PLANES:
                projected3d = original.planar_centroid(volume_center, plane)
                origin = origins.setdefault((cycle, plane), projected3d.copy())
                outer, npix, nhull = original.silhouette(mask, affine, plane)
                projection_center, checked_np, pixel_area = projection_geometry(mask, affine, plane)
                if npix != checked_np:
                    raise ValueError('Projection count disagrees with existing silhouette')
                outer_poly = Polygon(outer)
                if not outer_poly.is_valid or outer_poly.area <= 0:
                    raise ValueError('Invalid original outer contour')
                contours.append({'phase_id': pid, 'cycle_index': cycle, 'plane': plane,
                    'coordinate_names': COORDINATES[plane], 'outer_contour_RAS_mm': outer.tolist(),
                    'projection_center_RAS_mm': projection_center.tolist(), 'local_origin_RAS_mm': origin.tolist()})
                for name, method, count in original.METHODS:
                    sampled = original.equiarc(outer, count)
                    polygon = Polygon(sampled)
                    if not polygon.is_valid or polygon.area <= 0:
                        raise ValueError('Invalid equally spaced contour polygon')
                    center_local, radius, diagnostic = fit_with_diagnostics(sampled-origin, method)
                    center = center_local+origin
                    residual = np.linalg.norm(sampled-center, axis=1)-radius
                    area, projection_area = float(np.pi*radius**2), npix*pixel_area
                    record = {
                        'subject': 'nix', 'region': 'whole_heart', 'phase_id': pid, 'cycle_index': cycle,
                        'phase_percent_within_cycle': source['phase_percent_within_cycle'],
                        'derived_time_from_R0_ms': source['derived_time_from_R0_ms'],
                        'rr_interval_ms': source['rr_interval_ms'], 'is_reference_phase': source['is_reference_phase'],
                        'reference_phase_id': source['reference_phase_id'], 'plane': plane, 'method': name,
                        'library_method': method, 'sample_count': count, 'status': diagnostic['status'],
                        'solver_success': diagnostic['solver_success'], 'any_bound_hit': diagnostic['any_bound_hit'],
                        'fit_center_1_mm': float(center[0]), 'fit_center_2_mm': float(center[1]),
                        'projection_center_1_mm': float(projection_center[0]), 'projection_center_2_mm': float(projection_center[1]),
                        'projected_3d_centroid_center_1_mm': float(projected3d[0]), 'projected_3d_centroid_center_2_mm': float(projected3d[1]),
                        'local_origin_1_mm': float(origin[0]), 'local_origin_2_mm': float(origin[1]),
                        'radius_mm': radius, 'circle_area_mm2': area, 'projection_area_mm2': projection_area,
                        'hull_raster_area_mm2': nhull*pixel_area, 'outer_contour_polygon_area_mm2': float(outer_poly.area),
                        'sampled_polygon_area_mm2': float(polygon.area), 'pixel_area_mm2': pixel_area,
                        'circle_area_error_vs_projection_percent': 100*(area/projection_area-1),
                        'circle_area_error_vs_sampled_polygon_percent': 100*(area/polygon.area-1),
                        'hull_area_excess_percent': 100*(nhull/npix-1),
                        'center_offset_vs_projection_mm': float(np.linalg.norm(center-projection_center)),
                        'radial_fit_RMS_mm': float(np.sqrt(np.mean(residual**2))),
                        'target_touches_CT_boundary': source['provenance']['touches_source_image_boundary'],
                    }
                    rows.append(record)
                    diagnostics.append({'phase_id': pid, 'cycle_index': cycle, 'plane': plane, 'method': name,
                        'sampled_contour_RAS_mm': sampled.tolist(), **diagnostic})
            write_json(out/'status.json', {'status': 'running', 'expected_fits': 96, 'completed_fits': len(rows), 'last_phase': pid})
            print(pid, cycle, len(rows), flush=True)
        decorate_motion(rows)
        summary = summarize(rows)
        checks = {
            'exact_96_unique_fits': len(rows) == len({(r['phase_id'], r['plane'], r['method']) for r in rows}) == 96,
            'exact_16_summary_groups': len(summary) == 16,
            'zero_anchor_motion_error': all(r['motion_error_vs_projection_mm'] == 0 for r in rows if r['is_reference_phase']),
            'analytic_area_preservation': all(abs(r['circle_area_error_vs_sampled_polygon_percent']) < 1e-9 for r in rows if r['method'] == 'Area128'),
            'source_code_unchanged': all(digest(Path(p)) == sha for p, sha in original_hashes.items()),
            'baseline_unchanged': digest(baseline_path) == protocol['baseline_sha256'],
            'no_unreported_optimizer_failure': all(r['status'] == 'solver_not_converged' for r in rows if r['solver_success'] is False),
        }
        if not all(checks.values()):
            raise ValueError('Output contract checks failed: '+str(checks))
        write_csv(out/'phase_fits.csv', rows)
        write_csv(out/'summary.csv', summary)
        write_json(out/'solver_diagnostics.json', diagnostics)
        write_json(out/'contours.json', contours)
        write_json(out/'provenance.local.json', protocol)
        write_json(out/'checks.json', checks)
        write_readme(out, summary, protocol)
        output_hashes = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'status.json'}
        status = {'status': 'completed', 'evidence_status': 'exploratory_hypothesis_not_validated',
                  'expected_fits': 96, 'completed_fits': len(rows), 'summary_groups': len(summary),
                  'statuses': dict(Counter(r['status'] for r in rows)), 'output_sha256': output_hashes}
        write_json(out/'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except Exception as exc:
        write_json(out/'status.json', {'status': 'failed', 'completed_fits': len(rows),
                                     'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='full_geometry_v1 directory')
    run(parser.parse_args().base)
