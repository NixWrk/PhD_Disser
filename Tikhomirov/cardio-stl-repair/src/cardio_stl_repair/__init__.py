"""cardio-stl-repair — Python port of `Project/RFFI.m`.

Назначение: автоматическая починка триангулированных STL-сеток сердца
(закрытие дыр через морфологическое замыкание + повторное извлечение
поверхности через marching cubes), взвешенное прореживание точек,
экспорт в CSV.

Публичный API (когда будет реализован):
    heart_repair, heart_repair_test       — починка одного mesh
    task4, task4_export                   — fix + repair pipeline
    export_points_to_file                 — STL → CSV
    w_point, w_point_packet               — взвешенное прореживание
    parse_filename                        — разбор имени STL-файла
    prepare_stl_batch, stl_to_points_batch, packet_heart_repair
                                          — пакетные операции
"""

__version__ = "0.0.1"
