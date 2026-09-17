from pathlib import Path
p=Path('MATLAB_TRKG4_real_subjects/tools/build_heart_sphere_methods_report.py')
s=p.read_text(encoding='utf-8')
s=s.replace('        0.05,\n        f"Центр общий: (0, 0, 0) мм\\n$r_V=(abc)^{{1/3}}={r_volume:.2f}$ мм",',
'''        0.83,
        f"a = 42, b = 26, c = 18 мм\\\\nЦентр: (0, 0, 0) мм\\\\n$r_V=(abc)^{{1/3}}={r_volume:.2f}$ мм",''')
s=s.replace('сохраняемая $m_2$', 'сохраняемый момент $m_2$')
s=s.replace('ax.text(.5, .585,', 'ax.text(.5, .617,')
start=s.index('def _make_validation('); end=s.index('def _make_figures(',start)
replacement=r'''def _make_validation(path: Path) -> None:
    plt, _, _, _, _, Rectangle = _plotting()
    fig = plt.figure(figsize=(13.0, 7.5))
    ax = fig.add_axes((.075, .22, .45, .62))
    info = fig.add_axes((.59, .16, .39, .68))
    fig.suptitle("Рисунок 6. Подгонка и проверка на разных сборках и временных блоках",
                 color=PALETTE["ink"], y=.96, fontsize=14)
    for row in range(4):
        for col in range(6):
            kind, label = ("assembly_validation", "В") if row == 3 else (
                ("frame_validation", "К") if col in (1, 4) else ("train", "О"))
            ax.add_patch(Rectangle((col-.46, row-.40), .92, .80,
                         facecolor=PALETTE[kind], edgecolor="white", linewidth=2))
            ax.text(col, row, label, ha="center", va="center", color="white",
                    fontsize=13, fontweight="bold")
    ax.set(xlim=(-.52, 5.52), ylim=(3.52, -.52))
    ax.set_xticks(range(6), [f"B{i}" for i in range(1, 7)])
    ax.set_yticks(range(4), [f"Сборка {s}" for s in "ABCD"])
    ax.set_xlabel("Временные блоки; внутри блока несколько кадров", labelpad=12)
    ax.set_ylabel("Электродная сборка")
    ax.set_title("Пример разделения электрических откликов", fontsize=12, pad=15)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    info.axis("off")
    info.text(0, .98, "Правило фиксируется до подгонки", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["ink"])
    for y, kind, label in [
        (.85, "train", "О — подгонка: A–C × B1, B3, B4, B6"),
        (.75, "frame_validation", "К — проверка: A–C × B2, B5"),
        (.65, "assembly_validation", "В — новая сборка: D × все блоки")]:
        info.add_patch(Rectangle((0, y-.022), .045, .044,
                       facecolor=PALETTE[kind], transform=info.transAxes))
        info.text(.065, y, label, va="center", fontsize=10, color=PALETTE["ink"])
    info.text(0, .52, "Геометрическая сфера", va="top", color=PALETTE["blue"],
              fontsize=11, fontweight="bold")
    info.text(0, .455, "Параметры определяются по заявленной маске.\n"
              "Электрические отклики не используются для подгонки.",
              va="top", fontsize=10, color=PALETTE["ink"])
    info.text(0, .31, "Электрическая сфера", va="top", color=PALETTE["purple"],
              fontsize=11, fontweight="bold")
    info.text(0, .245, "Проверочные отклики не используются при построении.\n"
              "Они служат для оценки переноса на другие условия.",
              va="top", fontsize=10, color=PALETTE["ink"])
    info.text(0, .09, "Блоки и сборки выбираются заранее.\n"
              "Соседство кадров не обеспечивает независимость.\n"
              "Сравнение внутри модели не заменяет физическую валидацию.",
              va="top", fontsize=9.5, color=PALETTE["muted"])
    _finish_figure(fig, path)

'''
p.write_text(s[:start]+replacement+s[end:],encoding='utf-8',newline='\n')
