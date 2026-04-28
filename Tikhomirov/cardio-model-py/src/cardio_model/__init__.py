"""cardio-model-py — Python port of WolframMath-master.

Радиоимпедансная кардиография совмещённая с МРТ.
Прямой портов модуля `Project/RFFI.m` тут нет — он живёт в соседнем
репозитории `cardio-stl-repair/`.

Модули:
    models      — OneLayerModel, TwoLayerModel, SphereModel
    finders     — FindRoOne, FindRSphere, FindDRSphere*
    radial      — RadEvalMethod1..4
    geometry    — KubicInterpol, GetCoordAfterMove, sections
    volume      — VolumeBySection, VolumeByContour, SVfinder
    sphere_fit  — EqualSphere (Nelder-Mead подгонка)
    sistole     — SistoleContour, ValveConturMove
    pipeline    — EqualSphereMoveModelling, RadialEvaluation, TotalModelling
    dsp         — FirFilter, GetR, MyDeriv, MyIntegrate
    importers   — Reo32, Comsol
    data        — пациентские данные (Ivan, Alex, Artem)
"""

__version__ = "0.0.1"
