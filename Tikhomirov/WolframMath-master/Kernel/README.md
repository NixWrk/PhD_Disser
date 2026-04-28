# Описание библиотеки Kernel
Список каталогов:
- `core` - основное функциональное ядро модели, расчет объема
- `part1` - функции связанные с расчетами перемещения центра сердца при сокращении
- `history` - только исходные данные

Дерево файлов:
```
lib
│-core
│   Models.m___________
│                      OneLayerModel[ro1,a,b]
│                      TwoLayerModel[ro1,ro2,h,a,b]
│                      SphereModel[ro1,ro2,a,b,R,h,x,y]
│
│   ModelFinders.m_____FindRoOne
│                       
│   RadialEvaluation.m
│   SphereModel.m
│   VolumeCalc.m
│-part1
│   SistoleFunction.m
│   SphereMuvionFunction
```

## Core

## Part1
### SystoleFunction



