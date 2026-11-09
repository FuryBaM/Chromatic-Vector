# Применение метода Unipolar-Gambit

В Chromatic Vector добавлены два опциональных метода построения контуров:

```bash
python -m pip install -e '.[test]'
python -m chromatic_vector references/head_shoulder_crop.png head.svg \
  --mode illustrative --preset detail --contour-method unipolar-chaikin \
  --shape-report head.json --debug-dir debug/head

# Текущий адаптивный фильтр игры:
python -m chromatic_vector references/torso_crop.png torso.svg \
  --mode illustrative --preset detail --contour-method unipolar-adaptive
```

`--contour-method bounded` сохраняет вариант v0.6.1 и остаётся default. Новые
методы подключены к общим границам базовых фигур и к refinement overlays.
Они доступны в Python API через `contour_method='unipolar-chaikin'`.
Legacy `--quality precise` не изменён; для выбора нового метода нужен illustrative.

Проверены реальные исходники подключённого репозитория
[FuryBaM/Unipolar-Gambit](https://github.com/FuryBaM/Unipolar-Gambit)
на commit `6a826d823555baf789d68fced3825d5a26037527`:

- `Assets/Modules/Game.Rendering/BorderBuilder.cs`: активный `AdaptiveFilterPolyline`
  и сохранённый старый путь `TraceSimplifyAndSmooth` (RDP + `SmoothPolyline` / Chaikin).
- `Native/ProvinceEdges/ug_province_edges.cpp`: соответствующий нативный adaptive kernel.
- `Native/ProvinceEdges/README.md`: дополнительный marching-squares / distance-field
  рендеринг границ карты. Он строит текстуру расстояний; в этот SVG-конвертер
  перенесены геометрические фильтры. Полный шейдер карты не переносился.

В игровом коде активны tolerance=0.28, smoothStrength=0.22, sharpTurn=1.5.
Старый CV-фильтр использовал sharpTurn=0.65. В `unipolar-adaptive` исправлен именно
этот параметр; tolerance/smooth управляются параметрами CV и масштабом иллюстрации.
`unipolar-chaikin` использует исходное правило 3/4–1/4 и исходное разделение
замкнутого контура на две упорядоченные дуги для RDP. К результату добавлены
ограниченный Bézier fit, защита junctions, alpha, correction mask и проверка пересечений.

Зависимость от Unity/игры отсутствует. Репозиторий игры использовался только для
чтения исходников. MIT notice включён в `licenses/Unipolar-Gambit-MIT.txt`;
подробная атрибуция — `THIRD_PARTY_NOTICES.md`.

## Проверенный результат

Три колонки: `bounded` v0.6.1, UG adaptive, UG RDP + Chaikin.
Открыть `artifacts/unipolar_comparison/index.html`. PNG на 100/200/400%
получены прямым рендером каждого SVG. Два новых метода реально заново прогнаны
на head и torso; full sheet новым методом в этом эксперименте не прогонялся.

Chaikin визуально лучше округляет небольшие углы, adaptive сохраняет больше
ступенек. Число фигур сохранено: head 453, torso 381. Оставшиеся точные rect:

| Fixture | bounded | UG adaptive | UG Chaikin |
|---|---:|---:|---:|
| head | 54 | 34 | 34 |
| torso | 19 | 4 | 3 |

Одной смены фильтра недостаточно для устранения всей мозаики. Наивная логика v0.6.1
фиксировала даже небольшие растровые прямоугольники как геометрически точные.
В новых методах автоматическая фиксация ограничена достаточно крупными
прямоугольниками; ручные коррекции и alpha остаются точными. Часть углов закреплена
стыками нескольких цветовых областей. Их нельзя произвольно срезать независимо
с каждой стороны без изменения общей топологии и риска щелей.

Head RGB MAE: bounded 5.452; adaptive 5.420; Chaikin 5.497.
Torso RGB MAE: bounded 4.407; adaptive 4.392; Chaikin 4.441.
Это контролируемый компромисс: лучшее округление не даёт минимальную ошибку
к исходной пиксельной границе. SVG могут стать больше из-за дополнительных кривых.
Ни один метод не объявляется полностью решившим задачу художественной реконструкции.

Тесты: 77 passed, включая исходные 69, проверку коэффициентов исходного Chaikin,
порядка closed RDP, параметров adaptive, совпадения native-фильтра с Python,
общих реверсивных границ, перемещённого anchor изолированного цикла и сохранения
alpha/ручных коррекций. 10/10 сравнений legacy с исходной v0.5 совпали побайтно.
Метрики, причины fallback и decisions refinement доступны в JSON рядом с SVG.

Для воспроизведения сравнения из этого набора:

```bash
python -m pytest -q
python benchmarks/run_unipolar.py
# Только повторный рендер уже сохранённых SVG:
python benchmarks/run_unipolar.py --render-only
```

`ChromaticVector_Unipolar_Trial.zip` содержит текущие исходники, native CMake,
тесты, MIT notice, исходные изображения и эти сравнения. Большие исторические
benchmark/debug из предыдущих архивов в trial не включены. Это экспериментальный
вариант с явным выбором метода, а не обещание полного устранения всех квадратов.
