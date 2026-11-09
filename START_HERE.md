# Chromatic Vector Neural — компактная сборка

Код, обученная ONNX-модель, тесты и один пример изображения.
Архив включает веса для запуска через OpenCV CPU. PyTorch для запуска не нужен.

Установка из каталога ChromaticVector:

    python -m pip install -e '.[test]'
    python -m pytest -q
    python -m chromatic_vector references/head_shoulder_crop.png head.svg --mode illustrative --preset detail --contour-method neural --max-side 0 --simplify 1.1 --smooth .9 --max-error 1.1 --target-shapes 250 --detail-budget 100

Веса и metadata.json: chromatic_vector/assets/bezier_patch_v1/.
Программа автоматически использует эту копию модели. Для переобучения
предусмотрен training/train_bezier.py; потребуются PyTorch и ONNX.

Из компактного архива исключены большие отчёты, галереи, исторические релизы,
контрольная точка PyTorch и дубли весов. Ссылки README_NEURAL.md на отчёты
относятся к полной сборке. Готовые SVG и сравнение доступны отдельными файлами
в сообщении с результатом.
