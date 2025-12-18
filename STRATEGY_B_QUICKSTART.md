# Быстрый старт: Стратегия B для подавления динамических артефактов

## Шаг 1: Подготовка данных

### 1.1. Создание масок динамических объектов

```bash
# Активируйте окружение
conda activate gaussian_splatting

# Создайте маски для вашего датасета
python preprocessing/create_mask.py
```

Это создаст файлы `*_mask.npy` в папке с изображениями. Маски определяют динамические объекты (люди, роботы, машины).

**Примечание:** Убедитесь, что в проекте есть файл `yolo11n-seg.pt` для сегментации.

### 1.2. Структура датасета

После создания масок структура должна быть:
```
dataset/
└── your_scene/
    ├── images/
    │   ├── image1.jpg
    │   ├── image1_mask.npy  # ← маска
    │   ├── image2.jpg
    │   ├── image2_mask.npy  # ← маска
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin
```

## Шаг 2: Запуск обучения

### 2.1. Базовый запуск (с настройками по умолчанию)

```bash
python train.py -s dataset/your_scene --use_strategy_b
```

### 2.2. Расширенный запуск (с кастомными параметрами)

```bash
python train.py -s dataset/your_scene \
  --use_strategy_b \
  --strategy_b_sfm_threshold 0.5 \
  --strategy_b_densify_threshold 0.3 \
  --strategy_b_prune_threshold 0.7 \
  --strategy_b_prune_min_obs 10 \
  --iterations 30000
```

### 2.3. Параметры стратегии B

| Параметр | Значение по умолчанию | Диапазон | Описание |
|----------|----------------------|----------|----------|
| `--use_strategy_b` | False (флаг) | - | Включить стратегию B |
| `--strategy_b_sfm_threshold` | 0.5 | 0.0 - 1.0 | Порог фильтрации SfM точек. Ниже = агрессивнее |
| `--strategy_b_densify_threshold` | 0.3 | 0.0 - 1.0 | Порог блокировки денсификации. Ниже = агрессивнее |
| `--strategy_b_prune_threshold` | 0.7 | 0.0 - 1.0 | Порог удаления гауссианов. Ниже = агрессивнее |
| `--strategy_b_prune_min_obs` | 10 | 1 - 100 | Минимум наблюдений перед удалением |

## Шаг 3: Мониторинг

Во время обучения вы увидите:

```
================================================================================
STRATEGY B: Dynamic Filtering Enabled
================================================================================
[Strategy B] Loaded 50 masks for training cameras
[Strategy B] MaskAwareDensifier initialized
[Strategy B] DynamicGaussianTracker initialized
[Strategy B] AdaptivePruningScheduler initialized

[Strategy B-a] Filtering SfM points based on dynamic masks...
[Strategy B-a] Points before filtering: 50000
[Strategy B-a] Points removed (dynamic): 5000 (10.0%)
[Strategy B-a] Points kept (static): 45000 (90.0%)

Training progress: 10%|██████ | 3000/30000
[Strategy B-b] Filtered 150 gaussians from densification (on dynamic areas)
[Strategy B-c] Pruning 250 dynamic gaussians (threshold=0.7, min_obs=10)
```

## Шаг 4: Рендеринг и оценка

После обучения:

```bash
# Рендеринг тестовых изображений
python render.py -m output/your_model

# Оценка качества
python metrics.py -m output/your_model
```

## Примеры использования

### Пример 1: Сцена с людьми (агрессивная фильтрация)

```bash
python train.py -s dataset/scene_with_people \
  --use_strategy_b \
  --strategy_b_sfm_threshold 0.3 \
  --strategy_b_densify_threshold 0.2 \
  --strategy_b_prune_threshold 0.6
```

### Пример 2: Сцена с небольшим движением (консервативная)

```bash
python train.py -s dataset/scene_little_motion \
  --use_strategy_b \
  --strategy_b_sfm_threshold 0.7 \
  --strategy_b_densify_threshold 0.4 \
  --strategy_b_prune_threshold 0.8
```

### Пример 3: Полный набор параметров

```bash
python train.py \
  -s dataset/drjohnson \
  -m output/drjohnson_strategy_b \
  --use_strategy_b \
  --strategy_b_sfm_threshold 0.5 \
  --strategy_b_densify_threshold 0.3 \
  --strategy_b_prune_threshold 0.7 \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000
```

## Сравнение результатов

Для сравнения со стандартным 3DGS:

```bash
# Обучение БЕЗ стратегии B (baseline)
python train.py -s dataset/your_scene -m output/baseline

# Обучение СО стратегией B
python train.py -s dataset/your_scene -m output/strategy_b --use_strategy_b

# Сравнение метрик
python metrics.py -m output/baseline
python metrics.py -m output/strategy_b
```

## Настройка параметров

### Когда уменьшать пороги (агрессивнее фильтровать):
- Много движущихся объектов
- Сильные артефакты в baseline
- Готовы потерять немного деталей ради чистоты

### Когда увеличивать пороги (консервативнее):
- Мало движущихся объектов
- Baseline работает хорошо, но есть небольшие артефакты
- Важна детализация

## Устранение проблем

### Проблема: "No masks found"
**Решение:** Убедитесь, что файлы `*_mask.npy` находятся в папке с изображениями.

### Проблема: Слишком мало гауссианов
**Решение:** Увеличьте пороги:
```bash
--strategy_b_sfm_threshold 0.7 \
--strategy_b_densify_threshold 0.5 \
--strategy_b_prune_threshold 0.8
```

### Проблема: Артефакты остались
**Решение:** Уменьшите пороги:
```bash
--strategy_b_sfm_threshold 0.3 \
--strategy_b_densify_threshold 0.2 \
--strategy_b_prune_threshold 0.6
```

### Проблема: Out of memory
**Решение:** 
1. Уменьшите разрешение: `--resolution 2`
2. Уменьшите количество итераций: `--iterations 20000`
3. Используйте более агрессивную фильтрацию

## Дополнительная информация

- Полная документация: `strategies/strategy_B_dynamic_filtering/README.md`
- Исходный код компонентов:
  - SfM фильтрация: `strategies/strategy_B_dynamic_filtering/sfm_filtering.py`
  - Контролируемая денсификация: `strategies/strategy_B_dynamic_filtering/controlled_splitting.py`
  - Удаление гауссианов: `strategies/strategy_B_dynamic_filtering/gaussian_pruning.py`

## Ожидаемые результаты

✅ Меньше "призраков" от движущихся объектов  
✅ Чище статические области  
✅ Меньше гауссианов (более эффективная модель)  
✅ Лучшие метрики PSNR/SSIM на статичных частях  

⚠️ Немного медленнее обучение (~10-20%)  
⚠️ Требует больше GPU памяти для масок  
⚠️ Может потерять детали при слишком агрессивной фильтрации  
