# Knowledge Distillation for Countdown Reasoning

Проект по дистилляции знаний для компактной языковой модели `gemma-3-1b-it` на задаче **Countdown**.  
Цель — перенести способность teacher-модели решать арифметические задачи в student-модель с ограниченными вычислительными ресурсами.

## Задача

Даны несколько чисел и целевое значение.  
Необходимо построить арифметическое выражение, используя каждое число не более одного раза и только операции:

- `+`
- `-`
- `*`
- `/`

## Стек

- Python
- PyTorch
- Hugging Face Transformers
- TRL
- PEFT / QLoRA
- BitsAndBytes
- Hugging Face Datasets
- Pandas
- NumPy

## Что сделано

- Разработан воспроизводимый пайплайн дистилляции для `gemma-3-1b-it` на verified teacher-решениях.
- Реализован полный цикл экспериментов: подготовка датасета, обучение, инференс, автоматическая проверка корректности выражений и генерация submission-ready предсказаний.
- Проведено сравнение нескольких стратегий обучения:
  - `equation-only SFT`
  - hard-case fine-tuning
  - mixed targets
  - teacher expansion
  - correction stage

## Основной результат

Лучший рабочий baseline был получен на подходе:

**`equation-only SFT` + full verified teacher dataset**

### Лучшая offline-метрика
- **Dev accuracy:** `0.864`
- **Shadow accuracy:** `0.8617`

## Ключевые выводы

- Наиболее устойчивый результат дал обычный `SFT` на verified teacher-примерах с target в виде **только финального выражения**.
- Увеличение объёма train data улучшало качество до определённого предела, после чего наблюдалось плато.
- Более сложные стратегии (`hard-case fine-tuning`, `mixed targets`, `teacher expansion`) не превзошли baseline.
- Было выявлено расхождение между offline-метриками и public benchmark, что показало ограниченность внутреннего holdout как основной метрики отбора моделей.

## Структура проекта

```text
.
├── train_countdown.py
├── input/
│   ├── test_public.csv
│   └── sample_submission.csv
└── workdir/
    ├── data/
    ├── logs/
    ├── models/
    └── submissions/
