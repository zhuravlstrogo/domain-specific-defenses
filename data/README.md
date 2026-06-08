# Data

`eval_set_v0.jsonl` — первый ручной набор медицинских запросов для eval.

Основной датасет для `medical_safety` теперь готовится из CARES-18K:

```bash
python scripts/prepare_cares_dataset.py --split test --limit 300 --seed 42
```

Дефолтный путь eval-задачи:

```text
processed/cares_18k_v1.jsonl
```

Каждая строка содержит:

- `id`
- `category`: `benign`, `risky`, `edge_case`
- `subtype`
- `bypass_type`
- `prompt`
- `expected_behavior`
- `harm_if_failed`
- `severity`

Папки:

- `raw/`: внешние или сырые данные, если появятся.
- `processed/`: очищенные/подготовленные варианты датасетов.

Для MCQ bypass эксперимента:
- `processed/medqa_multiturn_robust_v1.jsonl`: локально подготовленный dataset из
  `dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust`.

Для CARES safety/bypass эксперимента:
- `processed/cares_18k_v1.jsonl`: локально подготовленный dataset из
  `HFXM/CARES-18K`.
- `harmful_level=0` мапится в `category=benign`.
- `harmful_level=1..3` мапится в `category=risky`.
- `method=direct` мапится в `bypass_type=none`; остальные `method` становятся
  bypass-срезами.
