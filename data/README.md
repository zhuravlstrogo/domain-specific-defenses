# Data

`eval_set_v0.jsonl` — первый ручной набор медицинских запросов для eval.

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
