# Source Code

Переиспользуемый код проекта.

- `domain_defenses/dataset.py`: загрузка JSONL в `inspect_ai.dataset.Sample`.
- `domain_defenses/policies.py`: baseline и defense system prompts.
- `domain_defenses/scoring.py`: LLM-as-judge scorer.
- `domain_defenses/tasks.py`: Inspect AI task.
- `domain_defenses/analysis.py`: преобразование `EvalLog` в таблицу и базовые метрики.
- `domain_defenses/mcq_prep.py`: подготовка MCQ датасета из HF в локальный JSONL.
- `domain_defenses/mcq_dataset.py`: загрузка MCQ JSONL в `inspect_ai.dataset.Sample`.
- `domain_defenses/mcq_parsing.py`: deterministic parser ответа (`1..4` / `A..D`).
- `domain_defenses/mcq_scoring.py`: deterministic MCQ scorer (`1..4` / `A..D`).
- `domain_defenses/mcq_tasks.py`: Inspect task для multi-turn MCQ robustness.
- `domain_defenses/mcq_analysis.py`: метрики `initial/post`, `flip`, `correct_to_incorrect`.
