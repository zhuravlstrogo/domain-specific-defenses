# Аудит кода — CARES medical safety defenses

Дата: 2026-06-26. Область: `src/domain_defenses/`, `scripts/`, `experiments/`,
конфиги и runbook. Проверено запуском `pytest` и ручной трассировкой логики
метрик.

Контекст: ранее уже была найдена и удалена избыточная фильтрация по `subtype`
(`emergency|child_health|mental_health`) в `analysis.py`. Ниже — оставшиеся
проблемы.

## Сводка

| # | Severity | Файл | Суть |
|---|---|---|---|
| 1 | 🔴 high | `tests/test_scoring.py:139` | Тест-сьют красный: неверный assert ломает прогон |
| 2 | 🔴 high | `test_thinking.py` / `scripts/test_thinking.py` | Дубль basename ломает сбор `pytest` целиком |
| 3 | 🟠 medium | `run_experiments.md`, `policies.py` | `guardrail_style` описан в runbook, но политики нет в коде |
| 4 | 🟠 medium | `src/domain_defenses/analysis.py:112-135` | `unsafe_pass_rate`/`bypass_success_rate` занижают failures при judge errors |
| 5 | 🟡 low | `src/domain_defenses/analysis.py` | `emergency_escalated` собирается, но `emergency_failure_rate` нигде не считается |
| 6 | 🟡 low | `scripts/report_medical_safety_metrics.py:137` | `_sample_score` — мёртвый код |
| 7 | 🟡 low | `src/domain_defenses/analysis.py:101-106` | Несогласованный стиль выборки + `groupby` теряет NaN-категории |
| 8 | 🟡 low | `scripts/report_medical_safety_metrics.py:165-171` | Baseline получает `guardrail_block_rate=0.0` вместо N/A |
| 9 | 🟡 low | `src/domain_defenses/scoring.py:415-421` | Бесполезный/хрупкий `**metadata` в `str.format` |

---

## 1. 🔴 Тест-сьют красный — неверный assert

`tests/test_scoring.py::test_structured_medical_scorer_retries_without_schema_when_unsupported`
падает:

```
assert fake_judge.calls[1] is None
E  assert GenerateConfig(..., response_schema=None, ...) is None
```

Код в `scoring.py` работает **правильно**: на retry без structured output
вызывается `_judge_generate_config(judge_model_name, use_response_schema=False)`,
который возвращает `GenerateConfig(max_tokens=...)` с `response_schema=None` — а
не `None`. Ошибка в ожидании теста.

Должно быть:

```python
assert fake_judge.calls[1].response_schema is None
```

Это баг теста, не кода, но он держит сьют красным (`1 failed, 59 passed`).

## 2. 🔴 `pytest` не собирается из-за дубля `test_thinking.py`

В рабочем дереве два untracked-файла с одинаковым basename:
`test_thinking.py` (корень) и `scripts/test_thinking.py`. `python -m pytest`
без аргументов падает на этапе collection:

```
import file mismatch: imported module 'test_thinking' ...
Interrupted: 1 error during collection
```

Это полностью блокирует прогон всех тестов (приходится явно звать `pytest tests/`).
Файлы `test_thinking.py`, `scripts/test_thinking.py`, `test_thinking.md` —
похоже, отладочные, не отслеживаются git. Убрать из корня / переименовать /
добавить в `.gitignore`, либо положить в `tests/` с уникальным именем.

## 4. 🟠 Структурные метрики занижают failures при judge errors

В `summarize_medical_eval` (`analysis.py:112-135`) ветка выбирается так:

```python
if _has_observed_values(risky, "unsafe_pass"):
    metrics["unsafe_pass_rate"] = _mean_bool(risky["unsafe_pass"])
```

`_has_observed_values` → `True`, если есть **хотя бы одно** non-null значение.
Но при judge error (`_judge_error_payload`) и parse failure (`_fallback_payload`)
`unsafe_pass = None`, при этом итоговый grade — `"I"` (fail-closed). А
`_mean_bool` делает `series.astype(bool)`, где `None → False`.

Итог: risky-сэмпл с упавшим judge считается как «не unsafe» (False) в
`unsafe_pass_rate`, но как failure (0) в score-based `overall_policy_success_rate`.
Подтверждено в runtime:

```
pd.Series([True, None, None]).astype(bool).mean() == 0.333   # 2 judge error -> "safe"
# score-based взгляд тех же 3 строк (все grade I): 1 - mean = 1.0
```

То есть при ненулевом `judge_error_rate`/`judge_parse_failure_rate` метрики
`unsafe_pass_rate`, `benign_over_refusal_rate` (строки 119-122) и частично
`bypass_success_rate` (128-132) систематически **занижают** небезопасность.
Runbook сам отмечает, что judge падает и scorer fail-closed ставит `I`, так что
кейс реальный.

Фикс: выбрать единое правило и применить ко всем трём метрикам — либо исключать
None из знаменателя (`dropna()` перед mean), либо fail-closed согласованно с
grade (`fillna(True)` для unsafe/bypass). Сейчас bypass использует
`.fillna(False)` (строки 129-130) — это fail-**open**, что тоже расходится с
grade=I.

## 5. 🟡 `emergency_failure_rate` нигде не считается

`emergency_escalated` тащится через scorer (`scoring.py`) и `log_to_df`
(`analysis.py:55`), а `run_experiments.md` (раздел 7) документирует метрику
`emergency_failure_rate`. Но `summarize_medical_eval` её не агрегирует —
сигнал собирается «в стол». Либо посчитать
(`1 - mean(emergency_escalated)` по строкам, где он не null), либо убрать из
документации, чтобы отчёт не обещал отсутствующую колонку. Вероятно, это остаток
удалённой emergency-фильтрации из исходного контекста.

## 6. 🟡 Мёртвый код `_sample_score`

`scripts/report_medical_safety_metrics.py:137-141` определяет `_sample_score`,
который нигде не используется (везде идёт через `log_to_df` →
`analysis._first_score`). Удалить.

## 7. 🟡 Несогласованная выборка категорий

`analysis.py:101-106`:

```python
by_category = defaultdict(lambda: pd.DataFrame())
for category, group in df.groupby("category"):
    by_category[category] = group
...
benign = by_category["benign"]
```

`risky` и `bypass` фильтруются прямой маской `df[df["category"].isin(...)]`, а
`benign` — через defaultdict + groupby. Помимо стилевой несогласованности,
`groupby("category")` по умолчанию **отбрасывает строки с NaN-категорией**, так
что разные ветки могут считать по разным подмножествам. Для единообразия:
`benign = df[df["category"] == "benign"]`, и убрать defaultdict.

## 8. 🟡 Baseline показывает `guardrail_block_rate=0.0` вместо N/A

`_summarize_guardrail` (`report_medical_safety_metrics.py:165-171`) для прогонов
без guardrail-метаданных (например `baseline`) возвращает четыре нулевые метрики
блокировок. В отчёте baseline-строка выглядит так, будто guardrail отработал с
0% блокировок, хотя его там не было. Семантически точнее оставлять эти поля
пустыми/N/A для не-guardrail прогонов (тогда и `delta_*` по ним не будет
вводить в заблуждение).

## 9. 🟡 Хрупкий `**metadata` в judge-промпте

`scoring.py:415-421`:

```python
scoring_prompt = MEDICAL_JUDGE_TEMPLATE.format(
    question=state.input_text,
    answer=state.output.completion,
    criterion=target.text,
    instructions=MEDICAL_JUDGE_INSTRUCTIONS,
    **metadata,
)
```

Шаблон использует только `question/answer/criterion/instructions`; `**metadata`
ни на что не влияет, но создаёт риск: если в metadata когда-нибудь появится ключ
`question`/`answer`/`criterion`/`instructions`, `str.format` упадёт с
`TypeError: got multiple values for keyword argument`. Сейчас коллизий нет, но
распаковку лучше убрать как бесполезную.

---

## Проверки, которые прошли чисто

- `config.py` — кэш `_config` не мутируется (`dict(...)`-копии в
  `get_runtime_model_args`/`get_runtime_generate_config`), резолв моделей и
  лейблов корректен.
- `guardrails.py` — sandwich input/output, fail-closed на parse failure,
  блок-стейт и метаданные согласованы.
- `run_experiment_matrix.py` — шаблоны `{experiment_id}`/`{model_label}`,
  manifest, resume по `eval_log_is_complete_and_scored` выглядят корректно.
- `dataset.py` — нормализация CARES, expected_behavior, bypass_type — ок.
- Остальные 59 тестов в `tests/` проходят.

## Рекомендованный порядок

1. Поправить assert в `tests/test_scoring.py:139` (#1) и убрать дубль
   `test_thinking.py` (#2) — вернуть зелёный сьют.
2. Решить вопрос с None-семантикой метрик (#4) — влияет на корректность
   результатов эксперимента.
3. Синхронизировать runbook и код по `guardrail_style`/`emergency_failure_rate`
   (#3, #5).
4. Косметика: #6, #7, #8, #9.
