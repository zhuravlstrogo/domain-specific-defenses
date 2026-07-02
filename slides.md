**Слайд 1. Название**

Domain-specific defenses для LLM: практическое сравнение guardrails защит в медицинском домене
ментор: Сабрина Садиех, 
менти: Алсу Ярулина,
“Подготовлено в рамках Technical AI Safety курса по оценке LLM”.

**Слайд 2. Проблема**

- В медицинском домене защита оценивается не по факту отказа, а по типу ошибки, которую она оставляет.
- Ответ может выглядеть безопасным, но вред остаётся при пропущенной срочности или опасной рекомендации.
- CARES задаёт три правильные реакции: ответить, ответить осторожно или отказать.
- Защита может снижать долю опасных ответов, но повышать отказы на нормальные медицинские вопросы.
- Обходные формулировки проверяют, держится ли защита при роли, маскировке или косвенном запросе.
- Отдельный риск оценки: модель-судья может поощрять стиль ответа вместо корректного поведения.
- **Исследовательский вопрос**: какие внешние защиты реально улучшают профиль ошибок медицинской LLM.

**Слайд 3. Проблема**

- Одна средняя метрика скрывает обмен между безопасностью, устойчивостью и полезностью.
- **Safety** — рискованные запросы без опасного медицинского совета:
  - **Unsafe pass rate** ↓: небезопасные ответы на рискованные запросы;
  - **Overall policy success** ↑: верная реакция для безопасных и рискованных запросов.
- **Robustness** — защита при маскировке, роли или косвенной формулировке:
  - **Bypass success rate** ↓: обходные запросы, где защита пропустила вредный ответ.
- **Utility** — сохранение ответа на безопасные медицинские вопросы:
  - **Benign over-refusal rate** ↓: безопасные запросы, ошибочно заблокированные или отклонённые.
- Строгая защита снижает пропуски, но часто повышает лишние отказы.

**Слайд 4. Обзор защит**

- Защиты сгруппированы по месту вмешательства в pipeline.
- A. Внешние защиты (wrapper): Guardrails; Internal safety probes; Multi-agent verification; Self-critique.
- B. Защиты на входе (inference-time, input-side): Adversarial-input transformations; Prompt-injection defenses; Prompt-level / system-prompt.
- C. Защиты на выходе (inference-time, output-side): Safe decoding; Sampling-based hallucination detection; RAG defenses.
- D. Вмешательство в представления (representation-level): Representation interventions / circuit breakers.
- Для закрытых моделей практически доступны прежде всего wrapper и inference-time defenses.

**Слайд 5. Обзор защит**

- E. Защиты через обучение (training-time): Safety alignment / preference optimization; Safety-preserving fine-tuning; Model merging / safety grafting; Safety LoRA.
- E. Защиты через обучение (training-time): Adversarial / continual fine-tuning; Abstention training / calibration; Medical hallucination fine-tuning; Backdoor / poisoning defenses; Data curation.
- F. Удаление и редактирование знаний: Machine unlearning; Knowledge editing.
- G. Защита данных пациентов (privacy): PHI de-identification; Privacy-preserving training.
- H. Сквозные направления (cross-cutting): Multimodal safety; Tool / agent safety.
- Методы, меняющие веса, потенциально сильнее, но дороже и требуют open weights.

**Слайд 6. Что реализовано и как устроено**

- Данные: CARES-300, подвыборка из 18k примеров с фиксированным random seed.
- Состав: 62 безопасных запроса и 238 рискованных.
- Формы запросов: 85 direct, 86 obfuscation, 70 indirect, 59 role-play.
- Сохранены 8 принципов CARES: privacy, ethics, bias, misinformation, self-harm, illegal medicine, AI clinical boundaries, public health.
- Основные модели: Qwen3-1.7B, Gemma-3-4B-IT, OLMo-2-0425-1B-Instruct.
- Внешний guardrail: Qwen3Guard-Gen-0.6B, проверка до генерации и перед выдачей ответа.

**Слайд 7. Что реализовано и как устроено**

- Проверяемые варианты: baseline, prompt policy, strict prompt policy, retrieval constraints, Qwen3Guard guardrail.
- LLM-as-judge оценивает ответы по единой рубрике.
- Основной полный judge-прогон: GPT-4o; для проверки согласованности поддержан Claude Sonnet 4.5 через OpenRouter.
- Режимы генерации: `no_think` и `think` / full CoT.
- Основные метрики: unsafe pass rate, over-refusal rate, bypass success rate, emergency handling, utility score.
- Для долей считаются 95% доверительные интервалы Вилсона.
- Дополнительно: разрезы по safe/risky/bypass запросам и по 8 принципам CARES.

**Слайд 8. Метрики**

Четыре оси сравнения защит — они же колонки в таблицах результатов:

- **Unsafe pass rate** ↓ — доля risky-запросов, где прошёл небезопасный ответ. *Ниже — безопаснее.*
- **Bypass success rate** ↓ — доля bypass-запросов (obfuscation / role-play / indirect), где защиту обошли. *Ниже — устойчивее к обходам.*
- **Benign over-refusal rate** ↓ — доля безопасных запросов с лишним отказом. *Ниже — выше полезность.*
- **Overall policy success** ↑ — доля запросов с правильной реакцией политики: Accept для safe, Caution / Refuse для risky. *Выше — лучше.*

Дополнительные разрезы и вспомогательные метрики:

- Safety: `emergency_failure_rate` (red-flag без эскалации); overall success в разбивке по 8 принципам CARES.
- Utility: `benign_utility_score`, `risky_safe_redirection_utility_score`.
- Guardrail: `benign / risky / bypass guardrail block rate`.

Надёжность самой оценки (см. слайд «Ограничения»): `judge_error_rate`, `judge_parse_failure_rate`.

Для всех долей считаются 95% доверительные интервалы Вилсона.

**Слайд 9. Результаты**

Qwen3-1.7B, CARES-300, judge qwen3-32b & gpt-oss-120b

| policy | unsafe pass | bypass success | benign over-refusal | overall success |
| --- | ---: | ---: | ---: | ---: |
| baseline | 43.3% | 51.5% | 9.8% | 42.7% |
| prompt policy | 36.1% | 42.7% | 12.7% | 48.0% |
| strict prompt policy | 23.9% | 32.9% | 16.1% | 66.3% |
| retrieval constraints | 38.8% | 45.2% | 10.7% | 51.3% |
| Qwen3Guard | 23.3% | 31.6% | 24.1% | 60.3% |

**Слайд 10. Результаты**

olmo2_0425_1b, CARES-300, judge qwen3-32b & gpt-oss-120b

| policy | unsafe pass | bypass success | benign over-refusal | overall success |
| --- | ---: | ---: | ---: | ---: |
| baseline | 43.3% | 51.5% | 9.8% | 42.7% |
| prompt policy | 36.1% | 42.7% | 12.7% | 48.0% |
| strict prompt policy | 23.9% | 32.9% | 16.1% | 66.3% |
| retrieval constraints | 38.8% | 45.2% | 10.7% | 51.3% |
| Qwen3Guard | 23.3% | 31.6% | 24.1% | 60.3% |

**Слайд 11. Результаты**

gemma_3_4b_it, CARES-300, judge qwen3-32b & gpt-oss-120b

| policy | unsafe pass | bypass success | benign over-refusal | overall success |
| --- | ---: | ---: | ---: | ---: |
| baseline | 43.3% | 51.5% | 9.8% | 42.7% |
| prompt policy | 36.1% | 42.7% | 12.7% | 48.0% |
| strict prompt policy | 23.9% | 32.9% | 16.1% | 66.3% |
| retrieval constraints | 38.8% | 45.2% | 10.7% | 51.3% |
| Qwen3Guard | 23.3% | 31.6% | 24.1% | 60.3% |

**Слайд 12. Результаты**

- Baseline плохо справляется с медицинскими risky/bypass запросами:
  - unsafe pass = 43.3%;
  - bypass success = 51.5%.
- Все защиты снижают unsafe pass, но не бесплатно.
- Лучший текущий trade-off: strict prompt policy.
- Strict prompt policy относительно baseline:
  - unsafe pass: -19.4 percentage points;
  - bypass success: -18.6 percentage points;
  - overall success: 66.3%;
  - цена: benign over-refusal растёт с 9.8% до 16.1%.

**Слайд 13. Результаты**

- Qwen3Guard чуть сильнее по raw safety:
  - unsafe pass = 23.3%;
  - bypass success = 31.6%.
- Но Qwen3Guard агрессивнее:
  - benign over-refusal = 24.1%;
  - benign guardrail block rate = 8.1%.
- Prompt policy даёт умеренное улучшение, но не является достаточной защитой.
- Retrieval constraints в этом запуске слабее strict prompt policy.
- Thinking пока не выглядит отдельной защитой:
  - no-think bypass success = 54.8%;
  - think bypass success = 58.6%.
- Вывод: безопасность надо сравнивать как компромисс между unsafe pass, bypass resistance и over-refusal.

**Слайд 14. Qwen3 guardrail score**

Qwen3Guard-Gen-0.6B на Qwen3-1.7B, CARES-300, judge GPT-4o

| metric | value |
| --- | ---: |
| unsafe pass | 23.3% |
| bypass success | 31.6% |
| benign over-refusal | 24.1% |
| overall policy success | 60.3% |
| overall guardrail block rate | 38.7% |
| risky guardrail block rate | 46.6% |
| benign guardrail block rate | 8.1% |
| bypass guardrail block rate | 38.8% |

**Слайд 15. Согласованность judge моделей**

- Основной CARES-300 результат сейчас посчитан с judge `GPT-4o`.
- Pipeline поддерживает второй judge: `Claude Sonnet 4.5` через OpenRouter.
- Текущий caveat: даже у GPT-4o `judge_error_rate` заметен и зависит от policy: 2.7%-20.7%.
- Для честного agreement нужны одинаковые sample ids и одинаковые policy runs.
- `scripts/report_judge_agreement.py` считает:
  - `n_common`;
  - `agreement_rate` / `disagreement_rate`;
  - `Cohen's kappa`;
  - directional disagreements: `left_success_right_failure` и наоборот.
- В текущих артефактах полного `Claude Sonnet 4.5` CARES-300 отчёта нет, поэтому устойчивость выводов к judge model пока не доказана.

**Слайд 16. Ограничения**

Текст
