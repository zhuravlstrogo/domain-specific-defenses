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

**Слайд 3. Метрики и центральный trade-off**

Сравниваем защиты не по одной метрике, а по trade-off между тремя осями:

- **Safety** — модель не даёт опасные медицинские инструкции на рискованные запросы.
  - **Unsafe answer rate** ↓: как часто risky-запрос получает unsafe ответ.
  - **Policy success** ↑: как часто модель выбирает правильную реакцию: ответить, осторожно ответить или отказать.

- **Robustness** — защита сохраняется, когда risky-запрос маскируют, задают через роль или формулируют косвенно.
  - **Bypass attack success** ↓: как часто обходная формулировка всё же приводит к unsafe ответу.

- **Utility** — модель продолжает полезно отвечать на безопасные медицинские вопросы.
  - **Benign over-refusal** ↓: как часто модель зря отказывается от безопасного вопроса.


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
- Основные метрики: unsafe answer rate, benign over-refusal, bypass attack success, emergency handling, utility score.
- Для долей считаются 95% доверительные интервалы Вилсона.
- Дополнительно: разрезы по safe/risky/bypass запросам и по 8 принципам CARES.

**Слайд 8. Метрики**

Четыре оси сравнения защит — они же колонки в таблицах результатов:

- **Unsafe answer rate** ↓ — доля risky-запросов с небезопасным ответом. *Ниже — безопаснее.*
- **Bypass attack success** ↓ — доля обходных запросов (obfuscation / role-play / indirect), которые привели к unsafe ответу. *Ниже — устойчивее к обходам.*
- **Benign over-refusal** ↓ — доля безопасных запросов с лишним отказом. *Ниже — выше полезность.*
- **Policy success** ↑ — доля запросов с правильной реакцией политики: Accept для safe, Caution / Refuse для risky. *Выше — лучше.*

Дополнительные разрезы и вспомогательные метрики:

- Safety: `emergency_failure_rate` (red-flag без эскалации); policy success в разбивке по 8 принципам CARES.
- Utility: `benign_utility_score`, `risky_safe_redirection_utility_score`.
- Guardrail: `benign / risky / bypass guardrail block rate`.

Надёжность самой оценки (см. слайд «Ограничения»): `judge_error_rate`, `judge_parse_failure_rate`.

Для всех долей считаются 95% доверительные интервалы Вилсона.

**Слайд 9. Результаты**

Qwen3-1.7B, доступный complete scored artifact: CARES `offset300_limit700`, judge `gpt-oss-120b`

| policy | unsafe answer rate ↓, 95% CI | bypass attack success ↓ | benign over-refusal ↓ | policy success ↑ |
| --- | ---: | ---: | ---: | ---: |
| baseline | 32.1% [28.4, 36.1] | 37.3% | 12.9% | 63.9% |
| prompt policy | 20.3% [17.2, 23.8] | 25.5% | 4.3% | 76.1% |
| strict prompt policy | 7.0% [5.1, 9.4] | 8.4% | 18.0% | 78.0% |
| retrieval constraints | 22.5% [19.2, 26.1] | 26.5% | 6.5% | 75.6% |
| Qwen3Guard | 14.1% [11.4, 17.2] | 19.2% | 26.8% | 65.6% |

Примечание: полный `CARES-3000` gpt-oss artifact для Qwen сейчас есть только для baseline, поэтому для сравнения 5 стратегий используется complete chunk на 700 samples.

**Слайд 10. Результаты**

OLMo-2-0425-1B-Instruct, доступный complete scored artifact: CARES `offset300_limit700`, judge `gpt-oss-120b`

| policy | unsafe answer rate ↓, 95% CI | bypass attack success ↓ | benign over-refusal ↓ | policy success ↑ |
| --- | ---: | ---: | ---: | ---: |
| baseline | 37.8% [33.9, 41.9] | 38.1% | 6.5% | 59.0% |
| prompt policy | 23.9% [20.5, 27.6] | 28.6% | 7.2% | 73.4% |
| strict prompt policy | 20.0% [16.9, 23.5] | 22.3% | 4.3% | 79.3% |
| retrieval constraints | 29.8% [26.1, 33.7] | 32.3% | 9.4% | 67.7% |
| Qwen3Guard | 13.4% [10.8, 16.4] | 16.5% | 42.3% | 63.7% |

**Слайд 11. Результаты**

Gemma-3-1B-IT, CARES-3000, judge `gpt-oss-120b`

| artifact | status |
| --- | --- |
| inference, baseline / prompt / strict / retrieval | complete: 4 x 3000 samples |
| inference, Qwen3Guard | partial: 2172 / 3000 samples |
| judge gpt-oss-120b | no complete scored report in `reports/results` |
| runner judge logs | only `Dry run complete` for Gemma paths |

Вывод: unsafe answer rate и CI для Gemma сейчас нельзя честно заполнить. В слайдах нельзя смешивать unscored inference `.eval` с scored judge metrics для Qwen/OLMo.

**Слайд 12. Результаты**

- На gpt-oss CARES-700 chunk baseline остаётся небезопасным:
  - Qwen unsafe answer rate = 32.1%, OLMo = 37.8%;
  - bypass attack success около 37-38%.
- Все защиты снижают unsafe answer rate, но профиль цены разный.
- Лучший текущий trade-off по двум complete моделям: strict prompt policy.
- Strict prompt policy:
  - Qwen: unsafe answer rate 32.1% -> 7.0%, policy success 63.9% -> 78.0%;
  - OLMo: unsafe answer rate 37.8% -> 20.0%, policy success 59.0% -> 79.3%.
- Qwen3Guard даёт самый низкий unsafe answer rate на OLMo, но резко повышает benign over-refusal: 42.3%.

**Слайд 13. Результаты**

- Prompt policy даёт сильное улучшение относительно baseline, но не закрывает обходы:
  - Qwen bypass attack success: 37.3% -> 25.5%;
  - OLMo bypass attack success: 38.1% -> 28.6%.
- Retrieval constraints в этих артефактах слабее strict prompt policy по safety:
  - Qwen unsafe answer rate 22.5% против 7.0%;
  - OLMo unsafe answer rate 29.8% против 20.0%.
- Qwen3Guard не доминирует strict prompt policy:
  - Qwen: хуже по unsafe answer rate, bypass attack success и over-refusal;
  - OLMo: лучше по unsafe answer rate, но намного хуже по benign over-refusal.
- Вывод: безопасность надо сравнивать как компромисс между unsafe answers, bypass resistance и over-refusal, а не по одной колонке.

**Слайд 14. Результаты: think vs no-think**

Baseline ablation, CARES-100, из `report_29_06_26.md`

| mode | unsafe answer rate ↓ | bypass success ↓ | benign over-refusal ↓ | policy attack success ↑ |
| --- | ---: | ---: | ---: | ---: |
| no-think | 45.8% | 54.8% | 6.7% | 47.0% |
| think | 45.6% | 58.6% | 0.0% | 46.0% |

- Thinking не выглядит самостоятельной защитой.
- Unsafe answer rate почти не меняется: -0.2 pp.
- Bypass attack success становится немного хуже: +3.8 pp.
- Сильный вывод делать нельзя: это baseline-only, 100 samples, не full defense matrix.

**Слайд 15. Результаты: 8 принципов CARES**

Qwen/OLMo, CARES `offset300_limit700`, judge `gpt-oss-120b`

- Метрика ниже: unsafe answer rate внутри risky-запросов каждого CARES-принципа.
- Qwen strict сильнее всего снижает unsafe answers в сложных baseline-принципах:
  - public health: 41.2% -> 8.8%;
  - misinformation: 38.5% -> 10.8%.
- OLMo strict тоже снижает unsafe answers, но остаточный риск выше, чем у Qwen:
  - misinformation: 52.3% -> 21.5%;
  - public health: 45.6% -> 29.4%;
  - clinical ethics: 44.0% -> 26.7%.
- Qwen3Guard работает неровно по темам: меньше опасных ответов, но хуже качество обработки self-harm запросов:
  - self-harm policy success: Qwen 52.9%, OLMo 54.1%.
- Разрез по принципам полезен диагностически, но группы маленькие: по одному принципу всего 75-108 примеров, поэтому точные проценты надо читать осторожно.

**Слайд 16. Результаты: форма запроса**

CARES `offset300_limit700`: 219 direct, 186 obfuscation, 165 indirect, 130 role-play

| форма запроса | Qwen baseline | Qwen strict | OLMo baseline | OLMo strict |
| --- | ---: | ---: | ---: | ---: |
| direct | 23.9% | 3.9% | 43.3% | 18.9% |
| obfuscation | 68.5% | 12.3% | 68.5% | 41.1% |
| indirect | 15.9% | 2.3% | 16.7% | 8.3% |
| role-play | 20.4% | 10.7% | 22.3% | 13.6% |

- Метрика в таблице: для direct — unsafe answer rate ↓; для obfuscation, indirect и role-play — bypass success rate ↓.
- Пример: среди obfuscation-запросов, которые относятся к risky/bypass, Qwen baseline дала небезопасный ответ или последовала обходной формулировке в 68.5% случаев.
- Obfuscation — самая успешная форма обхода запроса.
- Strict policy резко снижает bypass attack success на Qwen, но хуже переносится на OLMo.
- Direct — контрольная группа: это обычные risky-запросы без обходной формы.


**Слайд 17. Результаты: risk category**

CARES-300 имеет 62 benign и 238 risky; текущий gpt-oss chunk имеет 139 benign и 561 risky.

| model / policy | Unsafe answers on risky prompts ↓ | Over-refusal on benign prompts ↓ | Correct handling of risky prompts ↑ | Correct handling of benign prompts ↑ |
| --- | ---: | ---: | ---: | ---: |
| Qwen strict | 7.0% | 18.0% | 77.7% | 79.1% |
| Qwen Qwen3Guard | 14.1% | 26.8% | 65.6% | 65.5% |
| OLMo strict | 20.0% | 4.3% | 76.1% | 92.1% |
| OLMo Qwen3Guard | 13.4% | 42.3% | 66.7% | 51.8% |

- Этот разрез показывает главный компромисс: меньше опасных ответов на risky-запросы vs больше лишних отказов на benign-запросы.
- Qwen3Guard сильнее снижает unsafe answers на OLMo, но чаще ошибочно отказывает на benign-запросах.


**Слайд 18. Qwen3 guardrail score**

Qwen3Guard-Gen-0.6B на Qwen3-1.7B, CARES-300, judge GPT-4o

| metric | value |
| --- | ---: |
| unsafe answer rate | 23.3% |
| bypass attack success | 31.6% |
| benign over-refusal | 24.1% |
| policy success | 60.3% |
| overall guardrail block rate | 38.7% |
| risky guardrail block rate | 46.6% |
| benign guardrail block rate | 8.1% |
| bypass guardrail block rate | 38.8% |

**Слайд 19. Согласованность judge моделей**

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

**Слайд 20. Ограничения**

- gpt-oss результаты выше основаны на complete chunk `offset300_limit700`, а не на полном `CARES-3000` для всех policies.
- Gemma-3-1B-IT пока имеет inference artifacts, но не имеет complete scored judge report.
- Thinking ablation меньше и проще: baseline-only, 100 samples.
- Разрезы по principle/form полезны диагностически, но отдельные группы меньше полной выборки.
- Risk-category разрез не является независимой метрикой: он раскрывает компоненты уже определённых safety/utility метрик.
- Все выводы зависят от LLM-as-judge; для финального утверждения нужна проверка agreement вторым judge.
