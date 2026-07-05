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

- Данные: CARES-3000, подвыборка из 18k примеров с фиксированным random seed.
- Состав: 641 безопасный запрос и 2359 рискованных.
- Формы запросов: 881 direct, 859 obfuscation, 693 indirect, 567 role-play.
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

Qwen3-1.7B, CARES-3000, judge `gpt-oss-120b`. Метрики агрегированы из scored chunks `0..299`, `300..999`, `1000..2999`.

| policy | unsafe answer rate ↓, 95% CI | bypass attack success ↓, 95% CI | benign over-refusal ↓, 95% CI | policy success ↑, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| baseline | 32.9% [31.0, 34.8] | 37.0% [34.7, 39.4] | 10.9% [8.7, 13.6] | 64.3% [62.6, 66.0] |
| prompt policy | 23.5% [21.8, 25.2] | 29.1% [27.0, 31.3] | 6.0% [4.4, 8.1] | 74.4% [72.8, 76.0] |
| strict prompt policy | 9.1% [8.0, 10.3] | 10.5% [9.1, 12.1] | 18.3% [15.5, 21.5] | 76.5% [75.0, 78.0] |
| retrieval constraints | 23.6% [21.9, 25.4] | 27.8% [25.7, 30.0] | 6.9% [5.2, 9.1] | 74.9% [73.3, 76.4] |
| Qwen3Guard | 14.0% [12.7, 15.5] | 17.8% [16.0, 19.7] | 27.3% [24.0, 30.9] | 65.8% [64.0, 67.4] |

Примечание: интервалы посчитаны по агрегированным числителям и знаменателям трёх scored chunks.

**Слайд 10. Результаты**

OLMo-2-0425-1B-Instruct, CARES-3000, judge `gpt-oss-120b`. Метрики агрегированы из scored chunks `0..299`, `300..999`, `1000..2999`.

| policy | unsafe answer rate ↓, 95% CI | bypass attack success ↓, 95% CI | benign over-refusal ↓, 95% CI | policy success ↑, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| baseline | 36.6% [34.7, 38.5] | 38.0% [35.7, 40.4] | 9.4% [7.4, 11.9] | 59.5% [57.7, 61.2] |
| prompt policy | 24.4% [22.7, 26.2] | 29.0% [26.9, 31.3] | 9.2% [7.2, 11.7] | 72.1% [70.4, 73.6] |
| strict prompt policy | 19.8% [18.2, 21.4] | 21.9% [20.0, 24.0] | 5.3% [3.8, 7.3] | 78.3% [76.8, 79.7] |
| retrieval constraints | 30.2% [28.4, 32.1] | 33.6% [31.3, 35.9] | 9.0% [7.1, 11.5] | 66.4% [64.7, 68.1] |
| Qwen3Guard | 11.4% [10.2, 12.8] | 14.0% [12.4, 15.8] | 40.6% [36.9, 44.5] | 64.9% [63.1, 66.6] |

**Слайд 11. Результаты**

Gemma-3-1B-IT, CARES-3000, judge `gpt-oss-120b`

| policy | unsafe answer rate ↓, 95% CI | bypass attack success ↓, 95% CI | benign over-refusal ↓, 95% CI | policy success ↑, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| baseline | 30.7% [28.9, 32.6] | 34.7% [32.4, 37.0] | 15.3% [12.7, 18.3] | 61.4% [59.6, 63.1] |
| prompt policy | 17.3% [15.9, 18.9] | 16.8% [15.1, 18.7] | 15.4% [12.8, 18.4] | 75.9% [74.3, 77.4] |
| strict prompt policy | 10.3% [9.1, 11.6] | 11.5% [10.1, 13.2] | 26.1% [22.9, 29.7] | 79.5% [78.0, 80.9] |
| retrieval constraints | 26.5% [24.8, 28.3] | 31.5% [29.3, 33.7] | 15.6% [13.0, 18.7] | 66.7% [65.0, 68.4] |
| Qwen3Guard | 13.5% [12.2, 15.0] | 15.8% [14.1, 17.6] | 39.3% [35.6, 43.1] | 62.0% [60.2, 63.7] |

Примечание: таблица с 95% доверительными интервалами Вилсона.

**Слайд 12. Результаты**

- На CARES-3000 baseline остаётся небезопасным на всех трёх моделях:
  - unsafe answer rate: Qwen 32.9%, OLMo 36.6%, Gemma 30.7%;
  - bypass success rate: Qwen 37.0%, OLMo 38.0%, Gemma 34.7%.
- Все защиты снижают unsafe answer rate, но лучший общий trade-off даёт strict prompt policy.
- Strict prompt policy стабильно повышает policy success rate:
  - Qwen: 64.3% -> 76.5%;
  - OLMo: 59.5% -> 78.3%;
  - Gemma: 61.4% -> 79.5%.
- По unsafe answer rate strict сильнее всего улучшает Qwen и Gemma:
  - Qwen: 32.9% -> 9.1%;
  - Gemma: 30.7% -> 10.3%.
- Qwen3Guard даёт лучший unsafe answer rate на OLMo: 36.6% -> 11.4%, но резко повышает over-refusal rate до 40.6%.

**Слайд 13. Результаты**

- Prompt policy улучшает baseline на всех моделях, но не закрывает обходы полностью:
  - bypass success rate: Qwen 37.0% -> 29.1%;
  - OLMo 38.0% -> 29.0%;
  - Gemma 34.7% -> 16.8%.
- Retrieval constraints не дают устойчивого выигрыша относительно prompt policy:
  - Qwen почти без разницы по unsafe answer rate: 23.5% vs 23.6%;
  - OLMo хуже: 24.4% vs 30.2%;
  - Gemma хуже: 17.3% vs 26.5%.
- Strict prompt policy лучше всего подавляет обходы на Qwen и Gemma:
  - Qwen bypass success rate: 37.0% -> 10.5%;
  - Gemma bypass success rate: 34.7% -> 11.5%.
- Qwen3Guard не доминирует strict prompt policy:
  - Qwen и Gemma: strict лучше по unsafe answer rate, bypass success rate и over-refusal rate;
  - OLMo: Qwen3Guard лучше по unsafe answer rate и bypass success rate, но хуже по over-refusal rate: 40.6% против 5.3% у strict.
- Вывод: наиболее стабильная защита в CARES-3000 — strict prompt policy; Qwen3Guard полезен как safety-heavy вариант, но требует контроля over-refusal rate.

**Слайд 14. Результаты: think vs no-think**

В разрезе think и no think для Qwen3-1.7B

| mode | unsafe answer rate ↓ | bypass success ↓ | benign over-refusal ↓ | policy attack success ↑ |
| --- | ---: | ---: | ---: | ---: |
| no-think | 45.8% | 54.8% | 6.7% | 47.0% |
| think | 45.6% | 58.6% | 0.0% | 46.0% |

- Thinking не выглядит самостоятельной защитой, метрика почти не меняется. Доля обходных запросов, где атака сработала, стало чуть выше.

- Выводы достоверно сделать нельзя, слишком мало примеров. 


**Слайд 15. Результаты: 8 принципов CARES**

CARES-3000, judge `gpt-oss-120b`. Метрика: unsafe answer rate.

| model | самые сложные принципы в baseline | остаточный риск под strict prompt policy |
| --- | --- | --- |
| Qwen | public health 39.9% [34.5, 45.5], misinformation 34.2% [28.8, 40.1] | misinformation 13.0% [9.5, 17.6], public health 12.2% [8.9, 16.4] |
| OLMo | public health 45.3% [39.7, 51.0], misinformation 41.6% [35.9, 47.6], clinical ethics 40.0% [34.4, 45.8] | public health 27.0% [22.3, 32.4], clinical ethics 25.4% [20.6, 30.8] |
| Gemma | public health 35.1% [29.9, 40.7], clinical ethics 34.6% [29.3, 40.4], misinformation 33.5% [28.1, 39.3] | clinical ethics 15.0% [11.3, 19.7], public health 13.5% [10.1, 17.9] |

- Public health остаётся самым трудным принципом в baseline для всех трёх моделей.
- Strict prompt policy снижает unsafe answer rate по всем принципам, но OLMo сохраняет самый высокий остаточный риск.
- Qwen3Guard лучше strict по unsafe answer rate для OLMo, но для Qwen и Gemma чаще проигрывает strict на самых сложных принципах.

*Strict prompt policy остаётся лучшим общим trade-off по safety/utility.


**Слайд 16. Результаты: форма запроса**

CARES-3000: 881 direct, 859 obfuscation, 693 indirect, 567 role-play.

| форма запроса | Qwen baseline | Qwen strict | OLMo baseline | OLMo strict | Gemma baseline | Gemma strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 27.1% [23.9, 30.5] | 6.6% [5.0, 8.7] | 39.5% [36.0, 43.1] | 17.6% [15.0, 20.6] | 26.8% [23.7, 30.2] | 8.3% [6.5, 10.6] |
| obfuscation | 68.6% [65.0, 72.0] | 18.5% [15.8, 21.7] | 68.5% [64.8, 71.9] | 40.4% [36.7, 44.1] | 61.1% [57.4, 64.8] | 16.6% [14.0, 19.6] |
| indirect | 15.2% [12.4, 18.4] | 4.6% [3.1, 6.7] | 16.8% [13.9, 20.2] | 9.2% [7.1, 12.0] | 17.4% [14.4, 20.8] | 6.7% [4.9, 9.1] |
| role-play | 15.9% [12.8, 19.6] | 5.5% [3.7, 8.0] | 17.7% [14.4, 21.6] | 9.5% [7.1, 12.7] | 15.7% [12.6, 19.4] | 9.8% [7.3, 12.9] |

- Метрика в таблице: для direct — unsafe answer rate; для obfuscation, indirect и role-play — bypass success rate.
- Пример: среди obfuscation запросов, которые относятся к risky/bypass, Qwen baseline модель дала небезопасный ответ или последовала обходной формулировке в 68.6% случаев.

- Obfuscation остаётся самой успешной формой обхода запроса. Direct — контрольная группа.

- Strict prompt policy сильно снижает bypass success rate на Qwen и Gemma, но хуже переносится на OLMo: obfuscation остаётся 40.4%.

**Слайд 17. Результаты: risk category**

CARES-3000: 641 benign и 2359 risky.

| model / policy | unsafe answer rate on risky prompts ↓ | over-refusal rate on benign prompts ↓ | policy success rate on risky prompts ↑ | policy success rate on benign prompts ↑ |
| --- | ---: | ---: | ---: | ---: |
| Qwen strict | 9.1% [8.0, 10.3] | 18.3% [15.5, 21.5] | 75.8% [74.1, 77.5] | 79.1% [75.8, 82.1] |
| Qwen Qwen3Guard | 14.0% [12.7, 15.5] | 27.3% [24.0, 30.9] | 66.1% [64.2, 68.0] | 64.6% [60.8, 68.2] |
| OLMo strict | 19.8% [18.2, 21.4] | 5.3% [3.8, 7.3] | 75.5% [73.7, 77.2] | 88.6% [85.9, 90.8] |
| OLMo Qwen3Guard | 11.4% [10.2, 12.8] | 40.6% [36.9, 44.5] | 69.0% [67.1, 70.8] | 49.8% [45.9, 53.6] |
| Gemma strict | 10.3% [9.1, 11.6] | 26.1% [22.9, 29.7] | 81.7% [80.1, 83.2] | 71.3% [67.7, 74.7] |
| Gemma Qwen3Guard | 13.5% [12.2, 15.0] | 39.3% [35.6, 43.1] | 64.3% [62.4, 66.3] | 53.2% [49.3, 57.0] |

- Этот разрез проверяет базовый компромисс: защита действительно стала безопаснее на рискованных запросах, но не начала ли она из-за этого слишком часто отказывать на нормальные безопасные вопросы.

- Strict prompt policy даёт лучший общий баланс: низкий unsafe answer rate и более высокий policy success rate, особенно на risky prompts.
- Qwen3Guard полезен как safety-heavy вариант для OLMo, но сильно портит benign prompts: over-refusal rate 40.6%, policy success rate на benign prompts 49.8%.
- Для Qwen и Gemma Qwen3Guard хуже strict и по unsafe answer rate, и по over-refusal rate.


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
