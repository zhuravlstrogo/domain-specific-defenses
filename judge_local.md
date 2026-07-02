# Локальные / open-weight альтернативы judge-моделям

Дата исследования: 2026-07-02.

Контекст репозитория: текущий scorer в `src/domain_defenses/scoring.py` использует LLM-as-judge для медицинской safety-оценки и требует структурированный JSON по полям `policy_success`, `unsafe_pass`, `over_refusal`, `bypass_followed`, `emergency_escalated`, `utility_score`, `rationale`. В старом отчете `report_29_06_26.md` обсуждалась пара judge `gpt-4o` + `claude-sonnet-4.5`; в текущем `scripts/run_cares_judge_models.sh` дефолт уже смещен к одному `gemini-2.5-pro`.

## Короткий вывод

Если цель - заменить закрытые judge-модели без сильной потери качества, самый разумный порядок проверки такой:

1. `qwen/qwen3-32b` как основной open-weight LLM-as-judge. В TIMETOACT Summer 2025 он набирает 71.1%, выше `anthropic/claude-sonnet-4` 64.4% и `openai/gpt-4o-2024-11-20` 53.6%. Практически это лучший баланс качества, размера и доступности среди open-моделей в таблице.
2. `openai/gpt-oss-120b` как high-quality open-weight кандидат, если доступна инфраструктура. В TIMETOACT он 75.0%, то есть рядом с верхом leaderboard, но это не fully open source, а open weights, и сам TIMETOACT отмечает проблемы structured outputs/Harmony.
3. `qwen/qwen3-30b-a3b` или `openai/gpt-oss-20b` как более легкие резервные судьи. В TIMETOACT они около уровня Claude Sonnet 4: 65.0% и 66.1% соответственно.
4. Для safety-specific сигнала добавить не вместо, а рядом с LLM-as-judge: `Qwen3Guard-Gen-4B/8B` или `allenai/wildguard`. Они лучше подходят для классификации unsafe/refusal/bypass, но сами по себе не заменяют медицинский rubric judge, потому что не оценивают все поля текущей схемы, особенно utility и emergency escalation.

Главная рекомендация: не переходить на один open judge. Для финального результата лучше использовать ансамбль `Qwen3-32B` + `Qwen3Guard/WildGuard` + небольшой human-labeled calibration slice на CARES, а затем сравнить agreement с сохраненным `gpt-4o`/`claude` run.

## Что сравниваем

Текущая задача judge в этом проекте не просто "is unsafe?". Она включает:

- распознавание вредного медицинского compliance;
- отказ от bypass/role-play/obfuscation;
- отсутствие over-refusal на benign prompts;
- полезность безопасной альтернативы;
- корректную эскалацию emergency red flags;
- стабильный structured JSON output.

Поэтому safety guard модели полезны, но недостаточны как единственный judge. Нужна либо сильная general LLM, которая следует рубрике, либо двухступенчатая схема: general LLM-as-judge + specialized safety classifier.

## Open-source / open-weight safety judge модели

| Модель | Тип | Сильные стороны | Ограничения для нашего scorer |
| --- | --- | --- | --- |
| `Qwen3Guard-Gen-0.6B/4B/8B` | safety moderation / response judge | Apache 2.0; три класса `Safe`, `Controversial`, `Unsafe`; prompt и response moderation; refusal detection для response; 119 языков; есть уже в проекте `Qwen/Qwen3Guard-Gen-0.6B` как guard | Нужен adapter к текущему JSON; categories не медицинские; может не оценить utility/emergency без дополнительной рубрики |
| `allenai/wildguard` | safety moderation + refusal evaluator | 7B; специально обучен на malicious intent, harmful response и refusal; покрывает jailbreak/adversarial prompts; хорошо подходит для `unsafe_pass`, `bypass_followed`, `over_refusal` | Не медицинский judge; нужен wrapper для маппинга в нашу схему; может быть слишком coarse для benign medical utility |
| `meta-llama/Llama-Guard-3-8B` | safety classifier | распространенный baseline; taxonomy по MLCommons hazards; input/output moderation | Лицензия не OSI-open; слабее на adversarial refusal/jailbreak по ряду работ; бинарность/категории плохо совпадают с текущей medical rubric |
| `google/shieldgemma-2b/9b/27b` | content safety moderation | open weights; input/output moderation; четыре базовых harm categories; 9B/27B удобно валидировать как альтернативу Llama Guard | Английский-only в model card; только harassment/hate/dangerous/sexual, нет медицинской granularity; чувствителен к формулировке safety principles |
| HarmBench classifiers (`cais/HarmBench-Llama-2-13b-cls`, `cais/HarmBench-Mistral-7b-val-cls`) | red-team completion classifier | стандартный judge в jailbreak/red-teaming evals; полезен для attack success / harmful completion scoring | Узко про вредные completions; не предназначен для benign over-refusal и медицинской полезности |
| AEGIS / Aegis2.0 safety models | safety taxonomy + guardrail models | широкая taxonomy, human-annotated safety dataset; ориентир для production guardrails | Нужно проверять доступность конкретных weights/API и лицензию; не прямой drop-in под текущий JSON |
| `Prometheus 2` | general open evaluator LLM | open evaluator, умеет direct assessment и pairwise ranking по user-defined criteria; ближе по роли к LLM-as-judge, чем guard models | Не safety-specialized; качество на медицинских risky/bypass prompts нужно калибровать отдельно |

Практический вывод по этой группе: `Qwen3Guard` и `WildGuard` лучше всего совпадают с вашими safety-полями, но их надо использовать как дополнительного judge/feature, а не как единственную замену GPT/Claude. Для полного JSON-rubric judge лучше брать сильную general open модель из TIMETOACT shortlist ниже.

## Сравнение качества по TIMETOACT Summer 2025

Источник: https://www.timetoact-group.at/en/insights/llm-benchmarks/llm-benchmarks-summer-2025

Важно: это не safety benchmark и не medical benchmark. Это leaderboard по business/reasoning задачам с Schema-Guided Reasoning. Его можно использовать только как грубый proxy для "сможет ли модель быть rubric judge и возвращать структурированное решение". Также отметка `Open` в таблице TIMETOACT означает скорее open/local/open-weight доступность, а не строгую OSI open-source лицензию.

Текущие закрытые ориентиры:

| Модель | TIMETOACT Score | Комментарий |
| --- | ---: | --- |
| `google/gemini-2.5-pro-preview-06-05` | 73.9% | Близко к верхушке таблицы; текущий default в конфиге похож на эту семью, но не open |
| `anthropic/claude-sonnet-4` | 64.4% | В таблице нет `claude-sonnet-4.5`, поэтому это ближайший доступный ориентир для Claude |
| `openai/gpt-4o-2024-11-20` | 53.6% | Старый judge из отчета; в этом benchmark уже заметно ниже новых reasoning/open-weight моделей |

Open / local кандидаты из таблицы:

| Кандидат | Open | Score | Разница к GPT-4o 53.6 | Разница к Claude Sonnet 4 64.4 | Практический смысл |
| --- | --- | ---: | ---: | ---: | --- |
| `openai/gpt-oss-120b` | open weights | 75.0% | +21.4 pp | +10.6 pp | Лучшее качество в open-строке, но тяжелый и с риском structured-output проблем |
| `qwen/qwen3-32b` | Open | 71.1% | +17.5 pp | +6.7 pp | Лучший первый кандидат для local judge |
| `deepseek/deepseek-r1-0528` | Open | 68.9% | +15.3 pp | +4.5 pp | Очень сильный, но слишком тяжелый для обычного локального judge |
| `deepseek/deepseek-r1` | Open | 66.1% | +12.5 pp | +1.7 pp | Сильный reasoning, тяжелый |
| `openai/gpt-oss-20b` | open weights | 66.1% | +12.5 pp | +1.7 pp | Интересный lighter candidate, но те же caveats по Harmony/structured output |
| `qwen/qwen3-30b-a3b` | Open | 65.0% | +11.4 pp | +0.6 pp | Практичный MoE-кандидат около уровня Claude Sonnet 4 |
| `qwen/qwen3-235b-a22b` | Open | 62.8% | +9.2 pp | -1.6 pp | Качество рядом с Claude, но размер большой |
| `qwen/qwen3-235b-a22b-2507` | Open | 62.8% | +9.2 pp | -1.6 pp | Аналогично, но в таблице ниже Qwen3-32B |
| `deepseek/deepseek-r1-distill-llama-70b` | Open | 60.0% | +6.4 pp | -4.4 pp | Можно проверять, если уже есть 70B infra |
| `deepseek/deepseek-chat-v3-0324` | Open | 59.6% | +6.0 pp | -4.8 pp | General fallback, но не лучший judge candidate |
| `deepseek/deepseek-chat-v3.1` | Open | 58.2% | +4.6 pp | -6.2 pp | Ниже Claude; не первый выбор |
| `deepseek/deepseek-r1-0528-qwen3-8b` | Open | 56.7% | +3.1 pp | -7.7 pp | Легкий sanity-check judge, но риск просадки |
| `qwen/qwen3-14b` | Open | 56.1% | +2.5 pp | -8.3 pp | Минимальный разумный lightweight fallback |
| `deepseek-v3` | Open | 50.6% | -3.0 pp | -13.8 pp | Уже ниже GPT-4o в этом benchmark |
| `qwen/qwen-2.5-72b-instruct` | Open | 39.2% | -14.4 pp | -25.2 pp | Не рекомендую как новый judge; старые `limit100` runs в проекте тоже были проблемны |

Вывод из таблицы: если ориентироваться на "не просесть" относительно `gpt-4o` и ближайшего `claude-sonnet-4`, `qwen3-32b`, `gpt-oss-20b`, `qwen3-30b-a3b`, `deepseek-r1` и `deepseek-r1-0528` проходят quality bar. Если нужен pragmatic local вариант, shortlist сужается до `qwen3-32b`, `qwen3-30b-a3b`, `gpt-oss-20b`; если нужна максимальная точность и есть железо - `gpt-oss-120b` или большой `deepseek-r1`.

## Риски при замене judge

Свежие работы по safety evaluation показывают, что автоматический judge сам является источником ошибки:

- В июньской работе 2026 года про jailbreak judges dedicated classifier имел высокий recall, но over-flagging; prompted LLM judges сохраняли precision, но recall был нестабилен. Авторы рекомендуют репортить precision/recall judge на human-labeled slice и делать adversarial check judge.
- Работа "A Coin Flip for Safety" показывает, что на red-teaming distribution shifts LLM-as-judge может деградировать почти до случайного уровня; атаки могут увеличивать измеренный ASR за счет слабостей judge, а не реального harmful content.
- Для open-weight judge есть дополнительный риск white-box атаки на сам judge. Это не проблема для offline eval на фиксированном CARES, но важно, если judge будет частью публичного guardrail pipeline.

Для этого проекта риск особенно заметен на трех полях:

- `over_refusal`: safety classifiers часто склонны блокировать benign medical prompts и не видят полезность ответа.
- `bypass_followed`: general LLM judges могут быть обмануты harmless framing / refusal prefix.
- `utility_score` и `emergency_escalated`: guard модели обычно не обучены на такой медицинский rubric.

## Рекомендованная схема эксперимента

Минимальный честный план перед заменой:

1. Зафиксировать один CARES slice, например текущий `limit=300`, `seed=42`, `sample_shuffle=42`.
2. Запустить тех же outputs через:
   - `gpt-4o` или сохраненный старый run как reference;
   - `qwen3-32b`;
   - `qwen3-30b-a3b` или `gpt-oss-20b`;
   - `Qwen3Guard-Gen-8B` или `WildGuard` как auxiliary safety/refusal classifier.
3. Сравнить не только aggregate rates, но и disagreement rows:
   - risky false negative: open judge ставит success там, где закрытый judge видит unsafe_pass;
   - benign false positive: open judge ставит over_refusal там, где закрытый judge принимает ответ;
   - parse/schema errors;
   - disagreements по bypass samples отдельно.
4. Разметить вручную 50-100 disagreement examples, стратифицированно: benign, direct risky, obfuscate, indirect, role_play, emergency.
5. Выбрать judge по human agreement, а не только по TIMETOACT score.

Для итогового отчета лучше показывать:

- primary metrics по одному выбранному judge;
- agreement table между закрытым reference и open judge;
- error audit на disagreement slice;
- отдельный auxiliary safety classifier rate (`unsafe/refusal/bypass`) как robustness check.

## Drop-in shortlist

### Основной выбор

`qwen/qwen3-32b`

Причины:

- лучший practical open candidate в TIMETOACT после тяжелого `gpt-oss-120b`;
- score 71.1%, что выше и `gpt-4o-2024-11-20`, и `claude-sonnet-4` в этой таблице;
- Qwen family уже используется в проекте (`Qwen3-1.7B`, `Qwen3Guard-0.6B`), поэтому меньше интеграционного риска;
- должен лучше следовать JSON/rubric, чем чистый safety classifier.

Проверить перед финальным переходом:

- stable JSON output без markdown;
- schema parse failure rate;
- bias на benign over-refusal;
- sensitivity к medical emergency prompts.

### Второй judge в open ensemble

`qwen/qwen3-30b-a3b` или `openai/gpt-oss-20b`

`qwen3-30b-a3b` ближе к уже используемой Qwen экосистеме и в TIMETOACT почти равен Claude Sonnet 4. `gpt-oss-20b` чуть выше по score, но у TIMETOACT есть caveat про structured outputs у GPT-OSS/Harmony, что критично для текущего JSON scorer.

### Auxiliary safety judge

`Qwen/Qwen3Guard-Gen-8B` или `allenai/wildguard`

Использование:

- не просить их выставлять весь medical JSON;
- просить отдельные labels: prompt unsafe, response unsafe, refusal yes/no, jailbreak/bypass;
- использовать как cross-check и как feature для disagreement triage.

## Источники

- TIMETOACT GROUP, "LLM Benchmarks Summer 2025": https://www.timetoact-group.at/en/insights/llm-benchmarks/llm-benchmarks-summer-2025
- Qwen3Guard model card: https://huggingface.co/Qwen/Qwen3Guard-Gen-8B
- Qwen3Guard Technical Report: https://arxiv.org/abs/2510.14276
- WildGuard paper: https://arxiv.org/abs/2406.18495
- WildGuard model card: https://huggingface.co/allenai/wildguard
- Llama Guard 3 model card: https://huggingface.co/meta-llama/Llama-Guard-3-8B
- ShieldGemma model card: https://huggingface.co/google/shieldgemma-9b
- HarmBench repository: https://github.com/centerforaisafety/HarmBench
- HarmBench paper: https://arxiv.org/abs/2402.04249
- Benchmarking Open-Source Safety Guard Models, 2026: https://arxiv.org/abs/2605.28830
- How Reliable Is Your Jailbreak Judge?, 2026: https://arxiv.org/abs/2606.25487
- A Coin Flip for Safety, 2026: https://arxiv.org/abs/2603.06594
- Prometheus 2: https://arxiv.org/abs/2405.01535
