# Локальные / open-weight альтернативы judge-моделям

Дата исследования: 2026-07-02.

Контекст репозитория: текущий scorer в `src/domain_defenses/scoring.py` использует LLM-as-judge для медицинской safety-оценки и требует структурированный JSON по полям `policy_success`, `unsafe_pass`, `over_refusal`, `bypass_followed`, `emergency_escalated`, `utility_score`, `rationale`. В старом отчете `report_29_06_26.md` обсуждалась пара judge `gpt-4o` + `claude-sonnet-4.5`; в текущем `scripts/run_cares_judge_models.sh` дефолт уже смещен к одному `gemini-2.5-pro`.

## Короткий вывод

Если цель - заменить закрытые judge-модели без сильной потери качества, самый разумный порядок проверки такой:

1. Оставить `qwen/qwen-2.5-72b-instruct` как baseline-кандидата, потому что он уже использовался в проекте как локальная/API judge-модель, поддерживает `response_format` и `structured_outputs` на OpenRouter, а полный judge eval на `LIMIT=3000` стоит примерно $25-$31. Низкий TIMETOACT score не доказывает плохое качество именно для текущей medical safety rubric; это надо проверять agreement/human audit.
2. Проверить `qwen/qwen3-32b` как более дешевую open-weight альтернативу. В TIMETOACT Summer 2025 он набирает 71.1%, выше `anthropic/claude-sonnet-4` 64.4% и `openai/gpt-4o-2024-11-20` 53.6%, а полный judge eval стоит примерно $7-$8.
3. Проверить `qwen/qwen3-235b-a22b-2507` и `qwen/qwen3-30b-a3b-instruct-2507`: обе модели поддерживают structured outputs на OpenRouter и стоят дешевле `qwen-2.5-72b`; первая выглядит как сильный open-weight judge, вторая как дешевый sanity/ensemble judge.
4. Для safety-specific сигнала добавить не вместо, а рядом с LLM-as-judge: `Qwen3Guard-Gen-4B/8B`, `allenai/wildguard`, `openai/gpt-oss-safeguard-20b` или `nvidia/nemotron-3.5-content-safety`. Они лучше подходят для классификации unsafe/refusal/bypass, но сами по себе не заменяют медицинский rubric judge, потому что не оценивают все поля текущей схемы, особенно utility и emergency escalation.

Главная рекомендация: не переходить на один open judge без калибровки. Практичный стартовый набор для проверки: `qwen-2.5-72b-instruct` + `qwen3-32b` + auxiliary safety judge. Если `qwen-2.5-72b-instruct` показывает лучший agreement на CARES disagreement slice, его можно оставить основным: стоимость у него все еще на порядок ниже GPT-4o/Claude Sonnet.

## Стоимость полного judge eval для `LIMIT=3000`

Сценарий: `scripts/run_cares_model_inference.sh` запускает 3 main-модели (`qwen3_1_7b`, `gemma_3_1b_it`, `olmo2_0425_1b_instruct`), а каждая experiment config содержит 5 policies (`baseline`, `prompt_policy`, `strict_prompt_policy`, `retrieval_constraints`, `qwen3_guardrail`). Поэтому полный scoring поверх этих inference logs:

```text
3 main models * 5 policies * 3000 examples = 45000 judge calls
```

Оценка токенов:

- по локальным smoke logs средний `MEDICAL_JUDGE_TEMPLATE` prompt был 5467 символов;
- это примерно 1.4K input tokens на judge-вызов;
- completion - короткий JSON, считаю 150 output tokens;
- базовый расчет: 63.0M input tokens + 6.75M output tokens;
- расчет с буфером на более длинные ответы / tokenizer drift: 76.5M input + 8.1M output.

Если оценивать только один policy на каждую из 3 моделей, делить суммы примерно на 5.

Цены взяты из OpenRouter `/api/v1/models` на 2026-07-02; OpenRouter price fields указаны в долларах за token, в таблице ниже переведены в $/1M tokens.

| Judge model | OpenRouter price, $/1M in/out | Full eval базово | Full eval с буфером | Комментарий |
| --- | ---: | ---: | ---: | --- |
| `qwen/qwen-2.5-72b-instruct` | 0.36 / 0.40 | $25.38 | $30.78 | Уже использовался; хороший low-risk baseline; поддерживает structured outputs |
| `qwen/qwen3-32b` | 0.08 / 0.28 | $6.93 | $8.39 | Лучший первый cheap challenger по TIMETOACT; проверить over-refusal |
| `qwen/qwen3-30b-a3b-instruct-2507` | 0.048 / 0.193 | $4.34 | $5.25 | Очень дешевый второй judge; качество надо валидировать |
| `qwen/qwen3-235b-a22b-2507` | 0.09 / 0.10 | $6.34 | $7.70 | Сильный open-weight кандидат; неожиданно дешевый на OpenRouter |
| `qwen/qwen3-235b-a22b-thinking-2507` | 0.1495 / 1.495 | $19.51 | $23.55 | Проверять, только если нужен reasoning; output/reasoning может удорожать |
| `deepseek/deepseek-r1-0528` | 0.50 / 2.15 | $46.01 | $55.66 | Качественный reasoning fallback, но дороже Qwen-вариантов |
| `deepseek/deepseek-r1` | 0.70 / 2.50 | $60.98 | $73.80 | Сильный, но не лучший cost/quality для массового judge |
| `deepseek/deepseek-chat-v3.1` | 0.21 / 0.79 | $18.56 | $22.46 | Дешевле R1, но слабее как rubric judge |
| `openai/gpt-oss-120b` | 0.03 / 0.15 | $2.90 | $3.51 | Дешево; проверять JSON/schema и judge stability отдельно |
| `openai/gpt-oss-20b` | 0.029 / 0.14 | $2.77 | $3.35 | Дешевый auxiliary judge, не основной без калибровки |
| `openai/gpt-4o` | 2.50 / 10.00 | $225.00 | $272.25 | Закрытый reference; дорого для регулярного полного scoring |
| `anthropic/claude-sonnet-4.5` | 3.00 / 15.00 | $290.25 | $351.00 | Закрытый reference; лучше оставить для audit slice |
| `google/gemini-2.5-pro` | 1.25 / 10.00 | $146.25 | $176.62 | Текущий default в конфиге, но заметно дороже Qwen2.5-72B |

Вывод по цене: `qwen/qwen-2.5-72b-instruct` стоит примерно $30 за полный judge pass на 45K outputs. Это достаточно дешево, чтобы оставить его основным кандидатом и не оптимизировать стоимость преждевременно. Самая рациональная экономия - использовать закрытые GPT-4o/Claude Sonnet только для 300-1000 sample audit slice, а полный scoring делать `qwen-2.5-72b-instruct` или Qwen3 open-weight judge.

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
| `qwen/qwen-2.5-72b-instruct` | Open | 39.2% | -14.4 pp | -25.2 pp | Не отбрасывать: TIMETOACT слабый, но для нашего medical rubric это уже проверенный low-risk baseline; сравнивать по agreement/human audit |

Вывод из таблицы: если ориентироваться только на TIMETOACT, `qwen3-32b`, `gpt-oss-20b`, `qwen3-30b-a3b`, `deepseek-r1` и `deepseek-r1-0528` проходят quality bar относительно `gpt-4o` и ближайшего `claude-sonnet-4`. Но для текущего проекта TIMETOACT не должен автоматически выбивать `qwen-2.5-72b-instruct`: это не safety/medical benchmark, а уже использованная judge-модель может оказаться лучше именно на вашей рубрике. Поэтому shortlist для практической проверки: `qwen-2.5-72b-instruct`, `qwen3-32b`, `qwen3-235b-a22b-2507`, `qwen3-30b-a3b-instruct-2507`; закрытые `gpt-4o`/`claude-sonnet-4.5` оставить как reference на audit slice.

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
   - `qwen-2.5-72b-instruct` как основной local/API baseline;
   - `qwen3-32b`;
   - `qwen3-235b-a22b-2507` или `qwen3-30b-a3b-instruct-2507`;
   - `gpt-4o` или `claude-sonnet-4.5` на audit slice как закрытый reference;
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

### Основной baseline

`qwen/qwen-2.5-72b-instruct`

Причины:

- уже использовался в проекте как judge;
- OpenRouter price дает около $25-$31 за полный scoring 45K outputs;
- поддерживает `response_format` и `structured_outputs`;
- достаточно дешевый, чтобы прогнать полный `LIMIT=3000` без преждевременной оптимизации.

Проверить перед финальным переходом:

- stable JSON output без markdown;
- schema parse failure rate;
- bias на benign over-refusal;
- sensitivity к medical emergency prompts.

### Первый challenger

`qwen/qwen3-32b`

Причины: сильный TIMETOACT score, Qwen family, structured outputs, низкая цена полного judge eval около $7-$8. Главный риск - реальное agreement на CARES может быть хуже, чем у `qwen-2.5-72b-instruct`, несмотря на general benchmark.

### Второй judge в open ensemble

`qwen/qwen3-235b-a22b-2507` или `qwen/qwen3-30b-a3b-instruct-2507`

`qwen3-235b-a22b-2507` выглядит как сильный open-weight кандидат при низкой цене OpenRouter; `qwen3-30b-a3b-instruct-2507` стоит еще дешевле и годится как второй голос/санити-чек. `openai/gpt-oss-20b/120b` можно добавить позже, но только после отдельной проверки structured output стабильности.

### Auxiliary safety judge

`Qwen/Qwen3Guard-Gen-8B` или `allenai/wildguard`

Использование:

- не просить их выставлять весь medical JSON;
- просить отдельные labels: prompt unsafe, response unsafe, refusal yes/no, jailbreak/bypass;
- использовать как cross-check и как feature для disagreement triage.

## Источники

- TIMETOACT GROUP, "LLM Benchmarks Summer 2025": https://www.timetoact-group.at/en/insights/llm-benchmarks/llm-benchmarks-summer-2025
- OpenRouter model catalog API, prices checked 2026-07-02: https://openrouter.ai/api/v1/models
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
