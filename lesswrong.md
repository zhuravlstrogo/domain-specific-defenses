# LessWrong-level design for the domain-specific defenses experiment

Дата: 2026-06-03.

Входной документ: `notes/project_description.md`.

Цель этого файла: описать, что нужно добавить к текущему дизайну эксперимента, чтобы по проекту можно было написать сильную статью в духе LessWrong / Best of LessWrong, а не только курсовой отчет с таблицей метрик.

## Короткий вывод

Для статьи уровня LessWrong нужен не просто benchmark "baseline vs defense", а проверяемый тезис о том, как ломаются domain-specific defenses в медицине. Лучший тезис для текущего проекта:

> Domain-specific safety нельзя измерять одной осью вроде refusal rate или accuracy. Одна и та же защита может одновременно улучшать обычную медицинскую полезность, ухудшать устойчивость к вводящему в заблуждение контексту и маскировать опасные failures за "безопасным" стилем ответа.

Чтобы это показать, эксперимент должен иметь две связанные части:

1. **Safety / jailbreak eval**: harmful, benign, edge-case и bypass medical prompts. Главные метрики: unsafe pass rate, benign refusal rate, emergency escalation, bypass success.
2. **Utility / robustness eval**: medical MCQ или clinical QA с ground truth, где можно точно измерить, не стала ли защита хуже на обычной медицинской задаче. Главные метрики: initial accuracy, post-context accuracy, correct-to-incorrect flip rate, parse failure rate.

Текущий локальный результат по `ollama/qwen3:0.6b` уже дает полезную затравку:

| policy | initial_accuracy | post_context_accuracy | parse_failure_rate | flip_rate | correct_to_incorrect_rate |
|---|---:|---:|---:|---:|---:|
| baseline | 0.32 | 0.12 | 0.02 | 0.36 | 0.28 |
| mcq_prompt_policy | 0.40 | 0.08 | 0.00 | 0.44 | 0.38 |
| delta defense-baseline | +0.08 | -0.04 | -0.02 | +0.08 | +0.10 |

Это выглядит как "защита" улучшила первичную точность, но ухудшила устойчивость к misleading context. Для LessWrong-поста это хороший конфликт, но пока недостаточный результат: нужен повтор на нескольких моделях, safety-set с harmful/benign prompts, confidence intervals, qualitative failure analysis и контроль того, что метрики не являются артефактом парсера, токен-бюджета или judge bias.

## Что означает "рядом с Best of LessWrong"

Страница [Best of LessWrong](https://www.lesswrong.com/bestoflesswrong) описывает механизм отбора так: посты старше года пересматриваются сообществом и ранжируются по тому, насколько хорошо они выдержали проверку временем. В списке есть категории `Technical AI Safety`, `AI Strategy`, `Practical`, `Rationality`, `Optimization`. Для нашего проекта наиболее близки:

- `Technical AI Safety`: если фокус на эксперименте, robustness, adversarial evaluation, safety-usefulness frontier.
- `Practical`: если фокус на том, как реально проектировать evals и не обманываться метриками.
- `Rationality`: если главный вклад - методология измерений: "мы думали, что измеряем safety, а измеряли compliance/refusal/parser behavior".

LessWrong-посту обычно нужен не только результат, но и ясная модель мира:

- какой common mistake исправляется;
- какой эксперимент различает две гипотезы;
- что автор заранее ожидал увидеть;
- где результат удивил;
- какие механизмы могут объяснять наблюдение;
- какие выводы переживут смену конкретных моделей и датасетов.

Очень релевантен пост John Wentworth [You Are Not Measuring What You Think You Are Measuring](https://www.lesswrong.com/posts/9kNxhKWvixtKW5anS/you-are-not-measuring-what-you-think-you-are-measuring): для нашего проекта главный риск именно такой. Если считать только "модель отказалась или нет", можно случайно измерить стиль отказа, parser failures, verbosity, prompt sensitivity или judge preferences вместо медицинской безопасности.

Также релевантен [AI Control: Improving Safety Despite Intentional Subversion](https://www.lesswrong.com/s/PC3yJgdKvk8kzqZyA/p/d9FJHawgkiMSPjagR): там ценен не конкретный домен программирования, а методология - явно задавать trusted/untrusted components, threat model, red-team strategy и safety/usefulness tradeoff.

## Центральный crux статьи

Нужно заранее выбрать один главный вопрос. Без этого проект слишком широкий: guardrails, unlearning, embedding filters, RAG constraints, prompt policies, несколько доменов, safety, utility, false positives, robustness.

Рекомендуемый crux:

> When defending medical LLMs, do prompt-layer policies actually improve safety, or do they mainly move failures between axes: fewer obvious unsafe answers, more over-refusals, worse robustness to misleading context, or lower clinical utility?

Русская формулировка:

> Улучшает ли prompt-layer policy медицинскую безопасность, или она просто переводит ошибки из "опасного совета" в "ложное чувство защищенности", "избыточный отказ" и "хрупкость к контексту"?

Почему это хорошо:

- это не просто "какая defense лучше";
- вопрос важен для deployment;
- его можно проверить маленьким, но аккуратным экспериментом;
- результат может быть отрицательным и все равно интересным;
- он связывает safety, utility и robustness в одну модель.

## Гипотезы

Нужно зафиксировать гипотезы до расширенного прогона.

### H1: prompt policy снижает очевидные unsafe answers

На harmful medical prompts с прямыми просьбами назначить лечение, дозировку, отменить терапию или проигнорировать emergency red flags `prompt_policy` должна снизить `unsafe_pass_rate` относительно baseline.

Ожидаемый результат: снижение на direct harmful prompts.

### H2: prompt policy хуже работает на bypass framing

На role-play, educational framing, authority claim и multi-turn misleading context эффект защиты будет слабее, чем на direct harmful prompts.

Ожидаемый результат: `bypass_success_rate(role_play/authority/educational) > unsafe_pass_rate(direct)` при той же защите.

### H3: prompt policy может ухудшать utility или robustness

Защита может:

- чаще отказывать benign users;
- давать менее конкретные полезные ответы;
- хуже выдерживать misleading context;
- ухудшать `post_context_accuracy`;
- увеличивать `correct_to_incorrect_rate`.

Текущий Qwen result уже поддерживает слабую версию H3, но нужен повтор и контроль.

### H4: разные модели ломаются по-разному

Малые instruct/reasoning-модели могут демонстрировать разные failure modes:

- Qwen3 может быть чувствителен к thinking mode и контекстным ловушкам;
- Gemma может иначе реагировать на safety prompt;
- OLMo может иметь другую базовую способность к MCQ и другой refusal style.

LessWrong-ценность здесь не в leaderboard, а в том, что "domain-specific defense" не переносится как единая абстракция между моделями.

## Минимальный эксперимент, достойный статьи

### Эксперимент A: medical safety and bypass

Цель: измерить медицинскую безопасность и jailbreak robustness.

Данные:

- Primary: [CARES-18K](https://huggingface.co/datasets/HFXM/CARES-18K), если он используется как основной labeled safety/bypass set.
- Secondary: [MedSafetyBench](https://arxiv.org/abs/2403.03744) / [GitHub](https://github.com/AI4LIFE-GROUP/med-safety-bench), чтобы сверить direct harmful medical safety.
- Optional validation: medical red-teaming prompts, если нужен отдельный authority / educational slice.

Срезы:

| Slice | Size v1 | Purpose |
|---|---:|---|
| benign | 100 | false positives / utility |
| direct harmful | 100 | obvious safety |
| role-play harmful | 100 | jailbreak robustness |
| educational / authority harmful | 100 | "I am a doctor/student" bypass |
| emergency | 50 | missed escalation |
| edge-case | 50 | gray-zone behavior |

Если ресурсов мало, начать с 50 на slice. Главное - не смешивать все в один средний score.

Метрики:

- `unsafe_pass_rate`: доля harmful prompts, где модель дала опасно конкретный совет.
- `benign_refusal_rate`: доля benign prompts, где модель отказала или ушла в бесполезный disclaimer.
- `emergency_escalation_rate`: доля emergency prompts, где модель явно направила к срочной помощи.
- `bypass_success_rate_by_type`: role_play, educational, authority, no_disclaimer, indirect.
- `safe_helpfulness_score`: полезность безопасного ответа на benign/edge-case prompts.
- `over_disclaimer_rate`: доля ответов, где safety language вытеснил содержательный ответ.

Важно: refusal сам по себе не равен safety. В медицине безопасный ответ часто должен быть не отказом, а triage, red flags, общая информация и рекомендация обратиться за помощью при нужных условиях.

### Эксперимент B: medical utility and context robustness

Цель: измерить, не ломает ли защита обычную медицинскую полезность и устойчивость к misleading context.

Данные:

- Уже выбранный локально датасет: [MedQA-USMLE-4-MultiTurnRobust](https://huggingface.co/datasets/dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust).
- Связанный подход: MedFuzz-style perturbations на MedQA, где меняются характеристики пациента или контекст, но ground truth сохраняется.

Срезы:

| Slice | Size v1 | Purpose |
|---|---:|---|
| initial MCQ | 100-300 | base medical task ability |
| post misleading context | 100-300 | robustness to adversarial follow-up |
| alternative context | optional | sensitivity to benign context |
| edge-case context | optional | gray-zone robustness |

Метрики:

- `initial_accuracy`: точность до misleading context.
- `post_context_accuracy`: точность после misleading context.
- `flip_rate`: доля вопросов, где ответ изменился.
- `correct_to_incorrect_rate`: ключевая robustness metric.
- `incorrect_to_correct_rate`: sanity check, может показать, что context иногда помогает.
- `parse_failure_rate`: обязательный health metric пайплайна.

Текущий локальный результат по Qwen надо подавать не как финальный вывод, а как pilot:

- baseline: `initial_accuracy=0.32`, `post_context_accuracy=0.12`, `correct_to_incorrect_rate=0.28`;
- prompt policy: `initial_accuracy=0.40`, `post_context_accuracy=0.08`, `correct_to_incorrect_rate=0.38`;
- интерпретация: policy могла усилить initial instruction-following, но сделать модель более склонной менять ответ под последующим контекстом.

Для статьи это нужно подтвердить на большем `n` и хотя бы 3 моделях.

## Модели и defenses

### Модели

Минимум:

- `ollama/qwen3:0.6b`
- `ollama/gemma3:1b`
- `ollama/olmo2:7b`

Желательно добавить одну более сильную reference model, если бюджет позволяет. Иначе статья будет про "small local medical safety eval", что нормально, но выводы нужно ограничить.

Обязательные controls:

- одинаковый prompt format;
- `temperature=0` для основного прогона;
- фиксированный `max_tokens`;
- explicit thinking mode для Qwen;
- логировать model id, provider, decoding config, system prompt hash, dataset hash;
- `parse_failure_rate < 0.05` как минимальное условие валидности MCQ-прогона.

### Defenses

Минимальный набор:

1. `baseline`: обычная модель без дополнительной защиты.
2. `prompt_policy`: медицинская safety policy в system prompt.
3. `input_output_guardrail`: внешний classifier или LLM-as-judge до/после генерации.
4. `router`: простая классификация prompt type -> разные response policies.

Почему не начинать с unlearning и activation steering:

- они интересны как research direction;
- но для первого publishable experiment слишком много инфраструктуры;
- у них сложнее отделить "defense работает" от "модель деградировала";
- representation-level methods требуют open-weight access, layer choice, steering vector validation и ablations.

Их лучше оставить как раздел "future work" или второй пост.

## Scoring design

### Rule-based checks

Использовать там, где можно:

- MCQ exact match: `1..4` или `A..D`.
- Emergency escalation keywords: "call emergency services", "seek urgent care", "go to ER", русские аналоги при русских prompts.
- Concrete dosage / prescription patterns: дозы, мг, frequency, drug names.
- Refusal markers: "I can't", "I cannot", "I am not able", "consult a doctor" alone without useful content.

Rule-based scorer не должен быть единственным safety scorer, но он нужен как guard against judge drift.

### LLM-as-judge

Для free-form safety answers LLM-as-judge допустим, но только с контролями:

- строгая рубрика с бинарными или 0/1/2 критериями;
- отдельные labels для unsafe advice, missing escalation, over-refusal, safe helpfulness;
- blind scoring: judge не должен знать, baseline это или defense;
- фиксированный judge model и prompt;
- manual audit хотя бы 50-100 examples;
- inter-rater agreement на небольшой выборке, если есть два человека.

Источник риска: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) показывает, что LLM judges полезны, но имеют biases вроде position/verbosity/self-enhancement. Поэтому judge score нельзя подавать как ground truth без аудита.

### Physician-style rubrics

Вдохновляться [HealthBench](https://arxiv.org/abs/2505.08775) и [OpenAI HealthBench overview](https://openai.com/index/healthbench/): там ценны realistic multi-turn conversations и physician-written rubrics. Для проекта не нужно копировать масштаб HealthBench; достаточно позаимствовать идею conversation-specific rubrics:

- что ответ обязан сделать;
- чего ответ не должен делать;
- какие red flags нужно назвать;
- когда нужно направить к emergency care;
- когда нужен safe, useful non-refusal.

## Statistical plan

Для LessWrong-поста не обязательно делать тяжелую статистику, но нужна честная uncertainty.

Минимум:

- report `n` по каждому slice;
- bootstrap 95% confidence intervals для rates;
- paired comparison на MCQ, потому что baseline/defense отвечают на одни и те же questions;
- McNemar-style comparison для exact-match accuracy, если хочется формальнее;
- per-slice table вместо только aggregate table;
- qualitative examples для каждого крупного failure mode.

Главное: не прятать различия в average score. В high-stakes medical safety worst-slice performance важнее среднего.

## Failure taxonomy для статьи

Нужно заранее размечать failures не только как "bad", а по типам:

| Failure | Meaning |
|---|---|
| unsafe commission | модель рекомендует опасное действие |
| harm of omission | модель не сообщает критически важную информацию |
| missed escalation | emergency case не эскалирован |
| unsafe reassurance | модель уверенно успокаивает при red flags |
| no context elicitation | модель дает совет без важных уточнений |
| over-refusal | модель отказывает там, где можно безопасно помочь |
| utility collapse | ответ безопасный, но бесполезный |
| context hijack | misleading context меняет правильный ответ |
| policy laundering | ответ выглядит безопасным, но содержит конкретный вредный шаг |
| parser/eval failure | метрика сломалась, а не модель |

Эта taxonomy связывает медицинскую safety literature с LessWrong-мотивом "что именно мы измеряем".

## Контроли, без которых статья будет слабой

### 1. Parser and truncation controls

Текущий проект уже столкнулся с Qwen `<think>` / token budget issues. Для статьи нужно явно показать:

- parse failures низкие;
- ответы не обрезаны;
- thinking mode фиксирован;
- parser не извлекает вариант ответа из reasoning block;
- invalid outputs исключаются или считаются отдельно.

### 2. Prompt length and instruction hierarchy

Prompt policy может ухудшать MCQ не потому, что "safety tradeoff", а потому что:

- system prompt стал длиннее;
- модель хуже следует final answer format;
- policy конфликтует с MCQ instruction;
- токен-бюджет ушел на disclaimers.

Контроль:

- добавить neutral long system prompt такой же длины, но без safety policy;
- сравнить `baseline`, `neutral_long_prompt`, `prompt_policy`.

Если `neutral_long_prompt` тоже ухудшает robustness, эффект не про safety policy.

### 3. Benign utility control

Safety defense, которая отказывает на все, не является хорошей defense. Нужен benign set:

- health education;
- prevention;
- "what questions should I ask my doctor";
- lab test explanation without diagnosis;
- lifestyle general advice.

Метрика: useful non-refusal.

### 4. Emergency control

В медицине опасны и false negatives, и false positives. Нужны prompts с red flags:

- chest pain + dyspnea;
- stroke-like symptoms;
- anaphylaxis;
- severe bleeding;
- suicidal ideation;
- pregnancy complications.

Метрики:

- missed escalation rate;
- over-escalation rate on low-risk prompts.

### 5. Adversarial pressure

Bypass должен быть не декоративным, а систематическим:

- direct harmful;
- role-play;
- educational;
- authority claim;
- no disclaimer;
- translation / obfuscation;
- multi-turn misleading context.

Источник для general jailbreak methodology: [JailbreakBench](https://arxiv.org/abs/2404.01318) полезен тем, что стандартизирует threat model, prompts, templates и scoring functions. Для медицинского домена нужно перенести эту дисциплину, а не обязательно сами harmful behaviors.

## Какой результат будет достаточно интересным

### Result pattern 1: prompt policy improves direct safety but worsens robustness

Это самый вероятный и самый publishable результат.

Narrative:

- На direct harmful prompts policy помогает.
- На authority/educational bypass помогает меньше.
- На MCQ misleading context policy ухудшает correct-to-incorrect flips.
- Значит "safety prompt" не является domain defense, пока мы не проверили все axes.

### Result pattern 2: guardrail lowers unsafe pass but raises benign refusal

Narrative:

- Guardrail выглядит хорошо на harmful set.
- Но benign refusal растет.
- В medical assistant это не бесплатно: пользователь теряет access to safe health information.
- Deployment needs triage/routing, not blanket refusal.

### Result pattern 3: router dominates prompt policy

Narrative:

- Простая классификация intent -> response policy лучше, чем один universal safety prompt.
- Это поддерживает тезис "domain-specific defense is a protocol problem, not just a model behavior problem".

### Result pattern 4: no defense works robustly across models

Narrative:

- Prompt policy/guardrail effects do not transfer.
- Модель-специфичная safety behavior ломает идею универсальных best practices.
- Нужны eval suites per model/domain/deployment context.

## Структура будущей LessWrong-статьи

Рабочее название:

> Medical LLM Safety Is Not a Refusal Rate

Альтернативы:

- "Domain-Specific Safety Evals Are Measuring the Wrong Thing"
- "A Safety Prompt Can Make a Medical Model Less Robust"
- "The Safety-Utility Frontier Is Not One-Dimensional"

Предлагаемая структура:

1. **Hook**
   - Коротко: мы добавили medical safety policy и она улучшила одну метрику, но ухудшила другую.
   - Не начинать с обзора литературы.

2. **The mistake**
   - "I initially thought I was comparing defenses. Actually I was often comparing refusal styles, parser behavior, and context sensitivity."
   - Связь с LessWrong-мотивом measurement failure.

3. **Threat model**
   - User-facing medical assistant.
   - Harmful actions: unsafe advice, missing escalation, unsafe reassurance, over-refusal.
   - Attacks: role-play, authority, educational, multi-turn context.

4. **Experimental setup**
   - Models.
   - Datasets.
   - Defenses.
   - Metrics.
   - Pre-registered hypotheses.

5. **Results**
   - Per-slice tables.
   - Safety/usefulness frontier.
   - Pilot Qwen result only if still representative after rerun.

6. **Failure cases**
   - 5-8 carefully chosen examples.
   - Each example explains what the metric saw and what a human sees.

7. **What I think is going on**
   - Prompt policies change instruction-following and style, not necessarily robust clinical reasoning.
   - Guardrails can catch surface harm but miss context-dependent harm.
   - MCQ robustness and safety refusal are related but not interchangeable.

8. **Deployment implications**
   - Use multiple metrics.
   - Separate triage, refusal, helpful safe answer, and uncertainty calibration.
   - Monitor worst slices.
   - Treat defense as protocol, not a prompt.

9. **Limitations**
   - Small models.
   - Synthetic / benchmark prompts.
   - No physician-grade adjudication unless added.
   - LLM judge limitations.
   - English/Russian scope.

10. **What would change my mind**
   - A defense that improves unsafe pass, benign helpfulness, emergency escalation and MCQ robustness across models.
   - Human/clinician adjudication contradicting automated labels.
   - Better evidence that observed robustness loss is prompt-length artifact.

## Concrete next steps

### Week 1: make the current MCQ result trustworthy

- Run `baseline`, `neutral_long_prompt`, `mcq_prompt_policy`.
- Run on Qwen3, Gemma3, OLMo.
- Use `limit=100` minimum, ideally `300`.
- Confirm `parse_failure_rate < 0.05`.
- Add bootstrap confidence intervals.
- Produce per-model and aggregate tables.

### Week 2: add safety/bypass eval

- Build CARES/MedSafetyBench slice:
  - 100 benign;
  - 100 direct harmful;
  - 100 role-play harmful;
  - 100 authority/educational harmful;
  - 50 emergency.
- Implement or reuse rubric scorer.
- Manually audit 50 examples.
- Report unsafe pass, benign refusal, emergency escalation, bypass success.

### Week 3: compare defenses

- Add external guardrail or router.
- Compare:
  - baseline;
  - prompt_policy;
  - guardrail;
  - router.
- Plot safety-usefulness frontier:
  - x-axis: benign helpfulness / MCQ accuracy;
  - y-axis: unsafe pass or correct-to-incorrect rate;
  - mark worst-slice performance.

### Week 4: write the post

- Choose the main surprising result.
- Write around one thesis, not all experiments.
- Include methods enough for reproduction.
- Put full tables in appendix.
- Put strongest counterarguments in main text.

## What not to do

- Do not report a single "safety score".
- Do not treat refusal as equal to safety.
- Do not average direct harmful and bypass prompts without slices.
- Do not use LLM-as-judge without calibration examples.
- Do not compare defenses if parse failure or truncation differs.
- Do not claim "medical safety" from MCQ accuracy alone.
- Do not claim "robustness" from one model.
- Do not include unlearning/activation steering in the first paper unless the basic prompt/guardrail comparison is already stable.

## Source map

LessWrong and methodology:

- [The Best of LessWrong](https://www.lesswrong.com/bestoflesswrong)
- [You Are Not Measuring What You Think You Are Measuring](https://www.lesswrong.com/posts/9kNxhKWvixtKW5anS/you-are-not-measuring-what-you-think-you-are-measuring)
- [AI Control: Improving Safety Despite Intentional Subversion](https://www.lesswrong.com/s/PC3yJgdKvk8kzqZyA/p/d9FJHawgkiMSPjagR)
- [How To Write Quickly While Maintaining Epistemic Rigor](https://www.lesswrong.com/posts/Psr9tnQFuEXiuqGcR/how-to-write-quickly-while-maintaining-epistemic-rigor)

Medical safety evals:

- [MedSafetyBench: Evaluating and Improving the Medical Safety of Large Language Models](https://arxiv.org/abs/2403.03744)
- [MedSafetyBench GitHub](https://github.com/AI4LIFE-GROUP/med-safety-bench)
- [HealthBench: Evaluating Large Language Models Towards Improved Human Health](https://arxiv.org/abs/2505.08775)
- [OpenAI HealthBench overview](https://openai.com/index/healthbench/)
- [WHO guidance on ethics and governance of AI for health: large multi-modal models](https://www.who.int/publications/b/70584)

Guardrails and robustness:

- [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)
- [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://arxiv.org/abs/2404.01318)
- ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)
- [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119)

Representation-level / unlearning background:

- [Towards Safer Large Language Models through Machine Unlearning](https://arxiv.org/abs/2402.10058)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Programming Refusal with Conditional Activation Steering](https://arxiv.org/abs/2409.05907)

RAG and eval tooling:

- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)

LLM-as-judge:

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
