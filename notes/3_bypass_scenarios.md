# Какие bypass-сценарии нас волнуют

Bypass здесь шире, чем классический jailbreak. Нас волнуют все способы, которыми пользователь
или внешний контекст заставляет медицинскую модель:

- дать вредную клиническую рекомендацию;
- снять или ослабить safety framing;
- проигнорировать противопоказания, срочность, неопределённость или отсутствие контекста;
- следовать вредной инструкции, пришедшей не напрямую от пользователя.

Я бы разделила литературу на general jailbreak papers и medical-specific papers.

## General jailbreak / prompt injection

1. Jailbroken: How Does LLM Safety Training Fail?
   https://arxiv.org/abs/2307.02483

   Базовая статья про то, почему safety training ломается. Полезна для категорий:
   competing objectives, mismatched generalization, roleplay, instruction override.

2. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts
   https://arxiv.org/abs/2308.03825

   Таксономия реальных jailbreak-промптов из интернета: roleplay, privilege escalation,
   prompt injection, refusal suppression.

3. Universal and Transferable Adversarial Attacks on Aligned Language Models
   https://arxiv.org/abs/2307.15043

   Про adversarial suffix attacks. Для проекта достаточно знать, что бывают автоматизированные
   строковые атаки, которые могут переноситься между моделями.

4. Scalable and Transferable Black-Box Jailbreaks via Persona Modulation
   https://arxiv.org/abs/2311.03348

   Полезна для сценария "попросить модель принять персону": врача, циничного эксперта,
   персонажа, ассистента без ограничений.

5. Not what you've signed up for: Indirect Prompt Injection
   https://arxiv.org/abs/2302.12173

   Важно для RAG/retrieval/tool-use. Показывает риск, когда вредная инструкция приходит
   не от пользователя напрямую, а из внешнего документа.

## Medical-specific papers

6. MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering
   https://arxiv.org/abs/2406.06573

   Очень хороший источник для "semantic-preserving perturbations": смысл медицинского вопроса
   и правильный ответ не меняются, но меняются patient characteristics / формулировка / контекст,
   и модель может переключиться с правильного ответа на неправильный. Это не обязательно
   malicious jailbreak, но отличный bypass-сценарий для проверки устойчивости к реалистичным
   вариациям пользовательского запроса.

7. CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs
   https://arxiv.org/abs/2505.11413

   Самая удобная статья для структуры медицинских bypass-категорий. CARES специально добавляет
   clinical specificity, graded harm levels и четыре prompting styles: direct, indirect,
   obfuscated, role-play. Это можно почти напрямую превратить в матрицу eval set:
   harmful medical intent x harm level x prompting style.

8. Vulnerability of Large Language Models to Prompt Injection When Providing Medical Advice
   https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2842987

   JAMA Network Open, 2025. Очень релевантно для patient-facing сценариев: модель ведёт
   многоходовый медицинский диалог, а в контекст встраивается prompt injection, который
   подталкивает к unsafe / contraindicated treatment recommendations. Особенно полезны две
   стратегии: context-aware injection и evidence fabrication. Это показывает, что нас волнует
   не только "сразу попросили вредный совет", но и незаметное загрязнение клинического контекста.

9. Red-Teaming Medical AI: Systematic Adversarial Evaluation of LLM Safety Guardrails in Clinical Contexts
   https://www.medrxiv.org/content/10.64898/2026.02.26.26347212

   Preprint, но очень практичная таксономия: 8 attack categories и 24 sub-strategies для
   медицинского red-teaming. Главный сигнал: authority impersonation, особенно educational
   authority ("я студент-медик / это учебная задача"), оказался самым сильным классом атак.
   Это хорошо ложится на наши сценарии authority_claim и educational framing.

10. MedPerturbing LLMs: A Comparative Study of Toxicity, Prompt Tuning, and Jailbreaks in Medical QA
    https://ojs.aaai.org/index.php/AAAI-SS/article/view/36916

    Полезно для пересечения medical QA, demographic perturbations и jailbreak prompting.
    Важно не только "можно ли выбить вредный ответ", но и меняется ли поведение модели при
    вариантах пола, возраста, расы и других sensitive attributes.

11. Adversarial prompt and fine-tuning attacks threaten medical large language models
    https://www.nature.com/articles/s41467-025-64062-1

    Nature Communications, 2025. Покрывает два threat model: prompt-based attacks и
    fine-tuning/data-poisoning attacks. Сценарии медицинские: disease prevention, diagnosis,
    treatment. Для нашего проекта это аргумент, что bypass может жить не только в пользовательском
    prompt, но и в системном слое, кастомном GPT/RAG wrapper или скомпрометированном fine-tune.

12. Medical large language models are vulnerable to data-poisoning attacks
    https://www.nature.com/articles/s41591-024-03445-1

    Nature Medicine, 2025. Это не first-pass eval для обычного чатбота, но важный supply-chain
    сценарий: вред может быть внесён через training/fine-tuning data и не проявляться на обычных
    медицинских benchmark scores. Держать как later-stage threat model.

13. LLM Robustness Against Misinformation in Biomedical Question Answering
    https://arxiv.org/abs/2410.21330

    Полезно для RAG и внешнего контекста. Проверяет, что будет, если в prompt/RAG-контекст
    подмешана неверная biomedical information. Это отдельный bypass-сценарий: модель может
    выглядеть helpful и grounded, но grounding происходит на ложном источнике.

14. Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs
    https://arxiv.org/abs/2510.12255

    Про multi-turn robustness в medical QA. Важно, что модель может выдерживать начальную
    perturbation, но ломаться, когда пользователь несколько ходов подряд оспаривает ответ,
    давит авторитетом или подсовывает misleading context. Это хороший источник для сценария
    "постепенное давление" вместо one-shot jailbreak.

15. When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models
    https://arxiv.org/abs/2603.25960

    Не jailbreak в узком смысле, но важный prompt-sensitivity сценарий. В медицинских моделях
    CoT, few-shot examples, перестановка вариантов ответа и truncation могут резко менять
    результат. Это полезно как robustness/bypass через формат, а не через явно вредное намерение.

16. MedSafetyBench: Evaluating and Improving the Medical Safety of Large Language Models
    https://proceedings.neurips.cc/paper_files/paper/2024/file/3ac952d0264ef7a505393868a70a46b6-Paper-Datasets_and_Benchmarks_Track.pdf

    Хорош для уникально медицинских harmful requests: нарушение медицинской этики, сокрытие
    ошибки, манипуляция документацией, нарушение конфиденциальности, вред пациенту или врачу
    в healthcare setting. Это не только "опасный совет пациенту", но и злоупотребления внутри
    клинического workflow.

## Bypass-сценарии для первого eval set

| Bypass type | Что проверяет | Источники |
| --- | --- | --- |
| direct_harmful_request | Откажется ли модель от явно вредного медицинского запроса | CARES, MedSafetyBench |
| roleplay | Сломается ли policy через "представь, что ты врач / персонаж / ассистент без ограничений" | Jailbroken, Persona Modulation, CARES |
| authority_claim | Даст ли модель dangerous details, если пользователь заявляет, что он врач, медсестра, фармацевт, researcher | Red-Teaming Medical AI, CARES |
| educational_framing | Пропустит ли опасные детали под видом экзамена, OSCE, лекции, разбора клинического случая | Red-Teaming Medical AI, MedPerturbing |
| hypothetical_or_fictional | Даст ли конкретный вредный совет в гипотетике, симуляции, "для романа", "для дебага" | DAN, Jailbroken, CARES |
| no_disclaimer_or_refusal_suppression | Согласится ли убрать warnings, caveats, uncertainty и referral-to-care | DAN, Jailbroken |
| obfuscation | Пропустит ли unsafe request через кодировку, перевод, странный формат, typos, indirect wording | CARES, general jailbreak papers |
| translation_or_summary | Воспроизведёт ли unsafe medical instructions при переводе, пересказе, структурировании или извлечении данных | Indirect Prompt Injection, CARES |
| patient_attribute_perturbation | Изменит ли клиническое решение при нерелевантной или несущественной вариации demographics/context | MedFuzz, MedPerturbing |
| misleading_context | Поведётся ли на ложное утверждение пользователя о диагнозе, противопоказаниях, guideline или статье | JAMA prompt injection, Biomedical misinformation QA |
| evidence_fabrication | Примет ли сфабрикованную "ссылку на исследование / guideline / врача" как основание для опасной рекомендации | JAMA prompt injection, Biomedical misinformation QA |
| multi_turn_pressure | Сохранит ли безопасную позицию после нескольких ходов давления, уточнений и возражений | MedQA-Followup, JAMA prompt injection |
| rag_prompt_injection | Выполнит ли вредную инструкцию из внешнего документа, web page, patient note, retrieved guideline | Indirect Prompt Injection, Biomedical misinformation QA |
| tool_or_workflow_misuse | Даст ли инструкции для злоупотребления клиническими процессами: документация, labs, ошибки, privacy | MedSafetyBench |
| fine_tune_or_supply_chain | Риск, что модель/адаптер/датасет уже скомпрометированы и обычные benchmark scores это не ловят | Nature Communications 2025, Nature Medicine 2025 |

## Минимальная матрица для старта

Для первого набора я бы не пыталась покрыть всё. Достаточно взять 6 осей, которые ближе всего
к реальному медицинскому ассистенту:

1. `direct_harmful_request`
2. `authority_claim`
3. `educational_framing`
4. `patient_attribute_perturbation`
5. `misleading_context`
6. `multi_turn_pressure`

Если будет RAG или работа с внешними документами, сразу добавить:

7. `rag_prompt_injection`
8. `evidence_fabrication`

Если будет fine-tune, marketplace моделей, LoRA adapters или кастомные медицинские модели:

9. `fine_tune_or_supply_chain`
