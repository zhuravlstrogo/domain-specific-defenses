# Защитные стратегии LLM в медицинском домене — итоговая карта

Способы защиты медицинских языковых моделей: от опасных советов, галлюцинаций (выдуманных фактов) и утечки данных пациентов. Защиты сгруппированы по **месту вмешательства** — это определяет, какие из них работают с закрытой моделью (только через API), а какие требуют доступа к весам.

Обозначения: `[статья]` — публикация, `[код]` — репозиторий или веса, `[док]` — документация.

---

## ЧАСТЬ 1. Полная карта защит

### A. Внешние защиты (wrapper)
*Не меняют веса модели, а контролируют её вход и выход.*

**1. Guardrails (внешние guard-модели)**
*Отдельная модель-классификатор проверяет запрос пользователя и/или ответ основной модели и блокирует небезопасный контент. Базовый паттерн «guard-модель перед LLM».*
- Llama Guard: [статья](https://arxiv.org/abs/2312.06674) · [код](https://github.com/meta-llama/PurpleLlama)
- Llama Guard 3: [статья](https://arxiv.org/abs/2411.10414) · [веса](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
- ShieldGemma: [статья](https://arxiv.org/abs/2407.21772) · [веса](https://huggingface.co/google/shieldgemma-9b)
- Aegis 2.0 (NVIDIA): [статья](https://arxiv.org/abs/2501.09004)
- NeMo Guardrails (конструктор правил): [код](https://github.com/NVIDIA-NeMo/Guardrails) · [док](https://docs.nvidia.com/nemo/guardrails/latest/index.html)
- L2M3 — медицинский стек (Llama Guard 3 → NeMo → генератор): [статья](https://arxiv.org/abs/2409.17190)

**2. Internal safety probes (внутренние пробы-детекторы)**
*Лёгкий классификатор поверх скрытых состояний (hidden states) модели, распознающий небезопасное намерение или галлюцинацию по внутренним сигналам.*
- Semantic Entropy Probes: [статья](https://arxiv.org/abs/2406.15927)

**3. Multi-agent verification (перекрёстная проверка несколькими моделями)**
*Несколько моделей-агентов проверяют ответы друг друга; ошибка одной отлавливается остальными.*
- MedSentry (для медицины): [статья](https://arxiv.org/abs/2505.20824)
- Многоагентные проверочные циклы: [статья](https://arxiv.org/abs/2601.13268)

**4. Self-critique (самопроверка на инференсе)**
*Модель перечитывает собственный ответ и оценивает его безопасность до выдачи пользователю — без отдельной guard-модели и без обучения.*
- LLM Self-Defense: [статья](https://arxiv.org/abs/2308.07308)

---

### B. Защиты на входе (inference-time, input-side)
*Проверка и преобразование запроса до подачи в модель.*

**5. Adversarial-input transformations (преобразование входа против атак)**
*Защита от состязательных приписок (например, GCG-суффиксов): вход многократно слегка возмущается или перефразируется, и расхождение ответов выдаёт атаку.*
- SmoothLLM: [статья](https://arxiv.org/abs/2310.03684) · [код](https://github.com/arobey1/smooth-llm)
- Perplexity-фильтр + перефразирование: в [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)

**6. Prompt-injection defenses (защита от инъекций в промпт)**
*Отделяют пользовательские инструкции от данных, которые модель просто читает (например, из найденных документов), чтобы спрятанная в данных команда не исполнялась. Критично для RAG и агентов.*
- Spotlighting (Microsoft): [статья](https://arxiv.org/abs/2403.14720)
- StruQ (разделение инструкций и данных): [статья](https://arxiv.org/abs/2402.06363)

**7. Prompt-level / system-prompt (защита на уровне инструкции)**
*Системный промпт с правилами (медицинская этика, требование рассуждать пошагово), снижающий долю опасных и ошибочных ответов.*
- Med-HEAL (учёт галлюцинаций в контексте): [статья](https://arxiv.org/abs/2606.01301)
- Пошаговое рассуждение (CoT) против галлюцинаций — данные: [статья](https://www.medrxiv.org/content/10.1101/2025.02.28.25323115v2)

---

### C. Защиты на выходе (inference-time, output-side)
*Контроль на этапе генерации ответа, без изменения весов.*

**8. Safe decoding (безопасное декодирование)**
*На этапе генерации повышает вероятность безопасных токенов и подавляет токены, ведущие к вредному ответу.*
- SafeDecoding: [статья](https://arxiv.org/abs/2402.08983) · [код](https://github.com/uw-nsl/SafeDecoding)

**9. Sampling-based hallucination detection (детекция галлюцинаций по сэмплированию)**
*Модель генерирует несколько ответов на один вопрос; противоречия между ними указывают на галлюцинацию. Работает с закрытым API, без доступа к весам.*
- SelfCheckGPT: [статья](https://arxiv.org/abs/2303.08896) · [код](https://github.com/potsawee/selfcheckgpt) · [установка](https://pypi.org/project/selfcheckgpt/)

**10. RAG-защиты (ответ с опорой на retrieval)**
*Модель отвечает, опираясь на проверенную базу документов, а не только на параметрическую память; это снижает галлюцинации и даёт ссылку на источник. Саму базу нужно защищать от отравления (knowledge poisoning).*
- Сравнение вариантов RAG для клинической поддержки: [статья](https://www.mdpi.com/2079-9292/14/21/4227)
- Отравление базы (атака, мотивирующая защиту): [статья](https://arxiv.org/abs/2605.10253)
- Обзор RAG в медицине: [статья](https://www.mdpi.com/2673-2688/6/9/226)

---

### D. Вмешательство в представления (representation-level)
*Работа с внутренними активациями модели — глубже слов, но требует доступа к весам.*

**11. Representation interventions / circuit breakers (вмешательство в представления)**
*Модель дообучают так, чтобы внутренние представления вредного контента «перенаправлялись» в отказ или несвязный вывод. В отличие от обычного обучения отказам, устойчиво к незнакомым атакам. Сюда же входит activation steering — сдвиг активаций в нужном направлении на инференсе.*
- Circuit Breakers (Representation Rerouting): [статья](https://arxiv.org/abs/2406.04313) · [код](https://github.com/GraySwanAI/circuit-breakers)
- RepBend: [статья](https://aclanthology.org/2025.acl-long.1173.pdf)
- Contrastive Representation Learning: [статья](https://arxiv.org/abs/2506.11938) · [код](https://github.com/samuelsimko/crl-llm-defense)
- Adaptive Activation Steering: [статья](https://arxiv.org/abs/2406.00034)
- Обзор representation engineering: [статья](https://arxiv.org/abs/2502.17601)

---

### E. Защиты через обучение (training-time)
*Меняют веса модели. Требуют открытой модели и вычислительных ресурсов.*

**12. Safety alignment / preference optimization (выравнивание безопасности)**
*Обучение модели безопасному поведению через дообучение на примерах (SFT) и оптимизацию по человеческим предпочтениям (RLHF/DPO).*
- DPO: [статья](https://arxiv.org/abs/2305.18290)
- Constitutional AI / RLAIF (обучение по своду правил): [статья](https://arxiv.org/abs/2212.08073)
- Медицинский вариант — дообучение на MedSafetyBench: [статья](https://arxiv.org/abs/2403.03744) · [код](https://github.com/AI4LIFE-GROUP/med-safety-bench)

**13. Safety-preserving fine-tuning (сохранение безопасности при дообучении)**
*Приёмы, не дающие модели терять выравнивание при доменной донастройке под медицину — эффект «alignment tax», когда дообучение разрушает безопасность даже без злого умысла.*
- Почему дообучение ломает безопасность (Qi et al., ICLR 2024): [статья](https://arxiv.org/abs/2310.03693)

**14. Model merging / safety grafting (слияние моделей)**
*Объединение весов безопасной базовой и медицинской модели, переносящее безопасное поведение без переобучения.*
- Task Arithmetic: [статья](https://arxiv.org/abs/2212.04089)
- TIES-Merging: [статья](https://arxiv.org/abs/2306.01708)
- mergekit (инструмент): [код](https://github.com/arcee-ai/mergekit)

**15. Safety LoRA (модульный адаптер безопасности)**
*Небольшой съёмный модуль безопасности, добавляемый к модели вместо полного переобучения; дёшево и обратимо.*
- SafeLoRA: [статья](https://arxiv.org/abs/2405.16833)

**16. Adversarial / continual fine-tuning (дообучение на атаках)**
*Дообучение на состязательных примерах, чтобы модель научилась сопротивляться джейлбрейкам.*
- Towards Safe AI Clinicians (CFT, −62.7% к успеху атак): [статья](https://arxiv.org/abs/2501.18632)
- Latent Adversarial Training (обучение против атак на уровне активаций): [статья](https://arxiv.org/abs/2403.05030)

**17. Abstention training / calibration (обучение отказу и калибровка уверенности)**
*Модель учат отвечать «недостаточно данных» при неуверенности вместо выдумывания. Строгий вариант — conformal prediction, дающий статистические гарантии (ценно в клинике).*
- R-Tuning: [статья](https://arxiv.org/abs/2311.09677) · [код](https://github.com/shizhediao/R-Tuning)

**18. Medical hallucination fine-tuning (дообучение против медицинских галлюцинаций)**
*Дообучение на парах «галлюцинация → исправленный ответ» для клинических задач.*
- Данные MedHallu: [статья](https://arxiv.org/abs/2502.14302) · [сайт](https://medhallu.github.io)
- Данные Med-HALT: [статья](https://arxiv.org/abs/2307.15343) · [код](https://github.com/medhalt/medhalt)

**19. Backdoor / poisoning defenses (защита от закладок в обучающих данных)**
*Обнаружение и нейтрализация скрытых триггеров, внедрённых через отравленные обучающие данные (модель ведёт себя нормально, но «ломается» от особого сигнала).*
- Fine-pruning: [статья](https://arxiv.org/abs/1805.12185)
- Обзор backdoor-атак и защит: [статья](https://arxiv.org/abs/2007.08745)

**20. Data curation (чистка обучающих данных)**
*Профилактическая фильтрация вредного контента и дедупликация данных с персональной информацией до обучения, чтобы модель их не запомнила.*
- Дедупликация против запоминания: [статья](https://arxiv.org/abs/2107.06499)

---

### F. Удаление и редактирование знаний
*Точечная работа с тем, что модель уже содержит в весах.*

**21. Machine unlearning (целенаправленное забывание)**
*Удаление из весов конкретных знаний — опасных или приватных — без переобучения с нуля. Оговорка: «забытое» иногда удаётся восстановить, гарантии пока неполные.*
- WMDP + RMU (удаление опасных биознаний): [статья](https://arxiv.org/abs/2403.03218) · [код](https://github.com/centerforaisafety/wmdp) · [сайт](https://wmdp.ai)
- MedForget / CHIP (под HIPAA/GDPR, без переобучения): [статья](https://arxiv.org/abs/2512.09867) · [код](https://github.com/fengli-wu/MedForget)
- MLLMU-Med (мультимодальное): [статья](https://arxiv.org/abs/2508.04192)
- DuoLearn (под «право на забвение»): [статья](https://arxiv.org/abs/2511.19498)

**22. Knowledge editing (редактирование знаний)**
*Точечная правка одного конкретного факта в весах (например, устаревшей дозировки) без затрагивания остального.*
- MedEditBench / SGR-Edit: [статья](https://arxiv.org/abs/2506.03490) · [публикация](https://aclanthology.org/2026.eacl-long.219/)
- MedEdit: [статья](https://arxiv.org/abs/2309.16035)

---

### G. Защита данных пациентов (privacy)

**23. PHI de-identification (деидентификация)**
*Удаление из текста персональных идентификаторов пациента: имён, дат, адресов.*
- LPPA: [статья](https://arxiv.org/abs/2504.18569)

**23б. Privacy-preserving training (приватное обучение)**
*Методы обучения (DP-SGD, федеративное обучение), не дающие модели запоминать и затем воспроизводить персональные данные. В отличие от деидентификации, защищают сам процесс обучения, а не только данные.*
- DP-SGD (обучение с гарантией приватности): [статья](https://arxiv.org/abs/1607.00133)
- Детекция PII в NeMo Guardrails: [док](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview)

---

### H. Сквозные направления (cross-cutting)

**24. Multimodal safety (безопасность мультимодальных моделей)**
*Защита Med-VLM — моделей, работающих с медицинскими изображениями (рентген, патология), где атака может прийти через картинку, а не только через текст.*
- 3MAD (набор мультимодальных атак): [статья](https://arxiv.org/abs/2405.20775) · [код](https://github.com/dirtycomputer/O2M_attack)
- MLLMU-Med: [статья](https://arxiv.org/abs/2508.04192)

**25. Tool / agent safety (безопасность агентов)**
*Для моделей, выполняющих действия (запись в карту, заказ анализа): ограничение tool-вызовов, песочница, обязательное участие клинициста.*
- MedSentry: [статья](https://arxiv.org/abs/2505.20824)
- Защита действий в NeMo Guardrails: [док](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview)

---

### Бенчмарки для оценки защит
*Не защиты, а наборы вопросов и атак, на которых измеряют эффективность защиты.*

| Бенчмарк | Что измеряет | Ссылки |
|---|---|---|
| MedSafetyBench | Отказ модели от вредных медицинских запросов | [статья](https://arxiv.org/abs/2403.03744) · [код](https://github.com/AI4LIFE-GROUP/med-safety-bench) |
| CARES | Устойчивость к джейлбрейкам + штраф за избыточные отказы | [статья](https://arxiv.org/abs/2505.11413) |
| MedHallu | Способность распознавать галлюцинации | [статья](https://arxiv.org/abs/2502.14302) · [сайт](https://medhallu.github.io) |
| Med-HALT | Галлюцинации в медицинском QA | [статья](https://arxiv.org/abs/2307.15343) · [код](https://github.com/medhalt/medhalt) |
| WMDP-bio | Остаточные опасные биомедицинские знания | [сайт](https://wmdp.ai) · [код](https://github.com/centerforaisafety/wmdp) |
| Red-teaming medical AI | Таксономия из 8 типов атак (тест на Claude Sonnet 4.5) | [статья](https://www.medrxiv.org/content/10.64898/2026.02.26.26347212v1) |

---

## ЧАСТЬ 2. Что реально протестировать в эксперименте

### Уровень 1 — plug-and-play, совместимо с закрытым API
*Оборачивают любую модель снаружи, кода нужно минимум. С этого стоит начать.*

| Защита | Что требуется | Ссылки |
|---|---|---|
| Guard-модели (Llama Guard 3, ShieldGemma, Aegis) | Обращение к guard-модели | [LG3](https://arxiv.org/abs/2411.10414) · [SG](https://arxiv.org/abs/2407.21772) |
| Adversarial-input transforms (SmoothLLM, perplexity-фильтр) | Обёртка над входом | [статья](https://arxiv.org/abs/2310.03684) · [код](https://github.com/arobey1/smooth-llm) |
| Prompt-injection defense (Spotlighting) | Шаблон промпта | [статья](https://arxiv.org/abs/2403.14720) |
| RAG-защиты + порог уверенности извлечения | Подключение базы документов | [статья](https://www.mdpi.com/2079-9292/14/21/4227) |
| Sampling-based detection (SelfCheckGPT) | Несколько ответов → сверка | [статья](https://arxiv.org/abs/2303.08896) · [код](https://github.com/potsawee/selfcheckgpt) |
| Self-critique / multi-agent | Несколько обращений к модели | [MedSentry](https://arxiv.org/abs/2505.20824) |
| NeMo Guardrails | Установка пакета | [код](https://github.com/NVIDIA-NeMo/Guardrails) |

### Уровень 2 — нужны открытые веса, без обучения

| Защита | Что требуется | Ссылки |
|---|---|---|
| Activation steering | Доступ к активациям, ~40 примеров | [статья](https://arxiv.org/abs/2406.00034) |
| Safe decoding (SafeDecoding) | Доступ к логитам (вероятностям токенов) | [статья](https://arxiv.org/abs/2402.08983) · [код](https://github.com/uw-nsl/SafeDecoding) |
| Model merging (mergekit) | Два чекпойнта | [код](https://github.com/arcee-ai/mergekit) |
| Unlearning без переобучения (MedForget / CHIP) | Открытая модель | [статья](https://arxiv.org/abs/2512.09867) · [код](https://github.com/fengli-wu/MedForget) |
| Knowledge editing (MedEditBench) | Открытая модель | [статья](https://arxiv.org/abs/2506.03490) |

### Уровень 3 — требуют обучения модели

| Защита | Что требуется | Ссылки |
|---|---|---|
| Circuit Breakers / RMU | Обучение | [CB](https://arxiv.org/abs/2406.04313) · [код](https://github.com/GraySwanAI/circuit-breakers) · [RMU](https://github.com/centerforaisafety/wmdp) |
| Continual Fine-Tuning (CFT) | Обучение | [статья](https://arxiv.org/abs/2501.18632) |
| Safety alignment (DPO/KTO) | Данные предпочтений | [статья](https://arxiv.org/abs/2305.18290) |
| Safety LoRA | Обучение адаптера | [статья](https://arxiv.org/abs/2405.16833) |
| Abstention training (R-Tuning) | Обучение | [статья](https://arxiv.org/abs/2311.09677) · [код](https://github.com/shizhediao/R-Tuning) |

---

## Практический фильтр

**Закрытая модель (только API)** → доступен Уровень 1 + частично safe decoding (через logit_bias).
**Открытые веса** → разблокируются Уровни 2 и 3.

**Рекомендуемый минимальный эксперимент:** baseline без защит → +guard-модель → +adversarial-input transform → +RAG с порогом уверенности → +SelfCheckGPT. На каждом шаге измерять на **MedSafetyBench + CARES + MedHallu** сразу по трём осям: безопасность, польза (medical utility) и доля избыточных отказов.

Если после простых защит доля успешных атак на CARES остаётся выше 10% — переходить к representation-level (Circuit Breakers) или к unlearning (CHIP, работает без переобучения).
