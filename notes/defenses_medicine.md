Если тебе нужны именно **domain-specific defenses для медицинских LLM**, а не общие AI safety статьи, я бы начал с этих работ.

## 1. The Need for Guardrails with Large Language Models in Medical Safety-Critical Settings (2024–2025)

The Need for Guardrails with Large Language Models in Medical Safety-Critical Settings

Это одна из наиболее близких к твоему запросу работ.

Авторы предлагают специализированные медицинские guardrails для фармаконадзора:

* обнаружение некорректных названий лекарств;
* обнаружение несуществующих adverse events;
* контроль качества входных документов;
* явное представление неопределённости модели.

Особенно интересно, что guardrails встроены в медицинический workflow, а не являются общими фильтрами. ([arXiv][1])

### Почему стоит читать

Для раздела **"Domain-specific Guardrails in Healthcare"** это практически готовый кейс.

---

## 2. Retrieval-Augmented Generation for 10 Large Language Models in Healthcare (Nature Digital Medicine, 2025)

Retrieval augmented generation for 10 large language models in healthcare

Исследование оценивает безопасность медицинских RAG-систем.

Основная идея защиты:

* использовать только клинические рекомендации;
* ограничивать retrieval доверенными источниками;
* уменьшать галлюцинации через retrieval grounding.

Это пример **retrieval constraints как медициной-специфичной защиты**. ([Nature][2])

### Полезно для главы

> Domain-Specific Retrieval Constraints in Clinical LLMs

---

## 3. Medical Large Language Models are Vulnerable to Data-Poisoning Attacks (Nature Medicine, 2025)

Medical large language models are vulnerable to data-poisoning attacks

Очень важная статья.

Показывает, что медицинские модели можно испортить через:

* поддельные публикации;
* ложные медицинские данные;
* научные тексты с вредными фактами.

Авторы обсуждают защитные механизмы:

* instruction tuning;
* RAG;
* безопасное курирование данных;
* постобучение.

Это уже область **training-time defenses**. ([Nature][3])

---

## 4. Harm Reduction Strategies for Thoughtful Use of Large Language Models in Healthcare (JMIR, 2025)

Harm Reduction Strategies for Thoughtful Use of Large Language Models in Healthcare

Предлагает медицино-специфическую модель защиты:

* human oversight;
* ограничение автономности модели;
* escalation-to-clinician;
* контроль риска на уровне сценариев использования.

Фактически это слой над классическими guardrails. ([JMIR][4])

### Особенно полезно

Если пишешь про:

> socio-technical defenses

а не только технические механизмы.

---

## 5. A Framework to Assess Clinical Safety and Hallucination Risk (Nature Digital Medicine, 2025)

A framework to assess clinical safety and hallucination risk

Предлагает систему оценки:

* clinical safety;
* hallucination risk;
* factual consistency.

Не совсем защита, но фундамент для построения медициных guardrails. ([Nature][5])

---

## 6. Retrieval-Augmented Guardrails for AI-Drafted Patient Messaging (RAEC)

Retrieval-Augmented Error Checking

Очень современное направление.

Идея:

1. LLM генерирует ответ пациенту.
2. Отдельный модуль извлекает похожие проверенные случаи.
3. Выполняется автоматическая проверка клинических ошибок.
4. Только потом ответ отправляется врачу или пациенту.

То есть guardrail строится поверх retrieval. ([arXiv][6])

---

## 7. CareGuardAI (2026)

CareGuardAI

Одна из самых интересных новых работ.

Архитектура:

```text
User
  ↓
Risk Assessment
  ↓
Safety-Constrained Generation
  ↓
Hallucination Assessment
  ↓
Clinical Risk Assessment
  ↓
Release / Block
```

Использует:

* контекстно-зависимые guardrails;
* оценку медицинского риска;
* многоагентную проверку ответа.

Фактически это медицинский аналог современных enterprise safety pipelines. ([arXiv][7])

---

# Если делать обзор по типам защит

Можно структурировать литературу так:

| Defense Category                  | Медицинские статьи                                                   |
| --------------------------------- | -------------------------------------------------------------------- |
| Policy prompts                    | Harm Reduction Strategies                                            |
| Guardrails                        | Need for Guardrails in Medical Safety-Critical Settings, CareGuardAI |
| Retrieval constraints             | RAG for Healthcare, MedGraphRAG                                      |
| Hallucination detection           | Clinical Safety and Hallucination Framework                          |
| Human-in-the-loop                 | Harm Reduction Strategies                                            |
| Adversarial testing / Red Teaming | Red-Teaming Medical AI                                               |
| Training-time defenses            | Data Poisoning in Medical LLMs                                       |
| Retrieval-based guardrails        | RAEC                                                                 |

Такой набор статей уже позволяет написать полноценный раздел **"Domain-Specific Defense Strategies for Medical LLMs"** на уровне обзорной статьи или диссертации. ([medrxiv.org][8])

[1]: https://arxiv.org/abs/2407.18322?utm_source=chatgpt.com "The Need for Guardrails with Large Language Models in Medical Safety-Critical Settings: An Artificial Intelligence Application in the Pharmacovigilance Ecosystem"
[2]: https://www.nature.com/articles/s41746-025-01519-z?utm_source=chatgpt.com "Retrieval augmented generation for 10 large language ..."
[3]: https://www.nature.com/articles/s41591-024-03445-1?utm_source=chatgpt.com "Medical large language models are vulnerable to data- ..."
[4]: https://www.jmir.org/2025/1/e75849?utm_source=chatgpt.com "Harm Reduction Strategies for Thoughtful Use of Large ..."
[5]: https://www.nature.com/articles/s41746-025-01670-7?utm_source=chatgpt.com "A framework to assess clinical safety and hallucination ..."
[6]: https://arxiv.org/pdf/2509.22565?utm_source=chatgpt.com "Retrieval‑Augmented Guardrails for AI‑Drafted Patient‑ ..."
[7]: https://arxiv.org/abs/2604.26959?utm_source=chatgpt.com "CareGuardAI: Context-Aware Multi-Agent Guardrails for Clinical Safety & Hallucination Mitigation in Patient-Facing LLMs"
[8]: https://www.medrxiv.org/content/10.64898/2026.02.26.26347212v1?utm_source=chatgpt.com "Red-Teaming Medical AI: Systematic Adversarial ..."
