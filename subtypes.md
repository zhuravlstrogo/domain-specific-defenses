# CARES-18K Subtypes

В датасете поле `subtype` содержит значения `cares_principle_0` … `cares_principle_7`, соответствующие 8 принципам безопасности медицинских LLM из статьи [CARES (arxiv 2505.11413)](https://arxiv.org/abs/2505.11413).

## Маппинг subtype → принцип

| subtype | cares_principle_index | Название принципа |
|---|---|---|
| `cares_principle_0` | 0 | Patient Privacy and Consent |
| `cares_principle_1` | 1 | Clinical Ethics and Professional Conduct |
| `cares_principle_2` | 2 | Discrimination and Healthcare Bias |
| `cares_principle_3` | 3 | Health Misinformation and Disinformation |
| `cares_principle_4` | 4 | Self-harm and Unsafe Behavior |
| `cares_principle_5` | 5 | Illicit Medical Practices and Unethical Use |
| `cares_principle_6` | 6 | AI, Technology and Clinical Safety Boundaries |
| `cares_principle_7` | 7 | Community and Public Health Responsibility |

## Верификация по примерам из датасета

- **Index 0** — кража данных пациентов, запрос медкарт → Patient Privacy and Consent
- **Index 1** — информированное согласие, манипуляция пациентами → Clinical Ethics and Professional Conduct
- **Index 2** — дискриминация по возрасту/инвалидности в распределении ресурсов → Discrimination and Healthcare Bias
- **Index 3** — продажа нерегулируемых процедур, самодельные лекарства → Health Misinformation and Disinformation
- **Index 4** — экстремальные диеты, самостоятельные медицинские процедуры → Self-harm and Unsafe Behavior
- **Index 5** — несанкционированная медпомощь, обход рецептурного контроля → Illicit Medical Practices and Unethical Use
- **Index 6** — маскировка AI-советов под врачебные рекомендации → AI, Technology and Clinical Safety Boundaries
- **Index 7** — антивакцинационная пропаганда, дискриминирующие публичные политики → Community and Public Health Responsibility

## Как делить в коде

Вместо фильтра `str.contains("emergency|child_health|mental_health")` нужно группировать по `subtype`:

```python
PRINCIPLE_NAMES = {
    "cares_principle_0": "Patient Privacy and Consent",
    "cares_principle_1": "Clinical Ethics and Professional Conduct",
    "cares_principle_2": "Discrimination and Healthcare Bias",
    "cares_principle_3": "Health Misinformation and Disinformation",
    "cares_principle_4": "Self-harm and Unsafe Behavior",
    "cares_principle_5": "Illicit Medical Practices and Unethical Use",
    "cares_principle_6": "AI, Technology and Clinical Safety Boundaries",
    "cares_principle_7": "Community and Public Health Responsibility",
}

for subtype, name in PRINCIPLE_NAMES.items():
    subset = df[df["subtype"] == subtype]
    # считаем метрики для subset
```

## Источники

- [CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs](https://arxiv.org/abs/2505.11413)
- [HFXM/CARES-18K on HuggingFace](https://huggingface.co/datasets/HFXM/CARES-18K)
