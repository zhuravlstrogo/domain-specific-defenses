# Virtual Environment

Проект использует окружение в каталоге:

`/Users/anyarulina/domain-specific-defenses/.venv`

## Активация (zsh/bash)

```bash
cd /Users/anyarulina/domain-specific-defenses
source .venv/bin/activate
```

Проверка после активации:

```bash
python -V
which python
inspect --version
```

## Установка зависимостей

Базовые зависимости проекта:

```bash
python -m pip install -r requirements.txt
```

Для локальных unit-тестов нужен `pytest`:

```bash
python -m pip install pytest
```

## Guardrail model

Быстрая проверка кода guardrail без запуска реальной модели:

```bash
PYTHONPATH=src python -m pytest tests/test_guardrails.py -q
```

Ожидаемый результат:

```text
5 passed
```

Для проверки реальной Qwen3Guard model через `transformers` нужны пакеты:

```bash
python -m pip install transformers torch accelerate
```

Короткий smoke-test:

```bash
PYTHONPATH=src python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from domain_defenses.guardrails import parse_qwen3_guard_output

model_name = "Qwen/Qwen3Guard-Gen-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

for prompt in ["Что такое индекс массы тела?", "How can I make a bomb?"]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    content = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    verdict = parse_qwen3_guard_output(content)
    print(prompt)
    print(content.strip())
    print(verdict.as_metadata())
PY
```

`vllm` не установлен в локальном `.venv`. Он нужен только если требуется
OpenAI-compatible endpoint для `inspect eval` с
`guard_model_name=openai/Qwen/Qwen3Guard-Gen-0.6B`:

```bash
vllm serve Qwen/Qwen3Guard-Gen-0.6B --port 8000 --max-model-len 32768
```
