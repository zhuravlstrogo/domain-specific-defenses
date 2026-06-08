# Deployment: GPU inference on Tesla T4 16 GB

Цель профиля `t4_hf` в [config/config.yaml](config/config.yaml): запускать основной LLM-инференс и режим с защитой `qwen3_guardrail` на одной GPU Tesla T4 16 GB без отдельного Ollama/OpenRouter сервера.

## Ресурсы

Минимальный профиль подходит для текущего конфига:

- GPU: 1 x Tesla T4 16 GB
- CPU: 4 vCPU
- RAM: 16 GB
- Disk: 60 GB

Этого достаточно для `Qwen/Qwen3-0.6B` как основной модели и `Qwen/Qwen3Guard-Gen-0.6B` как guard-модели в `torch_dtype: float16`. GPU A2 16 GB тоже подходит для этого профиля, но обычно будет медленнее T4. Для `qwen3-1.7b` тоже должно хватить, но при OOM уменьшайте `runtime.t4_hf.generate.max_connections` и `runtime.t4_hf.model_args.batch_size` до `1-2`.

Запрашивайте больше ресурсов, если переходите на 7B+ модели или хотите держать несколько vLLM-серверов одновременно:

- GPU VRAM: 24-48 GB
- RAM: 32 GB
- Disk: 100 GB+

## Установка окружения

Команды ниже рассчитаны на Linux GPU VM с установленным NVIDIA driver.

```bash
git clone <repo-url>
cd domain-specific-defenses

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# CUDA wheel для PyTorch. Если на образе другая CUDA, используйте индекс,
# рекомендованный PyTorch для этого драйвера.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch

python -m pip install -r requirements.txt
```

Проверка CUDA:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

При первом запуске Hugging Face скачает веса моделей. Если системный диск маленький или нужен persistent cache:

```bash
export HF_HOME=/path/to/persistent/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

## Конфиг

Основной GPU-профиль находится в `runtime.t4_hf`:

- `main_model`: основная LLM для baseline и prompt-based защит.
- `guard_model`: Qwen3Guard для `policy=qwen3_guardrail`.
- `grade_model`: модель-грейдер для `medical_safety`.
- `model_args.device: cuda:0`: принудительный запуск на T4.
- `batch_size` и `max_connections`: главные ручки скорости/памяти.

Для более качественной, но более тяжёлой основной модели:

```bash
-T main_model_key=qwen3-1.7b
```

## Запуск MCQ robustness

Baseline без внешней защиты:

```bash
source .venv/bin/activate

inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  -T runtime=t4_hf \
  -T policy=baseline \
  -T thinking=no_think \
  --limit 20 \
  --max-samples 4 \
  --log-dir logs
```

Prompt-policy защита:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  -T runtime=t4_hf \
  -T policy=mcq_prompt_policy \
  -T thinking=no_think \
  --limit 20 \
  --max-samples 4 \
  --log-dir logs
```

Qwen3Guard sandwich:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_robustness \
  -T runtime=t4_hf \
  -T policy=qwen3_guardrail \
  -T thinking=no_think \
  --limit 20 \
  --max-samples 2 \
  --log-dir logs
```

Для multi-turn варианта замените task:

```bash
inspect eval experiments/medical_mcq_robustness_eval.py@medical_mcq_multiturn \
  -T runtime=t4_hf \
  -T policy=qwen3_guardrail \
  -T thinking=no_think \
  --limit 20 \
  --max-samples 2 \
  --log-dir logs
```

## Запуск medical safety

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
  -T runtime=t4_hf \
  -T policy=baseline \
  --limit 20 \
  --max-samples 4 \
  --log-dir logs
```

С guardrail:

```bash
inspect eval experiments/medical_safety_eval.py@medical_safety \
  -T runtime=t4_hf \
  -T policy=qwen3_guardrail \
  --limit 20 \
  --max-samples 2 \
  --log-dir logs
```

## Настройка скорости

Если GPU недогружена:

1. Увеличьте `runtime.t4_hf.model_args.batch_size` до `8`.
2. Увеличьте `runtime.t4_hf.generate.max_connections` до `8`.
3. Увеличьте `--max-samples` до того же значения.

Если появляется CUDA OOM:

1. Уменьшите `--max-samples`.
2. Уменьшите `batch_size` и `max_connections` до `1-2`.
3. Оставьте `thinking=no_think`.
4. Используйте `main_model_key=qwen3-0.6b`.

После каждого изменения смотрите фактическую память:

```bash
watch -n 1 nvidia-smi
```
