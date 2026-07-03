# Deployment: GPU inference on Tesla T4 16 GB

Цель профиля `t4_hf` в [configs/config.yaml](configs/config.yaml): запускать основной LLM-инференс и режим с защитой `qwen3_guardrail` на одной GPU Tesla T4 16 GB без отдельного Ollama/OpenRouter сервера.

## Ресурсы

Минимальный профиль подходит для текущего конфига:

- GPU: 1 x Tesla T4 16 GB
- CPU: 4 vCPU
- RAM: 16 GB
- Disk: 60 GB

Этого достаточно для `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it` и `allenai/OLMo-2-0425-1B-Instruct` как основных моделей, а также для `Qwen/Qwen3Guard-Gen-0.6B` как guard-модели в `dtype: float16`. GPU A2 16 GB тоже подходит для этого профиля, но обычно будет медленнее T4. При OOM уменьшайте `runtime.t4_hf.generate.max_connections` и `runtime.t4_hf.model_args.batch_size` до `1-2`.

Рекомендуемый профиль для ускоренного CARES-инференса на A10:

- GPU: 1 x Tesla A10 24 GB
- CPU: 8 vCPU
- RAM: 32 GB
- Disk: 120 GB SSD

Этот профиль подходит для более агрессивных настроек `batch_size`/`max_connections` на текущих 1B-4B моделях: например, `8/8` для no-think и `4/4` для think-режима, если `nvidia-smi` показывает достаточный запас VRAM.

Запрашивайте больше ресурсов, если переходите на 7B+ модели или хотите держать несколько vLLM-серверов одновременно:

- GPU VRAM: 24-48 GB
- RAM: 32 GB
- Disk: 100 GB+

## Установка окружения

Команды ниже рассчитаны на Linux GPU VM. Для Tesla T4 сначала установите NVIDIA driver из раздела ниже, затем создавайте Python-окружение.

```bash
git clone <repo-url>
cd domain-specific-defenses

sudo apt update
sudo apt install python3-venv python3-full
sudo apt install python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# CUDA wheel для PyTorch. Если на образе другая CUDA, используйте индекс,
# рекомендованный PyTorch для этого драйвера.
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
python3 -m pip install -r requirements.txt
```

Проверьте версии ML-зависимостей:

```bash
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda); print(transformers.__version__)"
```

Если запуск падает с ошибкой `module 'torch' has no attribute 'float8_e8m0fnu'`, значит установлена слишком новая версия `transformers` для текущего PyTorch. Переустановите совместимую версию:

```bash
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch
python -m pip install --upgrade 'transformers>=4.51,<4.57' accelerate
```

Если `pip` долго висит на dependency resolving или сеть до PyPI нестабильна, сделайте быстрый downgrade конкретного пакета без resolver. Это обычно достаточно, если зависимости уже были установлены вместе с `transformers 5.x`:

```bash
python -m pip uninstall -y transformers
python -m pip install --no-cache-dir --no-deps --progress-bar off --timeout 120 --retries 10 transformers==4.56.2
python -m pip check
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda); print(transformers.__version__)"
```

Если `pypi.org` таймаутится даже с большим timeout, повторите установку через зеркало:

```bash
python -m pip install --no-cache-dir --no-deps --progress-bar off --timeout 300 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==4.56.2
```

Та же команда в одну строку, если shell ломает многострочный ввод:

```bash
python -m pip install --no-cache-dir --no-deps --progress-bar off --timeout 300 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.56.2
```

После установки с `--no-deps` проверьте импорт. Если ошибка говорит, что `huggingface-hub>=0.34.0,<1.0 is required`, установите совместимые зависимости отдельно:

```bash
python -m pip install --no-cache-dir --progress-bar off --timeout 300 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface-hub>=0.34.0,<1.0" "tokenizers>=0.22.0,<=0.23.0" "safetensors>=0.4.3"
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda); print(transformers.__version__)"
```

Альтернативные индексы:

```bash
python -m pip install --no-cache-dir --no-deps --progress-bar off --timeout 300 --retries 10 \
  -i https://mirrors.aliyun.com/pypi/simple \
  transformers==4.56.2

python -m pip install --no-cache-dir --no-deps --progress-bar off --timeout 300 --retries 10 \
  -i https://pypi.mirrors.ustc.edu.cn/simple \
  transformers==4.56.2
```

## Установка Ollama

Ollama больше не нужен для основных CARES matrix-конфигов: текущие
`cares_qwen3_1_7b.yaml`, `cares_gemma_3_1b_it.yaml` и
`cares_olmo_2_0425_1b_instruct.yaml` идут через `provider: t4_hf`. Ollama
остаётся опциональным только для локальных smoke-тестов и ручных запусков вне
основной матрицы.

Установить Ollama на Linux VM:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Проверить установку:

```bash
which ollama
ollama --version
```

Поднять сервис в отдельном `tmux` pane/session:

```bash
ollama serve
```

В другом терминале при необходимости скачайте локальную модель для smoke-теста:

```bash
ollama pull qwen3:1.7b
ollama pull gemma3:1b
```

Проверить, что модель отвечает:

```bash
ollama run qwen3:1.7b "hello"
ollama run gemma3:1b "hello"
ollama ps
ollama list
```

Если `ollama` не найден после установки, откройте новый shell или проверьте,
что `/usr/local/bin` есть в `PATH`.

## Установка NVIDIA driver

Если `nvidia-smi` не найден, но `ubuntu-drivers devices` показывает NVIDIA GPU, для Tesla T4 ставьте server-драйвер и перезагрузите VM.

```bash
sudo apt update
ubuntu-drivers devices
sudo apt install -y nvidia-driver-595-server 
sudo reboot
```

Не используйте `nvidia-driver-595-open` как основной вариант для T4: на некоторых cloud images он устанавливается, но kernel module не поднимается.

Для GPU A2/A16 на Ubuntu 24.04 `ubuntu-drivers devices` может показать, например:

```text
model    : GA107GL [A2 / A16]
driver   : nvidia-driver-595-open - distro non-free recommended
driver   : nvidia-driver-580 - distro non-free
driver   : nvidia-driver-535 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

Для A2/A16 можно начать с рекомендованного пакета:

```bash
sudo apt install -y nvidia-driver-595-open
sudo reboot
```

Если рекомендованный `*-open` драйвер не загружается на конкретном cloud image, используйте обычный server-драйвер той же ветки:

```bash
sudo apt install -y nvidia-driver-595-server
sudo reboot
```

Сообщения вида `udevadm hwdb is deprecated` и `ERROR:root:aplay command not found` при `ubuntu-drivers devices` не критичны для CUDA-инференса, если NVIDIA GPU отображается в списке устройств.

Проверка драйвера после reboot:

```bash
nvidia-smi
```

Проверка CUDA в PyTorch выполняется только после установки окружения и PyTorch:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Если запускать проверку вне активированного `.venv`, используйте `python3`, но `torch` будет доступен только после установки PyTorch в это окружение. Ошибка `python: command not found` означает, что системного алиаса `python` нет; после `source .venv/bin/activate` команда `python` должна указывать на `.venv/bin/python`.

Если после установки драйвера `nvidia-smi` пишет `couldn't communicate with the NVIDIA driver`, значит пакет установлен, но kernel module не загрузился. Сначала соберите диагностику:

```bash
uname -r
dkms status
lsmod | grep -E 'nvidia|nouveau' || true
sudo modprobe nvidia
journalctl -k -b | grep -iE 'nvidia|nouveau|nvrm|secure|dkms' | tail -n 80
```

На cloud image чаще всего помогает установка headers/build tools и переход с `*-open` на обычный server-драйвер:

```bash
sudo apt update
sudo apt install -y "linux-headers-$(uname -r)" build-essential
sudo apt purge -y 'nvidia-*' 'libnvidia-*'
sudo apt autoremove -y
sudo apt install -y nvidia-driver-595-server
sudo reboot
```

Если `nvidia-driver-595-server` тоже не загружается, повторите с более консервативной веткой:

```bash
sudo apt purge -y 'nvidia-*' 'libnvidia-*'
sudo apt autoremove -y
sudo apt install -y nvidia-driver-535-server
sudo reboot
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

Важно: при многострочном запуске `\` должен стоять в конце каждой продолжаемой строки. Если первая строка не заканчивается на `\`, shell запустит `inspect eval` без аргументов `-T`, а следующая строка завершится ошибкой `-T: command not found`.

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
4. Используйте `main_model_key=olmo2-0425-1b-instruct` или `main_model_key=qwen3-1.7b` для ручного override.

После каждого изменения смотрите фактическую память:

```bash
watch -n 1 nvidia-smi
```
