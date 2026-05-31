#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: Basics and first contact with Inspect AI
# 
# Welcome to the first tutorial in our AI Safety Evaluations course.
# 
# **What you'll learn:**
# 
# - Connect Inspect AI to a language model (locally via Ollama, or via cloud API)
# - Run your first evaluation
# - Understand how tasks are structured: dataset → solver → scorer
# - View and analyze results with `inspect view`
# - Create single choice and multiple choice benchmarks
# - Analyzing position bias in multiple-choice tasks
# 
# **By the end:** You'll have a working evaluation pipeline and understand how to build your own benchmarks.

# ---
# ## Prerequisites: Model setup
# 
# > **💡 Inspect AI only needs a model name** — the model itself can come from anywhere.
# 
# **In this tutorial, we'll use Ollama and Perplexity, SambaNova as examples**, but you can substitute any provider: OpenAI, Anthropic, Google, local inference servers, or any OpenAI-compatible endpoint.
# 
# **Cost note:** Cloud APIs have small free tiers and then charge per token. Local models (Ollama) are completely free. For this course, a local model is sufficient for all assignments.
# 
# See [Inspect AI models docs](https://inspect.ai-safety-institute.org.uk/models.html) for the full list.

# ---
# ## Part 1: Local environment setup (Ollama)
# 
# Running evaluations locally gives you complete control and privacy.

# ### 1.1. Installing Ollama
# 
# **Before running this notebook, install Ollama:**
# 
# **macOS:** `brew install ollama` or download from https://ollama.ai/download
# 
# **Linux:**
# ```bash
# curl -fsSL https://ollama.ai/install.sh | sh
# ```
# 
# **Windows:** download from https://ollama.ai/download
# 
# After installation, start the server and download a model:
# ```bash
# ollama serve
# ollama pull llama2        # or use deepseek-r1:1.5b for a smaller option
# ```

# ### 1.2. Check Ollama connection

# In[1]:


import requests

def check_ollama():
    """Check Ollama connection and show installed models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama returned an error")
            return False

        models = response.json().get('models', [])
        total_size = sum(m['size'] for m in models)

        print("✅ Ollama is running!")
        print(f"\n📊 Installed models: {len(models)} ({total_size / 1e9:.1f} GB total)")

        for m in models:
            print(f"   - {m['name']}: {m['size'] / 1e9:.2f} GB")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Start it with: ollama serve")

check_ollama()


# ---
# ## Part 2: Basic Inspect setup

# ### 2.1. Install Inspect AI

# In[ ]:


get_ipython().system('pip install -q inspect-ai openai')
print("✅ Installed: inspect-ai, openai")


# ## Assignment 1: 'Hello world' in eval
# 
# Let's create the simplest possible evaluation!
# 
# **To do:** add one more `Sample()` to the dataset.

# In[4]:


from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact, match, model_graded_fact, choice, pattern
from inspect_ai.solver import (
    generate, system_message, chain_of_thought, 
    prompt_template, multiple_choice
)


# In[5]:


@task
def hello_model():
    """Test your model setup with simple questions."""
    return Task(
        dataset=[
            Sample(
                input="Say 'Hello world!' and nothing else.",
                target="Hello world!"
            ),
            Sample(
                input="2+2=",
                target="4"
            ),
            Sample(
                input="What is the surname of Sheldon from The Big Bang Theory?",
                target="Cooper"
            ),
            Sample(
                input="What is the opposite of 'hot'?",
                target="cold"
            ),
            Sample(
                input="Write the number five as a digit.",
                target="5"
            ),
            Sample(
                input="Finish the sequence: 1, 2, 3,",
                target="4"
            )
        ],
        solver=[generate()],
        scorer= match(
            location="end",        # where to look for the answer: "begin", "end", "any", "exact"
            ignore_case=True,      # ignore case when comparing
            numeric=False          # treat as numeric comparison (normalizes numbers, different punctuation rules)
        )
    )


# **Run the evaluation:**
# 
# This will take a minute or two depending on your hardware.

# In[6]:


eval(
    hello_model,
    model="ollama/llama2",
    # limit=1  # Uncomment to test with just 1 sample
)


# ---
# ## Part 3: API setup (optional)
# 
# If you don't have a GPU or want to test cloud models, you can use API providers.
# 
# ### 3.1. Perplexity setup
# 
# 1. Get an API key at https://www.perplexity.ai/settings/api
# 2. Set it in the cell below

# In[ ]:


import os
YOUR_API_KEY = ''
os.environ['PERPLEXITY_API_KEY'] = YOUR_API_KEY  # Replace with your key


# In[ ]:


eval(
    hello_model,
    model="perplexity/sonar",
    # limit=1  # Uncomment to test with just 1 sample
)


# ## Sambanova 
# 
# Get API KEY https://cloud.sambanova.ai/apis here 

# In[ ]:


# !pip install SambaNova


# In[ ]:


from sambanova import SambaNova

YOUR_API_KEY = ''
os.environ['SAMBANOVA_API_KEY'] = YOUR_API_KEY  # Replace with your key


# In[ ]:


eval(
    hello_model,
    model="sambanova/DeepSeek-V3.1",
    # limit=1  # Uncomment to test with just 1 sample
)


# ---
# ## Part 4: Viewing results with Inspect view
# 
# Every evaluation saves a log file. `inspect view` opens a web UI to explore them.
# 
# ### 4.1. Launch Inspect view
# 
# 1. In terminal, from the notebook's folder, run: `inspect view`
# 2. Open in browser: http://localhost:7575
# 
# This will:
# 1. Show all evaluation logs in an interactive interface
# 2. Allow you to drill down into individual samples
# 
# **Alternative options:**
# 
# ```bash
# # View logs from a specific directory
# inspect view --log-dir ./experiment-logs
# 
# # Use a different port
# inspect view --port 8080
# ```
# 
# **Troubleshooting:**
# 
# - If `inspect: command not found` → try `python -m inspect_ai view`
# - If the page won't load → check that you're in the correct folder (logs are saved relative to where you run evaluations)

# In[ ]:


# # Uncomment and run this cell if inspect view shows no logs

# import os
# from pathlib import Path
# 
# log_files = list(Path("./logs").glob("*.eval")) if Path("./logs").exists() else []
# 
# if not log_files:
#     print("❌ No log files found")
#     print("   Run at least one eval() in this notebook first")
#     print(f"   Current directory: {os.getcwd()}")
# else:
#     print(f"✅ Found {len(log_files)} log file(s):")
#     for f in sorted(log_files)[-5:]:  # show last 5
#         print(f"   {f.name}")


# ### Assignment 2: Explore your logs
# 
# In the Inspect view UI you can:
# 
# - **See overall accuracy** for each evaluation run
# - **Click on individual samples** to see the model's response
# - **Compare runs** with different models or parameters
# - **Filter by metadata** (e.g., show only "hard" problems)
# - **Export results** for further analysis
# 
# **Tip:** Keep `inspect view` running in a separate terminal while you work through this notebook. It auto-refreshes when new evaluations complete.

# ---
# ## Part 5: Understanding benchmark structure
# 
# ### 5.1. Task components overview
# 
# Every Inspect `Task` consists of:
# 
# ```
# Task {
#     dataset: [Sample, Sample, ...],    # data to evaluate on
#     solver: [Solver, Solver, ...],     # how to process
#     scorer: Scorer,                    # how to score
#     **parameters                       
# }
# ```
# 
# **Component flow:**
# ```
# Dataset (Samples) → Solver(s) → Model → Scorer → Results
# ```

# ### 5.2. Sample structure
# 
# A Sample contains input/target pairs with optional metadata:

# In[ ]:


# Example of a fully-featured Sample
sample_example = Sample(
    input="Question or prompt",
    target="Expected answer",
    id="unique_id",
    choices=["Option A", "Option B", "Option C"],  # For multiple choice
    metadata={
        "category": "math",
        "difficulty": "hard"
    }
)

print("Sample components:")
print(f"  - input: {sample_example.input}")
print(f"  - target: {sample_example.target}")
print(f"  - choices: {sample_example.choices}")
print(f"  - metadata: {sample_example.metadata}")


# ---
# ## Part 6: Understanding solvers
# 
# ### 6.1. What is a solver?
# 
# A **solver** is a function that transforms a **TaskState** (the prompt + conversation history) and optionally calls the model to generate a response.
# 
# **Think of solvers as middleware that:**
# 1. Modifies the prompt (prompt engineering)
# 2. Calls the model (generation)
# 3. Processes the response (extraction, critique, etc.)
# 
# ### 6.2. The solver pipeline
# 
# Solvers are chained together in a pipeline:
# 
# ```
# Input Sample
#     ↓
# [Solver 1: system_message]
#     ↓
# [Solver 2: prompt_template]
#     ↓
# [Solver 3: chain_of_thought]
#     ↓
# [Solver 4: generate]
#     ↓
# Model Output → Scorer → Final Result
# ```
# 
# Each solver receives the TaskState, modifies it, and passes it to the next solver.
# 
# ### 6.3. TaskState - the core data structure
# 
# Every solver operates on a **TaskState** containing:
# 
# ```
# TaskState {
#     messages: list[ChatMessage],  # Conversation history
#     output: ModelOutput,          # Final model output
#     user_prompt: str,             # Current user prompt
#     input_text: str,              # Original input
#     metadata: dict,               # Sample metadata
#     choices: list[str],           # For multiple choice
#     model: ModelName,             # Current model
#     sample_id: int | str,         # Sample identifier
# }
# ```

# ---
# ## Part 7: Built-in solvers
# 
# **system_message**
# ```python
# system_message(
#     message: str        # REQUIRED - the system prompt
# )
# ```
# 
# **prompt_template**
# ```python
# prompt_template(
#     template: str       # REQUIRED - use {prompt} as placeholder
# )
# ```
# 
# **chain_of_thought**
# ```python
# chain_of_thought(
#     template: str = None   # optional - custom CoT prompt (default: "Let's think step by step")
# )
# ```
# 
# **generate**
# ```python
# generate(
#     max_tokens: int = None,      # optional - limit response length
#     temperature: float = None,   # optional - 0.0 = deterministic, 1.0 = creative
#     top_p: float = None,         # optional - nucleus sampling
#     stop_seqs: list[str] = None  # optional - stop generation at these strings
# )
# ```
# 
# **multiple_choice**
# ```python
# multiple_choice(
#     cot: bool = False,              # optional - add chain-of-thought
#     multiple_correct: bool = False, # optional - allow multiple answers
#     shuffle: bool = False           # optional - randomize choice order
# )
# ```
# 
# **Typical pipeline:**
# ```
# system_message → prompt_template → chain_of_thought → generate
# ```
# 
# `multiple_choice()` replaces the entire chain - it handles prompting and generation internally.
# 
# **Viewing solver execution:** in `inspect view`, click any sample → messages tab shows each solver's contribution.

# ### 7.1 system_message()
# 
# **Purpose:** prepend a system role message to guide model behavior.
# 
# **When to use:**
# - establish the model's role or persona
# - set global guidelines or constraints
# - define the evaluation context
# 

# In[ ]:


@task
def example_system_message():
    """
    Demonstrates system_message() solver.
    The system prompt tells the model to be concise.
    """
    return Task(
        dataset=[
            Sample(input="What is 15 * 8?", target="120"),
            Sample(input="What is 99 + 1?", target="100"),
        ],
        solver=[
            system_message("You are a calculator. Reply with only the number, nothing else."),
            generate()
        ],
        scorer=match(numeric=True),
    )

# Run and check the Messages tab in inspect view
eval(example_system_message, model="ollama/llama2")


# ### 7.2 prompt_template()
# 
# **Purpose:** substitute variables into a template to reformat prompts.
# 
# **When to use:**
# - add specific output format requirements
# - include examples or demonstrations
# - structure prompts consistently
# - add reasoning steps or breakdowns
# 

# In[ ]:


STEP_BY_STEP_TEMPLATE = '''
Solve this problem step by step:

Problem: {prompt}

Structure:
1. Understand the problem
2. Plan your approach
3. Solve it
4. Final answer format: ANSWER: <value>
'''.strip()

@task
def example_prompt_template():
    """
    Demonstrates prompt_template() solver.
    The template adds structure to the prompt.
    """
    return Task(
        dataset=[
            Sample(input="What is 25 * 4?", target="100"),
            Sample(input="What is 144 / 12?", target="12"),
        ],
        solver=[
            system_message("You are a math tutor."),
            prompt_template(STEP_BY_STEP_TEMPLATE),
            generate()
        ],
        scorer=match(numeric=True),
    )

# Run and see how the template structures the prompt
eval(example_prompt_template, model="ollama/llama2")


# ### 7.3 chain_of_thought()
# 
# **Purpose:** ask the model to "think step by step" before answering.
# 
# **When to use:**
# - math and logic problems
# - multi-step reasoning tasks
# - when you want to see the model's thought process

# In[ ]:


@task
def example_chain_of_thought():
    """
    Demonstrates chain_of_thought() solver.
    Compare accuracy with and without CoT in inspect view.
    """
    return Task(
        dataset=[
            Sample(
                input="If Alice has 3 apples and Bob gives her 2 more, how many does she have?",
                target="5"
            ),
            Sample(
                input="A train travels 100 km in 2 hours. At this rate, how far in 5 hours?",
                target="250"
            ),
        ],
        solver=[
            system_message("Solve the problem. End with: ANSWER: <number>"),
            chain_of_thought(),
            generate()
        ],
        scorer=match(numeric=True),
    )

eval(example_chain_of_thought, model="ollama/llama2")


# ### 7.5. multiple_choice()
# 
# Special solver for A/B/C/D questions. Handles formatting and answer extraction automatically.
# 
# **When to use:**
# - multiple choice questions (use instead of generate)
# - must have letter target: "A", "B", "C", etc.
# 
# **Note:** When using `multiple_choice()`, use `choice()` as the scorer.

# In[ ]:


@task
def example_multiple_choice_with_cot():
    """
    Demonstrates multiple_choice(cot=True).
    Model reasons before selecting an answer.
    """
    return Task(
        dataset=[
            Sample(
                input="Light travels faster than sound. If you see lightning and hear thunder 3 seconds later, approximately how far away was the strike?",
                choices=["100 meters", "1 kilometer", "3 kilometers", "10 kilometers"],
                target="B"  # ~1 km (sound travels ~340 m/s)
            ),
        ],
        solver=multiple_choice(cot=True),
        scorer=choice(),
    )

eval(example_multiple_choice_with_cot, model="ollama/llama2")


# ### 7.6 Other solvers
# 
# **self_critique()** - Have model refine its own answer
# ```python
# solver=[generate(), self_critique()]
# ```
# 
# **use_tools()** - Enable tool/function calling
# ```python
# solver=[use_tools(calculator()), generate()]
# ```

# ---
# ## Part 8: Single choice tasks
# 
# Single choice tasks present the model with limited options to select from.
# 
# ### 8.1. Simple yes/no classification
# 
# The simplest single choice - binary classification:

# In[ ]:


@task
def yes_no_classification():
    return Task(
        dataset=[
            Sample(
                input="Is Python a programming language?",
                target="Yes"
            ),
            Sample(
                input="Is water dry?",
                target="No"
            ),
            Sample(
                input="Is the Earth round?",
                target="Yes"
            ),
        ],
        solver=[
            system_message("Answer 'Yes' or 'No'. Be concise."),
            generate()
        ],
        scorer=exact(),
    )

eval(
    yes_no_classification,
    model="ollama/llama2"
)


# ### 8.2. Multi-class classification
# 
# In multi-class classification, the model must choose from 3+ categories. This is common for:
# - sentiment analysis (positive / negative / neutral)
# - topic classification (sports / politics / tech / ...)
# - intent detection (question / command / statement)
# 
# ---
# 
# ### Task 2: Build a sentiment classifier
# 
# **Your goal:** Create a sentiment classification task with at least 4 samples.
# 
# **Note:**
# - `system_message` defines the classes and output format
# - `target` must exactly match one of your class labels

# In[ ]:


@task
def sentiment_classification():
    return Task(
        dataset = # YOUR CODE HERE
        solver = # YOUR CODE HERE
        scorer = # YOUR CODE HERE
    )


eval(
    # YOUR CODE HERE
)


# ### 8.3. Single choice with explanation
# 
# Collect both choice and reasoning:

# In[ ]:


@task
def choice_with_reasoning():
    PROMPT = '''
Classify as True or False:

Statement: {prompt}

Provide:
1. REASONING: [Your explanation]
2. ANSWER: [True or False]
    '''.strip()

    return Task(
        dataset=[
            Sample(
                input="The Earth is flat.",
                target="False"
            ),
            Sample(
                input="Water boils at 100°C at sea level.",
                target="True"
            ),
        ],
        solver=[
            chain_of_thought(),
            prompt_template(PROMPT),
            generate()
        ],
        scorer=pattern(r'ANSWER:\s*(True|False)'),
    )

eval(choice_with_reasoning, model='ollama/llama2')


# ---
# ## Part 9: Multiple choice tasks
# 
# ### 9.1. Understanding multiple choice in Inspect
# 
# Key rules:
# - `choices`: list of answer options (no letters — they're added automatically)
# - `target`: letter of correct answer ("A", "B", "C", or "D")
# - Use `multiple_choice()` solver + `choice()` scorer

# ### 9.2. Multiple choice with metadata
# 
# Metadata lets you filter and analyze results in `inspect view`.

# In[ ]:


@task
def mc_with_metadata():
    return Task(
        dataset=[
            Sample(
                input="Capital of Japan?",
                choices=["Seoul", "Tokyo", "Bangkok", "Beijing"],
                target="B",
                metadata={
                    "difficulty": "easy",
                    "category": "geography"
                }
            ),
            Sample(
                input="What is the Heisenberg Uncertainty Principle?",
                choices=[
                    "Cannot know both position and momentum precisely",
                    "Energy cannot be created or destroyed",
                    "All matter has wave-particle duality",
                    "Time always moves forward"
                ],
                target="A",
                metadata={
                    "difficulty": "hard",
                    "category": "physics"
                }
            ),
        ],
        solver=multiple_choice(),
        scorer=choice(),
    )

# Run and check results in inspect view - filter by metadata!
eval(mc_with_metadata, model="ollama/llama2")


# ### 9.6. Multiple correct Answers
# 
# When multiple answers are valid:

# In[ ]:


@task
def mc_multiple_correct():
    return Task(
        dataset=[
            Sample(
                input="Which are programming languages?",
                choices=["Python", "HTML", "JavaScript", "CSS"],
                target=["A", "C"]  # Python, JavaScript
            ),
            Sample(
                input="Which continents border the Atlantic Ocean?",
                choices=["Africa", "Asia", "Europe", "South America"],
                target=["A", "C", "D"]  # Africa, Europe, South America
            ),
        ],
        solver=[
            system_message("Select ALL correct answers. You may choose multiple options."),
            multiple_choice(multiple_correct=True)
        ],
        scorer=choice(),
    )

eval(mc_multiple_correct, model="ollama/llama2")


# ---
# ## Part 10: Composing solvers together
# 
# ## Quick reference
# 
# | Task type | Solvers | Scorer |
# |-----------|---------|--------|
# | Simple Q&A | `system_message() + generate()` | `match()` |
# | Reasoning | `chain_of_thought() + generate()` | `match()` |
# | Structured output | `prompt_template() + generate()` | `pattern()` |
# | Classification | `system_message() + generate()` | `exact()` |
# | Multiple choice | `multiple_choice()` | `choice()` |
# | MC + reasoning | `multiple_choice(cot=True)` | `choice()` |

# ---
# ## Assignment 3: Analyzing position bias in multiple choice
# 
# Language models can develop **position bias** - a tendency to favor certain answer positions (like more often picking "A" or "C") regardless of content.
# 
# In this assignment, you will:
# 1. Generate a set of simple math questions in multiple-choice format
# 2. Create two versions of the dataset:
#    - **Biased:** correct answer is always position A
#    - **Unbiased:** correct answer position is randomized
# 3. Run evaluations on both and compare results
# 4. Analyze whether the model shows position bias
# 
# ⚠️ **Note on methodology:** this is a minimal experiment to get you started. Comparing "all-A" vs "randomized" datasets is a quick sanity check, but it's not the most rigorous way to measure position bias.
# 
# Feel free to extend the assignment if you want a deeper analysis!

# In[ ]:


import random
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

# For reproducibility
random.seed(42)


# ### Step 1: Generate questions
# 
# First, create a helper function that generates simple questions with known correct answers.
# 
# **Function spec:**
# - Input: `n` (number of problems to generate)
# - Output: list of tuples `(question_text, correct_answer)`
# - Example output: `[("What is 5 + 3?", "8"), ("What is 12 - 4?", "8"), ...]`
# 
# **This is just an example.** You can implement your own generator with different content - trivia, vocabulary, geography, etc. Just make sure the model can reasonably answer them.

# In[ ]:


def generate_questions(n: int) -> list[tuple[str, int]]:
    """
    Generate n simple math problems.

    Args:
        n: number of problems to generate

    Returns:
        List of (question_text, correct_answer) tuples
    """
    problems = []

    for _ in range(n):
        # YOUR CODE HERE
        # Generate a simple addition or subtraction problem
        # Hint: use random.randint() for numbers, random.choice() for operation
        pass

    return problems


# ===== TESTS =====
test_questions = generate_questions(5)

assert len(test_questions) == 5, f"Expected 5 questions, got {len(test_questions)}"
assert all(isinstance(q, tuple) and len(q) == 2 for q in test_questions), "Each question must be a tuple of (question_text, answer)"
assert all(isinstance(q[0], str) and isinstance(q[1], str) for q in test_questions), "Both question and answer must be strings"
assert all(len(q[0]) > 0 and len(q[1]) > 0 for q in test_questions), "Question and answer cannot be empty"

print("\nSample output:")
for q, a in test_questions:
    print(f"  {q} → {a}")


# ### Step 2: Create wrong answers (distractors)
# 
# For multiple choice, we need plausible wrong answers.
# 
# **Function spec:**
# - Input: `correct_answer` (int)
# - Output: list of 3 wrong answers (ints), all different from correct and from each other
# 
# **Tip:** generate distractors close to the correct answer (e.g., ±1, ±2, ±10) to make them plausible.

# In[ ]:


def generate_distractors(correct: str, n: int = 3) -> list[str]:
    """
    Generate n plausible wrong answers.

    For numeric answers: generates nearby numbers.
    For other types: you'll need to customize this.

    Args:
        correct: the correct answer (string)
        n: number of distractors to generate

    Returns:
        List of n distinct wrong answers (strings)
    """
    distractors = set()

    offsets = # YOUR CODE HERE

    while len(distractors) < n:

        # YOUR CODE HERE
        # Generate offsets and find wrong answers that are close to correct but not equal
        pass

    return list(distractors)

# ===== TESTS =====
test_distractors = generate_distractors("10", n=3)

assert len(test_distractors) == 3, f"Expected 3 distractors, got {len(test_distractors)}"
assert all(isinstance(d, str) for d in test_distractors), "All distractors must be strings"
assert "10" not in test_distractors, "Distractors must not include the correct answer"
assert len(set(test_distractors)) == 3, "All distractors must be unique"

print(f"   Distractors for '10': {test_distractors}")


# ### Step 3: Create multiple choice samples
# 
# Now create a function that converts questions into multiple-choice format.
# 
# **Function spec:**
# - Input: 
#   - `questions` - list of `(question_text, correct_answer)` tuples
#   - `correct_position` - where to place correct answer:
#     - `None` → randomize position for each question
#     - `0` → always position A
#     - `1` → always position B
#     - `2` → always position C
#     - `3` → always position D
# - Output:
#     - list of `Sample` objects
# 
# ⚠️ **Note on `Sample` type:**
# 
# `Sample` is an Inspect AI class. For multiple choice, you create it like this:
# ```python
# Sample(
#     input="What is 2 + 2?",           # question text
#     choices=["3", "4", "5", "6"],     # list of options: list[str]
#     target="B"                         # letter of correct answer (A/B/C/D)
# )
# ```

# In[ ]:


def create_samples(
    questions: list[tuple[str, str]], 
    correct_position: int | None = None
) -> list[Sample]:
    """
    Convert questions to multiple-choice Samples.

    Args:
        questions: list of (question_text, correct_answer) tuples
        correct_position: 
            None → randomize position (A/B/C/D) for each question
            0 → correct answer always at position A
            1 → correct answer always at position B
            2 → correct answer always at position C  
            3 → correct answer always at position D

    Returns:
        List of Sample objects ready for Inspect AI.
        Each Sample has:
            - input: str (the question)
            - choices: list[str] (4 options, no letters)
            - target: str (correct letter: "A", "B", "C", or "D")
    """
    samples = []

    for question, correct in questions:

        # YOUR CODE HERE

        # 1. Generate 3 distractors (use generate_distractors() function)
        # 2. Build list of 4 options
        # 3. Place correct answer at the right position:
        #    - If correct_position is None → put all options together and shuffle
        #    - Otherwise → put correct at that index
        # 4. Determine target letter based on where correct ended up
        # 5. Create Sample(input=..., choices=..., target=...)
        pass

    return samples


# ===== TESTS =====
test_q = [("What is 2 + 2?", "4"), ("What is 10 - 3?", "7"), ("What is 5 + 5?", "10")]
samples_fixed = create_samples(test_q, correct_position=0)
samples_random = create_samples(test_q, correct_position=None)

assert len(samples_fixed) == len(test_q), f"Expected {len(test_q)} samples, got {len(samples_fixed)}"
assert all(hasattr(s, 'input') and hasattr(s, 'choices') and hasattr(s, 'target') for s in samples_fixed), "Each sample must have 'input', 'choices', and 'target' attributes"
assert all(len(s.choices) == 4 for s in samples_fixed), "Each sample must have exactly 4 choices"
assert all(s.target == "A" for s in samples_fixed), "With correct_position=0, all targets should be 'A'"
assert all(s.choices[0] == correct for s, (_, correct) in zip(samples_fixed, test_q)), "With correct_position=0, correct answer should be first in choices"
assert all(s.target in "ABCD" for s in samples_random), "Target must be one of A, B, C, D"

# Check that correct answer is actually at the target position
for s, (_, correct) in zip(samples_random, test_q):
    target_index = "ABCD".index(s.target)
    assert s.choices[target_index] == correct, f"Correct answer '{correct}' should be at position {s.target}, but found '{s.choices[target_index]}'"


# ### Step 4: Create the tasks
# 
# Now wrap your datasets into Inspect Tasks.
# 
# **Already provided:** task structure. You just need to call your functions.

# In[ ]:


@task
def position_bias_task(
    questions: list[tuple[str, int]],
    correct_position: int | None = None
):
    """
    Multiple choice evaluation task.

    Args:
        questions: list of (question_text, correct_answer) tuples
        correct_position: None for random, 0-3 for fixed position
    """
    samples = # YOUR CODE HERE

    return Task(
        dataset=samples,
        solver = # YOUR CODE HERE
        scorer = # YOUR CODE HERE
    )


# ### Step 5: Generate questions and run evaluations
# 
# Generate questions once, then run two evaluations:
# 1. **Biased:** correct answer always at position A
# 2. **Unbiased:** correct answer position randomized

# In[ ]:


MODEL = # YOUR CODE HERE
N_QUESTIONS = # YOUR CODE HERE

random.seed(42)
questions = generate_questions(N_QUESTIONS)

eval(
    position_bias_task, 
    model=MODEL,
    task_args={"questions": questions, "correct_position": 0}
)

eval(
    position_bias_task, 
    model=MODEL,
    task_args={"questions": questions, "correct_position": None}
)


# ### Step 7: Analyze your results
# 
# Open `inspect view` and examine both evaluation runs.
# 
# 1. **Accuracy comparison:**
#    - Biased task accuracy: ____%
#    - Unbiased task accuracy: ____%
#   
# 2. **Your analysis**
#     - What do the numbers show? (just the facts)
#     - What patterns do you notice?
#     - What might explain them? What doesn't fit your explanation?
#     - What else did you try, and what did you learn?
#    
#        etc.

# ### Bonus challenges (optional)
# 
# If you want to explore further:
# 
# 1. **Try different models** - Do all models show the same bias pattern?
# 
# 2. **Add Chain-of-Thought** - Does `multiple_choice(cot=True)` reduce position bias?
# 
# 3. **More positions** - What if you have 5 or 6 choices instead of 4?
# 
# 4. **Statistical test** - Is the position preference statistically significant? (Chi-squared test)
# 
# 5. **Content factors** - What else might affect position bias?

# In[ ]:




