#!/usr/bin/env python
# coding: utf-8

# # Tutorial 4: Evaluating LLM Agents on Mathematical Reasoning
# 
# Welcome to the fourth tutorial in our AI Safety Evaluations course.
# 
# So far you have evaluated models as **passive responders** — the model reads a prompt,
# produces an answer, and you score it. But many real-world AI systems are **agents**: they
# observe, reason, act (e.g. call a tool), observe the result, and repeat. Evaluating agents
# is harder because the model's behaviour is no longer a single forward pass — it is a
# *multi-step trajectory* where mistakes compound and new failure modes (infinite loops,
# tool misuse, hallucinated tool calls) appear.
# 
# In this tutorial you will build and evaluate a simple **ReAct agent** that solves
# math problems by calling calculator and algebra tools. You will see first-hand how
# scaffolding choices — prompts, tool sets, message limits, output formatting — affect
# agent performance, and you will practice a basic dev/test workflow for iterating on
# an agent without overfitting to your evaluation set. Think of it as a toy,
# simplified version of a real elicitation pipeline like the
# [METR Elicitation Protocol](https://evaluations.metr.org/elicitation-protocol/).
# 
# **What you'll learn:**
# 
# - Define custom tools for inspect_ai agents
# - Build a ReAct agent and iteratively improve it on a dev set
# - Develop intuition for how scaffolding choices affect agent performance
# 
# **By the end:** **You'll have a working agent evaluation pipeline and hands-on experience with the kind of iteration loop used in real-world agent evals.**

# ## 1. Setup
# 
# **Model choice.** We recommend picking a model that isn't too powerful — ideally one that occasionally
# stumbles on arithmetic so you can observe the effect of giving it tools. The examples
# below use `qwen2.5:3b`, but feel free to swap in any model you have access to (just make sure it supports tool calling). The
# main goal is to see how well the model uses the tools it's given, so don't worry if
# the tool-augmented score ends up lower than plain generation — that's a valid and
# interesting finding, not a sign that something is broken. Conversely, if your model
# solves everything perfectly even without tools, consider switching to a harder dataset
# (e.g. the full MATH instead of MATH-500, or a competition-math set like AIME) — just
# make sure to note this in your write-up.
# 
# > **Resource note:** Agent evaluations generate many more LLM calls than simple Q&A evals
# > because each problem may involve multiple reasoning steps. All `eval()` calls in this
# > notebook use a `limit` parameter to cap the number of samples processed. Adjust
# > `EVAL_LIMIT` and `MAX_MESSAGES` if your machine is slow.

# In[ ]:


get_ipython().system('pip install sympy datasets scipy -q')
print("✅ Installed: sympy, datasets, scipy")


# In[ ]:


import re
import math
import random
from textwrap import dedent
from collections import defaultdict
from scipy.stats import norm

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.agent import react
from inspect_ai.solver import generate, use_tools, system_message, TaskState
from inspect_ai.scorer import (
    Score, Target, model_graded_qa, scorer, accuracy, stderr, match, mean
)
from inspect_ai.tool import tool


# In[ ]:


# Configure model -- replace with what is available in your environment.
# E.g. 'ollama/qwen2.5:7b', 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5'
# Must support tool calling (look for "tools" tag in Ollama).

MODEL = "ollama/qwen2.5:3b"

RANDOM_SEED = 42
EVAL_LIMIT = 30        # max samples per eval run (raise if your machine is fast)
MAX_MESSAGES = 20      # max back-and-forth messages per agent trajectory


# A few helper functions for extracting and displaying results. You don't need to modify
# these — just run the cell and move on.

# In[ ]:


def get_acc(log):
    """Extract accuracy (or mean) from the first scorer in a log."""
    m = log.results.scores[0].metrics
    return (m.get("accuracy") or m.get("mean")).value


def _first_score(sample):
    """Get the first Score object from a sample, regardless of scorer name."""
    scores = sample.scores
    first_key = list(scores.keys())[0]
    val = scores[first_key]
    return val[0] if isinstance(val, list) else val


def print_results(label, log):
    """Pretty-print per-sample results from an eval log."""
    acc = get_acc(log)
    print(f"{'=' * 60}")
    print(f"  {label}   accuracy: {acc:.0%}")
    print(f"{'=' * 60}")
    for i, sample in enumerate(log.samples, 1):
        sc = _first_score(sample)
        msgs = len(sample.messages)
        expl = (sc.explanation or "")[:60]
        print(f"  [{sc.value}] #{i:2d}  msgs={msgs:2d}  "
              f"target={sample.target[:20]:>20s}  {expl}")
    print()


# ## 2. Tools and Agent Architecture
# 
# ### Why agents?
# 
# A standard LLM evaluation looks like this: you hand the model a question, it produces
# an answer, and a scorer checks whether the answer is correct. The model has *one shot*.
# 
# But many practical AI systems need to **interact with the world** — search the web, run
# code, query a database, or call an API — before they can answer. These systems are
# called **agents**. An agent follows a loop:
# 
# 1. **Observe** the current state (the question, plus any tool outputs so far).
# 2. **Think** about what to do next.
# 3. **Act** by calling a tool (or submitting a final answer).
# 4. **Observe** the tool's result, then go back to step 2.
# 
# This loop continues until the agent decides it has enough information to submit a final
# answer, or until a safety limit (maximum number of steps) is reached.
# 
# ### Why not just give the model tools and let it figure things out?
# 
# Because *access* to tools is not enough. The model also needs:
# 
# - A **system prompt** that tells it what tools exist and how to use them
# - A **loop structure** that feeds tool results back so the model can reason about them
# - A **stopping criterion** so the model knows when and how to submit its final answer
# - **Message limits** to prevent runaway loops that burn tokens without progress
# 
# This combination of prompt + loop + stopping logic is called **scaffolding**, and it
# can make or break agent performance. Whether an agent actually helps depends on the
# task, the tools, the prompt, and the model — scaffolding is not a guaranteed win, but
# understanding it is essential for evaluating agent systems.

# ### Approaches to giving a model tools
# 
# The simplest way to add tools in inspect_ai is the **`use_tools()` + `generate()`** pattern.
# `use_tools()` registers a list of tool functions so the model can call them, and
# `generate()` runs a **single generation**. The model may call tools during that generation,
# but the scaffolding never interrupts — tool calls and the final answer are produced in
# one continuous flow, with no structured pause to reconsider.
# ```python
# solver = [
#     system_message("You have access to calculator tools."),
#     use_tools([add(), multiply()]),
#     generate(),
# ]
# ```
# 
# This is easy to set up but fragile: if a tool returns something unexpected mid-generation,
# the model is already committed to a line of reasoning and may not change course.
# 
# ### The ReAct pattern
# 
# **ReAct** (Reason + Act) is a scaffolding pattern that introduces an explicit pause after
# every tool call. The model reasons and calls one tool; the scaffolding then appends the
# result to the context and starts a **fresh generation** — so the model reconsiders its plan
# from scratch before deciding the next step. This closed feedback loop makes it much easier
# to recover from unexpected tool results or multi-step reasoning chains.
# ```
# ┌─────────────────────────────────────────────────────────┐
# │                    ReAct Agent Loop                      │
# │                                                         │
# │   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
# │   │  THINK   │───>│   ACT    │───>│ OBSERVE  │──┐      │
# │   │          │    │          │    │          │  │      │
# │   │ "I need  │    │ call     │    │ tool     │  │      │
# │   │  to add  │    │ add(a,b) │    │ returns  │  │      │
# │   │  these"  │    │          │    │ "111915" │  │      │
# │   └──────────┘    └──────────┘    └──────────┘  │      │
# │        ^                                        │      │
# │        └────────────────────────────────────────┘      │
# │                                                         │
# │   Loop continues until the agent calls submit()         │
# │   or the message limit is reached.                      │
# └─────────────────────────────────────────────────────────┘
# ```
# 
# In **inspect_ai**, the `react()` solver implements this pattern. It:
# 1. Sends the problem with a system prompt describing available tools
# 2. Lets the model think and call one tool at a time
# 3. Appends the tool result to the context and starts a fresh generation
# 4. Repeats until the model calls `submit()` (a built-in action added automatically)
# 
# The `submit()` tool is special — it signals the end of the loop and its argument
# becomes the agent's final answer. You never define `submit()` yourself; `react()`
# adds it for you.

# ### Defining tools in inspect_ai
# 
# An agent needs *actions* it can take in the world. In inspect_ai, actions are
# **tools** — Python functions the model can call during its reasoning loop.
# 
# The `@tool` decorator registers a function so that inspect_ai can:
# 1. **Describe** it to the model (via the docstring and type hints — these become the
#    tool schema the model sees).
# 2. **Execute** it when the model emits a tool-call message and return the result.
# 
# The pattern is a factory function (decorated with `@tool`) that returns an `async`
# inner function. The inner function's docstring and parameter annotations are what the
# model actually sees, so clear names and descriptions directly affect agent performance.
# 
# Below we define four arithmetic tools ourselves. Notice how each one:
# - Takes typed parameters with descriptive `Args:` docstrings
# - Returns a **string** (tool outputs are always strings in inspect_ai)
# - Wraps execution in `try/except` so the agent gets a readable error instead of a crash
# 
# > **Built-in tools:** inspect_ai also ships with ready-made tools for common agent tasks — you don't need to write these yourself:
# > - `bash()` — run shell commands
# > - `python()` — execute Python code
# > - `web_search()` — search the web
# > - `web_browser()` — full browser interaction
# > - `text_editor()` — read and edit files
# >
# > For the full list see the [Tools section of the inspect_ai documentation](https://inspect.ai-safety-institute.org.uk/tools.html).
# > In this tutorial we write our own tools from scratch to understand the mechanics,
# > but in practice you will often mix custom tools with these built-in ones.

# In[ ]:


@tool
def add():
    async def execute(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        try:
            return str(float(a) + float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute


@tool
def subtract():
    async def execute(a: float, b: float) -> str:
        """Subtract b from a.

        Args:
            a: Number to subtract from.
            b: Number to subtract.
        """
        try:
            return str(float(a) - float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute


@tool
def multiply():
    async def execute(a: float, b: float) -> str:
        """Multiply two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        try:
            return str(float(a) * float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute


@tool
def divide():
    async def execute(a: float, b: float) -> str:
        """Divide a by b.

        Args:
            a: Dividend.
            b: Divisor (must not be zero).
        """
        try:
            b_val = float(b)
            if b_val == 0:
                return "Error: division by zero."
            return str(float(a) / b_val)
        except Exception as e:
            return f"Error: {e}"
    return execute


# ## Assignment 1: Create a `modular_arithmetic` tool
# 
# Cryptography and number theory problems often require modular arithmetic —
# for example, computing $7^{1000} \mod 13$ as part of a larger proof.
# 
# Following the pattern above, implement a `modular_arithmetic` tool with a clear docstring.

# In[ ]:


@tool
def modular_arithmetic():
    async def execute(a: int, b: int) -> str:
        """Compute a mod b.

        YOUR DESRIPTION

        """
        # YOUR CODE HERE
        raise NotImplementedError
    return execute


# Now let's see the tool in action. We run a small eval so the model attempts to use your tool. Don't worry if the model gets the answer wrong — what matters here is that the tool itself is correctly defined.

# In[ ]:


_test_samples_mod = [
    Sample(input="What is 1000000 mod 397? Reply with just the number.", target="354"),
    Sample(input="What is 100 mod 10? Reply with just the number.", target="0"),
]

_log_mod_test = eval(
    Task(
        dataset=_test_samples_mod,
        solver=react(
            prompt="You have a modular_arithmetic(a, b) tool. Use it, then submit the result.",
            tools=[modular_arithmetic()],
            attempts=1,
        ),
        scorer=match(numeric=True),
        message_limit=10,
    ),
    model=MODEL,
)[0]

print_results("modular_arithmetic tool test", _log_mod_test)


# > **Checking tool usage.** Run **`inspect view`** in the terminal to see the full
# > message trace for each sample — tool calls and their responses appear as explicit
# > steps. Alternatively, inspect `log.samples[i].messages` directly: each tool
# > call appears as a message with `role="assistant"` and a non-empty **`tool_calls`** field.

# ---
# 1. Did the model actually *use* your tool, or did it answer without using it?
#    Open the eval log in the inspect_ai viewer (`inspect view`) and check the trace —
#    you should see explicit tool call steps between the initial question and the final answer.
# 2. If the model skipped the tool, adjust the prompt in the `react()` call above and
#    re-run until the model uses it. What did you change?
# 
# **Your answer:**

# In[ ]:


ARITH_TOOLS = [add(), subtract(), multiply(), divide(), modular_arithmetic()]


# ## 3. Toy Evaluation — Three Solver Architectures
# 
# Before touching a real benchmark, let's try out three different solver architectures
# on a small set of hand-crafted problems. These 12 word problems use numbers large
# enough that a model without tools might make arithmetic errors.
# 
# We will compare three solver architectures:
# 
# - **Plain generation** - reads the question, produces an answer in one shot;
#   solver: `generate()`
# - **Naive tool loop** - gets access to tools; `generate()` runs once and may call
#   some tools; solver: `use_tools()` + `generate()`
# - **ReAct agent** - explicit think-act-observe loop with a `submit()` action to stop;
#   solver: `react()`
#   
# The goal here is to see how each architecture behaves — both in terms of accuracy
# and how the solver actually runs — before moving to a real benchmark.

# In[ ]:


TOY_SAMPLES = [
    Sample(
        input=(
            "A semiconductor factory produced 48,397 chips on Monday "
            "and 63,518 chips on Tuesday. How many chips were produced in total?"
        ),
        target="111915",
    ),
    Sample(
        input=(
            "A government reserve had 874,203 barrels of oil. "
            "After an emergency release, 295,867 barrels were distributed. "
            "How many barrels remain in the reserve?"
        ),
        target="578336",
    ),
    Sample(
        input=(
            "A logistics company ships 4,738 containers, each holding 2,659 units. "
            "How many units are shipped in total?"
        ),
        target="12598342",
    ),
    Sample(
        input=(
            "A national census counted 8,743,291 residents across 6,473 districts. "
            "If residents are distributed equally, how many full residents "
            "are assigned per district?"
        ),
        target="1350",
    ),
    Sample(
        input=(
            "A satellite completes a full orbit every 397 minutes. "
            "After exactly 1,000,000 minutes of operation, how many minutes "
            "have passed since the last complete orbit?"
        ),
        target="354",
    ),
    Sample(
        input=(
            "A hospital ordered 12,475 boxes of supplies at 387 dollars per box. "
            "They received a bulk discount of 843,750 dollars off the total. "
            "How much did the hospital pay after the discount?"
        ),
        target="3984075",
    ),
    Sample(
        input=(
            "A city has 14 times as many residents as municipal employees. "
            "If the total number of residents and employees together is 489,375, "
            "how many municipal employees does the city have?"
        ),
        target="32625",
    ),
    Sample(
        input=(
            "An airline flew 3,847 domestic flights and 2,964 international flights "
            "last month. Each flight used an average of 8,753 liters of fuel. "
            "How many liters of fuel were used in total?"
        ),
        target="59616683",
    ),
    Sample(
        input=(
            "A clock tower rings a bell every 1,873 seconds. "
            "After 10,000,000 seconds have elapsed since midnight, "
            "how many seconds ago did the bell last ring?"
        ),
        target="53",
    ),
    Sample(
        input=(
            "A farm harvested 247,839 kg of wheat and 184,672 kg of barley. "
            "The grain is loaded into trucks that carry exactly 4,750 kg each. "
            "How many full truckloads can be made from all the grain?"
        ),
        target="91",
    ),
    Sample(
        input=(
            "A global streaming platform has 1,847,293,847,291 seconds of video content. "
            "Given that a day has 86,400 seconds, how many full days of content "
            "does the platform have?"
        ),
        target="21380715",
    ),
    Sample(
        input=(
            "A country's economy grew by 3,847 dollars per citizen in a year. "
            "The country has 847,293,847 citizens. "
            "What was the total economic growth in dollars?"
        ),
        target="3259539429409",
    ),
]


# ## 3a. Approach 0: Plain generation (no tools)
# 
# The simplest baseline: give the model a system prompt and ask it to solve the problem
# directly. The solver is just `generate()` — a single forward pass with no tool access.
# 
# We score with `match(numeric=True)`, which extracts the first number from the model's
# response and compares it to the target.

# In[ ]:


SIMPLE_PROMPT = dedent("""    You are a math solver. Read the problem carefully, compute the answer,
    and respond with the final numeric result.
""")

log_toy_gen = eval(
    Task(
        dataset=TOY_SAMPLES,
        solver=[system_message(SIMPLE_PROMPT), generate()],
        scorer=match(numeric=True),
    ),
    model=MODEL,
)[0]

print_results("Approach 0: generate() only", log_toy_gen)


# ---
# 1. Did the model get everything right, or did it make arithmetic errors on the larger numbers?
# 2. If there were errors, what do you think caused them?
# 
# **Your answer:**

# ## 3b. Approach A: Naive tool loop
# 
# Now we give the model access to our arithmetic tools via `use_tools()`, followed by a
# single `generate()`. The model *can* call tools, but the solver doesn't enforce any
# structure: it may call one tool, multiple tools, or none at all, and it generates a
# final answer in the same pass.
# 
# > In practice this pattern is rarely used on its own — without scaffolding, whether the
# > model actually uses the tools is largely unpredictable.

# In[ ]:


NAIVE_LOOP_PROMPT = dedent("""    You are a math solver with access to calculator tools.
    Break each problem into arithmetic steps and call one tool per step.
""")

log_toy_naive = eval(
    Task(
        dataset=TOY_SAMPLES,
        solver=[
            system_message(NAIVE_LOOP_PROMPT),
            use_tools(ARITH_TOOLS),
            generate(),
        ],
        scorer=match(numeric=True),
    ),
    model=MODEL,
)[0]

print_results("Approach A: use_tools + generate (naive loop)", log_toy_naive)


# ---
# 1. Did having access to tools improve results compared to the baseline?
# 2. Did the model actually use the tools, or did it ignore them?
# 
# **Your answer:**

# ## 3c. Approach B: ReAct agent
# 
# Now let's use the `react()` solver — the full ReAct loop described in the introduction.
# Notice the prompt explicitly tells the model to use tools and call `submit()` at the end.

# In[ ]:


REACT_PROMPT_V1 = dedent("""    You are a math solver with access to calculator tools.
    Break each problem into arithmetic steps and call one tool per step.
    Don't calculate anything without tools.
    After getting the final numeric result, call submit() with ONLY the number.
""")

log_toy_react = eval(
    Task(
        dataset=TOY_SAMPLES,
        solver=react(prompt=REACT_PROMPT_V1, tools=ARITH_TOOLS, attempts=1),
        scorer=match(numeric=True),
        message_limit=20,
    ),
    model=MODEL,
)[0]

print_results("Approach B: react() with simple prompt", log_toy_react)


# ## Comparing the three approaches
# 
# Let's see the results side by side. Pay attention not just to accuracy but also to the
# number of messages — more messages means more LLM calls, which means more cost and latency.

# In[ ]:


TOY_LOGS = [log_toy_gen, log_toy_naive, log_toy_react]
TOY_LABELS = ["generate only", "naive tool loop", "react v1"]

print(f"{'Approach':<25s} {'Acc':>5s}  {'Avg msgs':>8s}  {'Max msgs':>8s}")
print("-" * 50)
for label, log in zip(TOY_LABELS, TOY_LOGS):
    acc = get_acc(log)
    msg_list = [len(s.messages) for s in log.samples]
    avg_m = sum(msg_list) / len(msg_list)
    max_m = max(msg_list)
    print(f"{label:<25s} {acc:>4.0%}   {avg_m:>7.1f}   {max_m:>7d}")


# ---
# 1. Which approach performed best? Was it also the most expensive (most messages)?
# 2. Open the eval and compare the naive tool loop and the ReAct
#    agent traces. Did the model actually use the tools in both cases, and how does the
#    structure of the traces differ?
# 
# **Your answer:**

# ## 4. Adding a Symbolic Algebra Tool
# 
# Arithmetic tools help with computation, but many math problems require solving
# equations — "find x such that 3x + 7 = 22". A small model can't do this reliably
# without tools, but SymPy (a Python symbolic math library) can solve it exactly.
# 
# ## Assignment 2: Create the `sympy_solve` tool
# 
# Implement a tool that takes an equation string (e.g. `"2*x + 5 = 21"` or `"3*x**2 - 12 = 0"`)
# and returns the solutions using SymPy. The tool should:
# 
# 1. Parse the equation — if it contains `=`, split into left and right sides and solve `left - right = 0`
# 2. If there's no `=`, treat the input as an expression equal to zero
# 3. Solve for the symbol `x`
# 4. Return the solutions as a string, or `"No solution found."` if empty
# 5. Handle errors gracefully

# In[ ]:


@tool
def sympy_solve():
    async def execute(equation: str) -> str:
        '''
        YOUR DESCRIPTION
        '''
        # YOUR CODE HERE
        raise NotImplementedError
    return execute


# In[ ]:


# Run the eval — the model should use your tool to solve both equations.
_log_sympy_test = eval(
    Task(
        dataset=[
            Sample(
                input="Solve for x: 2*x + 5 = 21. Reply with just the number.",
                target="8",
            ),
            Sample(
                input="Solve for x: x**2 - 9 = 0. What are the solutions? Reply with just the numbers separated by comma.",
                target="-3, 3",
            ),
        ],
        solver=react(
            prompt="You have a sympy_solve(equation) tool. Use it to solve the equation, then submit the result.",
            tools=[sympy_solve()],
            attempts=1,
        ),
        scorer=match(numeric=True),
        message_limit=10,
    ),
    model=MODEL,
)[0]

print_results("sympy_solve tool test", _log_sympy_test)


# ---
# 1. Did the model use the `sympy_solve` tool, or did it try to solve the equations
#    in its head? (Check the message counts.)
# 2. If it didn't use the tool, try adjusting the prompt in the `react()` call.
#    What wording helped?
# 
# **Your answer:**

# In[ ]:


# Bundle all tools
ARITH_TOOLS = [add(), subtract(), multiply(), divide(), modular_arithmetic()]
ALL_TOOLS = ARITH_TOOLS + [sympy_solve()]


# ## 5. Loading the MATH-500 Benchmark
# 
# Now that we have a working agent with tools, let's evaluate it on a real benchmark.
# **MATH-500** is a 500-question subset of the MATH dataset (Hendrycks et al., 2021),
# covering competition-level math across seven subjects. It's available on Hugging Face.
# 
# Not all subjects benefit equally from our tools — Geometry and Counting & Probability
# involve spatial reasoning and combinatorics that our calculator tools can't help with.
# We'll focus on the four subjects where arithmetic and algebra tools are most relevant:
# Algebra, Intermediate Algebra, Number Theory, and Prealgebra.
# 
# ### Extracting answers from MATH solutions
# 
# MATH stores answers inside `\boxed{...}` in the solution string. We need a helper to
# extract them:

# In[ ]:


TOOL_SUBJECTS = [
    "Algebra", "Number Theory", "Prealgebra", "Intermediate Algebra",
]


def extract_boxed(solution):
    """Extract the content of the last \\boxed{...} in a MATH solution string."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return solution.strip()
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start:i - 1].strip()


def record_to_sample(record):
    """Convert a MATH-500 record into an inspect_ai Sample."""
    target = record.get("answer") or extract_boxed(record["solution"])
    return Sample(
        input=record["problem"],
        target=target,
        metadata={
            "level": int(record["level"]),
            "subject": record["subject"],
        },
    )


# In[ ]:


full_dataset = hf_dataset(
    path="HuggingFaceH4/MATH-500",
    split="test",
    sample_fields=record_to_sample,
    cached=True,
)

print(f"Total MATH-500: {len(full_dataset)} samples")

subject_counts = defaultdict(int)
for s in full_dataset:
    subject_counts[s.metadata["subject"]] += 1

print(f"\n{'Subject':<30s} {'Count':>5s}")
print("-" * 37)
for subj in sorted(subject_counts):
    marker = " <-- tool-friendly" if subj in TOOL_SUBJECTS else ""
    print(f"{subj:<30s} {subject_counts[subj]:>5d}{marker}")


# ## Dev/test split
# 
# A core principle in evaluation: **never tune on your test set.** Iterating on the same
# data you use for final scoring inflates results and makes them meaningless. We split the
# tool-friendly subset into a small **dev set** (10%) for prompt and scaffolding iteration,
# and a larger **test set** (90%) reserved for final evaluation only.
# 
# This mirrors the elicitation workflow described in [METR's Guidelines for Capability Elicitation](https://evaluations.metr.org/elicitation-protocol/): iterate against the dev set until failures stabilize, then run the test set once.
# 
# > **Adjust to your hardware.** If even the dev set takes too long, reduce `split_point`
# > or set a smaller `EVAL_LIMIT`. The point is rapid iteration — you can always increase
# > the test set size later.

# In[ ]:


tool_dataset = [s for s in full_dataset if s.metadata["subject"] in TOOL_SUBJECTS]
print(f"Tool-friendly subset: {len(tool_dataset)} samples")

random.seed(RANDOM_SEED)
random.shuffle(tool_dataset)

split_point = int(len(tool_dataset) * 0.1)
DEV_SET = tool_dataset[:split_point]
TEST_SET = tool_dataset[split_point:]

print(f"DEV_SET:  {len(DEV_SET)} samples")
print(f"TEST_SET: {len(TEST_SET)} samples")


# ## 6. Scoring Mathematical Answers
# 
# For the toy problems we used `match(numeric=True)`, which works when answers are plain
# numbers. But MATH answers can be fractions (`3/7`), expressions (`2\sqrt{5}`), or
# formatted in different equivalent ways — a simple string match will miss many correct answers.
# 
# We covered `model_graded_qa()` in notebook 3. Here we apply it to math: the key challenge
# is writing a grading prompt that handles equivalent notations (e.g. `1/2` vs `0.5` vs `\frac{1}{2}`).
# Note that in notebook 3 we deliberately hid the reference answer from the grading model — here
# there's no reason to do that, so you can pass the correct answer directly.
# 
# > **Trade-off:** Model-graded scoring is slower and noisier, but catches equivalences that
# > string matching misses. For production evals you might want a more capable model for
# > grading than the one being tested — think about what makes sense for your setup.
# 
# ## Assignment 3: Define the math scorer
# 
# Define a `math_scorer` using `model_graded_qa()`. Write a grading prompt that instructs
# the model to judge mathematical equivalence and respond with **C** (correct) or **I** (incorrect),
# and choose which model should do the grading. Think about what edge cases matter: different
# notations, equivalent fractions, simplified vs unsimplified forms.

# In[ ]:


SCORER_MODEL = # YOUR CODE HERE 

GRADING_INSTRUCTIONS = # YOUR CODE HERE

MATH_SCORER = # YOUR CODE HERE


# In[ ]:


# Run the scorer on two toy samples where we know the answer.
_scorer_test_samples = [
    Sample(input="What is 1+1?", target="2"),
    Sample(input="What is 10/4?", target="5/2"),
]

_log_scorer_test = eval(
    Task(
        dataset=_scorer_test_samples,
        solver=[system_message("Answer the math question. Reply with just the answer."), generate()],
        scorer=MATH_SCORER,
    ),
    model=MODEL,
)[0]

print_results("Scorer sanity check", _log_scorer_test)
print("Check: does the scorer correctly mark equivalent answers as C?")


# ---
# 1. Did your scorer correctly handle the fraction equivalence (`10/4` vs `5/2`)?
# 2. What failure modes can you imagine for model-graded scoring?
# 
# **Your answer:**

# ## 7. Dev-Set Iteration — Building and Improving Your Agent
# 
# This is the core of agent evaluation: run on the dev set, see where the agent struggles,
# and systematically improve it. The iteration loop:
# 
# 1. Run the current agent on the dev set
# 2. Inspect failures — *why* did the agent get these wrong?
# 3. Hypothesize an improvement (better prompt? more tools? output formatting?) — and check
#    whether the answer was actually correct but the scorer marked it wrong
# 4. Implement the change and re-evaluate on the dev set
# 5. Repeat

# ## Assignment 4: Build and improve a ReAct agent
# 
# Your goal is to build a ReAct agent and iterate on it using the dev set.
# Does adding tools and scaffolding help on this task, and by how much?
# Start by running a basic ReAct agent on the dev set, then look at the failures in the logs and iterate.
# 
# **Step 1 — Run a baseline.** Write a system prompt and run `react()` with `ALL_TOOLS` on the dev set.
# 
# **Step 2 — Inspect failures.** Open the logs and look at what went wrong. Common things to consider:
# - Is the model ignoring tools and is failing?
# - Is the answer mathematically correct but formatted wrong (e.g. `0.5` instead of `1/2`)?
# - Is the model getting lost in multi-step problems?
# 
# **Step 3 — Iterate.** Based on what you see, try to improve. Some directions:
# - Make the system prompt more explicit about strategy or answer format
# - Add tools that cover operations the model struggles with (e.g. a single `calculator` tool, `gcd`, `factorial` or something very different)
# - Add a format-extraction step after the `react()` loop if formatting is the main issue
# - Reconsider the scorer if correct answers are being marked wrong
# 
# For each configuration you try, store the result as a `(description, log)` tuple in `DEV_RUNS` at the bottom — this will let you compare all your attempts in one table. Try at least two configurations.

# In[ ]:


MY_REACT_PROMPT = # YOUR CODE HERE

log_attempt_1 = eval(
    Task(
        dataset=DEV_SET,
        solver=react(
            prompt=MY_REACT_PROMPT,
            tools=ALL_TOOLS,
            attempts=1,
        ),
        scorer=MATH_SCORER,
        message_limit=MAX_MESSAGES,
    ),
    model=MODEL,
    limit=EVAL_LIMIT,
)[0]

print_results("Attempt 1 (baseline)", log_attempt_1)
DEV_RUNS = [("Attempt 1 (baseline)", log_attempt_1)]


# In[ ]:


# Look at the failures in the logs and decide what to change:
# prompt, tools, scorer, add a format-extraction step after react() or something else.
# Duplicate this cell for each new configuration. Give each log a new name.

# YOUR CODE HERE: define your updated prompt, tools, scorer, or solver

log_attempt_2 = eval(
    Task(
        dataset=DEV_SET,
        solver=# YOUR CODE HERE,
        scorer=# YOUR CODE HERE,
        message_limit=MAX_MESSAGES,
    ),
    model=MODEL,
    limit=EVAL_LIMIT,
)[0]

description_2 = # YOUR CODE HERE
print_results(description_2, log_attempt_2)
DEV_RUNS.append((description_2, log_attempt_2))


# In[ ]:


print(f"{'Configuration':<40s} {'Dev Accuracy':>12s}")
print("=" * 54)
for description, log in DEV_RUNS:
    acc = get_acc(log)
    print(f"{description:<40s} {acc:>12.0%}")


# ---
# 1. What modifications did you try? Which had the biggest impact?
# 2. What was the best dev-set accuracy you achieved? What configuration produced it?
# 3. Did any change that you expected to help actually hurt? Why might that be?
# 4. Look at the individual failures. What are the most common error types?
# 
# **Your answer:**

# ## 8. Test-Set Evaluation
# 
# You've iterated on the dev set and found your best configuration. Now it's time for the
# moment of truth: evaluating on the held-out test set.
# 
# > **Run this section only once**, with your best configuration. Re-running and picking
# > the best result would be "test-set contamination" — the same as peeking at a test set
# > in ML.
# 
# ## Assignment 5: Run and analyze your best configuration on the test set
# 
# Run your best agent configuration on the held-out test set, report accuracy with a 95%
# confidence interval, and break down performance by subject and difficulty level. Note
# anything that stands out.

# In[ ]:


def wilson_ci(n_correct, n_total, confidence_level=0.95):
    """Wilson score interval for a binomial proportion."""
    z = norm.ppf(0.5 + confidence_level / 2)
    p_hat = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total
    ) / denom
    return max(0, centre - margin), min(1, centre + margin)


# ## Assignment 5.1: Run on the test set
# 
# Use your best configuration and run it to evaluate the test set

# In[ ]:


log_test = eval(
    Task(
        dataset=TEST_SET,
        # YOUR CODE HERE
    ),
    model=MODEL,
    limit=len(TEST_SET),
)[0]

n_test = # YOUR CODE HERE
n_correct_test = # YOUR CODE HERE

lo, hi = wilson_ci(n_correct_test, n_test)
print(f"Test accuracy : {n_correct_test / n_test:.1%}")
print(f"95% Wilson CI : [{lo:.1%}, {hi:.1%}]")
print(f"n = {n_test}")


# ## Assignment 5.2: Breakdown by subject and difficulty level
# 
# There aren't enough samples per category to draw firm conclusions — that's fine.
# Just see if anything stands out.
# 
# Iterate over logs and produce **two separate tables**: one grouped
# by subject, one by difficulty level. 
# 
# Your output should look like this:
# 
# **By subject:**
# 
# | Subject | Correct | Total | Acc |
# |---|---:|---:|---:|
# | Algebra | 1 | 10 | 10% |
# | Precalculus | 3 | 6 | 50% |
# | ... | | | |
# 
# **By level:**
# 
# | Level | Correct | Total | Acc |
# |---|---:|---:|---:|
# | 1 | 4 | 5 | 80% |
# | 2 | 3 | 6 | 50% |
# | ... | | | |
# 
# The actual numbers will appear in the cell below.

# In[ ]:


subject_stats = defaultdict(lambda: [0, 0])
level_stats = defaultdict(lambda: [0, 0])

for sample in log_test.samples:
    sc = _first_score(sample)
    correct = 1 if sc.value == "C" else 0
    subj = sample.metadata.get("subject", "unknown")
    lvl = sample.metadata.get("level", 0)
    subject_stats[subj][0] += correct
    subject_stats[subj][1] += 1
    level_stats[lvl][0] += correct
    level_stats[lvl][1] += 1

print(f"{'Subject':<30s} {'Correct':>7s} {'Total':>5s} {'Acc':>6s}")
print("-" * 50)
for subj in sorted(subject_stats):
    c, t = subject_stats[subj]
    print(f"{subj:<30s} {c:>7d} {t:>5d} {c/t:>6.0%}")

print()
print(f"{'Level':<30s} {'Correct':>7s} {'Total':>5s} {'Acc':>6s}")
print("-" * 50)
for lvl in sorted(level_stats):
    c, t = level_stats[lvl]
    print(f"Level {lvl:<24d} {c:>7d} {t:>5d} {c/t:>6.0%}")


# **Your results:**
# 
# Fill in the tables once you've run the cell above.
# 
# **By subject:**
# 
# | Subject | Correct | Total | Acc |
# |---|---:|---:|---:|
# | Algebra | | | |
# | Precalculus | | | |
# | ... | | | |
# 
# **By difficulty level:**
# 
# | Level | Correct | Total | Acc |
# |---|---:|---:|---:|
# | 1 | | | |
# | 2 | | | |
# | ... | | | |

# ## Final comparison: dev vs test

# In[ ]:


print(f"{'Configuration':<30s} {'Dev Acc':>8s}  {'Test Acc':>8s}")
print("=" * 50)
for name, log in DEV_RUNS:
    acc = get_acc(log)
    print(f"{name:<30s} {acc:>8.0%}  {'--':>8s}")
print(f"{'best agent (TEST)':<30s} {'--':>8s}  {n_correct_test / n_test:>8.1%}")
print(f"\n95% CI on test: [{lo:.1%}, {hi:.1%}]")


# ---
# 1. How does the test accuracy compare to the dev accuracy? If there's a gap,
#    what might explain it?
# 2. Which subjects and difficulty levels does the agent handle best? Worst?
# 3. Look at the confidence interval. Is it narrow enough to be useful, or would
#    you want more test samples?
# 4. If you were to improve this agent further, where would you focus?
# 
# **Your answer:**

# ## Bonus assignment: Error Analysis
# 
# Pick 5-10 test samples that the agent got wrong and classify each failure. Here's one
# possible taxonomy — feel free to use your own:
# 
# - **Tool misuse:** Called the wrong tool or with wrong arguments
# - **Reasoning error:** Correct tool use but flawed multi-step reasoning
# - **Format error:** Correct answer but wrong format in `submit()`
# - **Inherent difficulty:** Problem requires reasoning beyond the tool set
# - **Grading error:** The agent was actually right but the scorer got it wrong
# - **Other:** Anything that doesn't fit the above
# 
# This kind of qualitative error analysis is essential in practice — aggregate metrics
# tell you *how much* the agent struggles, but error analysis tells you *why*.

# If you'd like to explore the logs in code, use the cell below — otherwise just read
# them directly and delete it.

# In[ ]:


# YOUR CODE HERE


# ---
# 1. What was the most common failure mode?
# 2. Which failure modes could be fixed with better prompting vs. better tools
#    vs. a more capable model?
# 3. Did you find any grading errors? What does that imply about using model-graded scoring?
# 
# **Your answer:**
