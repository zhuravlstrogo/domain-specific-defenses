#!/usr/bin/env python
# coding: utf-8

# # Tutorial 2: Evaluating LLMs on MMLU
# 
# Welcome to the second tutorial in our AI Safety Evaluations course.
# 
# Benchmark evaluation is a core skill in applied ML, but the statistical side is often
# treated as an afterthought — a single accuracy number gets reported, and model differences
# are treated as real without checking whether they could arise from chance alone.
# In this tutorial you will get hands-on experience running evaluations with the inspect_ai
# library and applying basic statistical methods to interpret the results rigorously.
# 
# **What you'll learn:**
# 
# - Load and prepare a benchmark dataset
# - Compute confidence intervals for accuracy
# - Compare models statistically
# - Perform power analysis to plan evaluation size
# 
# **By the end:** **You'll have a statistically rigorous evaluation pipeline that can tell you not just how accurate a model is, but whether observed differences between models are real.**

# ## 1. Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from string import ascii_uppercase
from typing import Tuple, List

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, FieldSpec
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from inspect_ai.log import EvalLog


# In[2]:


# Configure models -- replace with what is available in your environment.
# Examples: 'ollama/llama3.2', 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5'

MODEL_A = "ollama/llama2"        # weaker / baseline model
MODEL_B = "ollama/qwen2:latest"  # stronger / comparison model


# ## 2. Loading MMLU
# 
# `hf_dataset` is inspect_ai's loader for Hugging Face datasets. It downloads the data
# and wraps each record in a `Sample` -- the standard container that flows through every
# inspect_ai pipeline. A `Sample` carries the model input, the expected target, optional
# answer choices, and arbitrary metadata you want to keep around.
# 
# MMLU stores the correct answer as an integer (0 = A, 1 = B, 2 = C, 3 = D).
# The quickest way to load a dataset is with `FieldSpec`, which maps column names to
# `Sample` fields. Let's try it first and see what we get.

# In[3]:


dataset_raw = hf_dataset(
    path="cais/mmlu",
    name="all",
    split="test",
    sample_fields=FieldSpec(
        input="question",
        target="answer",           # raw MMLU answer is an integer index 0-3
        metadata=["choices", "subject"]
    ),
    cached=True
)

sample = dataset_raw[0]
print("input   :", sample.input[:80], "...")
print("target  :", sample.target,  "  <- integer index, not a letter!")
print("choices :", sample.metadata.get("choices"))


# The `target` came out as an integer -- but inspect_ai's `multiple_choice()` solver
# and `choice()` scorer expect it to be a letter (`"A"`, `"B"`, `"C"`, or `"D"`).
# When the automatic mapping isn't enough, inspect_ai lets you pass a
# **record-to-sample function** that receives the full raw record and returns a `Sample`
# you construct yourself.

# In[14]:


def record_to_sample(record: dict) -> Sample:
    """
    Convert a raw MMLU record to an inspect_ai Sample.

    MMLU stores the correct answer as an integer index (0=A, 1=B, 2=C, 3=D).
    We convert it to the corresponding uppercase letter so it matches the
    format expected by the choice() scorer.
    """
    answer_idx = int(record["answer"])
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=ascii_uppercase[answer_idx],   # 0->'A', 1->'B', ...
        metadata=dict(subject=record.get("subject"))
    )


dataset = hf_dataset(
    path="cais/mmlu",
    name="all",
    split="test",
    sample_fields=record_to_sample,
    cached=True
)

sample = dataset[0]
print("target  :", sample.target, " <- letter now")
print("choices :", sample.choices)


# ## Assignment 1: Create your working subset
# 
# All experiments in this notebook will run on a subject subset small enough to evaluate
# quickly. `Dataset.filter()` takes a predicate over `Sample` objects; the `metadata`
# field gives access to anything set in `record_to_sample` -- here, the MMLU subject tag.
# 
# We define `astronomy_subset` as a reference example. Pick any subject or subjects from the [MMLU subject list](https://huggingface.co/datasets/cais/mmlu#task-descriptions) with at least 50 questions so later analyses are statistically meaningful. Create MY_SUBSET and use it in all subsequent exercises.

# In[15]:


# Reference subset used in worked examples
astronomy_subset = dataset.filter(
    lambda s: s.metadata.get("subject") == "astronomy"
)
print(f"Astronomy: {len(astronomy_subset)} questions")


MY_SUBSET = # YOUR CODE HERE

print(f"My subset: {len(MY_SUBSET)} questions")


# ## 3. Running an evaluation
# 
# Every inspect_ai evaluation is described by a `Task`, which bundles three things:
# 
# - **dataset** -- the questions
# - **solver** -- the chain of steps that produces a model response;
#   `multiple_choice()` formats the prompt with lettered options and parses the model's choice
# - **scorer** -- the function that grades the response;
#   `choice()` checks whether the selected letter matches the target
# 
# The `@task` decorator registers the function so inspect_ai can discover it by name
# from the CLI or pass it directly to `eval()`.

# In[7]:


@task
def mmlu_subset(subset):
    """Minimal MMLU task for any subject subset."""
    return Task(
        dataset=subset,
        solver=[multiple_choice()],
        scorer=choice()
    )


# Calling `eval()` runs the task and returns a **list of `EvalLog` objects** -- one per
# (task, model) pair. Everything you need is in this object; there is no need to read
# log files from disk.
# 
# The two most useful attributes:
# - `log.results.scores` -- list of scorer results, each with a `metrics` dict
#   (`"accuracy"`, `"stderr"`, etc.)
# - `log.samples` -- list of `EvalSample` objects with per-question inputs, outputs, and scores

# In[16]:


logs: List[EvalLog] = eval(
    mmlu_subset(astronomy_subset),
    model=MODEL_A,
    limit=10        # evaluate only the first 10 questions
)

log = logs[0]      # one task -> one log
print("Status  :", log.status)
print("Model   :", log.eval.model)
print("Accuracy:", log.results.scores[0].metrics["accuracy"].value)


# ## 4. From `EvalLog` to a DataFrame
# 
# ## Assignment 2: Implement `log_to_df`
# 
# The aggregate accuracy in `log.results` is useful for a quick check, but for the
# statistical analyses ahead we need a flat table: **one row per (question, epoch)**
# with a numeric `score` column.
# 
# `log.samples` is a list of `EvalSample` objects. Each one has:
# - `.id` -- question identifier
# - `.epoch` -- which run this belongs to (relevant when `epochs > 1`)
# - `.scores` -- a dict mapping scorer name to `Score`; the `Score.value` for `choice()` is
#   `"C"` (correct) or `"I"` (incorrect)
# - `.metadata` -- the metadata dict you set in `record_to_sample`
# 
# Implement `log_to_df` so that it converts an `EvalLog` into a DataFrame with columns
# `id`, `epoch`, `score` (1/0), and `subject`. The smoke test below will verify the shape.

# In[17]:


def log_to_df(log: EvalLog) -> pd.DataFrame:
    """
    Convert an EvalLog to a DataFrame with one row per (question, epoch).

    Columns:
        id      – question identifier
        epoch   – epoch index (0 if epochs=1)
        score   – 1 if correct, 0 otherwise
        subject – MMLU subject tag from metadata

    The choice() scorer stores the result as "C" (correct) or "I" (incorrect).
    """
    # YOUR CODE HERE
    raise NotImplementedError

# =================================== TESTS ===================================
df_test = log_to_df(log)

assert set(df_test.columns) >= {"id", "epoch", "score", "subject"}
assert df_test["score"].isin([0, 1]).all()
assert len(df_test) == 10

print(df_test.head())
print(f"\nAccuracy: {df_test['score'].mean():.1%}")


# ## 5. Confidence intervals
# 
# 
# A single accuracy number carries uncertainty: the eval used a finite set of questions
# sampled from a much larger space. The paper (ss2.1, ss3.1) shows how to quantify this
# using the CLT standard error.
# 
# 
# ## Assignment 3: Implement `ci_accuracy_basic` and `ci_accuracy`
# 
# **`ci_accuracy_basic(scores, ci)`** -- the simple case where every question is answered
# exactly once. `scores` is a plain numpy array of 0s and 1s. Use Eq. 1 from the paper.
# 
# **`ci_accuracy(df, ci)`** -- the general case that handles multiple runs per question
# (`epochs > 1`). When K runs exist for a question, average their scores first, then
# compute the SE across question-level averages. Pooling all K×n individual answers
# would undercount variance -- answers to the same question across epochs are correlated.
# 

# In[ ]:


def ci_accuracy_basic(scores: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    CLT-based confidence interval for accuracy -- single run per question (K = 1).

    Parameters
    ----------
    scores : 1-D array of per-question binary scores (0 or 1)
    ci     : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, mean_accuracy, upper_bound)
    """
    # YOUR CODE HERE
    raise NotImplementedError


def ci_accuracy(df: pd.DataFrame, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    CLT-based confidence interval for accuracy, supporting multiple epochs (K >= 1).

    Parameters
    ----------
    df : DataFrame returned by log_to_df, with columns 'id', 'score', 'epoch'
    ci : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, mean_accuracy, upper_bound)
    """
    # YOUR CODE HERE
    raise NotImplementedError


# In[ ]:


# =================================== TESTS ===================================
def _make_df(ids, scores, epochs=None):
    if epochs is None:
        epochs = [0] * len(ids)
    return pd.DataFrame({"id": ids, "score": scores, "epoch": epochs})

# ci_accuracy_basic
l, m, u = ci_accuracy_basic(np.ones(10))

assert l == 1.0 and u == 1.0, "perfect accuracy: CI should collapse to 1"

l, m, u = ci_accuracy_basic(np.zeros(10))

assert l == 0.0 and u == 0.0, "zero accuracy: CI should collapse to 0"

scores3 = np.array([1, 1, 0, 1, 0], dtype=float)
l, m, u = ci_accuracy_basic(scores3)

assert l < 0.6 < u, f"0.6 not in [{l:.3f}, {u:.3f}]"

np.random.seed(42)
s = np.random.binomial(1, 0.75, 200).astype(float)
l95, _, u95 = ci_accuracy_basic(s, 0.95)
l99, _, u99 = ci_accuracy_basic(s, 0.99)

assert (u99 - l99) > (u95 - l95), "99% CI must be wider than 95%"
assert np.isclose(l95, 0.6819421067148456, atol=10e-2)
assert np.isclose(u95, 0.8080578932851544, atol=10e-2)

# ci_accuracy (K=1 should match basic)
df3 = _make_df([1,2,3,4,5], scores3.tolist())
l_df, _, u_df = ci_accuracy(df3)
l_ar, _, u_ar = ci_accuracy_basic(scores3)

assert np.isclose(l_df, l_ar) and np.isclose(u_df, u_ar), "K=1 must match basic version"

# ci_accuracy (K=3 should give narrower CI on average)
np.random.seed(0)
rows_k1, rows_k3 = [], []
for q in range(30):
    p = np.random.uniform(0.3, 0.9)
    rows_k1.append({"id": q, "score": int(np.random.binomial(1, p)), "epoch": 0})
    for e in range(3):
        rows_k3.append({"id": q, "score": int(np.random.binomial(1, p)), "epoch": e})

l1, _, u1 = ci_accuracy(pd.DataFrame(rows_k1))
l3, _, u3 = ci_accuracy(pd.DataFrame(rows_k3))
print(f"K=1 width: {u1-l1:.3f}")
print(f"K=3 width: {u3-l3:.3f}  (narrower on average)")
print("\n✓ All tests passed!")


# ## 6. Visualising how CIs shrink
# 
# Two things make confidence intervals narrower: more questions (larger n) and more
# runs per question (larger K). Your task is to visualise those effects.
# 
# ## Assignment 4.1: Plot CI width vs number of epochs
# 

# In[25]:


k_values    = [5, 6, 7, 8, 9, 10]
accuracies    = []
ci_lowers     = []
ci_uppers     = []

# YOUR CODE HERE

plt.figure(figsize=(8, 4))
plt.fill_between(k_values, ci_lowers, ci_uppers, alpha=0.25, label="95% CI")
plt.plot(k_values, accuracies, "o-", lw=2, label="Accuracy")
plt.xlabel("Number of runs per question (K)")
plt.ylabel("Accuracy")
plt.title(f"{MODEL_A} on MMLU-subset — accuracy and CI vs k")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# ---
# 1. Look at how fast the band narrows.
#    At what point does running another epoch stop being worth it?
# 2. Does more K change your estimate of the model's accuracy, or just your confidence in it?
# 3. What does this tell you about how to allocate your evaluation budget?
# 
# **Your answer:**

# ## Assignment 4.2: Compute and plot CI width vs n
# 
# For each sample size n in `range(10, len(question_ids)+1, 10)`, slice both DataFrames
# to the first n question IDs, compute `ci_accuracy`, and record the CI width.
# Then plot width vs n.

# In[ ]:


question_ids  = # YOUR CODE HERE
dataset_sizes = range(10, len(question_ids) + 1, 10)
accuracies    = []
ci_lowers     = []
ci_uppers     = []

# YOUR CODE HERE

plt.figure(figsize=(8, 4))
plt.fill_between(dataset_sizes, ci_lowers, ci_uppers, alpha=0.25, label="95% CI")
plt.plot(dataset_sizes, accuracies, "o-", lw=2, label="Accuracy")
plt.xlabel("Number of questions (n)")
plt.ylabel("Accuracy")
plt.title(f"{MODEL_A} on MMLU-subset — accuracy and CI vs n")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# ---
# 1. At what n does the accuracy line start to feel stable?
# 2. Compare that number to the size of `MY_SUBSET` — are you in the reliable region?
# 3. Compare this curve to the one in 4.1. What is the difference in what K and n actually buy you?
# 
# **Your answer:**

# ## 7. Comparing two models
# 
# Reporting two accuracy numbers side by side doesn't tell you whether the gap is real
# or just noise. The paper (§4.2) describes a **paired test**: because both models answer
# the same questions, you can compute per-question score differences and test whether
# their mean differs significantly from zero. This removes question-difficulty variance
# and yields a lower standard error than treating the two runs as independent samples.
# 

# ## Assignment 5: Compare two models
# 
# `run_and_get_scores` and `compare_models_paired` are provided. Complete
# `significance_by_paired_ttest` and use it to compare the two models on `MY_SUBSET`.
# 
# Implement `significance_by_paired_ttest` and compare MODEL_A and MODEL_B.

# In[15]:


def run_and_get_scores(model_name: str, dataset, epochs: int = 1) -> np.ndarray:
    """Run eval and return mean-per-question scores, sorted by question id."""
    print(f"  Running {model_name} ...")
    run_logs = eval(mmlu_subset(dataset), model=model_name, epochs=epochs)
    df = log_to_df(run_logs[0])
    return df.groupby("id")["score"].mean().sort_index().values


def significance_by_paired_ttest(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> Tuple[float, float, bool]:
    """
    Paired t-test between two sets of per-question scores.

    Returns (p_value, mean_difference scores1 - scores2, is_significant).
    """
    assert len(scores1) == len(scores2), "arrays must cover the same questions"

    alternative = "two-sided" if two_tailed else "greater"

    _, p_value  = # YOUR CODE HERE
    mean_diff   = # YOUR CODE HERE

    return p_value, mean_diff, bool(p_value < alpha)


def compare_models_paired(
    model_a: str,
    model_b: str,
    dataset,
    alpha: float = 0.05,
    two_tailed: bool = True,
    epochs_a: int = 1,
    epochs_b: int = 1,
) -> Tuple[float, float, bool]:
    """
    Evaluate both models on the same dataset and run a paired t-test.

    Returns (p_value, mean_difference A - B, is_significant).
    """
    scores_a = run_and_get_scores(model_a, dataset, epochs=epochs_a)
    scores_b = run_and_get_scores(model_b, dataset, epochs=epochs_b)
    return significance_by_paired_ttest(scores_a, scores_b, alpha, two_tailed)


# In[18]:


# =================================== TESTS ===================================
p, d, sig = significance_by_paired_ttest(np.array([1,2,3]), np.array([1,2,3]))

assert np.isclose(d, 0.0) and not sig

p, d, sig = significance_by_paired_ttest(
    np.array([1,1,1,1,1]), np.array([0,0,0,0,0]), two_tailed=False
)

assert sig and d > 0

print("All tests passed!")


# In[ ]:


# YOUR CODE HERE


# ---
# 1. Write down the p-value and the mean difference you got.
# 2. Is the gap significant? Is it large enough to matter in practice?
# 3. What would change your conclusion: more questions, a different subject, or a different model pair?
# 
# **Your answer:**

# ## 8. Interval estimation of the accuracy difference
# 
# In Assignment 5 you got a yes/no significance decision. Here you will estimate the size of the gap and its uncertainty: a confidence interval on the difference gives both pieces of information at once.
# 
# ## Assignment 6: Estimate the accuracy gap
# 
# Implement `ci_accuracy_for_difference` to compute a 95% CI on the per-question score differences.
# 
# Compute and report the confidence interval on MODEL_A − MODEL_B.
# 

# In[18]:


# YOUR CODE HERE


# ---
# 1. Write down the interval. Does it contain zero?
# 2. How does this relate to the significance test in Assignment 5 — do they tell the same story?
# 3. Which result is more informative — the p-value or the interval? Why?
# 
# **Your answer:**

# ## 9. Power analysis
# 
# Before running an expensive evaluation, it is worth asking: how many questions do we
# need to detect a meaningful difference with adequate statistical power?
# The paper (§5) derives the minimum detectable effect as a function of sample size n,
# question-level variance ω², and within-model variance σ².
# 
# ## Assignment 7: 'Estimate variance components'
# 
# Implement `estimate_variance_components` and report the MDE for `MY_SUBSET` at α = 0.05, power = 80%.

# In[19]:


def estimate_variance_components(
    logs_a: List[EvalLog],
    logs_b: List[EvalLog],
) -> dict:
    """
    Estimate omega2, sigma2_a, sigma2_b from two EvalLog objects (see ss5 of the paper).

    Both logs must cover the same set of questions. Use epochs >= 2 so that
    within-question variance can be estimated.

    Returns dict with keys 'omega2', 'sigma2_a', 'sigma2_b'.
    """

    # YOUR CODE HERE

    return {
        "omega2":   ...,
        "sigma2_a": ...,
        "sigma2_b": ...,
    }


def minimum_detectable_effect(
    n: int,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    ka: int = 1,
    kb: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """MDE for a paired model comparison (Eq. 10 in the paper)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * np.sqrt(
        (omega2 + sigma2_a / ka + sigma2_b / kb) / n
    ))


# In[19]:


print("Running pilot evals ...")
logs_a = eval(mmlu_subset(MY_SUBSET), model=MODEL_A, epochs=2, limit=15)
logs_b = eval(mmlu_subset(MY_SUBSET), model=MODEL_B, epochs=2, limit=15)

params = estimate_variance_components(logs_a, logs_b)
print(f"omega2  = {params['omega2']:.4f}")
print(f"sigma2_A = {params['sigma2_a']:.4f}")
print(f"sigma2_B = {params['sigma2_b']:.4f}")

mde = minimum_detectable_effect(n=len(MY_SUBSET), **params)
print(f"\nWith n={len(MY_SUBSET)} questions -> MDE = {mde:.1%}")
print("(smallest gap detectable at 80% power, alpha=0.05)")


# ---
# 1. What MDE did you get for `MY_SUBSET`? Is that gap practically meaningful?
# 2. If the MDE is larger than the gap you observed in Assignment 5,
#    what does that say about your earlier result?
# 
# **Your answer:**

# ## Assignment 8: Implement `required_sample_size`
# 
# `minimum_detectable_effect` computes delta given n. Implement its inverse:
# given a target delta, return the minimum n needed. Use the sample-size formula
# from ss5 of the paper (Eq. 9). Verify it passes the round-trip check,
# then use it to compute how many questions you would need to detect a 5% and a 10%
# accuracy gap on `MY_SUBSET`.

# In[ ]:


# --- Assignment 7 -----------------------------------------------------------
def required_sample_size(
    delta: float,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    ka: int = 1,
    kb: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum number of questions needed to detect `delta` at the given power."""
    # YOUR CODE HERE
    raise NotImplementedError


# =================================== TESTS ===================================
n_needed = required_sample_size(delta=0.05, **params)
print(f"Questions needed to detect delta=5%: {n_needed}")

mde_check = minimum_detectable_effect(n=n_needed, **params)

assert abs(mde_check - 0.05) < 0.005, f"Round-trip failed: MDE={mde_check:.3f}"

print("Round-trip check passed!")


# In[1]:


# YOUR CODE HERE


# ---
# 1. How many questions do you need to detect a 5% gap? A 10% gap?
# 2. Does `MY_SUBSET` have enough questions to be a useful benchmark for comparing these two models?
# 
# **Your answer:**

# ## Assignment 9: Compare a model with itself: baseline vs chain-of-thought
# 
# The `multiple_choice()` solver we have used so far prompts the model to answer
# directly. inspect_ai also provides a `chain_of_thought` that asks the model
# to reason step by step before giving its final answer.
# 
# Using the paired comparison infrastructure from Section 7, evaluate the same model
# twice on the same subset — once with the default solver and once with
# `chain_of_thought` — and test whether the difference in accuracy is statistically
# significant. Does reasoning help? Is the effect consistent across subjects?

# In[ ]:


# YOUR CODE HERE


# ---
# 1. Does chain-of-thought help? By how much, and is it significant?
# 2. Does the result surprise you? What might explain it?
# 3. Would you expect the same pattern on a different subject?
# 
# **Your answer:**

# ## Bonus assignment: Clustered Standard Errors
# 
# Some benchmarks contain groups of related questions — for example, several questions
# about the same passage in reading comprehension tasks like DROP or RACE. In such cases
# the standard CLT confidence interval is anti-conservative: questions within a group are
# correlated, so the effective sample size is smaller than n. Miller (2024) addresses this
# with clustered standard errors (§2.2).
# 
# Using a reading comprehension benchmark of your choice, implement the clustered
# confidence interval (Eq. 4 from the paper) and compare it against the naive CLT interval.
# How much wider is the clustered interval? Does the difference depend on the benchmark?
# Then compare two models on the same benchmark using the clustered paired standard error
# (Eq. 8) — does the conclusion about which model is better change compared to the
# naive paired test?

# In[ ]:


# YOUR CODE HERE

