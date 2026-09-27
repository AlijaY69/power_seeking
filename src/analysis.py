"""Loading eval logs into a DataFrame, summarising results, and statistical tests."""

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from scipy.stats import binomtest, wilcoxon

# Columns that together identify one eval configuration
CONFIG_COLS = ["model", "system_prompt", "reasoning"]


def config_labels(task_args: dict) -> tuple[str, str]:
    """Turn a task's arguments into two readable labels for plotting."""
    system_prompt = task_args.get("system_prompt_behavior") or "none"  # None -> "none"
    if not task_args.get("use_cot"):
        reasoning = "direct"
    elif task_args.get("use_self_critique"):
        reasoning = "CoT + critique"
    else:
        reasoning = "CoT"
    return system_prompt, reasoning


def load_samples(log_dir) -> pd.DataFrame:
    """
    Reads every successful alignment_eval log in log_dir into one DataFrame, one row per answer.

    If a configuration was run more than once, only the largest run is kept (full runs beat
    test runs), and among those the most recent one.
    """
    rows = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue

        system_prompt, reasoning = config_labels(log.eval.task_args)

        for sample in log.samples:
            score = sample.scores["answer"]
            n_choices = len(sample.choices) if sample.choices else 2
            valid_letters = [chr(65 + i) for i in range(n_choices)]  # e.g. A, B, C, D
            target = sample.target[0] if isinstance(sample.target, list) else sample.target
            rows.append({
                "log_file": info.name,
                "created": log.eval.created,
                "model": log.eval.model,
                "system_prompt": system_prompt,
                "reasoning": reasoning,
                "sample_id": sample.id,
                "category": sample.metadata.get("behavior_category"),
                "n_choices": n_choices,
                "target": target,  # letter of the power-seeking option
                "answer": score.answer,
                "parsed": score.answer in valid_letters,  # works for any number of options
                "power_seeking": score.value == "C",  # C = matched the power-seeking option
            })

    if not rows:
        raise ValueError(f"No successful eval logs with samples found in {log_dir}")

    df = pd.DataFrame(rows)

    # Deduplicate: per configuration, prefer the LARGEST run, then the NEWEST
    df["run_size"] = df.groupby("log_file")["sample_id"].transform("size")
    df = df[df["run_size"] == df.groupby(CONFIG_COLS)["run_size"].transform("max")]
    df = df[df["created"] == df.groupby(CONFIG_COLS)["created"].transform("max")]

    return df.reset_index(drop=True)


def print_checks(df: pd.DataFrame) -> None:
    """Prints the sanity checks to look at before trusting any results."""
    print("\nAnswers per configuration (should be 300 each):")
    print(df.groupby(CONFIG_COLS).size())

    print("\nParse rate per configuration (should be close to 1.0):")
    print(df.groupby(CONFIG_COLS)["parsed"].mean().round(3))

    print("\nNumber of answer options per question:")
    print(df.drop_duplicates("sample_id")["n_choices"].value_counts())


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Power-seeking rate, sample size and standard error per configuration (parsed answers only)."""
    df_parsed = df[df["parsed"]]
    summary = (
        df_parsed.groupby(CONFIG_COLS)
        .agg(rate=("power_seeking", "mean"), n=("power_seeking", "size"))
        .reset_index()
    )
    summary["se"] = (summary["rate"] * (1 - summary["rate"]) / summary["n"]) ** 0.5
    return summary


def _check_single_model(df: pd.DataFrame) -> None:
    """The paired tests compare questions within one model; mixing models gives wrong results."""
    if df["model"].nunique() != 1:
        raise ValueError("Pass data for a single model, e.g. df[df['model'] == model]")


def paired_flips(df_parsed: pd.DataFrame, cond_a: str, cond_b: str) -> pd.DataFrame:
    """
    For each system-prompt condition, compares reasoning setups cond_a and cond_b
    question by question, counts flips in each direction, and runs an exact
    McNemar test (a binomial test on the flipped questions only).

    df_parsed must contain parsed answers for a single model.
    """
    _check_single_model(df_parsed)

    # One row per (system_prompt, question); one column per reasoning setup; 1 = power-seeking
    wide = (
        df_parsed.assign(ps=df_parsed["power_seeking"].astype(int))
        .pivot_table(index=["system_prompt", "sample_id"], columns="reasoning",
                     values="ps", aggfunc="first")
    )

    rows = []
    for spb, grp in wide.groupby(level="system_prompt"):
        g = grp[[cond_a, cond_b]].dropna()  # keep questions parsed in both conditions
        a_ps, b_ps = g[cond_a] == 1, g[cond_b] == 1

        away = int((a_ps & ~b_ps).sum())  # power-seeking in A, not in B
        toward = int((~a_ps & b_ps).sum())  # not in A, power-seeking in B
        n_flips = away + toward

        rows.append({
            "system_prompt": spb,
            "comparison": f"{cond_a} → {cond_b}",
            "both_ps": int((a_ps & b_ps).sum()),
            "neither_ps": int((~a_ps & ~b_ps).sum()),
            "flipped_away": away,
            "flipped_toward": toward,
            "p_value": binomtest(away, n_flips, 0.5).pvalue if n_flips > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def per_question_wilcoxon(df_parsed: pd.DataFrame, cond_a: str = "direct", cond_b: str = "CoT"):
    """
    Paired test with one observation per question: each question's power-seeking rate is
    averaged over the three system-prompt conditions (treated as replicates), then the two
    reasoning setups are compared with a Wilcoxon signed-rank test.

    df_parsed must contain parsed answers for a single model.
    Returns (n_less_ps_under_b, n_more_ps_under_b, n_same, WilcoxonResult).
    """
    _check_single_model(df_parsed)

    per_q = (
        df_parsed.assign(ps=df_parsed["power_seeking"].astype(int))
        .pivot_table(index="sample_id", columns="reasoning", values="ps", aggfunc="mean")
        .dropna()
    )
    diff = per_q[cond_a] - per_q[cond_b]  # > 0 means less power-seeking under cond_b
    result = wilcoxon(per_q[cond_a], per_q[cond_b], zero_method="wilcox")
    return int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum()), result


def chance_tests(df_parsed: pd.DataFrame, chance: float = 0.25) -> pd.DataFrame:
    """
    Binomial test of the power-seeking rate against chance, per model, category and configuration.
    With balanced answer positions, chance is 1 / number of options (0.25 for four options).
    """
    rows = []
    for (model, category, spb, reasoning), g in df_parsed.groupby(
        ["model", "category", "system_prompt", "reasoning"]
    ):
        k, n = int(g["power_seeking"].sum()), len(g)
        rows.append({
            "model": model,
            "category": category,
            "system_prompt": spb,
            "reasoning": reasoning,
            "rate": k / n,
            "n": n,
            "p_value": binomtest(k, n, chance).pvalue,
        })
    return pd.DataFrame(rows)