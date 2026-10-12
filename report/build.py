"""Build every table and figure the report uses from saved result files and configs only.

Outputs report/tables/*.tex (\\input by report.tex) and copies run figures into report/figures/.
Tables hold numbers only; captions live in report.tex.
usage: PYTHONPATH=. python report/build.py
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
TABLES = REPO / "report" / "tables"
FIGURES = REPO / "report" / "figures"


def rd(path: str):
    p = RES / path
    return json.loads(p.read_text()) if p.suffix == ".json" else [json.loads(l) for l in p.read_text().splitlines()]


def cfg(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body.strip() + "\n")


def sci(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def f3(x) -> str:
    return f"{x:.3f}"


def gen_stats(name: str) -> dict:
    rows = rd(f"generations/{name}/generations.jsonl")
    return {"n": len(rows), "tokens": st.mean(r["n_new_tokens"] for r in rows)}


def t_data() -> None:
    d = rd("data/summary.json")
    s = d["summary"]
    persona = json.loads((REPO / "data/persona_lora/ramsay_style_preferences.json").read_text())
    hold = cfg("lora_persona")["holdout"]
    body = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
            r" & \multicolumn{2}{c}{Split size} & \multicolumn{3}{c}{Mean tokens} & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
            r"Track & Train & Val & Prompt & Chosen & Rejected & Cut at 512 \\", r"\midrule"]
    for track, src, key, resp in (("SFT", "UltraChat-200k", "sft", "response"), ("HH-DPO", "HH-RLHF", "hh", "chosen"),
                                  ("UF-DPO", "UltraFeedback", "uf", "chosen")):
        tr, va = s[f"{key}_train"], s[f"{key}_val"]
        rej = tr.get("mean_rejected_tokens")
        body.append(f"{track} & {tr['n']:,} & {va['n']:,} & {tr['mean_prompt_tokens']:.0f} & "
                    f"{tr[f'mean_{resp}_tokens']:.0f} & {'--' if rej is None else f'{rej:.0f}'} & "
                    f"{100 * tr[f'{resp}_truncated_rate']:.1f}"
                    f"{'' if rej is None else ' / ' + format(100 * tr['rejected_truncated_rate'], '.1f')}\\% \\\\")
    body.append(f"LoRA-DPO & {len(persona) - hold} & {hold} & -- & -- & -- & -- \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("data.tex", "\n".join(body))


def t_config() -> None:
    sft, hh, uf, lo, gen = cfg("sft"), cfg("dpo_hh"), cfg("dpo_uf"), cfg("lora_persona"), cfg("generate")
    per = rd("persona/summary.json")
    s_sft, s_hh, s_uf = rd("sft/summary.json"), rd("dpo_hh/summary.json"), rd("dpo_uf/summary.json")
    ref = f"SFT step {s_sft['summary']['best_step']}"
    assert hh["init_sha256"] == uf["init_sha256"] == s_sft["best_sha256"]
    lora = f"LoRA, {per['trainable_params'] / 1e6:.2f}M ({100 * per['trainable_params'] / per['total_params']:.2f}\\%)"
    rows = [
        ("Trainable parameters", "full", "full", "full", lora),
        ("LoRA rank, $\\alpha$, dropout", "--", "--", "--", f"{lo['lora_r']}, {lo['lora_alpha']}, {lo['lora_dropout']}"),
        ("LoRA target modules", "--", "--", "--", "all linear layers"),
        ("Optimizer", "AdamW", "RMSprop", "RMSprop", "AdamW"),
        ("Peak learning rate", sci(sft["lr"]), sci(hh["lr"]), sci(uf["lr"]), sci(lo["lr"])),
        ("Schedule, warmup", f"cosine, {100 * sft['warmup_ratio']:.0f}\\%", "constant, none", "constant, none",
         f"{lo['lr_scheduler']}, {100 * lo['warmup_ratio']:.0f}\\%"),
        ("Weight decay", sft["weight_decay"], "0", "0", "0"),
        ("Gradient clipping", sft["grad_clip"], hh["grad_clip"], uf["grad_clip"], "1.0"),
        ("Batch $\\times$ accumulation", f"{sft['per_device_batch']} $\\times$ {sft['grad_accum']}", f"1 $\\times$ {hh['grad_accum']}",
         f"1 $\\times$ {uf['grad_accum']}", f"{lo['per_device_batch']} $\\times$ {lo['grad_accum']}"),
        ("Effective batch", sft["per_device_batch"] * sft["grad_accum"], hh["grad_accum"], uf["grad_accum"],
         lo["per_device_batch"] * lo["grad_accum"]),
        ("Precision", "fp16 mixed", "fp16 mixed", "fp16 mixed", per["precision"]),
        ("Max length (prompt)", f"{sft['max_length']} (256)", f"{hh['max_length']} (256)", f"{uf['max_length']} (256)",
         f"{lo['max_length']} ({lo['max_prompt_length']})"),
        ("Epochs (optimizer steps)", f"{sft['epochs']} ({s_sft['summary']['n_steps']})", f"{hh['epochs']} ({s_hh['summary']['n_steps']})",
         f"{uf['epochs']} ({s_uf['summary']['n_steps']})", f"{lo['epochs']} ({rd('persona/log_history.jsonl')[-1]['step']})"),
        ("DPO $\\beta$", "--", hh["beta"], uf["beta"], lo["beta"]),
        ("Reference model", "--", ref, ref, "adapter disabled"),
        ("Seed", sft["seed"], hh["seed"], uf["seed"], lo["seed"]),
    ]
    body = [r"\begin{tabular}{l l l l l}", r"\toprule", r" & SFT & HH-DPO & UF-DPO & LoRA-DPO \\", r"\midrule"]
    body += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    dec = [("Decoding", "greedy"), ("Temperature, top-$p$", "not used"), ("Max new tokens", gen["max_new_tokens"]),
           ("Repetition penalty", gen["repetition_penalty"]), ("Length penalty", "none")]
    body += [r"\midrule", r"\multicolumn{5}{l}{\textit{Evaluation decoding}} \\"]
    body += [f"{k} & " + " & ".join([str(v)] * 4) + r" \\" for k, v in dec]
    body += [r"Chat template & Qwen default & Qwen default & Qwen default & SmolLM2 default \\", r"\bottomrule", r"\end{tabular}"]
    write("config.tex", "\n".join(body))


def t_training() -> None:
    sft, hh, uf, per = rd("sft/summary.json"), rd("dpo_hh/summary.json"), rd("dpo_uf/summary.json"), rd("persona/summary.json")
    body = [r"\begin{tabular}{l c c c c c c}", r"\toprule",
            r"Run & Step 0 & Best step & Best & Final & $\Delta\log\pi_\theta(y_w)$ & $\Delta\log\pi_\theta(y_l)$ \\",
            r"\midrule",
            (f"SFT & {sft['summary']['first_val_loss']:.4f} & {sft['summary']['best_step']} & "
             f"{sft['summary']['best_val_loss']:.4f} & {sft['summary']['final_val_loss']:.4f} & -- & -- \\\\")]
    for name, s in (("HH-DPO", hh), ("UF-DPO", uf)):
        m = s["summary"]
        body.append(f"{name} & {m['step0']['val_accuracy']:.3f} / {m['step0']['val_margin']:.3f} & {m['best_step']} & "
                    f"{m['best_val_accuracy']:.3f} / {m['best_val_margin']:.3f} & {m['final_val_accuracy']:.3f} / {m['final_val_margin']:.3f} & "
                    f"${m['delta_logp_policy_w_at_best']:+.2f}$ & ${m['delta_logp_policy_l_at_best']:+.2f}$ \\\\")
    s0, ev = per["step0_eval"], per["summary"]["evals"]
    body.append(f"LoRA-DPO & {s0['eval_rewards/accuracies']:.3f} / {s0['eval_rewards/margins']:.3f} & "
                f"{rd('persona/log_history.jsonl')[-1]['step']} & {ev[-1]['eval_rewards/accuracies']:.3f} / {ev[-1]['eval_rewards/margins']:.3f} & "
                f"{ev[-1]['eval_rewards/accuracies']:.3f} / {ev[-1]['eval_rewards/margins']:.3f} & "
                f"${ev[-1]['eval_logps/chosen'] - s0['eval_logps/chosen']:+.2f}$ & -- \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("training.tex", "\n".join(body))


JOBS = [("HH-DPO vs SFT", "hh_dpo_vs_sft", "safety"), ("UF-DPO vs SFT", "uf_dpo_vs_sft", "general"),
        ("LoRA vs base", "lora_vs_base", "style"), ("LoRA vs base", "lora_vs_base", "usefulness")]


def t_results() -> None:
    cols = []
    for label, name, rubric in JOBS:
        s = rd(f"eval/{name}/summary_{rubric}.json")["summary"]
        cols.append([s["n"], f"{s['win']} / {s['tie']} / {s['loss']}", f"{s['unknown']} / {s['failure']}",
                     s["retried_pairs"] + s["retried_pairs_swapped"], f3(s["win_rate_with_ties"]),
                     f3(s["win_rate_without_ties"])])
    names = ["Pairs $n$", "Wins / ties / losses", "Unknown / failures", "Pairs retried", "Win rate (ties as half)",
             "Win rate without ties"]
    body = [r"\begin{tabular}{l c c c c}", r"\toprule",
            r" & HH-DPO vs SFT & UF-DPO vs SFT & \multicolumn{2}{c}{LoRA-DPO vs SmolLM2} \\",
            r"\cmidrule(lr){4-5}",
            r"Rubric & safety & general & style & usefulness \\", r"\midrule"]
    body += [f"{n} & " + " & ".join(str(c[i]) for c in cols) + r" \\" for i, n in enumerate(names)]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("results.tex", "\n".join(body))


def by_cat(name: str, rubric: str) -> dict:
    return rd(f"eval/{name}/summary_{rubric}.json")["summary"]["by_category"]


def cat_tokens(gen: str) -> dict:
    out: dict = {}
    for r in rd(f"generations/{gen}/generations.jsonl"):
        out.setdefault(r.get("category") or "all", []).append(r["n_new_tokens"])
    return {k: st.mean(v) for k, v in out.items()}


def t_categories() -> None:
    hh = by_cat("hh_dpo_vs_sft", "safety")
    cats = json.loads((REPO / "eval_sets/hh_rlhf_eval_300_seed42.json").read_text())
    cat_of = {r["id"]: r["category"] for r in cats}
    tok = {}
    for gen in ("sft__hh_rlhf_eval", "dpo-hh__hh_rlhf_eval"):
        acc: dict = {}
        for r in rd(f"generations/{gen}/generations.jsonl"):
            acc.setdefault(cat_of[r["id"]], []).append(r["n_new_tokens"])
        tok[gen] = {k: st.mean(v) for k, v in acc.items()}
    order = sorted(hh, key=lambda c: -hh[c]["win_rate_with_ties"])
    body = [r"\begin{tabular}{l c c r r}", r"\toprule", r"Category & Wins / ties / losses & Win rate & SFT tokens & DPO tokens \\", r"\midrule"]
    for c in order:
        v = hh[c]
        body.append(f"{c.replace('_', ' ')} & {v['win']} / {v['tie']} / {v['loss']} & {f3(v['win_rate_with_ties'])} & "
                    f"{tok['sft__hh_rlhf_eval'][c]:.0f} & {tok['dpo-hh__hh_rlhf_eval'][c]:.0f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("hh_categories.tex", "\n".join(body))

    sty, use = by_cat("lora_vs_base", "style"), by_cat("lora_vs_base", "usefulness")
    body = [r"\begin{tabular}{l c c c c}", r"\toprule",
            r"Category & \multicolumn{2}{c}{Style} & \multicolumn{2}{c}{Usefulness} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r" & W / T / L & Win rate & W / T / L & Win rate \\", r"\midrule"]
    for c in sorted(sty):
        a, b = sty[c], use[c]
        body.append(f"{c.replace('_', ' ')} & {a['win']} / {a['tie']} / {a['loss']} & {f3(a['win_rate_with_ties'])} & "
                    f"{b['win']} / {b['tie']} / {b['loss']} & {f3(b['win_rate_with_ties'])} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("persona_categories.tex", "\n".join(body))


def t_generation() -> None:
    sets = [("SFT, AlpacaEval", ["sft__alpaca_eval"]), ("UF-DPO, AlpacaEval", ["dpo-uf__alpaca_eval"]),
            ("SFT, HH eval", ["sft__hh_rlhf_eval"]), ("HH-DPO, HH eval", ["dpo-hh__hh_rlhf_eval"]),
            ("SmolLM2 base, persona", ["smollm2-360m-instruct__persona_style"]),
            ("LoRA-DPO, persona", ["lora-persona__persona_style"])]
    body = [r"\begin{tabular}{l r r}", r"\toprule", r"Outputs & $n$ & Mean new tokens \\", r"\midrule"]
    for label, names in sets:
        g = [gen_stats(n) for n in names]
        n = sum(x["n"] for x in g)
        toks = sum(x["tokens"] * x["n"] for x in g) / n
        body.append(f"{label} & {n} & {toks:.1f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("generation.tex", "\n".join(body))


def wr(outs: list) -> tuple[float, int]:
    w, t, l = outs.count("win"), outs.count("tie"), outs.count("loss")
    return (w + 0.5 * t) / (w + t + l), w + t + l


def t_length() -> dict:
    cols = []
    for run in ("dpo_hh", "dpo_uf"):
        best = rd(f"{run}/summary.json")["summary"]["best_step"]
        rows = rd(f"{run}/val/step_{best:04d}.jsonl")
        lo = [r for r in rows if r["len_w"] > r["len_l"]]
        sh = [r for r in rows if r["len_w"] < r["len_l"]]
        corr = st.correlation([r["len_w"] - r["len_l"] for r in rows], [r["margin"] for r in rows])
        cols.append([best, f"{sum(r['correct'] for r in lo) / len(lo):.3f} ({len(lo)})",
                     f"{sum(r['correct'] for r in sh) / len(sh):.3f} ({len(sh)})", f"{corr:.3f}"])
    names = ["Selected step", "Accuracy, chosen longer ($n$)", "Accuracy, chosen shorter ($n$)",
             "Correlation, length gap and margin"]
    body = [r"\begin{tabular}{l c c}", r"\toprule", r" & HH-DPO & UF-DPO \\", r"\midrule"]
    body += [f"{n} & {cols[0][i]} & {cols[1][i]} \\\\" for i, n in enumerate(names)]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("length.tex", "\n".join(body))
    return {}


def numbers(extra: dict) -> None:
    vals = {"hhHarmless": f"{rd('data/summary.json')['summary']['hh_train']['subsets']['harmless-base']:,}",
            "personaEpOneMargin": f3(rd("persona/summary.json")["summary"]["evals"][0]["eval_rewards/margins"]),
            **extra}
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()]
    write("numbers.tex", "\n".join(lines))


def figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in ("sft_loss.png", "dpo_hh.png", "dpo_uf.png"):
        shutil.copy2(RES / "figures" / f, FIGURES / f)


if __name__ == "__main__":
    t_data()
    t_config()
    t_training()
    t_results()
    t_categories()
    t_generation()
    numbers(t_length())
    figures()
    print("tables:", sorted(p.name for p in TABLES.glob("*.tex")), "figures:", sorted(p.name for p in FIGURES.glob("*.png")))
