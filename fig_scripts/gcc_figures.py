"""Paper figures and tables, generated from the `analyzed` dict.

    import gcc_figures as F
    F.emit_all(analyzed, out_dir="paper_out")

Everything here reads what analyze_all/compute_paper_metrics already produced:
both survival curves come from analyzed["first_blunder"]["km"] and
analyzed["ponr"]["km"], and the trajectory bands from
analyzed["trajectories"], so a figure cannot disagree with the numbers the
dashboard reports. km_fallback() rebuilds a curve for older analyzed dicts that
predate the dashboard computing it, and warns loudly when it does.

Outputs (5.5in text block, 10pt Times):
    table1_outcomes.tex     W--D--L per cell, pooled score, cost
    table2_move_quality.tex Win% judgment mix + best-move rate
    fig1_failure.pdf        first blunder | point of no return, side by side
    fig2_trajectory.pdf     median Win% over the game, with interquartile band
"""
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODES = ("single", "multi", "hardcoded_multi")
DIFFS = ("low", "medium", "high")
BANDS = ("ok", "inaccuracy", "mistake", "blunder")
MODE_LABEL = {"single": "SAS", "multi": "Generated MAS",
              "hardcoded_multi": "Handcrafted MAS"}
MODE_SHORT = {"single": "SAS", "multi": "Gen. MAS", "hardcoded_multi": "Hand. MAS"}
# Okabe-Ito: colourblind-safe and still separable in greyscale print.
MODE_COLOR = {"single": "#0072B2", "multi": "#D55E00", "hardcoded_multi": "#009E73"}
# Categories that are not a played-out game result.
UNFINISHED = ("ply_cap_incomplete", "illegal_move_failure", "consensus_failure",
              "context_overflow")

FIG_WIDTH_IN = 5.565      # tight-bbox output lands on 396pt = 5.5in exactly
FIG_BASE_PT = 9           # "small": one point under the 10pt body
FONT_STACK = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


# ---------------------------------------------------------------------------
# What this sweep actually contains
# ---------------------------------------------------------------------------
# analyze_all summarises every mode and difficulty in its ORDER constants, so a
# group key exists even when nothing ran in it -- with n=0 and None statistics.
# Presence therefore has to be tested on n, not on the key. A prototype sweep
# typically has no single-agent arm and may cover one difficulty, so everything
# below iterates what is present rather than the full MODES/DIFFS.

def present_modes(analyzed: Dict[str, Any]) -> tuple:
    groups = analyzed.get("groups", {})
    return tuple(m for m in MODES if (groups.get(f"mode:{m}") or {}).get("n", 0))


def present_diffs(analyzed: Dict[str, Any]) -> tuple:
    groups = analyzed.get("groups", {})
    return tuple(d for d in DIFFS
                 if any((groups.get(f"cell:{d}:{m}") or {}).get("n", 0)
                        for m in present_modes(analyzed)))


# ---------------------------------------------------------------------------
# Survival curves
# ---------------------------------------------------------------------------

def km_fallback(runs: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """Kaplan-Meier curve built here, for analyzed dicts that predate the
    dashboard computing it. Emits a warning at the call site -- the figures are
    meant to render the dashboard's numbers, and silently substituting a second
    implementation is how the two drift apart.

    A run whose event field is None ended without the event and is CENSORED at
    last_sys_ply; dropping those instead biases the estimate early, because the
    games that never had the event are disproportionately the longest-lived.
    """
    exits = []
    for r in runs:
        last = r.get("last_sys_ply")
        if last is None:
            continue                      # never moved: no exposure
        at = r.get(field)
        exits.append((int(at), True) if at is not None else (int(last), False))

    # points is a LIST of per-ply dicts, matching the dashboard's km blocks
    # exactly, so downstream code never has to know which one it is holding.
    points = [{"ply": 0, "at_risk": len(exits), "events": 0, "F": 0.0}]
    surviving = 1.0
    for ply in sorted({p for p, had in exits if had}):
        at_risk = sum(1 for p, _ in exits if p >= ply)
        events = sum(1 for p, had in exits if had and p == ply)
        surviving *= (1 - events / at_risk)
        points.append({"ply": ply, "at_risk": at_risk, "events": events,
                       "F": 1 - surviving})
    return {"points": points,
            "n": len(exits),
            "risk_grid": [{"ply": t, "at_risk": sum(1 for p, _ in exits if p >= t)}
                          for t in range(10, 201, 10)]}


def at_risk_at(km_block: Dict[str, Any], ply: int) -> Optional[int]:
    """Games still under observation at `ply`, from the dashboard's risk grid.

    The grid is stored every 10 plies, so any readout grid built from multiples
    of 10 resolves; ply 0 is the whole group by definition. Returns None if the
    ply is off the grid, which the caller renders as a dash rather than guessing.
    """
    if ply == 0:
        return km_block.get("n")
    for row in km_block.get("risk_grid", []):
        if row["ply"] == ply:
            return row["at_risk"]
    return None


def km_from_points(points: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Re-key the dashboard's stored KM points by ply."""
    return {p["ply"]: {"F": p["F"], "at_risk": p["at_risk"], "events": p["events"]}
            for p in points}


def quantile_ply(curve: Dict[int, Dict[str, Any]], q: float = 0.5) -> Optional[int]:
    """First ply at which the estimate reaches q.

    The tolerance is load-bearing: F is a running product, so an exact half --
    15 events out of 30 -- comes out as 0.4999999999999999 and a bare `>= 0.5`
    steps past the true median to the next event ply.
    """
    for ply in sorted(curve):
        if curve[ply]["F"] >= q - 1e-9:
            return ply
    return None


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def _style(base_pt: int):
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42,               # TrueType; many venues reject Type 3
        "font.family": "serif", "font.serif": FONT_STACK,
        "mathtext.fontset": "stix", "font.size": base_pt,
        "axes.labelsize": base_pt, "legend.fontsize": base_pt - 1,
        "xtick.labelsize": base_pt - 1, "ytick.labelsize": base_pt - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "lines.linewidth": 1.3,
        # constrained_layout OFF: the at-risk rows are offset-point annotations
        # and it will not reserve space for them. subplots_adjust does.
        "figure.constrained_layout.use": False,
    })
    return plt


def fig_failure_dynamics(analyzed: Dict[str, Any], path: Path,
                         grid_step: int = 50, base_pt: int = FIG_BASE_PT,
                         width_in: float = FIG_WIDTH_IN, height_in: float = 3.5):
    """First blunder and point of no return, side by side on a shared y-axis.

    Sharing the axis is the point: the horizontal offset between a system's two
    curves is the lag from erring to being unrecoverably lost.

    Each line stops at that system's last observed event -- it is NOT extended to
    the end of follow-up, so a line ending means "no game had a first event after
    this", not "observation stopped". The at-risk row says how many games each
    stretch rests on. No confidence bands: they are pointwise, so curves whose
    bands overlap can still separate significantly, and showing them beside a
    log-rank result invites the opposite reading.
    """
    plt = _style(base_pt)
    from matplotlib.ticker import MultipleLocator, PercentFormatter
    from matplotlib.lines import Line2D

    runs = analyzed["runs"]
    modes = present_modes(analyzed)
    by_mode = {m: [r for r in runs if r["mode"] == m] for m in modes}

    def blocks(section, field, what):
        """The dashboard's km blocks for one event, or a warned-about fallback."""
        stored = analyzed.get(section, {}).get("km", {})
        out = {}
        for m in modes:
            key = f"mode:{m}"
            if key in stored:
                out[m] = stored[key]
            else:
                warnings.warn(
                    f"analyzed[{section!r}] has no {key!r}: rebuilding the "
                    f"{what} curve inside gcc_figures. Re-run the dashboard so "
                    f"the figure and the reported numbers share one "
                    f"implementation.", RuntimeWarning, stacklevel=3)
                out[m] = km_fallback(by_mode[m], field)
        return out

    fb = blocks("first_blunder", "first_blunder_ply", "first-blunder")
    pr = blocks("ponr", "ponr_ply", "point-of-no-return")
    panels = [("First blunder", fb), ("Point of no return", pr)]

    xmax = max(p["ply"] for _, blk in panels for b in blk.values()
               for p in b["points"])
    xmax = grid_step * (int(xmax) // grid_step + 1)
    grid = list(range(0, int(xmax) + 1, grid_step))
    row0, rowh = 30, 11

    fig, axes = plt.subplots(1, len(panels), figsize=(width_in, height_in),
                             dpi=200, sharey=True)
    # everything below the axes lives INSIDE the canvas, so it survives
    # plt.show() as well as savefig; negative figure coords are captured only by
    # bbox_inches='tight' and vanish silently in an interactive window.
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.40, wspace=0.08)

    for ax, (title, blk) in zip(axes, panels):
        for m in modes:
            curve = km_from_points(blk[m]["points"])
            plies = sorted(curve)
            # where="post": F holds until the next event. A straight line between
            # events would imply risk accrued smoothly between them.
            ax.step(plies, [100 * curve[p]["F"] for p in plies], where="post",
                    color=MODE_COLOR[m], label=MODE_LABEL[m])
        ax.set_title(title, fontsize=base_pt)
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, xmax)
        ax.set_xticks(grid)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        for k, m in enumerate(modes):
            for gp in grid:
                n = at_risk_at(blk[m], gp)
                ax.annotate("--" if n is None else str(n),
                            xy=(gp, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -(row0 + rowh * k)),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=base_pt - 2, color=MODE_COLOR[m])

    axes[0].set_ylabel("Games reached")
    # row labels on the left panel only: they name the same systems in both
    axes[0].annotate("Runs at risk", xy=(0, 0), xycoords="axes fraction",
                     xytext=(-12, -(row0 - rowh)), textcoords="offset points",
                     ha="right", va="top", fontsize=base_pt - 2, color="0.35")
    for k, m in enumerate(modes):
        axes[0].annotate(MODE_SHORT[m], xy=(0, 0), xycoords="axes fraction",
                         xytext=(-12, -(row0 + rowh * k)),
                         textcoords="offset points", ha="right", va="top",
                         fontsize=base_pt - 2, color=MODE_COLOR[m])
    fig.legend(handles=[Line2D([], [], color=MODE_COLOR[m], label=MODE_LABEL[m])
                        for m in modes],
               frameon=False, loc="lower center", ncol=len(modes),
               bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Ply", fontsize=base_pt, y=0.095)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def fig_trajectory(analyzed: Dict[str, Any], path: Path, grid_step: int = 20,
                   band: bool = True, min_n: Optional[int] = None,
                   max_ply: Optional[int] = None, base_pt: int = FIG_BASE_PT,
                   width_in: float = FIG_WIDTH_IN, height_in: float = 3.3):
    """Winning chances over the game: median and interquartile band per system.

    Straight from analyzed["trajectories"], which samples every second ply and
    carries `n` -- the games still running at that ply -- alongside the
    quantiles. Nothing is recomputed here.

    The row underneath is the counterpart of the survival figures' at-risk row,
    but the games have not "left a risk set": they simply finished, so it is
    labelled "Still playing". It matters for the same reason: n falls steeply,
    and without it the thin right-hand tail reads as well supported as the left.

    min_n trims to plies with at least that many games still running; max_ply
    truncates instead. For the paper, min_n=5 is a reasonable floor.

    grid_step must be EVEN: eval_trajectory_bands samples every second ply, so
    an odd readout ply has no sample and would print as a dash.
    """
    if grid_step % 2:
        raise ValueError(f"grid_step must be even (trajectories are sampled "
                         f"every 2 plies); got {grid_step}")
    plt = _style(base_pt)
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    modes = present_modes(analyzed)
    series = {m: (analyzed["trajectories"].get(f"mode:{m}") or []) for m in modes}
    n_at = {m: {p["ply"]: p["n"] for p in series[m]} for m in modes}
    row0, rowh = 30, 11

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.45)
    xmax = 0
    for m in modes:
        pts = [p for p in series[m]
               if (min_n is None or p["n"] >= min_n)
               and (max_ply is None or p["ply"] <= max_ply)]
        if not pts:
            continue
        xs = [p["ply"] for p in pts]
        xmax = max(xmax, xs[-1])
        if band:
            ax.fill_between(xs, [p["p25_wp"] for p in pts],
                            [p["p75_wp"] for p in pts],
                            color=MODE_COLOR[m], alpha=0.15, linewidth=0)
        ax.plot(xs, [p["median_wp"] for p in pts], color=MODE_COLOR[m],
                label=MODE_LABEL[m])
    # round UP to a tick, but leave an xmax already on one alone
    xmax = grid_step * -(-int(xmax) // grid_step)
    grid = list(range(0, int(xmax) + 1, grid_step))

    # 50% is the balance point; 25-point ticks put a gridline exactly on it
    ax.axhline(50, color="0.4", linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_ylabel("Win (LLM side)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, xmax)
    ax.set_xticks(grid)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    ax.annotate("Still playing", xy=(0, 0), xycoords="axes fraction",
                xytext=(-12, -(row0 - rowh)), textcoords="offset points",
                ha="right", va="top", fontsize=base_pt - 2, color="0.35")
    for k, m in enumerate(modes):
        ax.annotate(MODE_SHORT[m], xy=(0, 0), xycoords="axes fraction",
                    xytext=(-12, -(row0 + rowh * k)), textcoords="offset points",
                    ha="right", va="top", fontsize=base_pt - 2,
                    color=MODE_COLOR[m])
        for gp in grid:
            # ply 0 predates the first sample, where every game is still running
            n = len([r for r in analyzed["runs"] if r["mode"] == m]) if gp == 0 \
                else n_at[m].get(gp)
            ax.annotate("0" if n is None else str(n),
                        xy=(gp, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -(row0 + rowh * k)),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=base_pt - 2, color=MODE_COLOR[m])
    # below the "Ply" label, which sits below the still-playing rows
    ax.legend(frameon=False, loc="lower center", ncol=len(modes),
              bbox_to_anchor=(0.5, -0.75))
    ax.annotate("Ply", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -(row0 + rowh * len(modes) + 4)),
                textcoords="offset points", ha="center", va="top",
                fontsize=base_pt)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _tokens(x: Optional[float]) -> str:
    if x is None:
        return "--"
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"


def table1_outcomes(analyzed: Dict[str, Any], baseline: str = "single") -> str:
    r"""Outcomes per (difficulty x system), with pooled score and cost.

    Counts, not percentages: at n=10 a percentage is a multiple of 10 and only
    hides the denominator, and the cells are not all the same size. Unfinished
    runs ride in parentheses rather than taking their own columns, which would
    overflow the 5.5in text block.

    Rows and columns follow what the sweep contains. A prototype sweep with no
    single-agent arm simply has no such row, and the cost multiplier is taken
    against the cheapest system present instead -- expressing every system as a
    multiple of an arm that was not run would be meaningless.
    """
    groups = analyzed["groups"]
    modes, diffs = present_modes(analyzed), present_diffs(analyzed)

    def cell(d, m):
        g = groups.get(f"cell:{d}:{m}")
        return g if (g or {}).get("n", 0) else None

    def pooled(m):
        g = groups[f"mode:{m}"]
        cats = g.get("categories", {})
        return {"score": g["score_mean"],
                "tokens": (g["tokens_per_game"] or {}).get("mean"),
                "unfin": sum(cats.get(c, 0) for c in UNFINISHED)}

    stats = {m: pooled(m) for m in modes}
    ref = baseline if baseline in modes else min(
        modes, key=lambda m: stats[m]["tokens"] or float("inf"), default=None)
    base_tokens = stats[ref]["tokens"] if ref else None

    lines = [r"% requires \usepackage{booktabs}",
             r"\begin{tabular}{l" + "c" * len(diffs) + r"cr}", r"\toprule",
             "System & " + " & ".join(d.capitalize() for d in diffs)
             + r" & Score & Tokens/game \\", r"\midrule"]

    for m in modes:
        row = [MODE_LABEL[m]]
        for d in diffs:
            g = cell(d, m)
            if not g:
                row.append("--")
                continue
            cats = g.get("categories", {})
            unfin = sum(cats.get(c, 0) for c in UNFINISHED)
            row.append(f"{g['wins']}--{g['draws']}--{g['losses']} ({unfin})")
        p = stats[m]
        mult = (p["tokens"] / base_tokens
                if base_tokens and p["tokens"] else 1.0)
        row += [f"{p['score']:.2f}" if p["score"] is not None else "--",
                _tokens(p["tokens"])
                + (rf" ($\times${mult:.0f})" if mult > 1.5 else "")]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table2_move_quality(analyzed: Dict[str, Any]) -> str:
    r"""Win% judgment mix and best-move rate, pooled over all system moves.

    Percentages ARE right here, unlike Table 1: the denominator is ~1,600 moves,
    so a share is not a count in disguise. No confidence intervals -- the unit is
    the move and moves within a game are strongly correlated, so a binomial
    interval over thousands of pooled moves would understate the real
    uncertainty. The game-level test belongs in the text.
    """
    groups, per_mode = analyzed["groups"], analyzed["paper"]["per_mode"]
    lines = [r"% requires \usepackage{booktabs}",
             r"\begin{tabular}{lr" + "r" * len(BANDS) + r"r}", r"\toprule",
             r"System & Moves & " + " & ".join(b.capitalize() for b in BANDS)
             + r" & Best move \\", r"\midrule"]
    for m in present_modes(analyzed):
        g = groups[f"mode:{m}"]
        counts, total = g["quality_wp"], g["pooled_sys_moves"]
        if not total:
            continue
        best = per_mode.get(m, {}).get("best_rate")
        lines.append(" & ".join(
            [MODE_LABEL[m], f"{total:,}"]
            + [rf"{100 * counts.get(b, 0) / total:.1f}\%" for b in BANDS]
            + [rf"{100 * best:.1f}\%" if best is not None else "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def summary(analyzed: Dict[str, Any]) -> str:
    """The numbers the prose quotes, so they cannot drift from the figures."""
    runs = analyzed["runs"]
    modes = present_modes(analyzed)
    by_mode = {m: [r for r in runs if r["mode"] == m] for m in modes}
    stored_km = analyzed["first_blunder"]["km"]
    stored_pr = analyzed.get("ponr", {}).get("km", {})
    # log-rank is computed for every mode pair in the ORDER constant, so pairs
    # involving an arm this sweep did not run come back with chi2/p of None.
    logrank = {(t["a"], t["b"]): t for t in analyzed["first_blunder"]["logrank"]
               if t["kind"] == "mode" and t.get("p") is not None
               and t["a"] in modes and t["b"] in modes}

    out = ["system            n   W- D- L  unfin  score   tokens/game  "
           "med.1st-blunder  med.PoNR  ever-blunder"]
    for m in modes:
        g = analyzed["groups"][f"mode:{m}"]
        cats = g.get("categories", {})
        unfin = sum(cats.get(c, 0) for c in UNFINISHED)
        fb = km_from_points(stored_km[f"mode:{m}"]["points"])
        if f"mode:{m}" in stored_pr:
            pr = km_from_points(stored_pr[f"mode:{m}"]["points"])
        else:
            warnings.warn(
                f"analyzed['ponr'] has no 'mode:{m}': rebuilding the "
                f"point-of-no-return curve inside gcc_figures. Re-run the "
                f"dashboard so the summary and the reported numbers share one "
                f"implementation.", RuntimeWarning, stacklevel=2)
            pr = km_from_points(km_fallback(by_mode[m], "ponr_ply")["points"])
        out.append(f"{MODE_LABEL[m]:16} {g['n']:>2}  {g['wins']:>2}-{g['draws']:>2}"
                   f"-{g['losses']:>2}  {unfin:>4}   {g['score_mean']:.2f}  "
                   f"{_tokens(g['tokens_per_game']['mean']):>11}  "
                   f"{str(quantile_ply(fb)):>14}  {str(quantile_ply(pr)):>8}  "
                   f"{100 * stored_km[f'mode:{m}']['plateau']:>10.1f}%")
    if logrank:
        out.append("")
    for (a, b), t in logrank.items():
        out.append(f"  log-rank (first blunder) {MODE_LABEL[a]:>16} vs "
                   f"{MODE_LABEL[b]:<16} chi2={t['chi2']:.2f}  p={t['p']:.4f}")
    return "\n".join(out)


def emit_all(analyzed: Dict[str, Any], out_dir: str = "paper_out") -> List[Path]:
    """Write every paper artifact and print the prose numbers.

    Rows, columns and curves follow whichever systems and difficulties the sweep
    actually contains, so a prototype set with no single-agent arm needs no flag.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for name, text in (("table1_outcomes.tex", table1_outcomes(analyzed)),
                       ("table2_move_quality.tex", table2_move_quality(analyzed))):
        (out / name).write_text(text + "\n", encoding="utf-8")
        written.append(out / name)

    written.append(fig_failure_dynamics(analyzed, out / "fig1_failure.pdf"))
    if analyzed.get("trajectories"):
        written.append(fig_trajectory(analyzed, out / "fig2_trajectory.pdf"))

    (out / "summary.txt").write_text(summary(analyzed) + "\n", encoding="utf-8")
    written.append(out / "summary.txt")

    print(summary(analyzed))
    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    return written
