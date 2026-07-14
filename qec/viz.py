"""Plotting helpers for the repetition-code experiments."""
import matplotlib.pyplot as plt
import numpy as np


def plot_logical_vs_physical(df_corrected, df_uncoded, title, path, df_uncorrected=None,
                              code_label="code, corrected", theory_prefactor=3.0):
    """Classic QEC benchmark: logical error rate vs physical error rate, log-log.

    Below threshold the corrected curve should sit under the y=x (uncoded)
    line, bending over towards a p^2 slope. theory_prefactor is the leading
    combinatorial coefficient of the p^2 term (3 for the repetition code;
    pass a different value, or None to omit the curve, for other codes).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    p = df_corrected["p"].to_numpy()
    ax.plot(p, df_uncoded["logical_error_rate"], "o--", color="tab:gray", label="no code (bare qubit)")
    if df_uncorrected is not None:
        ax.plot(p, df_uncorrected["logical_error_rate"], "s--", color="tab:orange",
                 label="encoded, syndrome measured, no correction applied")
    ax.plot(p, df_corrected["logical_error_rate"], "o-", color="tab:blue", label=code_label)

    p_fine = np.linspace(min(p[p > 0]) if (p > 0).any() else 1e-3, max(p), 200)
    if theory_prefactor is not None:
        ax.plot(p_fine, theory_prefactor * p_fine ** 2, ":", color="tab:green",
                 label=rf"${theory_prefactor:g}p^2$ (theory)")
    ax.plot(p_fine, p_fine, ":", color="tab:gray", alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error probability p")
    ax.set_ylabel("logical error rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_syndrome_histogram(df, title, path):
    """How often each of the 4 syndrome patterns occurred, split by success/failure."""
    df = df.copy()
    df["syndrome"] = df["syn_a1"].astype(str) + df["syn_a0"].astype(str)
    order = ["00", "01", "11", "10"]
    labels = ["00 (no error)", "01 (d0)", "11 (d1)", "10 (d2)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 0.35
    x = np.arange(len(order))
    succ_counts = [len(df[(df.syndrome == s) & df.success]) for s in order]
    fail_counts = [len(df[(df.syndrome == s) & ~df.success]) for s in order]
    ax.bar(x - width / 2, succ_counts, width, label="corrected successfully", color="tab:blue")
    ax.bar(x + width / 2, fail_counts, width, label="logical error remained", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("trial count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_site_attribution(df, title, path):
    """Among *failed* trials, how often did each circuit site have an active error?

    Splits sites into gate categories (CX / H) vs the idle slot, answering
    "was the logical error caused by a specific gate, or by background
    (gate-uncorrelated) noise?"
    """
    failed = df[~df.success]
    site_counts = {}
    for sites_str in failed["sites_hit"]:
        if not sites_str:
            continue
        for s in sites_str.split(","):
            site_counts[s] = site_counts.get(s, 0) + 1

    if not site_counts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no failed trials", ha="center", va="center")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    def category(site):
        if "idle" in site:
            return "tab:gray"
        if "cx" in site:
            return "tab:blue"
        if site.startswith("h") or "_h" in site:
            return "tab:purple"
        return "tab:orange"

    items = sorted(site_counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    colors = [category(k) for k in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, counts, color=colors)
    ax.set_xlabel("number of failed trials where this site had an active error")
    ax.set_title(title)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    handles = [
        Patch(color="tab:blue", label="CX gate (encode/syndrome)"),
        Patch(color="tab:purple", label="H gate (basis change)"),
        Patch(color="tab:gray", label="idle slot (not gate-correlated)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pauli_attribution(df, title, path):
    """Among failed trials, breakdown of which Pauli error types (X/Y/Z) were present."""
    failed = df[~df.success]
    pauli_counts = {"X": 0, "Y": 0, "Z": 0}
    for paulis_str in failed["paulis_hit"]:
        if not paulis_str:
            continue
        for p in paulis_str.split(","):
            pauli_counts[p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"X": "tab:red", "Y": "tab:olive", "Z": "tab:blue"}
    ax.bar(pauli_counts.keys(), pauli_counts.values(),
           color=[colors[k] for k in pauli_counts])
    ax.set_ylabel("number of failed trials involving this Pauli type")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_count_vs_outcome(df, title, path):
    """Sanity/insight plot: how many injected errors did failed vs successful trials have?"""
    fig, ax = plt.subplots(figsize=(6, 4))
    max_n = int(df["n_errors"].max()) if len(df) else 0
    bins = np.arange(0, max_n + 2) - 0.5
    ax.hist(df[df.success]["n_errors"], bins=bins, alpha=0.6, label="success", color="tab:blue")
    ax.hist(df[~df.success]["n_errors"], bins=bins, alpha=0.6, label="failure", color="tab:red")
    ax.set_xlabel("number of injected errors in the trial")
    ax.set_ylabel("trial count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spatial_error_heatmap(df, title, path, data_pos, grid_shape=(3, 3)):
    """Where on the physical qubit grid did errors occur, among failed trials?

    data_pos: dict qubit_index -> (row, col), e.g. surface_code.DATA_POS.
    Reads df.attrs["logs"] (per-trial list of {"site","qubit","pauli"} dicts,
    with qubit labels like "d4") and df["success"].
    """
    logs = df.attrs.get("logs")
    if logs is None:
        raise ValueError("df must carry .attrs['logs'] from run_batch")

    rows, cols = grid_shape
    counts = np.zeros((rows, cols))
    for i, success in enumerate(df["success"]):
        if success:
            continue
        seen = set()
        for e in logs[i]:
            qlabel = e["qubit"]
            if not qlabel.startswith("d"):
                continue
            q = int(qlabel[1:])
            if q in seen:
                continue
            seen.add(q)
            r, c = data_pos[q]
            counts[r, c] += 1

    import textwrap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(counts, cmap="Reds")
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, int(counts[r, c]), ha="center", va="center",
                     color="black" if counts[r, c] < counts.max() * 0.6 else "white")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("\n".join(textwrap.wrap(title, 40)), fontsize=10)
    fig.colorbar(im, ax=ax, label="failed trials with an error on this qubit")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
