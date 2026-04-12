"""
CMS Explorer — Conditional Mean Shift Analysis
================================================
Streamlit page for exploring conditional mean shifts across all nodes,
conditioning sets, and contexts. Part of the thesis supplementary material.

Allows the user to:
  - Select any node as the target (not just SEVERITY)
  - Compare affect-domain vs. behavior-domain conditioning side by side
  - Adjust the conditioning magnitude (c) with a slider
  - View all six contexts simultaneously in a comparison grid

Fixes applied vs. original:
  - Domain color scheme unified with network viewer and analysis notebooks.
  - Removed set_page_config (handled by Home.py).
  - "Outcome node" renamed to "Target node" to avoid causal connotation.
  - Narrative box language softened to match thesis descriptive framing.
  - AFF_FEAR only conditioning set exposed in the UI.
  - Comparison chart caption clarifies sign/absolute-value treatment.
  - Footer clarifies that the CMS uses the Pearson correlation matrix, not the
    GGM partial-correlation matrix.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Navigation bar
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 8])
with nav_col1:
    if st.button("🏠 Home"):
        st.switch_page("Home.py")
with nav_col2:
    if st.button("🕸️ Network Viewer"):
        st.switch_page("pages/Network_Viewer.py")

st.markdown("---")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NODE_VARS = [
    "SEVERITY", "AFF_FEAR", "AFF_WORRY", "WORRY_HEALTH_SYSTEM",
    "USE2_MASK", "USE2_SPACE150", "USE2_HANDWASH20", "USE2_AVOID",
]

NODE_LABELS = {
    "SEVERITY": "Severity belief",
    "AFF_FEAR": "Fear",
    "AFF_WORRY": "Worry",
    "WORRY_HEALTH_SYSTEM": "Health-system worry",
    "USE2_MASK": "Mask wearing",
    "USE2_SPACE150": "Distancing",
    "USE2_HANDWASH20": "Handwashing",
    "USE2_AVOID": "Avoidance",
}

NODE_DOMAIN = {
    "SEVERITY": "belief",
    "AFF_FEAR": "affect",
    "AFF_WORRY": "affect",
    "WORRY_HEALTH_SYSTEM": "affect",
    "USE2_MASK": "behavior",
    "USE2_SPACE150": "behavior",
    "USE2_HANDWASH20": "behavior",
    "USE2_AVOID": "behavior",
}

# Unified color scheme — matches network viewer and analysis notebooks
DOMAIN_COLORS = {
    "belief": "#1565C0",
    "affect": "#C62828",
    "behavior": "#2E7D32",
    "conditioned": "#888888",
}

COND_SETS = {
    "Affect domain": [1, 2, 3],
    "Behavior domain": [4, 5, 6, 7],
    "AFF_FEAR only": [1],
}

CONTEXTS = [
    {"key": "W55_low",  "label": "Wave 55, low trust",  "wave": 55, "trust": "low",  "n": 376},
    {"key": "W55_high", "label": "Wave 55, high trust", "wave": 55, "trust": "high", "n": 309},
    {"key": "W56_low",  "label": "Wave 56, low trust",  "wave": 56, "trust": "low",  "n": 383},
    {"key": "W56_high", "label": "Wave 56, high trust", "wave": 56, "trust": "high", "n": 351},
    {"key": "W57_low",  "label": "Wave 57, low trust",  "wave": 57, "trust": "low",  "n": 336},
    {"key": "W57_high", "label": "Wave 57, high trust", "wave": 57, "trust": "high", "n": 407},
]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_correlation_matrices():
    """Load correlation matrices from instance CSV files."""
    base = Path(__file__).resolve().parent.parent
    data_dirs = [
        base / "results" / "networks" / "instances",
        base / "data" / "instances",
        base / "results" / "networks" / "kg_exports",
    ]

    # Also try relative paths
    for rel in [Path(".."), Path(".")]:
        data_dirs.append(rel / "results" / "networks" / "instances")
        data_dirs.append(rel / "data" / "instances")

    matrices = {}
    for ctx in CONTEXTS:
        wave, trust = ctx["wave"], ctx["trust"]
        fname = f"cosmo_nodes_wave{wave}_trust{trust}.csv"

        df = None
        for d in data_dirs:
            fpath = d / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                break

        if df is None:
            st.warning(f"Could not find {fname}")
            continue

        dat = df[NODE_VARS].apply(pd.to_numeric)
        R = dat.corr(method="pearson").values
        matrices[ctx["key"]] = R

    return matrices


def conditional_mean_shift(R, x_idx, y_idx, c=1.0):
    """Compute conditional mean shift: Delta_Y = Sigma_YX @ inv(Sigma_XX) @ c."""
    Sigma_YX = R[np.ix_(y_idx, x_idx)]
    Sigma_XX = R[np.ix_(x_idx, x_idx)]
    x_shift = np.full(len(x_idx), c)
    delta = Sigma_YX @ np.linalg.solve(Sigma_XX, x_shift)
    return delta


def compute_all_deltas(R, cond_name, c):
    """Compute displacement for all non-conditioned nodes."""
    x_idx = COND_SETS[cond_name]
    y_idx = [i for i in range(len(NODE_VARS)) if i not in x_idx]
    delta_y = conditional_mean_shift(R, x_idx, y_idx, c)

    full_delta = np.full(len(NODE_VARS), np.nan)
    for i, yi in enumerate(y_idx):
        full_delta[yi] = delta_y[i]

    return full_delta


# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.title("What moves a belief?")
st.markdown(
    "Explore how conditioning on the affect or behavior domain displaces "
    "any node in the network. All values are associational properties of "
    "the estimated Gaussian model — **not causal effects**."
)

# Explainer
with st.expander("How to read this", expanded=False):
    st.markdown(
        """
**What is this?** For each subgroup, we estimated how all eight variables
relate to each other (a Gaussian graphical model). This tool asks: if the
affect (or behavior) variables were elevated, how much would the model
expect a given target variable to shift?

**What matrix is used?** The CMS analysis uses the **Pearson correlation
matrix** computed from the within-context data — not the GGM partial-correlation
(precision) matrix. This means the displacements reflect marginal associational
structure, not conditional-dependence structure. See thesis Section 6.2 for
the full justification.

**What do the numbers mean?** A value like +0.66 SD means: if fear and
worry were one standard deviation above average, the model predicts the
target would sit about two-thirds of a standard deviation above its own
average. Larger values indicate tighter coupling.

**What to look for:**
- **Left vs. right panel:** affect conditioning (red) almost always
  produces bigger shifts in SEVERITY than behavior conditioning (green)
- **Low vs. high trust:** low-trust networks tend to show bigger shifts,
  though exceptions exist (e.g. Wave 56 high trust)
- **Switch the target node:** see how different variables respond
  differently to the same conditioning
"""
    )

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
matrices = load_correlation_matrices()

if not matrices:
    st.error(
        "No data files found. Place instance CSVs (cosmo_nodes_wave55_trustlow.csv etc.) "
        "in results/networks/instances/ or data/instances/."
    )
    st.stop()

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.sidebar.header("")

ctx_labels = {c["key"]: c["label"] for c in CONTEXTS}
selected_ctx_key = st.sidebar.selectbox(
    "Network context",
    options=[c["key"] for c in CONTEXTS if c["key"] in matrices],
    format_func=lambda k: ctx_labels[k],
)

target_node = st.sidebar.selectbox(
    "Target node",
    options=NODE_VARS,
    format_func=lambda v: f"{NODE_LABELS[v]} ({v})",
    index=0,
    help="Which variable's conditional displacement to highlight. "
         "This is the node whose expected shift is computed — not a causal outcome.",
)

c_value = st.sidebar.slider(
    "Conditioning shift (c, in SD units)",
    min_value=-2.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="How many SDs above (or below) the mean to set the conditioning variables.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Target:** {NODE_LABELS[target_node]}  \n"
    f"**Shift:** c = {c_value:+.1f} SD"
)

# ─────────────────────────────────────────────
# Compute displacements
# ─────────────────────────────────────────────
R = matrices[selected_ctx_key]
ctx_info = next(c for c in CONTEXTS if c["key"] == selected_ctx_key)

delta_affect = compute_all_deltas(R, "Affect domain", c_value)
delta_behavior = compute_all_deltas(R, "Behavior domain", c_value)
delta_fear_only = compute_all_deltas(R, "AFF_FEAR only", c_value)

target_idx = NODE_VARS.index(target_node)
target_in_affect_cond = target_idx in COND_SETS["Affect domain"]
target_in_behavior_cond = target_idx in COND_SETS["Behavior domain"]
target_in_fear_cond = target_idx in COND_SETS["AFF_FEAR only"]

# ─────────────────────────────────────────────
# Narrative box
# ─────────────────────────────────────────────
if abs(c_value) < 0.05:
    st.info("Drag the slider away from zero to see displacements.")
else:
    d_aff = delta_affect[target_idx] if not target_in_affect_cond else None
    d_beh = delta_behavior[target_idx] if not target_in_behavior_cond else None

    if d_aff is not None and d_beh is not None:
        ratio = abs(d_aff) / abs(d_beh) if abs(d_beh) > 0.001 else 99
        if ratio > 2:
            strength = "a substantially larger"
        elif ratio > 1.4:
            strength = "a clearly larger"
        elif ratio > 1.1:
            strength = "a slightly larger"
        else:
            strength = "a comparable"

        gap = abs(abs(d_aff) - abs(d_beh))

        st.success(
            f"**Affect conditioning is associated with {strength} displacement of "
            f"{NODE_LABELS[target_node]} than behavior conditioning** "
            f"in {ctx_info['label']}.  \n"
            f"Affect: **{abs(d_aff):.3f} SD** · "
            f"Behavior: **{abs(d_beh):.3f} SD** · "
            f"Gap: {gap:.3f} SD"
            + (f" · Ratio: {ratio:.1f}×" if ratio > 1.2 else "")
        )
    elif d_aff is not None:
        st.info(
            f"**{NODE_LABELS[target_node]}** is part of the behavior conditioning set, "
            f"so only affect-domain displacement is shown: **{abs(d_aff):.3f} SD**."
        )
    elif d_beh is not None:
        st.info(
            f"**{NODE_LABELS[target_node]}** is part of the affect conditioning set, "
            f"so only behavior-domain displacement is shown: **{abs(d_beh):.3f} SD**."
        )
    else:
        st.warning(f"**{NODE_LABELS[target_node]}** is in both conditioning sets.")


# ─────────────────────────────────────────────
# Side-by-side bar charts
# ─────────────────────────────────────────────
def make_bar_data(deltas, cond_name, target_idx):
    """Build a DataFrame for one conditioning set's bar chart."""
    cond_indices = set(COND_SETS[cond_name])
    rows = []
    for i, node in enumerate(NODE_VARS):
        is_cond = i in cond_indices
        is_target = i == target_idx
        d = deltas[i] if not np.isnan(deltas[i]) else 0.0

        if is_cond:
            status = "conditioned"
            domain_color = DOMAIN_COLORS["conditioned"]
        else:
            status = "target" if is_target else "other"
            domain_color = DOMAIN_COLORS[NODE_DOMAIN[node]]

        rows.append({
            "node": NODE_LABELS[node],
            "var": node,
            "delta": d,
            "abs_delta": abs(d),
            "domain": NODE_DOMAIN[node],
            "status": status,
            "color": domain_color,
            "label": "held fixed" if is_cond else f"{d:+.3f} SD",
            "order": i,
        })
    return pd.DataFrame(rows)


col_aff, col_beh = st.columns(2)

with col_aff:
    st.markdown("#### If fear and worry are elevated...")
    df_aff = make_bar_data(delta_affect, "Affect domain", target_idx)

    chart_aff = (
        alt.Chart(df_aff)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("node:N", sort=alt.EncodingSortField(field="order"), title=None),
            x=alt.X("abs_delta:Q", title="| Δ | (SD units)"),
            color=alt.Color("color:N", scale=None),
            opacity=alt.condition(
                alt.datum.status == "conditioned",
                alt.value(0.15),
                alt.value(0.9),
            ),
            tooltip=[
                alt.Tooltip("node:N", title="Variable"),
                alt.Tooltip("label:N", title="Displacement"),
                alt.Tooltip("domain:N", title="Domain"),
            ],
        )
        .properties(height=300)
    )

    target_rule = (
        alt.Chart(df_aff[df_aff["var"] == target_node])
        .mark_point(shape="diamond", size=120, filled=True, color="#1565C0")
        .encode(
            y=alt.Y("node:N", sort=alt.EncodingSortField(field="order")),
            x=alt.X("abs_delta:Q"),
        )
    )

    st.altair_chart(chart_aff + target_rule, use_container_width=True)

with col_beh:
    st.markdown("#### If protective behaviors are elevated...")
    df_beh = make_bar_data(delta_behavior, "Behavior domain", target_idx)

    chart_beh = (
        alt.Chart(df_beh)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("node:N", sort=alt.EncodingSortField(field="order"), title=None),
            x=alt.X("abs_delta:Q", title="| Δ | (SD units)"),
            color=alt.Color("color:N", scale=None),
            opacity=alt.condition(
                alt.datum.status == "conditioned",
                alt.value(0.15),
                alt.value(0.9),
            ),
            tooltip=[
                alt.Tooltip("node:N", title="Variable"),
                alt.Tooltip("label:N", title="Displacement"),
                alt.Tooltip("domain:N", title="Domain"),
            ],
        )
        .properties(height=300)
    )

    target_rule_beh = (
        alt.Chart(df_beh[df_beh["var"] == target_node])
        .mark_point(shape="diamond", size=120, filled=True, color="#1565C0")
        .encode(
            y=alt.Y("node:N", sort=alt.EncodingSortField(field="order")),
            x=alt.X("abs_delta:Q"),
        )
    )

    st.altair_chart(chart_beh + target_rule_beh, use_container_width=True)

# ─────────────────────────────────────────────
# AFF_FEAR only panel
# ─────────────────────────────────────────────
if not target_in_fear_cond:
    with st.expander("AFF_FEAR only conditioning", expanded=False):
        d_fear = delta_fear_only[target_idx]
        st.markdown(
            f"When only **Fear** is set to c = {c_value:+.1f} SD, "
            f"the implied displacement of **{NODE_LABELS[target_node]}** is "
            f"**{d_fear:+.3f} SD**."
        )
        df_fear = make_bar_data(delta_fear_only, "AFF_FEAR only", target_idx)
        chart_fear = (
            alt.Chart(df_fear)
            .mark_bar(cornerRadius=3)
            .encode(
                y=alt.Y("node:N", sort=alt.EncodingSortField(field="order"), title=None),
                x=alt.X("abs_delta:Q", title="| Δ | (SD units)"),
                color=alt.Color("color:N", scale=None),
                opacity=alt.condition(
                    alt.datum.status == "conditioned",
                    alt.value(0.15),
                    alt.value(0.9),
                ),
                tooltip=[
                    alt.Tooltip("node:N", title="Variable"),
                    alt.Tooltip("label:N", title="Displacement"),
                    alt.Tooltip("domain:N", title="Domain"),
                ],
            )
            .properties(height=300)
        )
        target_rule_fear = (
            alt.Chart(df_fear[df_fear["var"] == target_node])
            .mark_point(shape="diamond", size=120, filled=True, color="#1565C0")
            .encode(
                y=alt.Y("node:N", sort=alt.EncodingSortField(field="order")),
                x=alt.X("abs_delta:Q"),
            )
        )
        st.altair_chart(chart_fear + target_rule_fear, use_container_width=True)

# ─────────────────────────────────────────────
# Comparison grid across all contexts
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Δ({NODE_LABELS[target_node]}) across all contexts")

grid_rows = []
for ctx in CONTEXTS:
    if ctx["key"] not in matrices:
        continue
    R_ctx = matrices[ctx["key"]]
    d_a = compute_all_deltas(R_ctx, "Affect domain", c_value)
    d_b = compute_all_deltas(R_ctx, "Behavior domain", c_value)
    d_f = compute_all_deltas(R_ctx, "AFF_FEAR only", c_value)

    val_a = d_a[target_idx] if not target_in_affect_cond else None
    val_b = d_b[target_idx] if not target_in_behavior_cond else None
    val_f = d_f[target_idx] if not target_in_fear_cond else None

    grid_rows.append({
        "Context": ctx["label"],
        "n": ctx["n"],
        "Trust": ctx["trust"],
        "Affect Δ": f"{val_a:+.3f}" if val_a is not None else "in cond. set",
        "Behavior Δ": f"{val_b:+.3f}" if val_b is not None else "in cond. set",
        "AFF_FEAR only Δ": f"{val_f:+.3f}" if val_f is not None else "in cond. set",
        "Affect_val": val_a if val_a is not None else 0,
        "Behavior_val": val_b if val_b is not None else 0,
        "Fear_val": val_f if val_f is not None else 0,
    })

grid_df = pd.DataFrame(grid_rows)

# Visual comparison chart
if not target_in_affect_cond and not target_in_behavior_cond:
    chart_data = []
    for _, row in grid_df.iterrows():
        chart_data.append({
            "Context": row["Context"],
            "Conditioning": "Affect",
            "Δ": abs(row["Affect_val"]),
            "Trust": row["Trust"],
        })
        chart_data.append({
            "Context": row["Context"],
            "Conditioning": "Behavior",
            "Δ": abs(row["Behavior_val"]),
            "Trust": row["Trust"],
        })
    chart_df = pd.DataFrame(chart_data)

    comparison_chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Context:N", sort=[c["label"] for c in CONTEXTS], title=None),
            y=alt.Y("Δ:Q", title=f"|Δ({NODE_LABELS[target_node]})| (SD)"),
            color=alt.Color(
                "Conditioning:N",
                scale=alt.Scale(
                    domain=["Affect", "Behavior"],
                    range=[DOMAIN_COLORS["affect"], DOMAIN_COLORS["behavior"]],
                ),
            ),
            xOffset="Conditioning:N",
            tooltip=[
                alt.Tooltip("Context:N"),
                alt.Tooltip("Conditioning:N"),
                alt.Tooltip("Δ:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(comparison_chart, use_container_width=True)
    st.caption(
        "Bar heights show **absolute** displacement |Δ|. When c < 0, the raw Δ values "
        "are negative (the target is displaced downward); the chart shows magnitude only."
    )

# Table
st.dataframe(
    grid_df[["Context", "n", "Affect Δ", "Behavior Δ", "AFF_FEAR only Δ"]],
    use_container_width=True,
    hide_index=True,
)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Conditional mean shift explorer · Thesis supplementary material · "
    "Values are properties of the estimated Gaussian correlation structure (Pearson R) — "
    "not the GGM partial-correlation matrix and not causal effects. "
    f"Δ values are in SD units (c = {c_value:+.1f}). "
    "See thesis Section 6.2 for methodological details. · "
    "[Thesis & source code](https://github.com/imarouani/thesis-belief-explorer)"
)