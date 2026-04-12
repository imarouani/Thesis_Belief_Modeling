"""
Belief Network Explorer — Thesis Visualization Dashboard
=========================================================
Streamlit application for exploring GGM-estimated belief network instances
stored in the Belief Knowledge Graph (BKG).

Fixes applied vs. original:
  - n_cc now sourced from networks_summary.csv (correct estimation-level
    complete-case N) with fallback to network_instances.csv.
  - Sparse-network warning panel for contexts with few edges or zero bridge edges.
  - Stability filter: sidebar option to show only bootstrap-stable edges
    (prop_nonzero >= user-defined threshold).
  - Isolated node detection and explicit display in node metrics table.
  - Bootstrap stability summary panel (CI width, prop_nonzero) for focal edges.
  - BKG example queries section at the bottom of the page.
  - Bundled-view caption clarified as a summary projection.
  - n display now labelled "n (GGM)" to distinguish estimation N from stratum N.
  - Domain color scheme unified with CMS Explorer and analysis notebooks.
  - Edge sign colors improved for better visual contrast.
  - Bootstrap columns suppressed in bundled view (averaged CIs are not meaningful).
"""

from pathlib import Path
import tempfile

import altair as alt
import pandas as pd
import streamlit as st
from pyvis.network import Network

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
    if st.button("📊 CMS Explorer"):
        st.switch_page("pages/2_CMS_Explorer.py")

st.markdown("---")

# ─────────────────────────────────────────────
# Paths (adjusted: this file lives in pages/)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
KG_DIR = BASE_DIR / "results" / "networks" / "kg_exports"

if not KG_DIR.exists():
    alt_dir = (BASE_DIR / ".." / "results" / "networks" / "kg_exports").resolve()
    if alt_dir.exists():
        KG_DIR = alt_dir

NETWORK_DIR = BASE_DIR / "results" / "networks"
if not NETWORK_DIR.exists():
    NETWORK_DIR = (BASE_DIR / ".." / "results" / "networks").resolve()

# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.title("Network Viewer")

st.markdown(
    """
    Explore estimated GGM network instances across survey waves and trust strata.
    Networks were estimated with EBICglasso regularisation (γ = 0.5)
    and stored in the Belief Knowledge Graph (BKG). Robustness checks (γ = 0.75, thresholded, Spearman) are reported separately.

    **Main view** — nodes = variables; node size = node strength; edge thickness = |weight|  
    **Insight layer** — SEVERITY embeddedness metrics, affect vs. behaviour share,
    trust-stratum comparison, bootstrap stability  
    **BKG queries** — example Cypher queries for Neo4j (bottom of page)
    """
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FOCAL_NODE = "SEVERITY"

# Unified color scheme — matches CMS Explorer and analysis notebooks
DOMAIN_COLORS = {
    "belief":   "#1565C0",   # blue
    "affect":   "#C62828",   # red
    "behavior": "#2E7D32",   # green
}

SIGN_COLORS = {
    "positive": "#5B8DB8",   # muted blue — clearly distinct from negative
    "negative": "#E57373",   # muted red-orange
}

# Wave 57 high-trust has the lowest GS (2.303) but is not structurally collapsed.
KNOWN_DEGENERATE = set()  # No contexts are flagged as degenerate

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_data(kg_dir: Path, network_dir: Path):
    network_instances = pd.read_csv(kg_dir / "network_instances.csv")
    node_instances    = pd.read_csv(kg_dir / "node_instances.csv")
    variables         = pd.read_csv(kg_dir / "variables.csv")
    edge_instances    = pd.read_csv(kg_dir / "edge_instances.csv")

    # ── Fix: replace n_cc in network_instances with the correct
    #    complete-case estimation N from networks_summary.csv.
    summary_path = network_dir / "networks_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)[
            ["wave", "trust_stratum", "n_cc", "spec"]
        ].drop_duplicates(subset=["wave", "trust_stratum", "spec"])

        summary_lookup = summary.set_index(["wave", "trust_stratum", "spec"])["n_cc"].to_dict()

        def _correct_n(row):
            key = (row["wave"], row["trust_stratum"], row["spec"])
            return summary_lookup.get(key, row["n_cc"])

        network_instances["n_cc"] = network_instances.apply(_correct_n, axis=1)

    # ── Load context summary for bridge-edge counts (used in warnings)
    context_summary = None
    ctx_path = network_dir / "context_summary.csv"
    if ctx_path.exists():
        context_summary = pd.read_csv(ctx_path)

    return network_instances, node_instances, variables, edge_instances, context_summary


network_instances, node_instances, variables, edge_instances, context_summary = load_data(
    KG_DIR, NETWORK_DIR
)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def scale_value(x, min_x, max_x, out_min, out_max):
    if max_x == min_x:
        return (out_min + out_max) / 2
    return out_min + (x - min_x) * (out_max - out_min) / (max_x - min_x)


def get_other_node(row):
    if row["from"] == FOCAL_NODE:
        return row["to"]
    if row["to"] == FOCAL_NODE:
        return row["from"]
    return None


def compute_focal_metrics(edges_df: pd.DataFrame, variables_df: pd.DataFrame) -> dict:
    domain_map = variables_df.set_index("node_id")["domain"].to_dict()

    focal_edges = edges_df.loc[
        (edges_df["from"] == FOCAL_NODE) | (edges_df["to"] == FOCAL_NODE)
    ].copy()

    if focal_edges.empty:
        return {
            "S_total": 0.0,
            "S_aff":   0.0,
            "S_beh":   0.0,
            "P_aff":   0.0,
            "P_beh":   0.0,
            "focal_edges": focal_edges,
        }

    focal_edges["other_node"]   = focal_edges.apply(get_other_node, axis=1)
    focal_edges["other_domain"] = focal_edges["other_node"].map(domain_map)

    s_aff   = focal_edges.loc[focal_edges["other_domain"] == "affect",   "abs_weight"].sum()
    s_beh   = focal_edges.loc[focal_edges["other_domain"] == "behavior", "abs_weight"].sum()
    s_total = focal_edges["abs_weight"].sum()

    p_aff = s_aff / s_total if s_total > 0 else 0.0
    p_beh = s_beh / s_total if s_total > 0 else 0.0

    return {
        "S_total":     float(s_total),
        "S_aff":       float(s_aff),
        "S_beh":       float(s_beh),
        "P_aff":       float(p_aff),
        "P_beh":       float(p_beh),
        "focal_edges": focal_edges.sort_values("abs_weight", ascending=False).reset_index(drop=True),
    }


def build_bundled_view(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    def map_bundle(node_id: str, domain: str) -> str:
        if domain == "affect":
            return "AFFECT"
        if domain == "behavior":
            return "BEHAVIOR"
        return node_id

    nodes_tmp = nodes_df.copy()
    edges_tmp = edges_df.copy()

    nodes_tmp["bundle_id"]    = nodes_tmp.apply(lambda r: map_bundle(r["node_id"], r["domain"]), axis=1)
    edges_tmp["from_bundle"]  = edges_tmp.apply(lambda r: map_bundle(r["from"], r["from_domain"]), axis=1)
    edges_tmp["to_bundle"]    = edges_tmp.apply(lambda r: map_bundle(r["to"],   r["to_domain"]),   axis=1)

    bundled_nodes = (
        nodes_tmp.groupby("bundle_id", as_index=False)
        .agg(strength=("strength", "sum"), degree=("degree", "sum"))
    )

    def infer_domain(bundle_id: str) -> str:
        if bundle_id == "AFFECT":   return "affect"
        if bundle_id == "BEHAVIOR": return "behavior"
        return "belief"

    bundled_nodes["domain"] = bundled_nodes["bundle_id"].apply(infer_domain)
    bundled_nodes["role"]   = bundled_nodes["bundle_id"].apply(
        lambda x: "focal_belief" if x == FOCAL_NODE else "aggregated"
    )
    bundled_nodes["name"]    = bundled_nodes["bundle_id"]
    bundled_nodes["node_id"] = bundled_nodes["bundle_id"]

    bundled_edges = (
        edges_tmp.assign(
            edge_x=edges_tmp[["from_bundle", "to_bundle"]].min(axis=1),
            edge_y=edges_tmp[["from_bundle", "to_bundle"]].max(axis=1),
        )
        .groupby(["edge_x", "edge_y"], as_index=False)
        .agg(
            weight=("weight", "sum"),
            abs_weight=("abs_weight", "sum"),
        )
        .rename(columns={"edge_x": "from", "edge_y": "to"})
    )
    bundled_edges = bundled_edges.loc[bundled_edges["from"] != bundled_edges["to"]].copy()
    bundled_edges["sign"] = bundled_edges["weight"].apply(lambda x: "positive" if x >= 0 else "negative")
    bundled_edges["from_domain"] = bundled_edges["from"].apply(infer_domain)
    bundled_edges["to_domain"]   = bundled_edges["to"].apply(infer_domain)

    return bundled_nodes, bundled_edges


def make_network_html(
    plot_nodes: pd.DataFrame,
    plot_edges: pd.DataFrame,
    show_edge_labels: bool,
    use_physics: bool,
    stable_only: bool = False,
    stability_threshold: float = 0.5,
) -> str:
    net = Network(height="760px", width="100%", notebook=False, bgcolor="white", font_color="black")

    if use_physics:
        net.barnes_hut()
    else:
        net.set_options("""{"physics": {"enabled": false}}""")

    if stable_only and "prop_nonzero" in plot_edges.columns:
        plot_edges = plot_edges.loc[
            plot_edges["prop_nonzero"].fillna(0) >= stability_threshold
        ].copy()

    strength_min = plot_nodes["strength"].min()
    strength_max = plot_nodes["strength"].max()

    for _, row in plot_nodes.iterrows():
        node_size  = scale_value(row["strength"], strength_min, strength_max, 18, 60)
        node_color = DOMAIN_COLORS.get(row["domain"], "#999999")
        is_isolated = int(row["degree"]) == 0
        border_width = 3 if is_isolated else 1
        border_color = "#FF0000" if is_isolated else node_color

        title = (
            f"Node: {row['name']}<br>"
            f"Domain: {row['domain']}<br>"
            f"Role: {row['role']}<br>"
            f"Degree: {int(row['degree'])}<br>"
            f"Strength: {row['strength']:.3f}"
        )
        if is_isolated:
            title += "<br><b>⚠ Isolated node (degree = 0)</b>"

        net.add_node(
            row["node_id"],
            label=row["name"],
            title=title,
            size=node_size,
            color={"background": node_color, "border": border_color},
            borderWidth=border_width,
        )

    abs_min = plot_edges["abs_weight"].min() if not plot_edges.empty else 0
    abs_max = plot_edges["abs_weight"].max() if not plot_edges.empty else 1

    def _fmt(v, spec=".3f"):
        return format(v, spec) if pd.notna(v) else "—"

    for _, row in plot_edges.iterrows():
        edge_width = scale_value(row["abs_weight"], abs_min, abs_max, 1, 10)
        sign       = "positive" if row["weight"] >= 0 else "negative"
        edge_color = SIGN_COLORS[sign]

        edge_title = (
            f"Edge: {row['from']} ↔ {row['to']}<br>"
            f"Weight: {row['weight']:.3f}<br>"
            f"|Weight|: {row['abs_weight']:.3f}"
        )

        pnz = row.get("prop_nonzero", float("nan"))
        if pd.notna(row.get("boot_mean")):
            stability_flag = ""
            if pd.notna(pnz):
                if pnz < 0.5:
                    stability_flag = " ⚠ unstable"
                elif pnz >= 0.8:
                    stability_flag = " ✓ stable"

            edge_title += (
                f"<br>Bootstrap mean: {_fmt(row.get('boot_mean'))}<br>"
                f"CI: [{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]<br>"
                f"prop_nonzero: {_fmt(pnz)}{stability_flag}<br>"
                f"n_boot: {int(row['n_boot']) if pd.notna(row.get('n_boot')) else '—'}"
            )

        edge_kwargs = dict(
            source=row["from"],
            to=row["to"],
            width=edge_width,
            title=edge_title,
            color=edge_color,
        )
        if show_edge_labels:
            edge_kwargs["label"] = f"{row['weight']:.2f}"

        net.add_edge(**edge_kwargs)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        temp_path = Path(tmp_file.name)
    net.save_graph(str(temp_path))
    return temp_path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.sidebar.header("Filters")

wave_options    = sorted(network_instances["wave"].unique().tolist())
selected_wave   = st.sidebar.selectbox("Wave", wave_options)

trust_options   = sorted(
    network_instances.loc[network_instances["wave"] == selected_wave, "trust_stratum"]
    .unique().tolist()
)
selected_trust  = st.sidebar.selectbox("Trust stratum", trust_options)

gamma_options   = sorted(
    network_instances.loc[
        (network_instances["wave"] == selected_wave) &
        (network_instances["trust_stratum"] == selected_trust),
        "gamma_ebic",
    ].unique().tolist()
)
if len(gamma_options) == 1:
    selected_gamma = gamma_options[0]
    st.sidebar.markdown(f"**EBIC gamma:** {selected_gamma:.2f} *(only specification in BKG)*")
else:
    selected_gamma = st.sidebar.selectbox("EBIC gamma", gamma_options)

st.sidebar.markdown("---")
st.sidebar.header("Display options")
show_edge_labels    = st.sidebar.checkbox("Show edge labels", value=False)
use_physics         = st.sidebar.checkbox(
    "Auto-arrange nodes",
    value=True,
    help="When enabled, nodes are positioned automatically using a force-directed layout. "
         "Disable to freeze positions and drag nodes manually.",
)
bundle_domains      = st.sidebar.checkbox("Bundle affect and behavior (summary view)", value=False)
stable_only         = st.sidebar.checkbox("Show only bootstrap-stable edges", value=False)
stability_threshold = st.sidebar.slider(
    "Stability threshold (prop_nonzero ≥)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    disabled=not stable_only,
)

# ─────────────────────────────────────────────
# Resolve selected network
# ─────────────────────────────────────────────
selected_row = network_instances.loc[
    (network_instances["wave"]          == selected_wave) &
    (network_instances["trust_stratum"] == selected_trust) &
    (network_instances["gamma_ebic"]    == selected_gamma)
]

if selected_row.empty:
    st.error("No network instance found for the selected combination.")
    st.stop()

selected_row       = selected_row.iloc[0]
selected_network_id = selected_row["network_id"]

# ─────────────────────────────────────────────
# Prepare network data
# ─────────────────────────────────────────────
nodes_df = (
    node_instances.loc[node_instances["network_id"] == selected_network_id]
    .merge(variables[["node_id", "name"]], on="node_id", how="left")
    .copy()
)
edges_df = edge_instances.loc[edge_instances["network_id"] == selected_network_id].copy()

if nodes_df.empty or edges_df.empty:
    st.error("Selected network has no nodes or no edges in the KG export.")
    st.stop()

focal_metrics = compute_focal_metrics(edges_df, variables)

# ─────────────────────────────────────────────
# ⚠ Network health warning
# ─────────────────────────────────────────────
n_edges_total   = len(edges_df)
n_bridge_edges  = 0
is_known_degenerate = (selected_wave, selected_trust) in KNOWN_DEGENERATE

if context_summary is not None:
    ctx_row = context_summary.loc[
        (context_summary["wave"]          == selected_wave) &
        (context_summary["trust_stratum"] == selected_trust) &
        (context_summary["spec"]          == selected_row.get("spec", "g05"))
    ]
    if not ctx_row.empty:
        n_bridge_edges = int(ctx_row.iloc[0].get("n_bridge_edges", 0))

isolated_nodes = nodes_df.loc[nodes_df["degree"] == 0, "node_id"].tolist()

show_warning = (
    n_edges_total <= 8
    or n_bridge_edges == 0
    or len(isolated_nodes) > 0
    or is_known_degenerate
)

if show_warning:
    msgs = []
    if n_edges_total <= 8:
        msgs.append(
            f"**Sparse network**: only {n_edges_total} nonzero edges survive regularisation "
            f"(vs. 18–22 in other contexts). Structural metrics should be interpreted cautiously."
        )
    if n_bridge_edges == 0:
        msgs.append(
            "**Zero bridge edges**: no direct conditional association between "
            "psychological variables and behavior nodes was retained in this context. "
            "The focal belief has no direct behavioral connections in this network."
        )
    if isolated_nodes:
        msgs.append(
            f"**Isolated nodes** (degree = 0, shown with red border): "
            f"{', '.join(isolated_nodes)}. These nodes have no detectable conditional "
            f"association with any other variable under this specification."
        )
    if is_known_degenerate:
        msgs.append(
            "This context has been flagged for additional interpretive caution. "
            "See thesis Section 5 (Robustness and stability assessment) for details."
        )
    with st.expander("⚠ Network health warnings — click to expand", expanded=True):
        for msg in msgs:
            st.warning(msg)

# ─────────────────────────────────────────────
# Header metrics
# ─────────────────────────────────────────────
st.subheader("Selected network instance")

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Network",      selected_network_id)
m2.metric("Wave",         int(selected_row["wave"]))
m3.metric("Trust",        selected_row["trust_stratum"])
m4.metric("EBIC γ",       f"{float(selected_row['gamma_ebic']):.2f}")
m5.metric("n (GGM)",      int(selected_row["n_cc"]),
          help="Complete-case N used for GGM estimation (after listwise deletion on the node set).")
m6.metric("Edges",        n_edges_total)
gs_val = selected_row.get("global_strength", None)
m7.metric("Global strength", f"{float(gs_val):.3f}" if pd.notna(gs_val) else "—")

# ─────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex; gap:28px; align-items:center; margin-bottom:8px; flex-wrap:wrap;">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:#1565C0;"></span>
    <span>Belief (SEVERITY = focal)</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:#C62828;"></span>
    <span>Affect / Emotion</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:#2E7D32;"></span>
    <span>Behavior</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:26px;height:4px;background:#5B8DB8;"></span>
    <span>Positive edge</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:26px;height:4px;background:#E57373;"></span>
    <span>Negative edge</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block;width:16px;height:16px;border-radius:50%;
                 background:#1565C0;border:3px solid #FF0000;"></span>
    <span>Isolated node (degree = 0)</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Focal belief embeddedness
# ─────────────────────────────────────────────
st.subheader("Focal belief embeddedness (SEVERITY)")

i1, i2, i3, i4, i5 = st.columns(5)
i1.metric("S(SEVERITY)",      f"{focal_metrics['S_total']:.3f}",
          help="Total node strength: sum of absolute edge weights incident on SEVERITY.")
i2.metric("Affect coupling",  f"{focal_metrics['S_aff']:.3f}",
          help="Sum of |weights| on SEVERITY–affect edges.")
i3.metric("Behavior coupling",f"{focal_metrics['S_beh']:.3f}",
          help="Sum of |weights| on SEVERITY–behavior edges.")
i4.metric("Affect share",     f"{100 * focal_metrics['P_aff']:.1f}%",
          help="Proportion of SEVERITY strength attributable to affect connections.")
i5.metric("Behavior share",   f"{100 * focal_metrics['P_beh']:.1f}%",
          help="Proportion of SEVERITY strength attributable to behavior connections.")

share_df = pd.DataFrame({
    "Component": ["Affect", "Behavior"],
    "Share":     [focal_metrics["P_aff"], focal_metrics["P_beh"]],
})
share_chart = (
    alt.Chart(share_df)
    .mark_bar(cornerRadius=4)
    .encode(
        x=alt.X("Share:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Component:N", sort=["Affect", "Behavior"]),
        color=alt.Color("Component:N", scale=alt.Scale(
            domain=["Affect", "Behavior"],
            range=[DOMAIN_COLORS["affect"], DOMAIN_COLORS["behavior"]],
        ), legend=None),
        tooltip=[alt.Tooltip("Component:N"), alt.Tooltip("Share:Q", format=".1%")],
    )
    .properties(height=110)
)
st.altair_chart(share_chart, use_container_width=True)

# ─────────────────────────────────────────────
# View selection (bundled or full) + graph render
# ─────────────────────────────────────────────
if bundle_domains:
    plot_nodes, plot_edges = build_bundled_view(nodes_df, edges_df)
    st.caption(
        "⚠ **Bundled view enabled**: affect and behavior nodes are aggregated "
        "into single meta-nodes for visual simplicity. Edge weights are summed across "
        "the bundle. This is a **summary projection only** — not a re-estimated network. "
        "Do not interpret bundled edge weights as partial correlations. "
        "Bootstrap statistics are suppressed in this view because averaged CIs across "
        "different edges are not statistically meaningful."
    )
else:
    plot_nodes = nodes_df.copy()
    plot_edges = edges_df.copy()

if stable_only:
    n_before = len(plot_edges)
    stable_mask = plot_edges["prop_nonzero"].fillna(0) >= stability_threshold
    n_after     = stable_mask.sum()
    st.caption(
        f"Stability filter active (prop_nonzero ≥ {stability_threshold:.2f}): "
        f"showing {n_after} of {n_before} edges."
    )

graph_html = make_network_html(
    plot_nodes, plot_edges, show_edge_labels, use_physics,
    stable_only=stable_only, stability_threshold=stability_threshold,
)
st.components.v1.html(graph_html, height=780, scrolling=True)

# ─────────────────────────────────────────────
# Node metrics + Edge metrics
# ─────────────────────────────────────────────
st.markdown("---")
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Node metrics")
    node_display = (
        plot_nodes[["node_id", "name", "domain", "role", "degree", "strength"]]
        .sort_values("strength", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(node_display, use_container_width=True)

with right_col:
    st.subheader("Edge metrics")
    if bundle_domains:
        edge_display_cols = [c for c in ["from", "to", "weight", "abs_weight"]
                            if c in plot_edges.columns]
    else:
        edge_display_cols = [c for c in
            ["from", "to", "weight", "abs_weight", "boot_mean", "ci_low", "ci_high",
             "prop_nonzero", "n_boot"]
            if c in plot_edges.columns
        ]
    st.dataframe(
        plot_edges[edge_display_cols]
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# Focal belief neighbourhood + bootstrap stability
# ─────────────────────────────────────────────
st.subheader("Focal belief neighbourhood (SEVERITY)")

focal_edge_cols = [c for c in
    ["from", "to", "weight", "abs_weight", "boot_mean", "ci_low", "ci_high",
     "prop_nonzero", "n_boot"]
    if c in focal_metrics["focal_edges"].columns
]

if not focal_metrics["focal_edges"].empty:
    fe = focal_metrics["focal_edges"][focal_edge_cols].copy()

    # Stability classification
    if "prop_nonzero" in fe.columns:
        def _stability_label(p):
            if pd.isna(p):    return "unknown"
            if p >= 0.8:      return "stable (≥0.80)"
            if p >= 0.5:      return "moderate (0.50–0.79)"
            return "unstable (<0.50)"
        fe["stability"] = fe["prop_nonzero"].apply(_stability_label)

    if "ci_low" in fe.columns and "ci_high" in fe.columns:
        fe["ci_width"] = fe["ci_high"] - fe["ci_low"]
        fe["ci_excludes_zero"] = (
            (fe["ci_low"] > 0) | (fe["ci_high"] < 0)
        ).map({True: "yes", False: "no"})

    st.dataframe(fe.reset_index(drop=True), use_container_width=True)

    if "prop_nonzero" in fe.columns:
        n_stable   = (fe["prop_nonzero"].fillna(0) >= 0.8).sum()
        n_moderate = ((fe["prop_nonzero"].fillna(0) >= 0.5) & (fe["prop_nonzero"].fillna(0) < 0.8)).sum()
        n_unstable = (fe["prop_nonzero"].fillna(0) < 0.5).sum()
        st.caption(
            f"Bootstrap stability summary for focal edges — "
            f"stable (prop_nonzero ≥ 0.80): **{n_stable}** | "
            f"moderate (0.50–0.79): **{n_moderate}** | "
            f"unstable (<0.50): **{n_unstable}**"
        )
else:
    st.info(
        "SEVERITY has no edges in this network under the current specification. "
        "This indicates complete structural isolation of the focal belief."
    )

# ─────────────────────────────────────────────
# BKG example Cypher queries
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("BKG — Example Cypher Queries for Neo4j", expanded=False):
    st.markdown(
        """
These queries assume the BKG has been imported into Neo4j using `import_kg.cypher`.
They demonstrate the analytical advantage of the graph representation over flat CSV files.

---

**Query 1 — Retrieve SEVERITY node strength across all contexts**
```cypher
MATCH (ni:NodeInstance)-[:INSTANCE_OF]->(v:Variable {node_id: "SEVERITY"}),
      (net:NetworkInstance)-[:HAS_NODE_INSTANCE]->(ni),
      (net)-[:HAS_CONTEXT]->(ctx:Context)
RETURN ctx.wave AS wave,
       ctx.trust_stratum AS trust_stratum,
       ni.strength AS S_SEVERITY
ORDER BY ctx.wave, ctx.trust_stratum;
```

---

**Query 2 — Find all SEVERITY–behavior edges with |weight| > 0.10**
```cypher
MATCH (ei:EdgeInstance),
      (ei)-[:FROM]->(n1:NodeInstance)-[:INSTANCE_OF]->(v1:Variable {node_id: "SEVERITY"}),
      (ei)-[:TO]->(n2:NodeInstance)-[:INSTANCE_OF]->(v2:Variable {domain: "behavior"}),
      (net:NetworkInstance)-[:HAS_EDGE_INSTANCE]->(ei),
      (net)-[:HAS_CONTEXT]->(ctx:Context)
WHERE ei.abs_weight > 0.10
RETURN ctx.wave, ctx.trust_stratum, v2.node_id AS behavior_node,
       ei.weight, ei.abs_weight
ORDER BY ei.abs_weight DESC;
```

---

**Query 3 — Across which contexts does SEVERITY have zero behavior connections?**
```cypher
MATCH (net:NetworkInstance)-[:HAS_CONTEXT]->(ctx:Context)
WHERE NOT EXISTS {
  MATCH (net)-[:HAS_EDGE_INSTANCE]->(ei:EdgeInstance),
        (ei)-[:FROM|TO]->(ni:NodeInstance)-[:INSTANCE_OF]->(v:Variable {node_id: "SEVERITY"}),
        (ei)-[:FROM|TO]->(ni2:NodeInstance)-[:INSTANCE_OF]->(v2:Variable {domain: "behavior"})
  WHERE ni <> ni2
}
RETURN ctx.wave, ctx.trust_stratum;
```

---

**Query 4 — Which behavior node is SEVERITY's strongest neighbor, per context?**
```cypher
MATCH (ei:EdgeInstance),
      (ei)-[:FROM|TO]->(n1:NodeInstance)-[:INSTANCE_OF]->(v1:Variable {node_id: "SEVERITY"}),
      (ei)-[:FROM|TO]->(n2:NodeInstance)-[:INSTANCE_OF]->(v2:Variable {domain: "behavior"}),
      (net:NetworkInstance)-[:HAS_EDGE_INSTANCE]->(ei),
      (net)-[:HAS_CONTEXT]->(ctx:Context)
WHERE v1 <> v2
WITH ctx.wave AS wave, ctx.trust_stratum AS trust,
     v2.node_id AS behavior_node, ei.abs_weight AS w
ORDER BY wave, trust, w DESC
WITH wave, trust, COLLECT({node: behavior_node, weight: w})[0] AS top
RETURN wave, trust, top.node AS strongest_behavior_neighbor, top.weight AS weight
ORDER BY wave, trust;
```

---

**Query 5 — Compare global strength across strata within each wave**
```cypher
MATCH (net:NetworkInstance)-[:HAS_CONTEXT]->(ctx:Context)
RETURN ctx.wave AS wave,
       ctx.trust_stratum AS trust_stratum,
       net.global_strength AS global_strength
ORDER BY ctx.wave, ctx.trust_stratum;
```
        """
    )

st.markdown("---")
st.caption(
    "Data sourced from the Belief Knowledge Graph (BKG) — "
    "GGM estimated with EBICglasso via `qgraph` / `bootnet` (R). "
    "n (GGM) = complete-case N used for estimation. "
    "All edges are partial correlations; interpret as conditional associations, not causal effects."
)