"""
Belief Network Explorer — Thesis Supplementary Material
========================================================
Multi-page Streamlit application for exploring GGM-estimated belief
network instances and conditional mean shift analysis.
"""

import streamlit as st

st.set_page_config(page_title="Belief Network Explorer", layout="wide")

# Hide default sidebar nav for a clean landing page
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ───
st.markdown(
    """
    <div style="text-align:center; margin-top:60px;">
        <h1 style="font-size:2.4rem; margin-bottom:4px;">Belief Network Explorer</h1>
        <p style="font-size:1.05rem; color:#666; margin-bottom:0;">
            Supplementary material for
        </p>
        <p style="font-size:1.1rem; font-weight:600; margin-top:4px; margin-bottom:2px;">
            Computational Modeling of Belief Structure
        </p>
        <p style="font-size:0.95rem; color:#888; margin-top:0;">
            A Network-Based Representation of a Focal Belief in a Knowledge Graph
        </p>
        <p style="font-size:0.9rem; color:#999; margin-top:12px;">
            Iheb Marouani · Osnabrück University · Institute of Cognitive Science
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

# ─── Two clickable cards ───
col_left, col_spacer, col_right = st.columns([5, 1, 5])

with col_left:
    st.markdown(
        """
        <div style="
            border: 1.5px solid #ddd;
            border-radius: 12px;
            padding: 36px 28px;
            text-align: center;
            min-height: 220px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size:2.2rem; margin-bottom:12px;">🕸️</div>
            <h3 style="margin:0 0 10px 0;">Network Viewer</h3>
            <p style="color:#666; font-size:0.92rem; line-height:1.5;">
                Explore estimated GGM networks interactively.
                View nodes, edges, bootstrap stability, embeddedness
                metrics, and trust-stratum comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Network Viewer", use_container_width=True, type="primary"):
        st.switch_page("pages/Network_Viewer.py")

with col_right:
    st.markdown(
        """
        <div style="
            border: 1.5px solid #ddd;
            border-radius: 12px;
            padding: 36px 28px;
            text-align: center;
            min-height: 220px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size:2.2rem; margin-bottom:12px;">📊</div>
            <h3 style="margin:0 0 10px 0;">CMS Explorer</h3>
            <p style="color:#666; font-size:0.92rem; line-height:1.5;">
                Explore conditional mean shift analysis.
                Select any node as the target, compare affect vs. behavior
                conditioning, and view results across all six contexts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open CMS Explorer", use_container_width=True, type="primary"):
        st.switch_page("pages/2_CMS_Explorer.py")

# ─── Footer ───
st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#aaa; font-size:0.82rem;">
        Networks estimated with EBICglasso regularization (γ = 0.5) on
        COSMO survey data (Germany, waves 55–57).<br>
        All values are descriptive properties of the estimated conditional-dependence structure — not causal effects.<br><br>
        <a href="https://github.com/imarouani/thesis-belief-explorer" style="color:#888;">
            GitHub · Thesis &amp; source code
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)