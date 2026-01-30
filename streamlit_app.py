"""
Invoice Insights Agent — Streamlit UI

Upload an invoice image, generate insights, and see evaluation scores.
"""

import tempfile
from pathlib import Path

import streamlit as st
from main import run

st.set_page_config(page_title="Invoice Insights", layout="wide")
st.title("Invoice Insights Agent")

uploaded = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("Upload an invoice image to get started.")
    st.stop()

# Show the uploaded image
st.image(uploaded, caption=uploaded.name, width=400)

if st.button("Analyze Invoice"):
    # Save upload to a temp file so the pipeline can read it by path
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getvalue())
        tmp_path = f.name

    with st.spinner("Running pipeline (parsing → insights → evaluation)..."):
        try:
            result = run(tmp_path)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    if result.get("error"):
        st.error(result["error"])
        st.stop()

    # ── Insights ──────────────────────────────────────────────
    workflow = result.get("workflow", {})
    insights = workflow.get("insights", [])

    st.header("Generated Insights")
    st.caption(f"Parser: {workflow.get('parser_used', 'unknown')} · {len(insights)} insights")
    for i, insight in enumerate(insights, 1):
        st.markdown(f"**{i}.** {insight}")

    # ── Evaluation ────────────────────────────────────────────
    ev = result.get("evaluation", {})
    if ev.get("error"):
        st.error(f"Evaluation error: {ev['error']}")
        st.stop()

    st.header("Evaluation")

    # Overall score
    overall = ev.get("overall_score", 0)
    st.metric("Overall Score", f"{overall}/100")

    # Three judge columns
    col1, col2, col3 = st.columns(3)

    # ── Judge 1: Factual Completeness ─────────────────────────
    fc = ev.get("factual_completeness", {})
    with col1:
        st.subheader("Factual Completeness")
        if fc.get("error"):
            st.error(fc["error"])
        else:
            st.metric("Score", f"{fc.get('score', 'N/A')}%")
            sub1, sub2 = st.columns(2)
            sub1.metric("Accuracy", f"{fc.get('accuracy_score', 'N/A')}%")
            sub2.metric("Coverage", f"{fc.get('completeness_score', 'N/A')}%")

            # Per-insight verdicts
            per_insight = fc.get("per_insight", [])
            if per_insight:
                st.markdown("**Per-insight verdicts**")
                for item in per_insight:
                    label = item.get("label", "")
                    icon = {"factual": "✅", "hallucinated": "❌", "partial": "⚠️"}.get(label, "•")
                    text = f"{icon} Insight {item.get('insight', '?')}: **{label}**"
                    issue = item.get("issue")
                    if issue and issue != "null":
                        text += f" — {issue}"
                    st.markdown(text)

            # Covered / missing
            covered = fc.get("covered", [])
            missing = fc.get("missing", [])
            if covered:
                with st.expander(f"Covered data points ({len(covered)})"):
                    st.write(", ".join(covered))
            if missing:
                with st.expander(f"Missing data points ({len(missing)})"):
                    st.write(", ".join(missing))

            if fc.get("explanation"):
                with st.expander("Reasoning"):
                    st.write(fc["explanation"])

    # ── Judge 2: Quality ──────────────────────────────────────
    q = ev.get("quality", {})
    with col2:
        st.subheader("Quality")
        if q.get("error"):
            st.error(q["error"])
        else:
            st.metric("Score", f"{q.get('score', 'N/A')}/4")
            for criterion in ("clarity", "specificity", "diversity", "actionability"):
                c = q.get(criterion, {})
                if isinstance(c, dict):
                    st.markdown(f"**{criterion.title()}**: {c.get('label', 'N/A')} ({c.get('score', '?')}/4)")
                else:
                    st.markdown(f"**{criterion.title()}**: {c}")

            if q.get("explanation"):
                with st.expander("Reasoning"):
                    st.write(q["explanation"])

    # ── Judge 3: Parsing Consistency ──────────────────────────
    pc = ev.get("parsing_consistency", {})
    with col3:
        st.subheader("Parsing Consistency")
        if pc.get("skipped"):
            st.warning(pc["skipped"])
        elif pc.get("error"):
            st.error(pc["error"])
        else:
            st.metric("Score", f"{pc.get('score', 'N/A')}%")

            matches = pc.get("matches", [])
            mismatches = pc.get("mismatches", [])
            if matches:
                with st.expander(f"Matches ({len(matches)})"):
                    for m in matches:
                        st.markdown(f"**{m.get('field', '?')}**: {m.get('value', '')}")
            if mismatches:
                with st.expander(f"Mismatches ({len(mismatches)})"):
                    for m in mismatches:
                        st.markdown(
                            f"**{m.get('field', '?')}**\n"
                            f"- Vision: {m.get('source_a', '')}\n"
                            f"- OCR: {m.get('source_b', '')}"
                        )

            if pc.get("explanation"):
                with st.expander("Reasoning"):
                    st.write(pc["explanation"])

    # ── Raw JSON ──────────────────────────────────────────────
    with st.expander("Raw evaluation JSON"):
        st.json({k: v for k, v in ev.items() if k != "extracted_text"})
