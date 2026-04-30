import streamlit as st
import pandas as pd
import sqlite3
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from agent.classifier    import classify_requirement, gpt_second_opinion
from agent.clarification import (
    analyze_ambiguity,
    generate_clarification_questions,
    refine_requirement,
    auto_refine_ambiguous,
    check_and_fix_requirement,
    split_and_validate_input,
)
from agent.srs_generator import generate_srs_from_registry
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Requirements Elicitation Agent",
    page_icon  = "agent/agent_logo.png",
    layout     = "wide"
)

# ── Registry helpers ──────────────────────────────────────────────────────
DB_PATH = './data/requirements_registry.db'

def get_registry_df(project_name):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT * FROM requirements WHERE project_name = ?',
        (project_name,)
    ).fetchall()
    conn.close()
    cols = ['id', 'original_text', 'final_text', 'label',
            'confidence', 'iterations', 'status', 'timestamp', 'project_name']
    return pd.DataFrame(rows, columns=cols)

def add_to_registry(result, project_name):
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        'SELECT id FROM requirements WHERE final_text = ? AND project_name = ?',
        (result['final_text'], project_name)
    ).fetchone()
    if not existing:
        conn.execute('''
            INSERT INTO requirements
            (original_text, final_text, label, confidence,
             iterations, status, timestamp, project_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['original_text'], result['final_text'],
            result['label'],         result['confidence'],
            result['iterations'],    result['status'],
            datetime.now().isoformat(), project_name
        ))
        conn.commit()
    conn.close()

def rename_project(old_name, new_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'UPDATE requirements SET project_name = ? WHERE project_name = ?',
        (new_name, old_name)
    )
    conn.commit()
    conn.close()

def clear_registry(project_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM requirements WHERE project_name = ?', (project_name,))
    conn.commit()
    conn.close()

# ── Session state initialisation ──────────────────────────────────────────
defaults = {
    'project_name'          : 'My Project',
    'current_input'         : None,
    'questions'             : [],
    'analysis'              : None,
    'awaiting_answers'      : False,
    'iteration'             : 0,
    'refined_text'          : None,
    'registry_updated'      : False,
    'mode'                  : 'Interactive',
    # pending_clarifications: list of dicts for interactive clarification
    # each dict: {text, reason, index, questions, analysis}
    'pending_clarifications': [],
    'last_result'           : None,
    'last_results'          : None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.logo(image="agent/agent_logo.png", size="large", icon_image="agent/agent_logo.png")
    st.title("Requirements Agent")
    st.divider()

    st.subheader("Project Settings")
    project_name = st.text_input(
        "Project Name",
        value=st.session_state.project_name
    )
    if project_name.strip() and project_name.strip() != st.session_state.project_name:
        rename_project(st.session_state.project_name, project_name.strip())
        project_name = project_name.strip()
    st.session_state.project_name = project_name

    st.divider()
    st.subheader("Registry")

    # Always read fresh — reflects latest inserts immediately after st.rerun()
    sidebar_df = get_registry_df(project_name)

    with st.container(border=True):
        col_m1 = st.columns(1)[0]
        col_m1.metric("**Total Requirements**", len(sidebar_df))

        with st.container(border=True):
            col_m2, col_m3 = st.columns(2)
            col_m2.metric("Classified", len(sidebar_df[sidebar_df['status'] == 'classified']))
            col_m3.metric("Clarified", len(sidebar_df[sidebar_df['status'] == 'clarified']))

        with st.container(border=True):
            col_m4, col_m5 = st.columns(2)
            col_m4.metric("FR",  len(sidebar_df[sidebar_df['label'] == 'FR']))
            col_m5.metric("NFR", len(sidebar_df[
                sidebar_df['label'].str.startswith('NFR_', na=False)
            ]))

    if st.button("🗑️ Clear Registry", type="secondary"):
        clear_registry(project_name)
        st.session_state.registry_updated      = False
        st.session_state.pending_clarifications = []
        st.success("Registry cleared")
        st.rerun()

    st.divider()
    st.caption("AI Requirements Elicitation Agent")
    st.caption("Academic Project — BERT + GPT-4o")

# ── Main content ──────────────────────────────────────────────────────────
st.title("AI Requirements Elicitation Agent")
st.caption("Classify, clarify, and document software requirements automatically")

tab1, tab2, tab3 = st.tabs([
    "🔍 Elicit Requirements",
    "📊 Requirements Registry",
    "📄 Generate SRS",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Elicit Requirements
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Requirement Statement")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "Requirement",
            placeholder=(
                "e.g. The system shall allow users to reset their password via email."
            ),
            height=100,
            key="requirement_input"
        )
        st.components.v1.html(
            """
            <script>
            (function () {
                const MIN_H = 100, MAX_H = 400;
                function init() {
                    const ta = window.parent.document.querySelector(
                        'textarea[aria-label="Requirement"]'
                    );
                    if (!ta) { setTimeout(init, 150); return; }
                    ta.style.minHeight = MIN_H + 'px';
                    ta.style.maxHeight = MAX_H + 'px';
                    ta.style.overflowY = 'hidden';
                    function resize() {
                        ta.style.height = 'auto';
                        const h = Math.min(ta.scrollHeight, MAX_H);
                        ta.style.height = h + 'px';
                        ta.style.overflowY = ta.scrollHeight > MAX_H ? 'auto' : 'hidden';
                    }
                    ta.addEventListener('input', resize);
                    resize();
                }
                setTimeout(init, 300);
            })();
            </script>
            """,
            height=0,
        )
    with col2:
        st.markdown("##### Mode")
        mode = st.radio(
            "Processing mode",
            ["Interactive", "Auto"],
            help=(
                "Interactive: agent asks clarification questions\n"
                "\nAuto: GPT auto-refines vague requirements"
            )
        )

    classify_btn = st.button("Classify Requirement", type="primary")

    # ── New submission — reset all state ──────────────────────────────────
    if classify_btn and user_input.strip():
        st.session_state.current_input          = user_input.strip()
        st.session_state.awaiting_answers       = False
        st.session_state.questions              = []
        st.session_state.analysis               = None
        st.session_state.iteration              = 0
        st.session_state.refined_text           = None
        st.session_state.mode                   = mode
        st.session_state.pending_clarifications = []   # clear previous pending
        st.session_state.last_result            = None
        st.session_state.last_results           = None

        # ── Step 1: Split compound requirements ───────────────────────────
        with st.spinner("Analyzing input..."):
            split_result = split_and_validate_input(user_input.strip())

        requirements_to_process = split_result.get('requirements', [])
        # Force was_split=True whenever the pre-splitter or GPT produced multiple items
        was_split = split_result.get('was_split', False) or len(requirements_to_process) > 1
        if len(requirements_to_process) > 1:
            split_result['split_count'] = max(
                split_result.get('split_count', 0), len(requirements_to_process)
            )

        if was_split:
            n = split_result.get('split_count', len(requirements_to_process))
            if n > 5:
                st.info(
                    f"Input contained **{n} requirements** — each processed individually."
                )
            else:
                st.info(
                    f"Input contains **{n} requirements** — processing each separately."
                )

        # ── Step 2: Process each requirement ─────────────────────────────
        # Collect all results first — only rerun AFTER the loop is complete
        needs_rerun = False
        collected_results = []

        for i, req_item in enumerate(requirements_to_process):
            req_text = req_item['text']        # ← use req_text, NOT user_input
            is_vague = req_item.get('is_vague', False)
            reason   = req_item.get('reason', '')

            st.divider()

            if len(requirements_to_process) > 1:
                st.markdown(f"**Requirement {i + 1} of {len(requirements_to_process)}:**")

            st.markdown(f"**Requirement:** `{req_text}`")

            # ── Classify this individual requirement ──────────────────────
            with st.spinner("Classifying..."):
                label, confidence, all_probs = classify_requirement(req_text)

            # Show result
            col_a, col_b = st.columns(2)
            with col_a:
                if label == 'Ambiguous' or is_vague:
                    st.warning(
                        f"**Ambiguous** (confidence: {confidence:.2%})"
                        + (f" — {reason}" if reason else "")
                    )
                else:
                    st.success(f"**{label}** ({confidence:.2%})")
            with col_b:
                top_probs = dict(sorted(
                    all_probs.items(), key=lambda x: x[1], reverse=True
                )[:5])
                prob_df = pd.DataFrame(
                    list(top_probs.items()), columns=['Label', 'Probability']
                )
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # ── Route: Ambiguous or vague ─────────────────────────────────
            if label == 'Ambiguous' or is_vague:

                if mode == "Auto":
                    # Auto-refine without asking the user
                    with st.spinner("Auto-refining with GPT..."):
                        gpt_label   = gpt_second_opinion(req_text)
                        final_label = (
                            gpt_label
                            if gpt_label and gpt_label != 'Ambiguous'
                            else 'NFR_Other'
                        )
                        refined, assumption = auto_refine_ambiguous(req_text, final_label)
                        text_was_rewritten = refined.strip() != req_text.strip()
                        bert_label, bert_conf, _ = classify_requirement(refined)
                        if bert_label != 'Ambiguous' and bert_conf >= 0.75:
                            final_label = bert_label
                        else:
                            bert_conf = confidence

                    if assumption:
                        st.info(f"**Assumed:** {assumption}")
                    st.success(f"**Refined:** {refined}")
                    st.success(f"**Label:** {final_label}")

                    auto_refined_pre_grammar = refined   # save before grammar fix

                    # Quality / grammar check
                    with st.spinner("Checking requirement quality..."):
                        fixed_text, was_changed, issues = check_and_fix_requirement(refined)

                    if was_changed:
                        st.warning("**Quality issues detected and fixed:**")
                        for issue in issues:
                            st.markdown(f"  - {issue}")
                        st.markdown(f"**Original:** {refined}")
                        st.markdown(f"**Fixed:**    {fixed_text}")
                        refined = fixed_text

                    result = {
                        'original_text': user_input.strip(),
                        'final_text'   : refined,
                        'label'        : final_label,
                        'confidence'   : bert_conf,
                        'iterations'   : 1,
                        'status'       : 'clarified' if text_was_rewritten else 'classified',
                    }
                    add_to_registry(result, project_name)
                    collected_results.append({
                        'original_input'      : user_input.strip(),
                        'req_text'            : req_item['text'],
                        'label'               : label,
                        'confidence'          : confidence,
                        'top_probs'           : top_probs,
                        'is_vague'            : True,
                        'reason'              : reason,
                        'mode'                : 'Auto',
                        'status'              : 'clarified' if text_was_rewritten else 'classified',
                        'grammar_was_changed' : was_changed,
                        'grammar_issues'      : issues,
                        'grammar_original'    : auto_refined_pre_grammar,
                        'grammar_fixed'       : fixed_text,
                        'assumption'          : assumption,
                        'auto_refined'        : auto_refined_pre_grammar,
                        'final_label'         : final_label,
                        'final_conf'          : bert_conf,
                        'questions'           : None,
                        'answers'             : None,
                        'interactive_refined' : None,
                    })
                    st.success("Added to registry")
                    needs_rerun = True

                else:
                    # Interactive mode — queue for clarification form below.
                    # Generate questions NOW and store them in session state
                    # so they are not regenerated on every Streamlit rerun.
                    with st.spinner("Generating clarification questions..."):
                        analysis  = analyze_ambiguity(req_text)
                        questions = generate_clarification_questions(req_text, analysis)

                    st.session_state.pending_clarifications.append({
                        'text'      : req_text,
                        'reason'    : reason,
                        'index'     : i,
                        'analysis'  : analysis,
                        'questions' : questions,
                        'label'     : label,
                        'confidence': confidence,
                        'top_probs' : top_probs,
                    })
                    st.warning(
                        "This requirement needs clarification — "
                        "scroll down to answer the questions."
                    )

            # ── Route: Clear requirement ──────────────────────────────────
            else:
                req_text_pre_grammar = req_text   # save before grammar fix

                # Grammar + completeness check
                with st.spinner("Checking requirement quality..."):
                    fixed_text, was_changed, issues = check_and_fix_requirement(req_text)

                if was_changed:
                    st.warning("**Quality issues detected and fixed:**")
                    for issue in issues:
                        st.markdown(f"  - {issue}")
                    st.markdown(f"**Original:** {req_text}")
                    st.markdown(f"**Fixed:**    {fixed_text}")
                    req_text = fixed_text

                result = {
                    'original_text': user_input.strip(),
                    'final_text'   : req_text,
                    'label'        : label,
                    'confidence'   : confidence,
                    'iterations'   : 1,
                    'status'       : 'classified',
                }
                add_to_registry(result, project_name)
                collected_results.append({
                    'original_input'      : user_input.strip(),
                    'req_text'            : req_text_pre_grammar,
                    'label'               : label,
                    'confidence'          : confidence,
                    'top_probs'           : top_probs,
                    'is_vague'            : False,
                    'reason'              : '',
                    'mode'                : mode,
                    'status'              : 'classified',
                    'grammar_was_changed' : was_changed,
                    'grammar_issues'      : issues,
                    'grammar_original'    : req_text_pre_grammar,
                    'grammar_fixed'       : fixed_text,
                    'assumption'          : None,
                    'auto_refined'        : None,
                    'final_label'         : label,
                    'final_conf'          : confidence,
                    'questions'           : None,
                    'answers'             : None,
                    'interactive_refined' : None,
                })
                st.success(f"Added to registry as **{label}**")
                needs_rerun = True

        # ── Store collected results for persistent display ─────────────
        if len(collected_results) == 1:
            st.session_state.last_result  = collected_results[0]
        elif len(collected_results) > 1:
            st.session_state.last_results = collected_results

        # ── Rerun ONCE after the entire loop — not inside it ─────────────
        # Only rerun if there are no pending clarifications to show.
        # If there are pending clarifications we stay on the page to show forms.
        if needs_rerun and not st.session_state.pending_clarifications:
            st.session_state.registry_updated = True
            st.rerun()

    # ── Clarification forms — shown BELOW the classify button ─────────────
    # Render when there are pending interactive clarifications.
    # Questions are already stored in session state — no extra API calls.
    if st.session_state.pending_clarifications:
        st.divider()
        st.markdown("### Clarification Required")

        # Process one pending item at a time to keep the UI clean
        pending = st.session_state.pending_clarifications[0]

        st.markdown(
            f"**Requirement {pending['index'] + 1}:** `{pending['text']}`"
        )
        if pending.get('reason'):
            st.info(f"**Issue:** {pending['reason']}")

        questions = pending['questions']

        st.markdown("Please answer the following clarification questions:")

        with st.form("clarification_form"):
            answers = []
            for i, q in enumerate(questions):
                answer = st.text_input(
                    f"Q{i + 1}: {q['question']}",
                    key=f"answer_{pending['index']}_{i}"
                )
                answers.append(answer)

            submitted = st.form_submit_button("Submit Answers", type="primary")

        if submitted:
            current_text = pending['text']

            with st.spinner("Refining requirement based on your answers..."):
                refinement   = refine_requirement(current_text, questions, answers)
                refined_text = refinement['refined_requirement']
                text_was_rewritten = refined_text.strip() != current_text.strip()
                label, conf, all_probs_refined = classify_requirement(refined_text)

            # If still ambiguous after clarification — force classify with GPT
            if label == 'Ambiguous':
                with st.spinner("Force classifying with GPT..."):
                    gpt_label = gpt_second_opinion(refined_text)
                    label     = (
                        gpt_label
                        if gpt_label and gpt_label != 'Ambiguous'
                        else 'NFR_Other'
                    )

            refined_text_pre_grammar = refined_text   # save before grammar fix

            # Grammar + quality check on refined text
            with st.spinner("Checking requirement quality..."):
                fixed_text, was_changed, issues = check_and_fix_requirement(refined_text)

            if was_changed:
                st.warning("**Quality issues fixed:**")
                for issue in issues:
                    st.markdown(f"  - {issue}")
                refined_text = fixed_text

            st.success(f"**Refined:** {refined_text}")
            st.success(f"**Classified as:** {label} ({conf:.2%})")

            result = {
                'original_text': pending['text'],
                'final_text'   : refined_text,
                'label'        : label,
                'confidence'   : conf,
                'iterations'   : 1,
                'status'       : 'clarified' if text_was_rewritten else 'classified',
            }
            add_to_registry(result, project_name)

            top_probs_refined = dict(sorted(
                all_probs_refined.items(), key=lambda x: x[1], reverse=True
            )[:5])
            st.session_state.last_result = {
                'original_input'      : pending['text'],
                'req_text'            : pending['text'],
                'label'               : pending.get('label', 'Ambiguous'),
                'confidence'          : pending.get('confidence', 0.0),
                'top_probs'           : pending.get('top_probs', top_probs_refined),
                'is_vague'            : True,
                'reason'              : pending.get('reason', ''),
                'mode'                : 'Interactive',
                'grammar_was_changed' : was_changed,
                'grammar_issues'      : issues,
                'grammar_original'    : refined_text_pre_grammar,
                'grammar_fixed'       : fixed_text,
                'assumption'          : None,
                'auto_refined'        : None,
                'final_label'         : label,
                'final_conf'          : conf,
                'questions'           : questions,
                'answers'             : answers,
                'interactive_refined' : refined_text_pre_grammar,
            }

            # Remove the processed item and move to next
            st.session_state.pending_clarifications.pop(0)
            st.session_state.registry_updated = True
            st.rerun()

        # Show how many are left
        remaining = len(st.session_state.pending_clarifications)
        if remaining > 1:
            st.info(
                f"After submitting, {remaining - 1} more requirement(s) "
                f"will need clarification."
            )

    # ── Persistent result display ──────────────────────────────────────────
    if st.session_state.last_result is not None or st.session_state.last_results is not None:
        st.divider()
        st.markdown("### Last Processed Requirement")

        results_to_display = (
            st.session_state.last_results
            if st.session_state.last_results is not None
            else [st.session_state.last_result]
        )

        large_batch = len(results_to_display) > 5

        # Show input text only for small batches (large inputs would be huge)
        if not large_batch:
            original_input_disp = results_to_display[0].get('original_input', '')
            if original_input_disp:
                st.markdown(f"**Input:** `{original_input_disp}`")

        if large_batch:
            st.info(
                f"Input contained **{len(results_to_display)} requirements** "
                f"— each processed individually."
            )

        def _render_result(res):
            """Render the detail view for one result dict."""
            col_a, col_b = st.columns(2)
            with col_a:
                if res.get('label') == 'Ambiguous' or res.get('is_vague'):
                    reason_str = res.get('reason', '')
                    st.warning(
                        f"**Ambiguous** (confidence: {res.get('confidence', 0.0):.2%})"
                        + (f" — {reason_str}" if reason_str else "")
                    )
                else:
                    st.success(
                        f"**{res.get('label')}** ({res.get('confidence', 0.0):.2%})"
                    )
            with col_b:
                top_probs_disp = res.get('top_probs', {})
                if top_probs_disp:
                    prob_df_disp = pd.DataFrame(
                        list(top_probs_disp.items()), columns=['Label', 'Probability']
                    )
                    st.dataframe(prob_df_disp, hide_index=True, use_container_width=True)

            if res.get('mode') == 'Auto' and res.get('is_vague'):
                if res.get('assumption'):
                    st.info(f"**Assumed:** {res['assumption']}")
                if res.get('auto_refined'):
                    st.success(f"**Refined:** {res['auto_refined']}")

            if res.get('mode') == 'Interactive' and res.get('is_vague'):
                questions_disp = res.get('questions') or []
                answers_disp   = res.get('answers') or []
                if questions_disp:
                    st.markdown("**Clarification Q&A:**")
                    for q_item, a_item in zip(questions_disp, answers_disp):
                        st.markdown(f"- **Q:** {q_item['question']}")
                        if a_item:
                            st.markdown(f"  **A:** {a_item}")
                if res.get('interactive_refined'):
                    st.success(f"**Refined:** {res['interactive_refined']}")

            if res.get('grammar_was_changed'):
                st.warning("**Quality issues detected and fixed:**")
                for issue in (res.get('grammar_issues') or []):
                    st.markdown(f"  - {issue}")
                st.markdown(f"**Original:** {res.get('grammar_original', '')}")
                st.markdown(f"**Fixed:**    {res.get('grammar_fixed', '')}")

            final_lbl = res.get('final_label', res.get('label', ''))
            st.success(f"Added to registry as **{final_lbl}**")

        for idx, res in enumerate(results_to_display):
            final_label_disp = res.get('final_label', res.get('label', ''))
            conf_disp        = res.get('final_conf',  res.get('confidence', 0.0))
            status_disp      = res.get('status', 'classified')
            req_full         = res.get('req_text', '')
            req_short        = req_full[:60] + ('…' if len(req_full) > 60 else '')

            if large_batch:
                expander_label = (
                    f"#{idx + 1}  {req_short}   ·   "
                    f"{final_label_disp}  {conf_disp:.0%}   ·   {status_disp}"
                )
                with st.expander(expander_label):
                    _render_result(res)
            else:
                if len(results_to_display) > 1:
                    st.markdown(
                        f"**Requirement {idx + 1} of {len(results_to_display)}:**"
                    )
                _render_result(res)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Requirements Registry
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Requirements Registry — {project_name}")

    df = get_registry_df(project_name)

    if df.empty:
        st.info(
            "No requirements yet. "
            "Go to 'Elicit Requirements' to add some."
        )
    else:
        # Summary metrics
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("**Total Requirements**",     len(df))
            col2.metric("FR",        len(df[df['label'] == 'FR']))
            col3.metric("NFR",       len(df[df['label'].str.startswith('NFR_', na=False)]))
            col4.metric("Classified", len(df[df['status'] == 'classified']))
            col5.metric("Clarified", len(df[df['status'] == 'clarified']))

        st.divider()

        # Filter by label
        all_labels = ['All'] + sorted(df['label'].unique().tolist())
        selected   = st.selectbox("Filter by label", all_labels)
        filtered   = df if selected == 'All' else df[df['label'] == selected]

        # Table
        display_cols = ['final_text', 'label', 'confidence', 'status', 'timestamp']
        st.dataframe(
            filtered[display_cols].rename(columns={
                'final_text' : 'Requirement',
                'label'      : 'Label',
                'confidence' : 'Confidence',
                'status'     : 'Status',
                'timestamp'  : 'Timestamp',
            }),
            use_container_width = True,
            hide_index          = True
        )

        # Label distribution chart
        st.divider()
        st.markdown("#### Label Distribution")
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        st.bar_chart(label_counts.set_index('Label'))

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Generate SRS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Generate SRS Document")

    df         = get_registry_df(project_name)
    classified = df[~df['label'].isin(['Ambiguous', 'Requires Manual Review'])]

    if classified.empty:
        st.warning("No classified requirements yet. Add requirements first.")
    else:
        st.info(
            f"Ready to generate SRS from **{len(classified)}** "
            f"classified requirements."
        )

        with st.expander("Project Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                srs_project = st.text_input("Project Name",  value=project_name)
                srs_version = st.text_input("Version",        value="1.0")
                srs_org     = st.text_input("Organization",   value="")
            with col2:
                srs_authors = st.text_input("Author(s)",      value="")
                srs_status  = st.selectbox(
                    "Document Status",
                    ["Draft", "Review", "Approved", "Final"]
                )
                srs_desc = st.text_area(
                    "Project Description",
                    height=80,
                    placeholder="Brief description of the project..."
                )

        generate_btn = st.button("Generate SRS Document", type="primary")

        if generate_btn:
            context = {
                "project_name"    : srs_project,
                "version"         : srs_version,
                "date"            : datetime.now().strftime("%B %d, %Y"),
                "organization"    : srs_org,
                "authors"         : [srs_authors],
                "document_status" : srs_status,
                "description"     : srs_desc or f"Software requirements for {srs_project}",
                "intended_users"  : [
                    "Software developers",
                    "Requirements engineers",
                    "Project managers",
                ],
                "scope": (
                    srs_desc
                    or f"This document defines the requirements for {srs_project}."
                ),
            }

            with st.spinner("Generating SRS document — this may take 1-2 minutes..."):
                progress = st.progress(0,  text="Generating Introduction...")
                progress.progress(25,      text="Generating Overall Description...")
                progress.progress(50,      text="Generating Functional Requirements...")
                progress.progress(75,      text="Generating Non-Functional Requirements...")

                output_path = generate_srs_from_registry(
                    project_name = project_name,
                    db_path      = DB_PATH,
                    context      = context
                )

                progress.progress(100, text="Complete!")

            if output_path and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    docx_bytes = f.read()

                st.success("SRS document generated successfully!")
                st.download_button(
                    label     = "Download SRS Document",
                    data      = docx_bytes,
                    file_name = (
                        f"SRS_{srs_project.replace(' ', '_')}"
                        f"_v{srs_version}.docx"
                    ),
                    mime = (
                        "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document"
                    ),
                )
            else:
                st.error(
                    "SRS generation failed — "
                    "check your registry's requirements."
                )