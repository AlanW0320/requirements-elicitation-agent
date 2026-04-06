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

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Requirements Elicitation Agent",
    page_icon  = "<placeholder>",
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
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("<placeholder> Requirements Agent")
    st.divider()

    st.subheader("Project Settings")
    project_name = st.text_input(
        "Project Name",
        value=st.session_state.project_name
    )
    st.session_state.project_name = project_name

    st.divider()
    st.subheader("Registry")

    # Always read fresh — reflects latest inserts immediately after st.rerun()
    sidebar_df = get_registry_df(project_name)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Total",     len(sidebar_df))
    col_m2.metric("Clarified", len(sidebar_df[sidebar_df['status'] == 'clarified']))

    col_m3, col_m4 = st.columns(2)
    col_m3.metric("FR",  len(sidebar_df[sidebar_df['label'] == 'FR']))
    col_m4.metric("NFR", len(sidebar_df[
        sidebar_df['label'].str.startswith('NFR_', na=False)
    ]))

    if st.button("<placeholder> Clear Registry", type="secondary"):
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
    "<placeholder> Elicit Requirements",
    "<placeholder> Requirements Registry",
    "<placeholder> Generate SRS",
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

    classify_btn = st.button("<placeholder> Classify Requirement", type="primary")

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

        # ── Step 1: Split compound requirements ───────────────────────────
        with st.spinner("Analyzing input..."):
            split_result = split_and_validate_input(user_input.strip())

        requirements_to_process = split_result.get('requirements', [])
        was_split               = split_result.get('was_split', False)

        if was_split:
            st.info(
                f"<placeholder> Input contains **{split_result['split_count']} requirements** "
                f"— processing each separately."
            )

        # ── Step 2: Process each requirement ─────────────────────────────
        # Collect all results first — only rerun AFTER the loop is complete
        needs_rerun = False

        for i, req_item in enumerate(requirements_to_process):
            req_text = req_item['text']        # ← use req_text, NOT user_input
            is_vague = req_item.get('is_vague', False)
            reason   = req_item.get('reason', '')

            st.divider()

            if len(requirements_to_process) > 1:
                st.markdown(f"**Requirement {i + 1} of {len(requirements_to_process)}:**")

            st.markdown(f"<placeholder> `{req_text}`")

            # ── Classify this individual requirement ──────────────────────
            with st.spinner("Classifying..."):
                label, confidence, all_probs = classify_requirement(req_text)

            # Show result
            col_a, col_b = st.columns(2)
            with col_a:
                if label == 'Ambiguous' or is_vague:
                    st.warning(
                        f"<placeholder> **Ambiguous** (confidence: {confidence:.2%})"
                        + (f" — {reason}" if reason else "")
                    )
                else:
                    st.success(f"<placeholder> **{label}** ({confidence:.2%})")
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
                        bert_label, bert_conf, _ = classify_requirement(refined)
                        if bert_label != 'Ambiguous' and bert_conf >= 0.75:
                            final_label = bert_label
                        else:
                            bert_conf = confidence

                    if assumption:
                        st.info(f"<placeholder> **Assumed:** {assumption}")
                    st.success(f"<placeholder> **Refined:** {refined}")
                    st.success(f"<placeholder> **Label:** {final_label}")

                    # Quality / grammar check
                    with st.spinner("Checking requirement quality..."):
                        fixed_text, was_changed, issues = check_and_fix_requirement(refined)

                    if was_changed:
                        st.warning("<placeholder> **Quality issues detected and fixed:**")
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
                        'status'       : 'clarified',
                    }
                    add_to_registry(result, project_name)
                    st.success("<placeholder> Added to registry")
                    needs_rerun = True

                else:
                    # Interactive mode — queue for clarification form below.
                    # Generate questions NOW and store them in session state
                    # so they are not regenerated on every Streamlit rerun.
                    with st.spinner("Generating clarification questions..."):
                        analysis  = analyze_ambiguity(req_text)
                        questions = generate_clarification_questions(req_text, analysis)

                    st.session_state.pending_clarifications.append({
                        'text'     : req_text,
                        'reason'   : reason,
                        'index'    : i,
                        'analysis' : analysis,
                        'questions': questions,       # stored — not regenerated
                    })
                    st.warning(
                        "<placeholder> This requirement needs clarification — "
                        "scroll down to answer the questions."
                    )

            # ── Route: Clear requirement ──────────────────────────────────
            else:
                # Grammar + completeness check
                with st.spinner("Checking requirement quality..."):
                    fixed_text, was_changed, issues = check_and_fix_requirement(req_text)

                if was_changed:
                    st.warning("<placeholder> **Quality issues detected and fixed:**")
                    for issue in issues:
                        st.markdown(f"  - {issue}")
                    st.markdown(f"**Original:** {req_text}")
                    st.markdown(f"**Fixed:**    {fixed_text}")
                    req_text = fixed_text      # use the fixed version

                result = {
                    'original_text': user_input.strip(),
                    'final_text'   : req_text,   # ← fixed text, not raw input
                    'label'        : label,
                    'confidence'   : confidence,
                    'iterations'   : 1,
                    'status'       : 'classified',
                }
                add_to_registry(result, project_name)
                st.success(f"<placeholder> Added to registry as **{label}**")
                needs_rerun = True

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
        st.markdown("### <placeholder> Clarification Required")

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

            submitted = st.form_submit_button("<placeholder> Submit Answers", type="primary")

        if submitted:
            current_text = pending['text']

            with st.spinner("Refining requirement based on your answers..."):
                refinement   = refine_requirement(current_text, questions, answers)
                refined_text = refinement['refined_requirement']
                label, conf, _ = classify_requirement(refined_text)

            # If still ambiguous after clarification — force classify with GPT
            if label == 'Ambiguous':
                with st.spinner("Force classifying with GPT..."):
                    gpt_label = gpt_second_opinion(refined_text)
                    label     = (
                        gpt_label
                        if gpt_label and gpt_label != 'Ambiguous'
                        else 'NFR_Other'
                    )

            # Grammar + quality check on refined text
            with st.spinner("Checking requirement quality..."):
                fixed_text, was_changed, issues = check_and_fix_requirement(refined_text)

            if was_changed:
                st.warning("<placeholder> **Quality issues fixed:**")
                for issue in issues:
                    st.markdown(f"  - {issue}")
                refined_text = fixed_text

            st.success(f"<placeholder> **Refined:** {refined_text}")
            st.success(f"<placeholder> **Classified as:** {label} ({conf:.2%})")

            result = {
                'original_text': pending['text'],
                'final_text'   : refined_text,
                'label'        : label,
                'confidence'   : conf,
                'iterations'   : 1,
                'status'       : 'clarified',
            }
            add_to_registry(result, project_name)

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total",     len(df))
        col2.metric("FR",        len(df[df['label'] == 'FR']))
        col3.metric("NFR",       len(df[df['label'].str.startswith('NFR_', na=False)]))
        col4.metric("Clarified", len(df[df['status'] == 'clarified']))

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

        with st.expander("<placeholder> Project Details", expanded=True):
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

        generate_btn = st.button("<placeholder> Generate SRS Document", type="primary")

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

                st.success("<placeholder> SRS document generated successfully!")
                st.download_button(
                    label     = "<placeholder> Download SRS Document",
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
                    "<placeholder> SRS generation failed — "
                    "check your registry has requirements."
                )