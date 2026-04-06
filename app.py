import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from agent.classifier    import classify_requirement
from agent.clarification import (
    analyze_ambiguity, generate_clarification_questions,
    refine_requirement, auto_refine_ambiguous, check_and_fix_requirement
)
from agent.srs_generator import generate_srs_from_registry
from agent.classifier    import gpt_second_opinion

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Requirements Elicitation Agent",
    page_icon  = "<placeholder>",
    layout     = "wide"
)

# ── Registry helper ───────────────────────────────────────────────────────
DB_PATH = './data/requirements_registry.db'

def get_registry_df(project_name):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT * FROM requirements WHERE project_name = ?',
        (project_name,)
    ).fetchall()
    conn.close()
    cols = ['id','original_text','final_text','label',
            'confidence','iterations','status','timestamp','project_name']
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
            result['label'], result['confidence'],
            result['iterations'], result['status'],
            datetime.now().isoformat(), project_name
        ))
        conn.commit()
    conn.close()

def clear_registry(project_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM requirements WHERE project_name = ?', (project_name,))
    conn.commit()
    conn.close()

# ── Session state init ────────────────────────────────────────────────────
if 'project_name'    not in st.session_state:
    st.session_state.project_name    = "My Project"
if 'current_input'   not in st.session_state:
    st.session_state.current_input   = None
if 'questions'       not in st.session_state:
    st.session_state.questions       = []
if 'analysis'        not in st.session_state:
    st.session_state.analysis        = None
if 'awaiting_answers' not in st.session_state:
    st.session_state.awaiting_answers = False
if 'iteration'       not in st.session_state:
    st.session_state.iteration       = 0
if 'refined_text'    not in st.session_state:
    st.session_state.refined_text    = None
if 'registry_updated' not in st.session_state:
    st.session_state.registry_updated = False

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
    
    sidebar_df = get_registry_df(project_name)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Total",     len(sidebar_df))
    col_m2.metric("Clarified", len(sidebar_df[sidebar_df['status'] == 'clarified']))

    col_m3, col_m4 = st.columns(2)
    col_m3.metric("FR",  len(sidebar_df[sidebar_df['label'] == 'FR']))
    col_m4.metric("NFR", len(sidebar_df[sidebar_df['label'].str.startswith('NFR_', na=False)]))

    # st.metric("Total Requirements", len(df))
    # if not df.empty:
    #     classified = len(df[df['status'] == 'classified'])
    #     clarified  = len(df[df['status'] == 'clarified'])
    #     st.metric("Classified directly", classified)
    #     st.metric("Clarified via agent", clarified)

    if st.button("<placeholder> Clear Registry", type="secondary"):
        clear_registry(project_name)
        st.session_state.registry_updated = False
        st.success("Registry cleared")
        st.rerun()

    st.divider()
    st.caption("AI Requirements Elicitation Agent")
    st.caption("Academic Project — BERT + GPT-4o")

# ── Main content ──────────────────────────────────────────────────────────
st.title("AI Requirements Elicitation Agent")
st.caption("Classify, clarify, and document software requirements automatically")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "<placeholder> Elicit Requirements",
    "<placeholder> Requirements Registry",
    "<placeholder> Generate SRS"
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
            placeholder="e.g. The system shall allow users to reset their password via email.",
            height=100,
            key="requirement_input"
        )
    with col2:
        st.markdown("##### Mode")
        mode = st.radio(
            "Processing mode",
            ["Interactive", "Auto"],
            help=(
                "Interactive: agent asks you clarification questions\n"
                "Auto: GPT auto-refines vague requirements"
            )
        )

    classify_btn = st.button("<placeholder> Classify Requirement", type="primary")

    # ── Classification ────────────────────────────────────────────────────
    if classify_btn and user_input.strip():
        # ── Reset ALL session state for fresh requirement ─────────────────
        st.session_state.current_input     = user_input.strip()
        st.session_state.awaiting_answers  = False
        st.session_state.questions         = []
        st.session_state.analysis          = None
        st.session_state.iteration         = 0
        st.session_state.refined_text      = None
        st.session_state.pending_requirements = []
        st.session_state.mode              = mode  # save mode at classification time

        with st.spinner("Classifying..."):
            label, confidence, all_probs = classify_requirement(user_input.strip())

        st.session_state.last_label      = label
        st.session_state.last_confidence = confidence
        st.session_state.all_probs       = all_probs
        
        # Show confidence chart
        st.markdown("#### Classification Result")
        col_a, col_b = st.columns(2)

        with col_a:
            if label == 'Ambiguous':
                st.warning(f"<placeholder> **Ambiguous** (confidence: {confidence:.2%})")
            else:
                st.success(f"<placeholder> **{label}** (confidence: {confidence:.2%})")

        with col_b:
            top_probs = dict(sorted(
                all_probs.items(), key=lambda x: x[1], reverse=True
            )[:5])
            prob_df = pd.DataFrame(
                list(top_probs.items()),
                columns=['Label', 'Probability']
            )
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        # ── Handle ambiguous ──────────────────────────────────────────────
        if label == 'Ambiguous':
            if mode == "Auto":
                with st.spinner("Auto-refining with GPT..."):
                    gpt_label     = gpt_second_opinion(user_input.strip())
                    final_label   = gpt_label if gpt_label and gpt_label != 'Ambiguous' else 'NFR_Other'
                    refined, assumption = auto_refine_ambiguous(user_input.strip(), final_label)
                    bert_label, bert_conf, _ = classify_requirement(refined)
                    if bert_label != 'Ambiguous' and bert_conf >= 0.75:
                        final_label = bert_label

                st.info(f"<placeholder> **Assumed:** {assumption}")
                st.success(f"<placeholder> **Refined:** {refined}")
                st.success(f"<placeholder> **Label:** {final_label}")

                result = {
                    'original_text': user_input.strip(),
                    'final_text'   : refined,
                    'label'        : final_label,
                    'confidence'   : bert_conf if bert_label != 'Ambiguous' else confidence,
                    'iterations'   : 1,
                    'status'       : 'clarified'
                }
                add_to_registry(result, project_name)
                # ── Force sidebar to update immediately ───────────────────
                st.session_state.registry_updated = True
                st.rerun()
                st.success("<placeholder> Added to registry")

            else:
                # Interactive mode — generate questions
                with st.spinner("Analyzing ambiguity..."):
                    analysis  = analyze_ambiguity(user_input.strip())
                    questions = generate_clarification_questions(
                        user_input.strip(), analysis
                    )
                st.session_state.analysis         = analysis
                st.session_state.questions        = questions
                st.session_state.awaiting_answers = True
                st.session_state.iteration        = 1

        else:
            # Clear requirement — save directly
            result = {
                'original_text': user_input.strip(),
                'final_text'   : user_input.strip(),
                'label'        : label,
                'confidence'   : confidence,
                'iterations'   : 1,
                'status'       : 'classified'
            }
            add_to_registry(result, project_name)
            # ── Force sidebar to update immediately ───────────────────────
            st.session_state.registry_updated = True
            st.rerun()
            st.success("<placeholder> Added to registry")

    # ── Clarification questions form ──────────────────────────────────────
    if st.session_state.awaiting_answers and st.session_state.questions:
        st.divider()
        st.markdown(
            f"#### <placeholder> Clarification Needed "
            f"(Iteration {st.session_state.iteration}/{3})"
        )
        st.info(
            f"**Issue:** {st.session_state.analysis.get('summary', '')}"
        )

        answers = []
        with st.form("clarification_form"):
            for i, q in enumerate(st.session_state.questions):
                answer = st.text_input(
                    f"Q{i+1}: {q['question']}",
                    key=f"answer_{i}"
                )
                answers.append(answer)

            submitted = st.form_submit_button("Submit Answers", type="primary")

        if submitted:
            current_text = (
                st.session_state.refined_text
                or st.session_state.current_input
            )

            with st.spinner("Refining requirement..."):
                refinement   = refine_requirement(
                    current_text,
                    st.session_state.questions,
                    answers
                )
                refined_text = refinement['refined_requirement']
                label, conf, _ = classify_requirement(refined_text)

            st.session_state.refined_text = refined_text

            st.markdown(f"**<placeholder> Refined:** {refined_text}")

            if label != 'Ambiguous':
                st.success(f"<placeholder> Classified as **{label}** ({conf:.2%})")
                st.session_state.awaiting_answers = False

                result = {
                    'original_text': st.session_state.current_input,
                    'final_text'   : refined_text,
                    'label'        : label,
                    'confidence'   : conf,
                    'iterations'   : st.session_state.iteration,
                    'status'       : 'clarified'
                }
                add_to_registry(result, project_name)
                st.success("<placeholder> Added to registry")
                st.rerun()

            elif st.session_state.iteration < 3:
                # Still ambiguous — another round
                with st.spinner("Generating follow-up questions..."):
                    analysis  = analyze_ambiguity(refined_text)
                    questions = generate_clarification_questions(
                        refined_text, analysis
                    )
                st.session_state.analysis         = analysis
                st.session_state.questions        = questions
                st.session_state.awaiting_answers = True
                st.session_state.iteration        += 1
                st.rerun()

            else:
                # Max iterations — force classify with GPT
                with st.spinner("Force classifying..."):
                    gpt_label   = gpt_second_opinion(refined_text)
                    final_label = (
                        gpt_label
                        if gpt_label and gpt_label != 'Ambiguous'
                        else 'NFR_Other'
                    )
                st.warning(f"<placeholder> Max iterations reached — classified as {final_label}")
                st.session_state.awaiting_answers = False

                result = {
                    'original_text': st.session_state.current_input,
                    'final_text'   : refined_text,
                    'label'        : final_label,
                    'confidence'   : conf,
                    'iterations'   : 3,
                    'status'       : 'clarified'
                }
                add_to_registry(result, project_name)
                st.success("<placeholder> Added to registry")
                st.rerun()

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Requirements Registry
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Requirements Registry — {project_name}")

    df = get_registry_df(project_name)

    if df.empty:
        st.info("No requirements yet. Go to 'Elicit Requirements' to add some.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total",      len(df))
        col2.metric("FR",         len(df[df['label'] == 'FR']))
        col3.metric("NFR",        len(df[df['label'].str.startswith('NFR_', na=False)]))
        col4.metric("Clarified",  len(df[df['status'] == 'clarified']))

        st.divider()

        # Filter
        all_labels = ['All'] + sorted(df['label'].unique().tolist())
        selected   = st.selectbox("Filter by label", all_labels)
        filtered   = df if selected == 'All' else df[df['label'] == selected]

        # Display table
        display_cols = ['final_text', 'label', 'confidence', 'status', 'timestamp']
        st.dataframe(
            filtered[display_cols].rename(columns={
                'final_text' : 'Requirement',
                'label'      : 'Label',
                'confidence' : 'Confidence',
                'status'     : 'Status',
                'timestamp'  : 'Timestamp'
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

    df = get_registry_df(project_name)
    classified = df[
        ~df['label'].isin(['Ambiguous', 'Requires Manual Review'])
    ]

    if classified.empty:
        st.warning("No classified requirements yet. Add requirements first.")
    else:
        st.info(
            f"Ready to generate SRS from **{len(classified)}** "
            f"classified requirements."
        )

        # Project context form
        with st.expander("<placeholder> Project Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                srs_project = st.text_input("Project Name", value=project_name)
                srs_version = st.text_input("Version", value="1.0")
                srs_org     = st.text_input("Organization", value="")
            with col2:
                srs_authors = st.text_input("Author(s)", value="")
                srs_status  = st.selectbox(
                    "Document Status",
                    ["Draft", "Review", "Approved", "Final"]
                )
                srs_desc    = st.text_area(
                    "Project Description", height=80,
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
                "intended_users"  : ["Software developers", "Requirements engineers", "Project managers"],
                "scope"           : srs_desc or f"This document defines the requirements for {srs_project}."
            }

            with st.spinner("Generating SRS document — this may take 1-2 minutes..."):
                progress = st.progress(0, text="Generating Introduction...")
                progress.progress(25, text="Generating Overall Description...")
                progress.progress(50, text="Generating Functional Requirements...")
                progress.progress(75, text="Generating Non-Functional Requirements...")

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
                    label     = "<placeholder> Download SRS Document",
                    data      = docx_bytes,
                    file_name = f"SRS_{srs_project.replace(' ', '_')}_v{srs_version}.docx",
                    mime      = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("SRS generation failed — check your registry has requirements.")