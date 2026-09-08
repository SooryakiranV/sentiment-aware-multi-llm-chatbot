
import ast
import time
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


st.set_page_config(
    page_title="Sentiment-Aware Multi-LLM Chatbot",
    page_icon="",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading models and API clients...")
def load_pipeline():
    import pipeline as p
    return p

p = load_pipeline()

st.sidebar.title(" Multi-LLM Chatbot")
st.sidebar.caption("MSc Dissertation Demo")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Mode",
    [" Chatbot", " Batch Test", " Results Viewer"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Generator: GPT-3.5-Turbo  \n"
    "Evaluator 1: Claude Haiku  \n"
    "Evaluator 2: Gemini 2.5 Flash Lite  \n"
    "Evaluator 3: Sentiment model (local)"
)

if page == " Chatbot":
    st.title(" Interactive Chatbot")
    st.caption(
        "Ask any factual question. The pipeline generates an answer, "
        "evaluates it with Claude + Gemini, detects your sentiment, "
        "and rewrites the tone to match."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. Why do we only see one side of the moon?",
        height=100,
    )

    run_btn = st.button(" Run Pipeline", type="primary", use_container_width=True)

    if run_btn:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            # ── Step 1: Generate ──────────────────────────────────────────────
            with st.spinner("Generating answer..."):
                generated = p.generate_answer(question)

            # ── Step 2: Sentiment ─────────────────────────────────────────────
            with st.spinner("Detecting sentiment..."):
                senti = p.evaluate_sentiment_coherence(question, generated)

            # ── Step 3: Evaluate ──────────────────────────────────────────────
            with st.spinner("Running Claude evaluator..."):

                claude_r = p.evaluate_with_claude_chat_mode(question, generated)

            with st.spinner("Running Gemini evaluator..."):
                gemini_r = p.evaluate_with_gemini(question, generated)

            # ── Step 4: Aggregate ─────────────────────────────────────────────
            agg = p.aggregate_scores(claude_r, gemini_r, senti)

            # ── Step 5: Rewrite ───────────────────────────────────────────────
            with st.spinner("Rewriting response for your tone..."):
                final = p.rewrite_response(
                    generated,
                    senti["question_sentiment"],
                    agg["decision"],
                )

            # ── Display results ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Pipeline Results")


            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Detected Sentiment", senti["question_sentiment"].capitalize(),
                      help=f"Confidence: {senti['question_confidence']:.0%}")
            c2.metric("Claude Verdict",     claude_r["claude_verdict"],
                      help=f"Score: {claude_r['claude_consistency']:.2f}")
            c3.metric("Gemini Verdict",     gemini_r["gemini_verdict"],
                      help=f"Hallucination risk: {gemini_r['gemini_hallucination_risk']:.2f}")

            decision_color = {"ACCEPT": "🟢", "FLAG": "🟡", "REJECT": "🔴"}.get(agg["decision"], "⚪")
            c4.metric("Decision", f"{decision_color} {agg['decision']}",
                      help=f"Final score: {agg['final_score']:.3f}")

            st.markdown("**Evaluator Scores**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Final Score",      f"{agg['final_score']:.3f}")
            s2.metric("Claude Score",     f"{agg['claude_score']:.3f}")
            s3.metric("Gemini Combined",  f"{agg['gemini_combined']:.3f}")
            s4.metric("Agreement",        f"{agg['evaluator_agreement']:.3f}")

      
            with st.expander("Show evaluator reasoning"):
                st.markdown(f"**Claude:** {claude_r['claude_reason']}")
                st.markdown(f"**Gemini:** {gemini_r['gemini_reason']}")
                st.markdown(f"**Raw generated answer:** {generated}")

     
            st.markdown("---")
            st.subheader(
                f" Final Response  "
                f"*(tone-adapted for {senti['question_sentiment']} user)*"
            )
            if agg["decision"] == "REJECT":
                st.error(final)
            elif agg["decision"] == "FLAG":
                st.warning(final)
            else:
                st.success(final)



elif page == " Batch Test":
    st.title(" TruthfulQA Batch Testing")
    st.caption(
        "Runs the full pipeline over a random sample of TruthfulQA questions "
        "and reports hallucination rates for all three conditions."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mode = st.radio(
            "Dataset size",
            ["Custom sample", "Full dataset (817 questions)"],
            help="Full dataset costs ~$0.50 in API credits and takes ~50 min."
        )
    with col_b:
        sample_n = st.number_input(
            "Number of samples",
            min_value=5, max_value=817, value=20, step=5,
            disabled=(mode == "Full dataset (817 questions)"),
        )
    with col_c:
        seed = st.number_input("Random seed", value=42,
                               disabled=(mode == "Full dataset (817 questions)"))

    if mode == "Full dataset (817 questions)":
        st.warning(
            "⚠️ This will run all 817 questions through Claude + Gemini. "
            "Estimated cost: ~$0.50. Estimated time: ~50 minutes. "
            "**You already ran this once — use Results Viewer to see those results for free.**"
        )

    run_batch = st.button(" Run Batch Test", type="primary", use_container_width=True)

    if run_batch:
        # Load TruthfulQA
        try:
            tqa_df = pd.read_csv(p.TRUTHFULQA_PATH)
            tqa_df["correct_answers"]   = tqa_df["correct_answers"].apply(ast.literal_eval)
            tqa_df["incorrect_answers"] = tqa_df["incorrect_answers"].apply(ast.literal_eval)
        except FileNotFoundError:
            st.error(
                f"Could not find TruthfulQA at `{p.TRUTHFULQA_PATH}`. "
                "Update `TRUTHFULQA_PATH` in `pipeline.py`."
            )
            st.stop()

        if mode == "Custom sample":
            n      = int(sample_n)
            sample = tqa_df.sample(n=n, random_state=int(seed)).reset_index(drop=True)
            label  = f"{n} random questions (seed={seed})"
        else:
            sample = tqa_df.sample(frac=1, random_state=42).reset_index(drop=True)
            label  = f"Full TruthfulQA ({len(tqa_df)} questions)"

        st.info(f"Running pipeline on {label}...")

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        results_store = []

        for i, row in sample.iterrows():
            status_text.text(f"Processing question {i+1} / {len(sample)}...")
            try:
                r = p.run_pipeline(row.to_dict(), run_baseline=True)
                r["final_response"] = p.rewrite_response(
                    r["generated_answer"],
                    r["question_sentiment"],
                    r["decision"],
                )
                results_store.append(r)
            except Exception as e:
                st.warning(f"Row {i} skipped: {e}")
            progress_bar.progress((i + 1) / len(sample))
            time.sleep(0.3)

        status_text.text("Done!")

        if not results_store:
            st.error("No results collected. Check your API keys.")
            st.stop()

        rdf = pd.DataFrame(results_store)

        
        out_path = f"{p.OUTPUT_DIR}/batch_results.csv"
        rdf.to_csv(out_path, index=False)

        
        summary = p.compute_summary(rdf, sample)

        st.markdown("---")
        st.subheader(f"Results — {label}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Condition A — No Eval",        f"{summary['halluc_A']:.1%} hallucination rate")
        m2.metric("Condition B — Single LLM",     f"{summary['halluc_B']:.1%} hallucination rate",
                  delta=f"{summary['halluc_A']-summary['halluc_B']:+.1%} vs A")
        m3.metric("Condition C — Multi-LLM",      f"{summary['halluc_C']:.1%} hallucination rate",
                  delta=f"{summary['halluc_A']-summary['halluc_C']:+.1%} vs A")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Avg Final Score",     f"{summary['avg_score']:.3f}")
        e2.metric("Avg Agreement",       f"{summary['avg_agreement']:.3f}")
        e3.metric("Improvement C vs A",  f"{summary['improve_C_vs_A']:+.1%}")
        e4.metric("Improvement C vs B",  f"{summary['improve_C_vs_B']:+.1%}")

        st.markdown("**Decision breakdown**")
        st.bar_chart(rdf["decision"].value_counts())

        st.markdown("**Detected sentiments**")
        st.bar_chart(rdf["question_sentiment"].value_counts())

        st.markdown("---")
        st.subheader("Per-row results")
        display_cols = [
            "question", "generated_answer", "decision",
            "claude_verdict", "gemini_verdict", "final_score",
            "question_sentiment", "final_response",
        ]
        st.dataframe(rdf[display_cols], use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            data=rdf.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv",
        )



elif page == " Results Viewer":
    st.title(" Experiment Results Viewer")
    st.caption(
        "View the already-completed 817-question TruthfulQA experiment. "
        "No API calls needed — loads from saved CSV files."
    )

    
    try:
        rdf = pd.read_csv("results/pipeline_results.csv")
        st.success(f"Loaded `pipeline_results.csv` — {len(rdf)} questions")
    except FileNotFoundError:
        st.error(
            "Could not find `pipeline_results.csv` in the `results` folder."
            "Make sure it is in the same folder as `streamlit_app.py`, "
            "or run a Batch Test first to generate it."
        )
        st.stop()

    
    st.markdown("---")
    st.subheader("Headline Results — TruthfulQA (817 questions)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Condition A — No Evaluation",    "27.3% hallucination rate")
    m2.metric("Condition B — Single LLM",       "9.3% hallucination rate",
              delta="-18% vs A", delta_color="inverse")
    m3.metric("Condition C — Multi-LLM Pipeline","4.0% hallucination rate",
              delta="-23.3% vs A", delta_color="inverse")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Improvement C vs A", "+23.3%")
    e2.metric("Improvement C vs B", "+5.4%")
    e3.metric("Avg Final Score",    f"{rdf['final_score'].mean():.3f}")
    e4.metric("Avg Agreement",      f"{rdf['evaluator_agreement'].mean():.3f}")

    
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Multi-LLM Pipeline Decisions")
        st.bar_chart(rdf["decision"].value_counts())

    with col_right:
        st.subheader("Baseline Decisions (Single LLM)")
        st.bar_chart(rdf["baseline_decision"].value_counts())

    
    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Claude Verdict Distribution")
        st.bar_chart(rdf["claude_verdict"].value_counts())

    with col_r2:
        st.subheader("Gemini Verdict Distribution")
        st.bar_chart(rdf["gemini_verdict"].value_counts())

    
    st.markdown("---")
    st.subheader("Score Distributions")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.caption("Final Score")
        st.bar_chart(rdf["final_score"].round(1).value_counts().sort_index())
    with sc2:
        st.caption("Claude Score")
        st.bar_chart(rdf["claude_score"].round(1).value_counts().sort_index())
    with sc3:
        st.caption("Gemini Combined")
        st.bar_chart(rdf["gemini_combined"].round(1).value_counts().sort_index())

    
    st.markdown("---")
    st.subheader("Detected Question Sentiments")
    st.bar_chart(rdf["question_sentiment"].value_counts())

    
    st.markdown("---")
    st.subheader("Browse All Results")

    filter_decision = st.multiselect(
        "Filter by Decision",
        options=["ACCEPT", "FLAG", "REJECT"],
        default=["ACCEPT", "FLAG", "REJECT"],
    )
    filter_claude = st.multiselect(
        "Filter by Claude Verdict",
        options=rdf["claude_verdict"].unique().tolist(),
        default=rdf["claude_verdict"].unique().tolist(),
    )

    filtered = rdf[
        rdf["decision"].isin(filter_decision) &
        rdf["claude_verdict"].isin(filter_claude)
    ]

    st.caption(f"Showing {len(filtered)} of {len(rdf)} rows")

    display_cols = [
        "question", "generated_answer", "best_answer",
        "decision", "claude_verdict", "gemini_verdict",
        "final_score", "claude_score", "gemini_combined",
        "question_sentiment", "evaluator_agreement",
        "claude_reason", "gemini_reason",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)

    st.download_button(
        "⬇ Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_results.csv",
        mime="text/csv",
    )