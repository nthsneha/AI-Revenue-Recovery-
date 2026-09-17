"""
AI Revenue Recovery Agent — Streamlit dashboard.

Run:
    streamlit run dashboard.py

Wraps the existing LangGraph pipeline in agent/graph.py. Uses app.stream()
so each node's result renders as it completes rather than all at once.
"""

import json
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Revenue Recovery Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Measured on the held-out set. Update these if you retrain.
# --------------------------------------------------------------------------
MODEL_AUC = 0.6773
MODEL_BRIER = 0.2254
MODEL_PRECISION = 0.6178
MODEL_RECALL = 0.5609
MODEL_F1 = 0.5880
BAYES_AUC = 0.6761
BAYES_BRIER = 0.2257
BAYES_ACC = 0.6283
MAJORITY_BASELINE = 0.5257
LATENT_VARIANCE_SHARE = 0.388

DEMO_LOGS = {
    "Abandoned checkout — high value": (
        "[2026-09-04 10:15:32] INFO: Session user_8231 loaded checkout for "
        "cart_val 2500.0. User spent 12 mins on page. Connection closed by client."
    ),
    "Payment failure — low value": (
        "[2026-09-04 11:22:11] ERROR: Payment failed for user_1120. "
        "RZP_CODE: insufficient_funds. Amount: 15.00. Device: Mobile."
    ),
    "Overdue invoice": (
        "[2026-09-04 14:00:00] SYSTEM: Invoice inv_9921 for user_5531 is "
        "5 days past due. Amount: 450.00."
    ),
}

NODE_LABELS = {
    "triage": "Triage",
    "viability": "Viability Gate",
    "planner": "Orchestrator / Planner",
    "executor": "Executor",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def extract_text(content):
    """Gemini 3.x returns a list of content blocks, not a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return "".join(parts).strip()
    return str(content)


def get_tool_calls(msg):
    return getattr(msg, "tool_calls", None) or []


@st.cache_resource(show_spinner=False)
def load_agent():
    """Import the compiled graph once and reuse it across reruns."""
    from agent.graph import app
    return app


def probability_bar(prob, threshold):
    """Simple HTML meter showing the score against the gate threshold."""
    pct = max(0.0, min(1.0, prob)) * 100
    tpct = max(0.0, min(1.0, threshold)) * 100
    color = "#2e9e5b" if prob >= threshold else "#c0392b"
    return f"""
    <div style="position:relative;height:34px;background:#eceff3;
                border-radius:6px;overflow:hidden;margin:4px 0 2px 0;">
      <div style="width:{pct}%;height:100%;background:{color};
                  transition:width .4s ease;"></div>
      <div style="position:absolute;left:{tpct}%;top:0;height:100%;
                  border-left:2px dashed #333;"></div>
      <div style="position:absolute;left:10px;top:7px;color:#fff;
                  font-weight:600;font-size:14px;">{prob:.3f}</div>
    </div>
    <div style="font-size:12px;color:#777;margin-bottom:10px;">
      dashed line = gate threshold ({threshold:.2f})
    </div>
    """


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Recovery Agent")

    key_present = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if key_present:
        st.success("API key loaded", icon="✅")
    else:
        st.error("GEMINI_API_KEY not found", icon="⚠️")
        st.caption("Set it in .env or pass it into the container.")

    model_exists = os.path.exists("models/viability_model.pkl")
    if model_exists:
        st.success("Viability model loaded", icon="✅")
    else:
        st.warning("No .pkl — heuristic fallback", icon="⚠️")

    st.divider()

    threshold = st.slider(
        "Viability gate threshold",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="Events scoring below this are dropped without spending LLM calls.",
    )
    st.caption(
        "Note: this slider is for exploring the trade-off. The live pipeline "
        "uses the threshold hard-coded in `viability_node`."
    )

    recursion_limit = st.number_input(
        "Recursion limit", min_value=4, max_value=50, value=15,
        help="Caps planner/executor cycles so a loop can't run away.",
    )

    st.divider()
    st.caption("LangGraph · Gemini · XGBoost")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("AI Revenue Recovery Agent")
st.caption(
    "Autonomous pipeline that triages revenue-loss events from raw logs, "
    "scores recoverability, and orchestrates a recovery plan."
)

tab_live, tab_model, tab_about = st.tabs(
    ["Live Agent", "Model Validation", "How it works"]
)

# ==========================================================================
# TAB 1 — Live agent
# ==========================================================================
with tab_live:
    left, right = st.columns([3, 2])

    with left:
        choice = st.selectbox(
            "Scenario", list(DEMO_LOGS.keys()) + ["Custom log…"]
        )
        default_log = DEMO_LOGS.get(choice, "")
        log_input = st.text_area(
            "Raw application log", value=default_log, height=110,
            placeholder="[timestamp] LEVEL: message…",
        )

    with right:
        st.write("")
        st.write("")
        run = st.button("Run recovery agent", type="primary",
                        use_container_width=True, disabled=not key_present)
        st.caption("Each run costs several Gemini calls — mind free-tier quota.")

    st.divider()

    if run and log_input.strip():
        agent = load_agent()

        initial_state = {
            "unstructured_log": log_input.strip(),
            "metadata": {},
            "revenue_loss_mode": "",
            "is_viable": False,
            "messages": [],
        }

        # Placeholders so each node renders the moment it finishes.
        trace_box = st.container()
        metrics_row = st.empty()
        tool_box = st.container()

        seen_tools = []
        final_state = {}
        planner_turns = 0
        t0 = time.time()

        status = st.status("Running pipeline…", expanded=True)

        try:
            for chunk in agent.stream(
                initial_state, {"recursion_limit": int(recursion_limit)}
            ):
                for node, update in chunk.items():
                    final_state.update(update or {})
                    label = NODE_LABELS.get(node, node)

                    # ---------------- Triage ----------------
                    if node == "triage":
                        status.write(f"**{label}** — parsed log")
                        meta = update.get("metadata", {})
                        mode = update.get("revenue_loss_mode", "—")
                        with trace_box:
                            st.subheader("Triage")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("Loss mode", mode)
                            c2.json(meta)

                    # ---------------- Viability ----------------
                    elif node == "viability":
                        prob = update.get("recovery_probability", 0.0)
                        viable = update.get("is_viable", False)
                        status.write(
                            f"**{label}** — p={prob:.3f} → "
                            f"{'proceed' if viable else 'dropped'}"
                        )
                        with trace_box:
                            st.subheader("Viability Gate")
                            st.markdown(
                                probability_bar(prob, threshold),
                                unsafe_allow_html=True,
                            )
                            if viable:
                                st.success(
                                    f"Recoverable (p={prob:.3f}) — proceeding to planner"
                                )
                            else:
                                st.error(
                                    f"Not viable (p={prob:.3f}) — dropped before "
                                    "any LLM planning. Compute saved."
                                )

                    # ---------------- Planner ----------------
                    elif node == "planner":
                        planner_turns += 1
                        msgs = update.get("messages", [])
                        for m in msgs:
                            for tc in get_tool_calls(m):
                                seen_tools.append({
                                    "Tool": tc.get("name", "?"),
                                    "Arguments": json.dumps(
                                        tc.get("args", {}), ensure_ascii=False
                                    ),
                                })
                        status.write(f"**{label}** — turn {planner_turns}")

                    # ---------------- Executor ----------------
                    elif node == "executor":
                        status.write(f"**{label}** — tools returned")

                    # Re-render the tool table as it grows.
                    if seen_tools:
                        tool_box.empty()
                        with tool_box:
                            st.subheader("Tool executions")
                            st.dataframe(
                                pd.DataFrame(seen_tools),
                                use_container_width=True, hide_index=True,
                            )

            status.update(label="Pipeline complete", state="complete",
                          expanded=False)

        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.exception(exc)
            st.stop()

        elapsed = time.time() - t0

        # ---------------- Summary ----------------
        meta = final_state.get("metadata", {})
        with metrics_row.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Amount at risk", f"₹{meta.get('amount_at_risk', 0):,.2f}")
            m2.metric("Recovery probability",
                      f"{final_state.get('recovery_probability', 0):.3f}")
            m3.metric("Tools executed", len(seen_tools))
            m4.metric("Wall time", f"{elapsed:.1f}s")

        st.divider()
        if final_state.get("is_viable"):
            msgs = final_state.get("messages", [])
            if msgs:
                st.subheader("Agent output")
                st.success(extract_text(msgs[-1].content) or "(no text returned)")
        else:
            st.info(
                "Pipeline stopped at the viability gate — no recovery attempted."
            )

    elif run:
        st.warning("Paste a log first.")

# ==========================================================================
# TAB 2 — Model validation
# ==========================================================================
with tab_model:
    st.subheader("Is 0.677 AUC any good?")
    st.write(
        "In isolation that number looks weak. It isn't — it sits on the "
        "theoretical ceiling for this feature set. The training labels are "
        "generated stochastically with latent confounders that are "
        "deliberately withheld from the model, so a portion of the outcome "
        "is unknowable by construction, exactly as it would be in production."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f"{MODEL_AUC:.4f}",
              delta=f"{MODEL_AUC - BAYES_AUC:+.4f} vs ceiling")
    c2.metric("Brier score", f"{MODEL_BRIER:.4f}",
              delta=f"{MODEL_BRIER - BAYES_BRIER:+.4f} vs floor",
              delta_color="inverse")
    c3.metric("F1", f"{MODEL_F1:.4f}")
    c4.metric("Latent variance share", f"{LATENT_VARIANCE_SHARE:.1%}",
              help="Share of the logit variance carried by unobservable factors.")

    st.divider()

    lift = (MODEL_AUC - 0.5) / (BAYES_AUC - 0.5)
    st.markdown(f"### Captures **{lift:.1%}** of achievable signal")

    chart_df = pd.DataFrame(
        {"AUC": [0.5, MODEL_AUC, BAYES_AUC]},
        index=["Random baseline", "Our model", "Bayes ceiling"],
    )
    st.bar_chart(chart_df, height=260)

    st.divider()
    st.markdown("#### Why a ceiling exists")
    st.markdown(
        "- **Latent confounders** — a hidden per-user responsiveness trait, "
        "payment-channel outages, competitor promotions and pay-cycle timing "
        "all shift the outcome but are never exposed as features\n"
        "- **Stochastic labels** — outcomes are *sampled* from a probability "
        "rather than thresholded, so identical feature rows can differ\n"
        "- **Label noise** — 6% of outcomes are flipped, mimicking tracking error"
    )

    st.info(
        "An earlier version of the generator derived the label deterministically "
        "from the same columns used as features. Models scored near-perfectly "
        "while learning nothing generalisable. Knowing where the ceiling should "
        "be is what makes a suspiciously high score legible as a bug.",
        icon="💡",
    )

    st.caption(
        f"Bayes accuracy ceiling {BAYES_ACC:.4f} · majority baseline "
        f"{MAJORITY_BASELINE:.4f} · precision {MODEL_PRECISION:.4f} · "
        f"recall {MODEL_RECALL:.4f}"
    )

# ==========================================================================
# TAB 3 — About
# ==========================================================================
with tab_about:
    st.subheader("Pipeline")
    st.markdown(
        """
| Stage | What it does |
|---|---|
| **Triage** | Gemini parses the raw log into typed metadata via a Pydantic schema — user, amount, device, intent, loss mode |
| **Viability Gate** | XGBoost scores recoverability from the event *and* the user's payment history; below threshold the event is dropped before any planning |
| **Orchestrator / Planner** | Tool-calling LLM plans a recovery strategy and chains calls, feeding each result into the next decision |
| **Executor** | LangGraph `ToolNode` runs the selected tools and returns results to the planner |
"""
    )

    st.subheader("Available tools")
    st.dataframe(
        pd.DataFrame([
            ["check_alternate_payment_methods", "Look up saved UPI / cards"],
            ["generate_whatsapp_payment_link", "Create a Razorpay pay link"],
            ["apply_dynamic_discount", "Discount the cart to convert"],
            ["send_whatsapp_message", "Outreach over WhatsApp"],
            ["trigger_email_sequence", "Start an email drip"],
            ["query_bank_downtime_api", "Check gateway/bank health"],
        ], columns=["Tool", "Purpose"]),
        use_container_width=True, hide_index=True,
    )

    st.caption("All tools are mocked for the demo — no external calls are made.")