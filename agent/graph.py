import os
import pickle
import json
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Ensure you are running this relative to the project root or adjust imports
import agent.tools as my_tools

# -----------------
# 1. State Definition
# -----------------
class AgentState(TypedDict):
    unstructured_log: str
    metadata: dict
    revenue_loss_mode: str
    is_viable: bool
    recovery_probability: float
    messages: Annotated[Sequence[BaseMessage], add_messages]   

# -----------------
# 2. Pydantic Models for Phase 1
# -----------------
class TriageOutput(BaseModel):
    revenue_loss_mode: str = Field(
        description="Must be one of: PAYMENT_FAILURE, ABANDONED_CHECKOUT, INCOMPLETE_PENDING, OVERDUE_INVOICE, PARTIAL_PAYMENT"
    )
    user_id: str = Field(description="The user ID extracted from the log")
    amount: float = Field(description="The transaction amount or cart value extracted. Default to 0.0 if not found.", default=0.0)
    device_type: str = Field(description="Device type if mentioned (Mobile, Desktop, Tablet), else 'Unknown'", default="Unknown")
    time_since_last_action_mins: int = Field(description="Estimated minutes since last action based on log. Default to 30.", default=30)
    intent_score: float = Field(description="Estimated intent 0.0 to 1.0 based on how far the user got. Default 0.5.", default=0.5)

# -----------------
# 3. Node Implementations
# -----------------

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

def triage_node(state: AgentState):
    """Phase 1: Parse unstructured log into structured data."""
    print("--- [NODE: Triage] ---")
    log = state["unstructured_log"]
    
    # Use LLM with structured output to parse the log
    parser_llm = llm.with_structured_output(TriageOutput)
    prompt = f"Analyze the following raw application log and extract the revenue loss context:\n\nLOG: {log}"
    
    parsed = parser_llm.invoke(prompt)
    
    metadata = {
        "user_id": parsed.user_id,
        "amount_at_risk": parsed.amount,
        "device_type": parsed.device_type,
        "time_since_last_action": parsed.time_since_last_action_mins,
        "user_intent_score": parsed.intent_score
    }
    
    print(f"Extracted Mode: {parsed.revenue_loss_mode}")
    print(f"Extracted Metadata: {metadata}")
    
    return {"revenue_loss_mode": parsed.revenue_loss_mode, "metadata": metadata}


def viability_node(state: AgentState):
    """
    Phase 2: Combined recovery probability from current event state + user history.
    Uses the trained XGBoost model on joined features.
    Returns both is_viable (bool) and recovery_probability (float).
    """
    print("--- [NODE: Viability Gate] ---")

    meta = state.get("metadata", {})
    mode = state.get("revenue_loss_mode", "")
    user_id = meta.get("user_id", "")

    recovery_prob = 0.0
    is_viable = False

    try:
        # Load the trained model
        with open("models/viability_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_cols = model_data['feature_columns']

        # Load user history
        import pandas as pd
        try:
            history_df = pd.read_csv("dataset/user_history.csv")
            history_df['payment_reliability'] = (
                history_df['successful_payments'] / history_df['total_orders'].clip(lower=1)
            )
            user_row = history_df[history_df['user_id'] == user_id]
        except FileNotFoundError:
            user_row = pd.DataFrame()

        # Build the feature row from triage metadata
        row = {
            'amount_at_risk': meta.get('amount_at_risk', 500.0),
            'time_since_last_action': meta.get('time_since_last_action', 30),
            'user_intent_score': meta.get('user_intent_score', 0.5),
            'is_valuable_customer': False,
            'revenue_loss_mode': mode,
            'device_type': meta.get('device_type', 'Unknown'),
            'time_of_day': 'Morning',  # default; could be extracted from log timestamp
        }

        # Merge in user history features
        if not user_row.empty:
            h = user_row.iloc[0]
            row['total_orders'] = h.get('total_orders', 0)
            row['successful_payments'] = h.get('successful_payments', 0)
            row['failed_payments'] = h.get('failed_payments', 0)
            row['avg_order_value'] = h.get('avg_order_value', 500.0)
            row['last_successful_payment_days'] = h.get('last_successful_payment_days', 180)
            row['churn_risk_score'] = h.get('churn_risk_score', 0.5)
            row['payment_reliability'] = h.get('payment_reliability', 0.5)
            row['preferred_payment_method'] = h.get('preferred_payment_method', 'Unknown')
            row['preferred_channel'] = h.get('preferred_channel', 'Unknown')
            row['is_subscribed'] = h.get('is_subscribed', False)
            print(f"User history found: {int(row['total_orders'])} orders, "
                  f"reliability={row['payment_reliability']:.2f}, churn={row['churn_risk_score']:.2f}")
        else:
            # Neutral defaults for unknown user
            row['total_orders'] = 0
            row['successful_payments'] = 0
            row['failed_payments'] = 0
            row['avg_order_value'] = 500.0
            row['last_successful_payment_days'] = 180
            row['churn_risk_score'] = 0.5
            row['payment_reliability'] = 0.5
            row['preferred_payment_method'] = 'Unknown'
            row['preferred_channel'] = 'Unknown'
            row['is_subscribed'] = False
            print(f"No history found for {user_id}, using neutral defaults")

        # One-hot encode to match the trained model's feature set
        row_df = pd.DataFrame([row])
        categorical_cols = [
            'revenue_loss_mode', 'device_type', 'time_of_day',
            'preferred_payment_method', 'preferred_channel'
        ]
        row_df = pd.get_dummies(row_df, columns=categorical_cols)

        # Convert booleans to int
        bool_cols = row_df.select_dtypes(include=['bool']).columns
        row_df[bool_cols] = row_df[bool_cols].astype(int)

        # Align columns to the trained model (add missing as 0, drop extras)
        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0
        row_df = row_df[feature_cols]

        # Predict recovery probability
        recovery_prob = float(model.predict_proba(row_df)[:, 1][0])
        is_viable = recovery_prob >= 0.3  # threshold for attempting recovery

        print(f"Combined Recovery Probability: {recovery_prob:.3f}")

    except Exception as e:
        # Fallback heuristic if model/data isn't available
        print(f"Model unavailable ({e}), using heuristic fallback")
        intent = meta.get("user_intent_score", 0.5)
        amount = meta.get("amount_at_risk", 0.0)

        recovery_prob = intent * 0.5
        if amount > 100:
            recovery_prob += 0.2
        if amount < 50 and intent < 0.2:
            recovery_prob = 0.05

        is_viable = recovery_prob >= 0.3

    print(f"Is Recovery Viable? {is_viable} (probability: {recovery_prob:.3f})")
    return {"is_viable": is_viable, "recovery_probability": recovery_prob}


# Bind tools to LLM for Phase 3
llm_with_tools = llm.bind_tools(my_tools.ALL_TOOLS)

def planner_node(state: AgentState):
    """Phase 3: The LLM plans the recovery strategy and calls tools."""
    print("--- [NODE: Orchestrator / Planner] ---")

    messages = state.get("messages", [])
    new_messages = []

    if not messages:
        prob = state.get('recovery_probability', 0.0)
        urgency = 'HIGH' if prob < 0.4 else ('MEDIUM' if prob < 0.7 else 'LOW')
        sys_msg = HumanMessage(content=f"""
You are an autonomous AI Revenue Recovery Agent.
A revenue loss event has occurred.
Mode: {state['revenue_loss_mode']}
Metadata: {json.dumps(state['metadata'])}
Recovery Probability (combined event + user history score): {prob:.3f}
Urgency Level: {urgency}

Use the recovery probability to calibrate your strategy:
- HIGH urgency (prob < 0.4): Use aggressive tactics — discounts, immediate WhatsApp outreach, alternate payment methods.
- MEDIUM urgency (0.4-0.7): Use balanced tactics — payment link + email sequence.
- LOW urgency (prob > 0.7): Light touch — simple reminder is enough.

Your job is to draft a recovery pipeline using the available tools.
Analyze the failure and call the necessary tools sequentially or in parallel.
When you are done recovering the revenue, just say 'RECOVERY PIPELINE COMPLETE'.
""")
        messages = [sys_msg]
        new_messages.append(sys_msg)

    response = llm_with_tools.invoke(messages)
    new_messages.append(response)

    return {"messages": new_messages}

# Phase 4: Executor Node (Native LangGraph ToolNode)
executor_node = ToolNode(my_tools.ALL_TOOLS)


# -----------------
# 4. Routing Edges
# -----------------
def route_after_viability(state: AgentState):
    if state.get("is_viable"):
        return "planner"
    else:
        print("Transaction dropped due to low viability. Saving compute.")
        return END

def route_after_planner(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # If the LLM called a tool, go to executor
    if last_message.tool_calls:
        return "executor"
    # Otherwise, it means it's done
    return END

# -----------------
# 5. Graph Compilation
# -----------------
workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_node)
workflow.add_node("viability", viability_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.add_edge(START, "triage")
workflow.add_edge("triage", "viability")
workflow.add_conditional_edges("viability", route_after_viability, {"planner": "planner", END: END})
workflow.add_conditional_edges("planner", route_after_planner, {"executor": "executor", END: END})
workflow.add_edge("executor", "planner")

app = workflow.compile()
