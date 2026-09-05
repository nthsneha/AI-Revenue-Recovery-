import os
from dotenv import load_dotenv

# Load environment variables before importing the graph
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not found in environment. Please set it in .env file.")

from agent.graph import app

def run_demo():
    print("=============================================")
    print("   AI Revenue Recovery Agent - Hackathon Demo")
    print("=============================================")
    
    # 3 Example Logs to simulate different scenarios
    demo_logs = [
        # Scenario 1: Implicit - Abandoned Checkout (High Intent, High Value)
        "[2026-09-04 10:15:32] INFO: Session user_8231 loaded checkout for cart_val 2500.0. User spent 12 mins on page. Connection closed by client.",
        
        # Scenario 2: Explicit - Payment Failure (Low Intent, Low Value)
        "[2026-09-04 11:22:11] ERROR: Payment failed for user_1120. RZP_CODE: insufficient_funds. Amount: 15.00. Device: Mobile.",
        
        # Scenario 3: Implicit - Overdue Invoice
        "[2026-09-04 14:00:00] SYSTEM: Invoice inv_9921 for user_5531 is 5 days past due. Amount: 450.00."
    ]
    
    for i, log in enumerate(demo_logs, 1):
        print(f"\n\n>>> PROCESSING DEMO LOG {i} <<<")
        print(f"INPUT: {log}\n")
        
        initial_state = {
            "unstructured_log": log,
            "metadata": {},
            "revenue_loss_mode": "",
            "is_viable": False,
            "messages": []
        }
        
        try:
            # Run the LangGraph application
            result = app.invoke(initial_state)
            
            if not result.get("is_viable"):
                print("Result: Pipeline aborted at Viability Gate.")
            else:
                print("Result: Pipeline successfully orchestrated.")
                print("\n[LLM FINAL OUTPUT]")
                # The last message is typically the LLM confirming completion
                print(result['messages'][-1].content)
        except Exception as e:
            print(f"Error during graph execution: {e}")
            print("Did you set your GEMINI_API_KEY?")

if __name__ == "__main__":
    run_demo()
