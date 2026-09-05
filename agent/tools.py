from langchain_core.tools import tool

@tool
def check_alternate_payment_methods(user_id: str) -> str:
    """Checks if the user has alternate payment methods saved (e.g., UPI, other credit cards)."""
    print(f"[TOOL EXECUTION] Mock check_alternate_payment_methods for user {user_id}")
    return "User has a saved UPI ID and a Debit Card ending in 4432."

@tool
def generate_whatsapp_payment_link(amount: float, transaction_id: str) -> str:
    """Generates a unique Razorpay payment link to be sent via WhatsApp."""
    print(f"[TOOL EXECUTION] Mock generate_whatsapp_payment_link for transaction {transaction_id} amount {amount}")
    return f"https://rzp.io/i/{transaction_id}_wa"

@tool
def apply_dynamic_discount(cart_value: float, percentage: float) -> str:
    """Applies a dynamic discount to the cart to incentivize completion."""
    discount = cart_value * (percentage / 100)
    new_value = cart_value - discount
    print(f"[TOOL EXECUTION] Mock apply_dynamic_discount: {percentage}% off {cart_value}. New value: {new_value}")
    return f"Discount of {percentage}% applied. New cart value is {new_value}. Promo code: RECOVER{int(percentage)}"

@tool
def send_whatsapp_message(phone: str, message: str) -> str:
    """Sends a WhatsApp message to the user."""
    print(f"[TOOL EXECUTION] Mock send_whatsapp_message to {phone}: '{message}'")
    return "Message sent successfully."

@tool
def trigger_email_sequence(user_email: str, template_id: str) -> str:
    """Triggers an automated email sequence (e.g., 'cart_abandonment_1')."""
    print(f"[TOOL EXECUTION] Mock trigger_email_sequence to {user_email} using template {template_id}")
    return "Email sequence triggered successfully."

@tool
def query_bank_downtime_api(bank_name: str) -> str:
    """Checks if a specific bank's UPI or Netbanking is currently experiencing downtime."""
    print(f"[TOOL EXECUTION] Mock query_bank_downtime_api for {bank_name}")
    # Simulate that HDFC is down but SBI is up
    if "hdfc" in bank_name.lower():
        return f"{bank_name} is currently experiencing a 2-hour downtime."
    return f"{bank_name} systems are fully operational."

# List of all tools to bind to the LLM
ALL_TOOLS = [
    check_alternate_payment_methods,
    generate_whatsapp_payment_link,
    apply_dynamic_discount,
    send_whatsapp_message,
    trigger_email_sequence,
    query_bank_downtime_api
]
