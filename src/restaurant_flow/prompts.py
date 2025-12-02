"""Prompt templates for the restaurant flow agents."""


def get_intent_classification_prompt(customer_message: str, context: str) -> str:
    """Generate the intent classification prompt."""
    return f"""Message: "{customer_message}"
Context: {context}

Classify intent:
- menu_inquiry: asking about menu/prices
- order_request: placing order, checking order status, or modifying order
- reservation_request: booking table, checking/modifying reservation
- general_question: policies, hours, how-to questions
- complaint: dissatisfaction or problem
- unclear: ambiguous request
- other: none of the above

Use context to resolve references like "my last order" or "make it 2".
Set requires_escalation=true for complaints, anger, or manager requests.
Set confidence: high/medium/low.
"""


def get_menu_inquiry_prompt(customer_message: str, context: str = "") -> str:
    """Generate the menu inquiry prompt."""
    context_section = f"\nContext: {context}" if context and context != "New customer - no previous history." else ""
    return f"""Customer: {customer_message}{context_section}

Use menu_search tool to find matching items. Return dish names and prices.
If customer has dietary restrictions or allergies in context, prioritize suitable items.
"""


def get_order_handler_prompt(customer_message: str, context: str) -> str:
    """Generate the order handler prompt."""
    return f"""Message: '{customer_message}'
Context: {context}

MANDATORY STEPS FOR NEW ORDERS:
1. Check if customer name/phone is provided. If NOT provided:
   - Return order_status="awaiting_details" immediately

2. Use customer_lookup to get customer_id from name/phone

3. Use menu_search for EACH item to get menu_item_id and exact price

4. If customer has dietary restrictions or allergies in context, WARN if ordered items may conflict

5. CREATE the order by calling order_lookup with:
   - action='create'
   - customer_id=<from step 2>
   - items='[{{"menu_item_id": <id>, "quantity": <n>}}]'

6. Return the created order with order_id from the response

RESPONSE FORMAT:
- order_id: the ID returned from order_lookup create action
- items_ordered: [{{"menu": "Name", "price": "$X.XX", "quantity": N}}]
- total_amount: sum of (price × quantity) for all items
- order_status: "confirmed" if created, "awaiting_details" if missing info
"""


def get_reservation_handler_prompt(customer_message: str, context: str) -> str:
    """Generate the reservation handler prompt."""
    return f"""Message: '{customer_message}'
Context: {context}

MANDATORY STEPS FOR NEW RESERVATIONS:
1. Check if party_size, date, time, and customer info are provided. If NOT:
   - Return status="awaiting_details" immediately

2. Use customer_lookup to get customer_id from name/phone

3. Convert date/time formats:
   - "tomorrow" → actual date (YYYY-MM-DD)
   - "7pm" → "19:00"

4. CREATE the reservation by calling reservation_lookup with:
   - action='create'
   - customer_id=<from step 2>
   - party_size=<number>
   - reservation_date='YYYY-MM-DD'
   - reservation_time='HH:MM'

5. Return the created reservation with reservation_id from the response

RESPONSE FORMAT:
- reservation_id: the ID returned from reservation_lookup create action
- party_size, reservation_datetime, status: from the created reservation
- status: "confirmed" if created, "awaiting_details" if missing info
"""


def get_escalation_prompt(customer_message: str, intent: str) -> str:
    """Generate the escalation handler prompt."""
    return f"""Message: '{customer_message}' (flagged: {intent})

Respond with empathy. Apologize if appropriate. Offer solution or compensation. Provide manager contact. De-escalate professionally.
"""


def get_fallback_prompt(customer_message: str, intent: str, confidence: str) -> str:
    """Generate the fallback handler prompt."""
    return f"""Message: '{customer_message}' ({intent}, {confidence})

Ask clarifying questions if unclear. Guide to appropriate service. Offer menu/order/reservation options.
"""


def get_response_composer_prompt(customer_message: str, specialist_data: str) -> str:
    """Generate the response composer prompt."""
    return f"""Customer: {customer_message}

Data: {specialist_data}

Compose a friendly, professional response addressing the customer's request. Include next steps.
"""


def get_clarification_prompt(conversation_history: str, latest_message: str) -> str:
    """Generate prompt for analyzing if clarification is needed."""
    return f"""Analyze this restaurant conversation and determine if we have enough info to proceed.

CONVERSATION HISTORY:
{conversation_history}

LATEST MESSAGE: "{latest_message}"

REQUIRED INFO BY INTENT:
- order_request: customer_name, items (what they want to order)
- reservation_request: customer_name, party_size, date_time
- menu_inquiry: nothing required (can proceed immediately)
- general_question: nothing required
- complaint: nothing required

INSTRUCTIONS:
1. Identify the customer's intent
2. Extract any info provided so far (customer_name, items, party_size, date_time, etc.)
3. Check if all required info for that intent is available
4. If missing info, generate a natural clarifying question
5. Set is_ready=true ONLY if all required info is present

RESPONSE FORMAT:
- intent: detected intent
- is_ready: true/false
- missing_info: list of what's still needed
- clarification_question: friendly question to ask (empty if ready)
- collected_info: dict of extracted info
"""
