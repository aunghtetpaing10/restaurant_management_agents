# Memory System Test Scenario

## Scenario: Noah Chen's Order Modification

### Setup
Noah Chen (customer_id=4) has these preferences stored:
```
last_order_id: 29
recent_items: Caesar Salad, Burger
usual_party_size: 8
```

### Test Message
```
"Can you make it 2 for my last order for Noah Chen"
```

### Expected Flow

**1. receive_message()**
- Extracts "Noah Chen" from message
- Looks up customer_id=4
- Sets `current_customer_id=4`
- Logs: `[MEMORY] Identified customer_id: 4 (Noah Chen)`

**2. classify_intent()**
- Loads context:
  ```
  Preferences for customer 4:
    last_order_id: 29
    recent_items: Caesar Salad, Burger
    usual_party_size: 8
  ```
- Sees "make it 2" + "my last order" + context has `last_order_id`
- Classifies as: `order_request`
- Logs: `[CLASSIFY] Intent: order_request`

**3. handle_order()**
- Receives context with `last_order_id: 29` and `recent_items: Caesar Salad, Burger`
- Agent interprets:
  - "my last order" = order #29 (from context)
  - "make it 2" = change quantity to 2
- Calls: `order_lookup(action='lookup_by_id', order_id=29)`
- Returns order details
- Sets `order_status` to explain the modification request
- Example response: "Found your previous order #29 (Caesar Salad, Burger). You want to change the quantity to 2."

**4. deliver_response()**
- Composes friendly message explaining:
  - We found their order #29
  - They want quantity changed to 2
  - Next steps (confirm modification, etc.)

### Key Improvements Made

1. **Early customer_id extraction** - Happens in `receive_message()` before context is needed
2. **Context-aware intent classification** - Understands "make it 2" refers to last order
3. **Enhanced order handler prompt** - Explicitly handles modification requests
4. **Memory integration** - Preferences loaded and used throughout the flow

### How to Test

1. Ensure Noah Chen has preferences:
```sql
SELECT * FROM customer_preferences WHERE customer_id = 4;
```

2. Run the flow:
```bash
cd src
uv run python -m restaurant_flow.main
```

3. Check logs for:
- `[MEMORY] Identified customer_id: 4 (Noah Chen)`
- `[CLASSIFY] Intent: order_request`
- Context being passed to order handler
- Agent understanding the modification request

### Success Criteria

✅ Customer ID extracted early  
✅ Preferences loaded from database  
✅ Intent correctly classified as order_request  
✅ Agent understands "my last order" = order #29  
✅ Agent understands "make it 2" = quantity modification  
✅ Response explains what was found and what customer wants
