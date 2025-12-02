# Customer Preference Memory System

## Overview
Simple database-backed memory system to track customer preferences across conversations.

## Database Schema

### customer_preferences table
```sql
CREATE TABLE customer_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE,
    UNIQUE(customer_id, preference_key)
);

CREATE INDEX idx_preferences_customer ON customer_preferences(customer_id);
```

## Tool: CustomerPreferenceTool

### Actions

**1. get** - Retrieve a specific preference
```python
tool._run(
    action="get",
    customer_id=4,
    preference_key="usual_party_size"
)
# Returns: "usual_party_size: 4"
```

**2. set** - Save or update a preference
```python
tool._run(
    action="set",
    customer_id=4,
    preference_key="usual_party_size",
    preference_value="6"
)
# Returns: "Saved preference: usual_party_size = 6"
```

**3. get_all** - Retrieve all preferences for a customer
```python
tool._run(
    action="get_all",
    customer_id=4
)
# Returns:
# Preferences for customer 4:
#   last_order_id: 42
#   usual_party_size: 6
```

## Integration in RestaurantFlow

### State
```python
current_customer_id: int | None  # Database ID of current customer
```

### Helper Methods

**_update_memory(key, value)**
- Saves preference to database
- Requires `current_customer_id` to be set
- Logs: `[MEMORY] Saved preference: key = value`

**_get_context_summary()**
- Retrieves all preferences for current customer
- Returns formatted string for agent prompts
- Used by intent classifier, order handler, reservation handler

### Automatic Tracking

After successful order/reservation operations, the system automatically:
- Extracts `customer_id` from customer name in message
- Saves relevant preferences:
  - **Orders**: `last_order_id`, `recent_items`
  - **Reservations**: `last_reservation_id`, `usual_party_size`, `last_reservation_time`

## Example Flow

**Turn 1:**
```
Message: "Book table for 4 for Noah Chen"
→ Extracts customer_id=4 from lookup
→ Creates reservation #15
→ Saves: usual_party_size=4, last_reservation_id=15
```

**Turn 2:**
```
Message: "Change to 8pm"
→ Loads preferences: usual_party_size=4, last_reservation_id=15
→ Agent understands "change" refers to reservation #15
→ Updates reservation time
→ Updates: last_reservation_time=20:00
```

## Testing

Run direct database test:
```bash
python scripts/test_memory_direct.py
```

## Benefits

✅ **Persistent** - Survives application restarts  
✅ **Per-customer** - Each customer has isolated preferences  
✅ **Simple** - Just key-value pairs, no complex schema  
✅ **Automatic** - Preferences saved after successful operations  
✅ **Queryable** - Can analyze preferences across all customers  
✅ **Timestamped** - Track when preferences were created/updated
