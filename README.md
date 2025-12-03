# RestaurantFlow – Multi-Agent Restaurant Assistant (Demo)

This project is a **demo multi-agent system** for a restaurant assistant, built with [CrewAI Flow](https://crewai.com), SQLite, and MCP-compatible tools.

---

## Features

- **SQLite + MCP**
  - Schema for `customers`, `menu_items`, `orders`, `order_items`, `reservations`, `customer_preferences`.
  - Data seeded via `database/seed.sql` (50+ menu items, 18 customers).
  - Accessed through `mcp-server-sqlite` via `restaurant_flow.mcp_init`.

- **Custom MCP-Compatible Tools** (located in `src/restaurant_flow/tools/`):
  - `MenuSearchTool` – search menu by name/category.
  - `OrderLookupTool` – create orders, lookup by ID/phone.
  - `ReservationLookupTool` – create and lookup reservations.
  - `CustomerLookupTool` – search customers by name/phone/email.
  - `CustomerPreferenceTool` – DB-backed memory (`get`, `set`, `get_all`).

- **Agents** (see `src/restaurant_flow/agents.py`):
  - Intent Classifier
  - Menu Specialist
  - Order Handler
  - Reservation Agent
  - Response Composer
  - Escalation Agent (complaints)
  - Fallback Agent (unclear requests)
  - Clarification Agent (gathers missing info in interactive mode)

- **Flow Architecture** (see `src/restaurant_flow/main.py`):
  - `@start` **`classify_intent`**
    - Single-shot: calls Intent Classifier agent.
    - Interactive: runs Clarification Agent in a loop to gather `customer_name`, `items`, `party_size`, `date_time`, etc.
    - Stores result in `RestaurantState.classification` and `RestaurantState.clarification_info`.
    - Extracts `current_customer_id` and dietary/allergy info early.
  - `@router` **`route_intent`**
    - Pure routing logic based on `state.classification.intent` and `requires_escalation`.
  - `@listen` handlers:
    - `handle_menu` → Menu Specialist (`MenuResponse`).
    - `handle_order` → Order Handler (`OrderResponse`).
    - `handle_reservation` → Reservation Agent (`ReservationResponse`).
    - `handle_escalation` → Escalation Agent (`FinalResponse`).
    - `handle_fallback` → Fallback Agent (`FinalResponse`).
    - `deliver_response` → Response Composer (`FinalResponse`).

- **State & Memory** (see `src/restaurant_flow/models.py`):
  - `RestaurantState` holds:
    - `customer_message`, `classification`, `menu_response`, `order_response`, `reservation_response`, `final_response`.
    - `current_customer_id` and `clarification_info`.
  - `MemoryKeys` defines centralized keys like `LAST_ORDER_ID`, `USUAL_PARTY_SIZE`, `RECENT_ITEMS`, `DIETARY_RESTRICTIONS`, etc.
  - Handlers update memory via `CustomerPreferenceTool` and `_update_memory`.

---

## Installation

Requirements:

- Python **>=3.10 <3.14**
- [UV](https://docs.astral.sh/uv/) for dependency management

Install `uv` if needed:

```bash
pip install uv
```

From the project root (`restaurant_flow` folder), install dependencies:

```bash
uv sync
```

### 1. Configure environment variables (LLM + MCP)

You can either export these variables in your shell or create a `.env` file in the
project root (many tools, including the CrewAI CLI, will auto-load `.env`).

**Recommended `.env` example:**

```bash
# LLM configuration (defaults shown)
RESTAURANT_LLM_MODEL=ollama/llama3.1:8b
RESTAURANT_LLM_BASE_URL=http://localhost:11434

# OpenAI API key (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key

# SQLite database path for MCP (default is db/restaurant.db)
# If you follow the steps below exactly, you can omit this.
RESTAURANT_DB_PATH=db/restaurant.db

# MCP server command (defaults)
RESTAURANT_MCP_COMMAND=uvx
RESTAURANT_MCP_SERVER=mcp-server-sqlite
```

If you prefer a different LLM (e.g. OpenAI via LiteLLM), set `RESTAURANT_LLM_MODEL`
and `RESTAURANT_LLM_BASE_URL` accordingly.

### 2. Initialize the SQLite database

The Flow expects a SQLite database with the schema and seed data from
`database/schema.sql` and `database/seed.sql`.

From the project root, run (requires the `sqlite3` CLI):

```bash
# Create db directory if it doesn't exist
mkdir -p db

# Create schema
sqlite3 db/restaurant.db < database/schema.sql

# Seed data (customers, menu_items, etc.)
sqlite3 db/restaurant.db < database/seed.sql
```

By default, the MCP server will use `db/restaurant.db` via `RESTAURANT_DB_PATH` or
the built-in default in `mcp_init.py`.

---

## Running the Project

### 1. Single-Shot Flow (one-off message)

Runs the full Flow once using the message defined in `RestaurantState.customer_message` or passed in as input.

```bash
uv run python -m restaurant_flow.main
```

### 2. Interactive Chat with Clarification

Runs an interactive CLI chat. The Clarification Agent will ask follow-up questions to gather missing details before the main Flow runs.

```bash
uv run python -m restaurant_flow.main chat
```

You can type natural language like:

- `I want to order pizza` → it will ask for name / details.
- `Book a table` → it will ask for name, party size, date/time.

Type `quit` to exit.

### 3. Demo Scenarios (for evaluation)

The `demo.py` script runs several known-good scenarios through the Flow and prints pass/fail status.

```bash
uv run python demo.py
```

Scenarios include:

- Menu inquiry (desserts).
- Complete order for a **seeded customer** (e.g. Noah Chen).
- Complete reservation for a **seeded customer** (e.g. Harper Davis).
- Complaint (escalation).
- Unclear intent (fallback).

---

## Project Structure (Key Files)

- `src/restaurant_flow/main.py` – Flow definition (`classify_intent`, `route_intent`, handlers, `deliver_response`).
- `src/restaurant_flow/agents.py` – Agent factories and configurations.
- `src/restaurant_flow/models.py` – Pydantic models and `RestaurantState`.
- `src/restaurant_flow/tools/custom_tool.py` – `MenuSearchTool`, `OrderLookupTool`, `CustomerLookupTool`, `ReservationLookupTool`.
- `src/restaurant_flow/tools/preference_tools.py` – `CustomerPreferenceTool`.
- `database/schema.sql` & `database/seed.sql` – DB schema and seed data.
- `demo.py` – Single-shot demo scenarios.

For a deeper architectural explanation, see `ARCHITECTURE.md`.
