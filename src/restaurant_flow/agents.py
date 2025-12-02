"""Agent definitions for the restaurant flow."""

from crewai import LLM, Agent

from restaurant_flow.tools.custom_tool import (
    MenuSearchTool,
    OrderLookupTool,
    ReservationLookupTool,
    CustomerLookupTool,
)


def get_llm() -> LLM:
    """Get the configured LLM instance."""
    import os
    
    model = os.getenv("RESTAURANT_LLM_MODEL", "ollama/llama3.1:8b")
    base_url = os.getenv("RESTAURANT_LLM_BASE_URL", "http://localhost:11434")
    
    return LLM(model=model, base_url=base_url)


def create_intent_classifier(llm: LLM | None = None) -> Agent:
    """Create the intent classifier agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Intent Classifier",
        goal="Accurately classify customer messages into intent categories and detect escalation needs",
        backstory="You are an expert at understanding customer requests. You distinguish between asking about something vs actively doing something (e.g., 'how do I order?' vs 'I want to order pizza').",
        verbose=True,
        llm=llm,
        max_iter=3,
    )


def create_menu_specialist(llm: LLM | None = None) -> Agent:
    """Create the menu specialist agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Menu Specialist",
        goal="Search the menu database and provide accurate dish names, prices, and availability",
        backstory="You know the restaurant menu inside out. Always use menu_search tool to get current information - never guess prices or availability.",
        tools=[MenuSearchTool()],
        verbose=True,
        llm=llm,
        max_iter=3,
    )


def create_order_handler(llm: LLM | None = None) -> Agent:
    """Create the order handler agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Order Handler",
        goal="Create orders in the database and return the order_id",
        backstory="You MUST complete all steps: (1) customer_lookup for customer_id, (2) menu_search for menu_item_ids and prices, (3) order_lookup with action='create' to save the order. Do NOT stop before calling order_lookup create.",
        tools=[OrderLookupTool(), MenuSearchTool(), CustomerLookupTool()],
        verbose=True,
        llm=llm,
        max_iter=6,
    )


def create_reservation_agent(llm: LLM | None = None) -> Agent:
    """Create the reservation agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Reservation Agent",
        goal="Create reservations in the database and return the reservation_id",
        backstory="You MUST complete all steps: (1) customer_lookup for customer_id, (2) reservation_lookup with action='create' to save the booking. Do NOT stop before calling reservation_lookup create.",
        tools=[ReservationLookupTool(), CustomerLookupTool()],
        verbose=True,
        llm=llm,
        max_iter=5,
    )


def create_escalation_agent(llm: LLM | None = None) -> Agent:
    """Create the escalation handler agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Customer Service Manager",
        goal="Resolve customer complaints and concerns with empathy while offering concrete solutions",
        backstory="You are a senior manager with authority to offer solutions and compensation. You acknowledge feelings first, apologize sincerely, then provide actionable next steps.",
        verbose=True,
        llm=llm,
        max_iter=3,
    )


def create_fallback_agent(llm: LLM | None = None) -> Agent:
    """Create the fallback handler agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="General Support Agent",
        goal="Clarify unclear requests and guide customers to the right service",
        backstory="You help when requests are ambiguous. Ask specific clarifying questions and offer clear options: menu inquiries, placing orders, or making reservations.",
        verbose=True,
        llm=llm,
        max_iter=3,
    )


def create_response_composer(llm: LLM | None = None) -> Agent:
    """Create the response composer agent."""
    if llm is None:
        llm = get_llm()
        
    return Agent(
        role="Response Composer",
        goal="Transform specialist data into warm, professional customer responses with clear next steps",
        backstory="You craft the final message customers see. Use all provided data, maintain a friendly tone, and always include what happens next or how to get more help.",
        verbose=True,
        llm=llm,
        max_iter=3,
    )
