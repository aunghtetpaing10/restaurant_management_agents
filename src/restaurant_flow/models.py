"""Pydantic models for the restaurant flow."""

from typing import List

from pydantic import BaseModel, Field


class MemoryKeys:
    """Centralized memory keys for customer preferences."""
    
    # Order-related
    LAST_ORDER_ID = "last_order_id"
    RECENT_ITEMS = "recent_items"
    FAVORITE_ITEMS = "favorite_items"
    
    # Reservation-related
    LAST_RESERVATION_ID = "last_reservation_id"
    USUAL_PARTY_SIZE = "usual_party_size"
    LAST_RESERVATION_TIME = "last_reservation_time"
    
    # Menu-related
    RECENT_MENU_SEARCHES = "recent_menu_searches"
    
    # Dietary preferences
    DIETARY_RESTRICTIONS = "dietary_restrictions"
    ALLERGIES = "allergies"


class IntentClassification(BaseModel):
    """Intent classifier structured output."""

    intent: str = Field(
        description="Detected intent category (menu_inquiry, order_request, reservation_request, general_question, complaint, unclear, other)"
    )
    requires_escalation: bool = Field(
        default=False,
        description="True if the message contains a complaint, urgent issue, or requires human intervention",
    )
    confidence: str = Field(
        default="high",
        description="Confidence level in the classification: high, medium, or low",
    )


class MenuResponse(BaseModel):
    """Structured output for menu inquiries."""

    menu_items: List[str] = Field(
        description="List of menu item names found (e.g., ['Korean BBQ Chicken Wings', 'Caesar Salad'])"
    )
    prices: List[float] = Field(
        description="Prices for each menu item in order", default_factory=list
    )


class OrderItemDetail(BaseModel):
    """Structured summary for each ordered item."""

    menu: str = Field(description="Menu item name (e.g., 'Caesar Salad')")
    price: str = Field(description="Price string including currency (e.g., '$12.99')")
    quantity: int = Field(description="Quantity ordered")


class OrderResponse(BaseModel):
    """Structured output for order inquiries."""

    order_id: int | None = Field(
        default=None,
        description="ID of the order (if known or after creation)",
    )
    items_ordered: List[OrderItemDetail] = Field(
        default_factory=list,
        description="List of ordered items with menu name, price string, and quantity",
    )
    total_amount: float = Field(description="Total amount of the order")
    order_status: str = Field(description="Status of the order")

    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle string-encoded lists."""
        if isinstance(obj, dict) and isinstance(obj.get("items_ordered"), str):
            import json

            try:
                obj["items_ordered"] = json.loads(obj["items_ordered"])
            except (json.JSONDecodeError, TypeError):
                obj["items_ordered"] = []
        return super().model_validate(obj)


class ReservationResponse(BaseModel):
    """Structured output for reservation inquiries."""

    reservation_id: int | None = Field(
        default=None, description="Reservation ID if available"
    )
    party_size: int | None = Field(
        default=None, description="Number of guests (if known)"
    )
    reservation_datetime: str | None = Field(
        default=None, description="Reservation date/time (if known)"
    )
    status: str = Field(
        description="Status of reservation (confirmed, waitlisted, cancelled, etc.)"
    )
    special_requests: str = Field(
        default="",
        description="Any special requests from the customer",
    )


class FinalResponse(BaseModel):
    """Structured output for the final composed response."""

    customer_message_summary: str = Field(
        description="Brief summary of what the customer asked"
    )
    final_response: str = Field(
        description="The polished, professional final response to send to the customer"
    )


class RestaurantState(BaseModel):
    """State model for the restaurant flow."""

    customer_message: str = "What are your options for beverages?"
    classification: IntentClassification | None = None
    menu_response: MenuResponse | None = None
    order_response: OrderResponse | None = None
    reservation_response: ReservationResponse | None = None
    final_response: FinalResponse | None = None
    current_customer_id: int | None = Field(
        default=None, description="Database ID of current customer for memory tracking"
    )
