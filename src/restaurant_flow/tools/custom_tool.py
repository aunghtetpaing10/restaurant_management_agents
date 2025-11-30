import ast
from typing import Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool


class MenuSearchInput(BaseModel):
    """Input schema for MenuSearchTool."""

    query: str = Field(
        ..., description="Guest question or keywords about menu items or categories (e.g. 'vegan pasta, appetizers')"
    )


class MenuSearchTool(BaseTool):
    """Search restaurant menu items and return matching dishes."""

    name: str = "menu_search"
    description: str = (
        "Use this tool to answer guest questions about menu items. It queries the restaurant"
        " SQLite database (through the MCP SQLite server) and returns matching dishes with"
        " category, price, availability, and description."
    )
    args_schema: Type[BaseModel] = MenuSearchInput

    def _run(self, query: str) -> str:
        # Import here to avoid circular dependency and ensure MCP is initialized
        from restaurant_flow.mcp_init import get_mcp_tools
        
        try:
            tools = get_mcp_tools()
        except Exception as e:
            return f"MenuSearchTool error: Failed to initialize MCP tools - {str(e)}"
        
        # ToolCollection is iterable, find the read_query tool
        read_query_tool = None
        try:
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == "read_query":
                    read_query_tool = tool
                    break
        except Exception as e:
            return f"MenuSearchTool error: Failed to find read_query tool - {str(e)}"

        if read_query_tool is None:
            return "MenuSearchTool error: read_query tool is not available from MCP server."

        cleaned_query = query.replace("'", "''")

        sql = (
            "SELECT name, category, price, description, is_available "
            "FROM menu_items "
            f"WHERE (name LIKE '%{cleaned_query}%' OR description LIKE '%{cleaned_query}%' OR category LIKE '%{cleaned_query}%')"
        )

        sql += " ORDER BY category, name LIMIT 10"

        try:
            # CrewAIMCPTool uses .run() method, not .invoke()
            response = read_query_tool.run(query=sql)
        except Exception as e:
            return f"MenuSearchTool error: Failed to execute query - {str(e)}"

        # The MCP read_query tool returns a string representation of a Python list
        # Parse it to get the actual data
        try:
            if isinstance(response, str):
                # Parse Python literal (list of dicts with single quotes)
                results = ast.literal_eval(response)
            elif isinstance(response, list):
                results = response
            elif isinstance(response, dict):
                if not response.get("success", True):
                    return f"MenuSearchTool error: {response.get('error', 'Unknown error')}"
                results = response.get("results", response.get("data", []))
            else:
                return f"MenuSearchTool received unexpected response format: {type(response).__name__}"
        except (ValueError, SyntaxError) as e:
            return f"MenuSearchTool error: Failed to parse response - {str(e)}"
        if not results:
            return "No matching menu items found for that query."

        lines = []
        for item in results:
            availability = "Available" if item.get("is_available") else "Unavailable"
            price_value = item.get("price")
            if price_value is None:
                price_text = "Price unavailable"
            else:
                price_text = f"${float(price_value):.2f}"

            line = (
                f"{item.get('name', 'Unknown')} ({item.get('category', 'Unknown')}): "
                f"{price_text} - {availability}. "
                f"{item.get('description', 'No description provided.')}"
            )
            lines.append(line)

        return "\n".join(lines)


class OrderLookupInput(BaseModel):
    """Input schema for OrderLookupTool."""

    action: str = Field(
        ...,
        description="Action to perform: 'create' to create new order, 'lookup_by_id' to check status by order ID, 'lookup_by_phone' to check status by phone number",
    )
    customer_id: int = Field(
        default=None,
        description="ID of the customer (required for 'create' action)",
    )
    items: str = Field(
        default=None,
        description="JSON string of order items for 'create' action: [{\"menu_item_id\": 1, \"quantity\": 2}, ...]",
    )
    order_id: int = Field(
        default=None,
        description="Order ID to lookup (required for 'lookup_by_id' action)",
    )
    phone: str = Field(
        default=None,
        description="Customer phone number to lookup orders (required for 'lookup_by_phone' action)",
    )


class OrderLookupTool(BaseTool):
    """Create orders and check order status in the database."""

    name: str = "order_lookup"
    description: str = (
        "Use this tool to create new orders OR check order status. "
        "Actions: "
        "1. 'create' - Create new order (requires customer_id and items). "
        "2. 'lookup_by_id' - Check order status by order ID. "
        "3. 'lookup_by_phone' - Check all orders for a phone number. "
        "Returns order details including status, items, and total."
    )
    args_schema: Type[BaseModel] = OrderLookupInput

    def _run(
        self,
        action: str,
        customer_id: int = None,
        items: str = None,
        order_id: int = None,
        phone: str = None,
    ) -> str:
        # Import here to avoid circular dependency
        from restaurant_flow.mcp_init import get_mcp_tools

        try:
            tools = get_mcp_tools()
        except Exception as e:
            return f"OrderLookupTool error: Failed to initialize MCP tools - {str(e)}"

        # Find write_query and read_query tools
        write_query_tool = None
        read_query_tool = None
        try:
            for tool in tools:
                if hasattr(tool, "name"):
                    if tool.name == "write_query":
                        write_query_tool = tool
                    elif tool.name == "read_query":
                        read_query_tool = tool
        except Exception as e:
            return f"OrderLookupTool error: Failed to find MCP tools - {str(e)}"

        if not read_query_tool:
            return "OrderLookupTool error: read_query tool not available."

        # Route to appropriate action
        if action == "create":
            return self._create_order(
                customer_id, items, write_query_tool, read_query_tool
            )
        elif action == "lookup_by_id":
            return self._lookup_by_id(order_id, read_query_tool)
        elif action == "lookup_by_phone":
            return self._lookup_by_phone(phone, read_query_tool)
        else:
            return f"OrderLookupTool error: Unknown action '{action}'. Use 'create', 'lookup_by_id', or 'lookup_by_phone'."

    def _create_order(
        self, customer_id: int, items: str, write_query_tool, read_query_tool
    ) -> str:
        """Create a new order."""
        if not customer_id or not items:
            return "OrderLookupTool error: customer_id and items are required for 'create' action."

        if not write_query_tool:
            return "OrderLookupTool error: write_query tool not available for creating orders."

        # Parse items JSON
        try:
            import json

            items_list = json.loads(items)
        except json.JSONDecodeError as e:
            return f"OrderManagementTool error: Invalid items JSON - {str(e)}"

        # Calculate total price first
        total_price = 0.0
        for item in items_list:
            menu_item_id = item.get("menu_item_id")
            quantity = item.get("quantity", 1)

            # Get price from menu_items
            price_sql = f"SELECT price FROM menu_items WHERE id = {menu_item_id}"
            try:
                price_response = read_query_tool.run(query=price_sql)
                price_results = ast.literal_eval(price_response)
                if price_results:
                    price = float(price_results[0].get("price", 0))
                    total_price += price * quantity
            except Exception as e:
                return f"OrderManagementTool error: Failed to get price for item {menu_item_id} - {str(e)}"

        # Create order
        create_order_sql = (
            f"INSERT INTO orders (customer_id, total_amount, order_status) "
            f"VALUES ({customer_id}, {total_price}, 'in_progress')"
        )

        try:
            write_query_tool.run(query=create_order_sql)
        except Exception as e:
            return f"OrderManagementTool error: Failed to create order - {str(e)}"

        # Get the order ID - query the most recent order for this customer
        get_order_id_sql = (
            f"SELECT id FROM orders WHERE customer_id = {customer_id} "
            f"ORDER BY id DESC LIMIT 1"
        )
        try:
            order_id_response = read_query_tool.run(query=get_order_id_sql)
            order_id_results = ast.literal_eval(order_id_response)
            order_id = order_id_results[0].get("id")
        except Exception as e:
            return f"OrderManagementTool error: Failed to get order ID - {str(e)}"

        # Insert order items
        for item in items_list:
            menu_item_id = item.get("menu_item_id")
            quantity = item.get("quantity", 1)

            # Get price again for order_items
            price_sql = f"SELECT price FROM menu_items WHERE id = {menu_item_id}"
            try:
                price_response = read_query_tool.run(query=price_sql)
                price_results = ast.literal_eval(price_response)
                price = float(price_results[0].get("price", 0))
            except Exception:
                price = 0.0

            insert_item_sql = (
                f"INSERT INTO order_items (order_id, menu_item_id, quantity, price) "
                f"VALUES ({order_id}, {menu_item_id}, {quantity}, {price})"
            )

            try:
                write_query_tool.run(query=insert_item_sql)
            except Exception as e:
                return f"OrderManagementTool error: Failed to add item {menu_item_id} - {str(e)}"

        return (
            f"Order created successfully!\n"
            f"Order ID: {order_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Total Price: ${total_price:.2f}\n"
            f"Status: In Progress\n"
            f"Items: {len(items_list)}"
        )

    def _lookup_by_id(self, order_id: int, read_query_tool) -> str:
        """Look up order status by order ID."""
        if not order_id:
            return "OrderLookupTool error: order_id is required for 'lookup_by_id' action."

        # Get order details with customer info
        order_sql = (
            "SELECT o.id, o.customer_id, o.order_status, o.order_datetime, o.total_amount, "
            "c.first_name, c.last_name, c.phone "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            f"WHERE o.id = {order_id}"
        )

        try:
            order_response = read_query_tool.run(query=order_sql)
            order_results = ast.literal_eval(order_response)
        except Exception as e:
            return f"OrderLookupTool error: Failed to lookup order - {str(e)}"

        if not order_results:
            return f"No order found with ID {order_id}."

        order = order_results[0]

        # Get order items
        items_sql = (
            "SELECT oi.quantity, oi.price, m.name "
            "FROM order_items oi "
            "JOIN menu_items m ON oi.menu_item_id = m.id "
            f"WHERE oi.order_id = {order_id}"
        )

        try:
            items_response = read_query_tool.run(query=items_sql)
            items_results = ast.literal_eval(items_response)
        except Exception:
            items_results = []

        # Format response
        response = (
            f"Order #{order.get('id')}\n"
            f"Status: {order.get('order_status', 'Unknown')}\n"
            f"Customer: {order.get('first_name')} {order.get('last_name')}\n"
            f"Phone: {order.get('phone', 'N/A')}\n"
            f"Order Date: {order.get('order_datetime')}\n"
            f"Total: ${float(order.get('total_amount', 0)):.2f}\n"
        )

        if items_results:
            response += "\nItems:\n"
            for item in items_results:
                response += f"  - {item.get('name')} x{item.get('quantity')} (${float(item.get('price', 0)):.2f} each)\n"

        return response

    def _lookup_by_phone(self, phone: str, read_query_tool) -> str:
        """Look up all orders for a customer by phone number."""
        if not phone:
            return "OrderLookupTool error: phone is required for 'lookup_by_phone' action."

        # Get customer ID from phone
        customer_sql = f"SELECT id, first_name, last_name FROM customers WHERE phone = '{phone}'"

        try:
            customer_response = read_query_tool.run(query=customer_sql)
            customer_results = ast.literal_eval(customer_response)
        except Exception as e:
            return f"OrderLookupTool error: Failed to lookup customer - {str(e)}"

        if not customer_results:
            return f"No customer found with phone number {phone}."

        customer = customer_results[0]
        customer_id = customer.get("id")

        # Get all orders for this customer
        orders_sql = (
            "SELECT id, order_status, order_datetime, total_amount "
            "FROM orders "
            f"WHERE customer_id = {customer_id} "
            "ORDER BY order_datetime DESC"
        )

        try:
            orders_response = read_query_tool.run(query=orders_sql)
            orders_results = ast.literal_eval(orders_response)
        except Exception as e:
            return f"OrderLookupTool error: Failed to lookup orders - {str(e)}"

        if not orders_results:
            return f"No orders found for {customer.get('first_name')} {customer.get('last_name')} (phone: {phone})."

        # Format response
        response = f"Orders for {customer.get('first_name')} {customer.get('last_name')} (phone: {phone}):\n\n"

        for order in orders_results:
            response += (
                f"Order #{order.get('id')} - {order.get('order_status')}\n"
                f"  Date: {order.get('order_datetime')}\n"
                f"  Total: ${float(order.get('total_amount', 0)):.2f}\n\n"
            )

        return response.strip()


class ReservationLookupInput(BaseModel):
    """Input schema for ReservationLookupTool."""

    action: str = Field(
        ...,
        description="Action to perform: 'create' to create new reservation, 'lookup_by_id' to check status by reservation ID, 'lookup_by_phone' to check reservations by phone number",
    )
    customer_id: int = Field(
        default=None,
        description="ID of the customer (required for 'create' action)",
    )
    party_size: int = Field(
        default=None,
        description="Number of people in the party (required for 'create' action)",
    )
    reservation_date: str = Field(
        default=None,
        description="Date of reservation in YYYY-MM-DD format (required for 'create' action)",
    )
    reservation_time: str = Field(
        default=None,
        description="Time of reservation in HH:MM format (required for 'create' action)",
    )
    reservation_id: int = Field(
        default=None,
        description="Reservation ID to lookup (required for 'lookup_by_id' action)",
    )
    phone: str = Field(
        default=None,
        description="Customer phone number to lookup reservations (required for 'lookup_by_phone' action)",
    )


class ReservationLookupTool(BaseTool):
    """Create reservations and check reservation status in the database."""

    name: str = "reservation_lookup"
    description: str = (
        "Use this tool to create new reservations OR check reservation status. "
        "Actions: "
        "1. 'create' - Create new reservation (requires customer_id, party_size, date, time). "
        "2. 'lookup_by_id' - Check reservation status by reservation ID. "
        "3. 'lookup_by_phone' - Check all reservations for a phone number. "
        "Returns reservation details including status, date, time, and party size."
    )
    args_schema: Type[BaseModel] = ReservationLookupInput

    def _run(
        self,
        action: str,
        customer_id: int = None,
        party_size: int = None,
        reservation_date: str = None,
        reservation_time: str = None,
        reservation_id: int = None,
        phone: str = None,
    ) -> str:
        # Import here to avoid circular dependency
        from restaurant_flow.mcp_init import get_mcp_tools

        try:
            tools = get_mcp_tools()
        except Exception as e:
            return f"ReservationLookupTool error: Failed to initialize MCP tools - {str(e)}"

        # Find write_query and read_query tools
        write_query_tool = None
        read_query_tool = None
        try:
            for tool in tools:
                if hasattr(tool, "name"):
                    if tool.name == "write_query":
                        write_query_tool = tool
                    elif tool.name == "read_query":
                        read_query_tool = tool
        except Exception as e:
            return f"ReservationLookupTool error: Failed to find MCP tools - {str(e)}"

        if not read_query_tool:
            return "ReservationLookupTool error: read_query tool not available."

        # Route to appropriate action
        if action == "create":
            return self._create_reservation(
                customer_id, party_size, reservation_date, reservation_time, write_query_tool, read_query_tool
            )
        elif action == "lookup_by_id":
            return self._lookup_by_id(reservation_id, read_query_tool)
        elif action == "lookup_by_phone":
            return self._lookup_by_phone(phone, read_query_tool)
        else:
            return f"ReservationLookupTool error: Unknown action '{action}'. Use 'create', 'lookup_by_id', or 'lookup_by_phone'."

    def _create_reservation(
        self, customer_id: int, party_size: int, reservation_date: str, reservation_time: str, write_query_tool, read_query_tool
    ) -> str:
        """Create a new reservation."""
        if not customer_id or not party_size or not reservation_date or not reservation_time:
            return "ReservationLookupTool error: customer_id, party_size, reservation_date, and reservation_time are required for 'create' action."

        if not write_query_tool:
            return "ReservationLookupTool error: write_query tool not available for creating reservations."

        # Create reservation - combine date and time into datetime
        reservation_datetime = f"{reservation_date} {reservation_time}:00"
        create_reservation_sql = (
            f"INSERT INTO reservations (customer_id, party_size, reservation_datetime, status) "
            f"VALUES ({customer_id}, {party_size}, '{reservation_datetime}', 'confirmed')"
        )

        try:
            write_query_tool.run(query=create_reservation_sql)
        except Exception as e:
            return f"ReservationLookupTool error: Failed to create reservation - {str(e)}"

        # Get the reservation ID - query the most recent reservation for this customer
        get_reservation_id_sql = (
            f"SELECT id FROM reservations WHERE customer_id = {customer_id} "
            f"ORDER BY id DESC LIMIT 1"
        )
        try:
            reservation_id_response = read_query_tool.run(query=get_reservation_id_sql)
            reservation_id_results = ast.literal_eval(reservation_id_response)
            reservation_id = reservation_id_results[0].get("id")
        except Exception as e:
            return f"ReservationLookupTool error: Failed to get reservation ID - {str(e)}"

        return (
            f"Reservation created successfully!\n"
            f"Reservation ID: {reservation_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Party Size: {party_size}\n"
            f"Date: {reservation_date}\n"
            f"Time: {reservation_time}\n"
            f"Status: Confirmed"
        )

    def _lookup_by_id(self, reservation_id: int, read_query_tool) -> str:
        """Look up reservation status by reservation ID."""
        if not reservation_id:
            return "ReservationLookupTool error: reservation_id is required for 'lookup_by_id' action."

        # Get reservation details with customer info
        reservation_sql = (
            "SELECT r.id, r.customer_id, r.reservation_datetime, r.party_size, r.special_requests, r.status, "
            "c.first_name, c.last_name, c.phone "
            "FROM reservations r "
            "JOIN customers c ON r.customer_id = c.id "
            f"WHERE r.id = {reservation_id}"
        )

        try:
            reservation_response = read_query_tool.run(query=reservation_sql)
            reservation_results = ast.literal_eval(reservation_response)
        except Exception as e:
            return f"ReservationLookupTool error: Failed to lookup reservation - {str(e)}"

        if not reservation_results:
            return f"No reservation found with ID {reservation_id}."

        reservation = reservation_results[0]

        # Format response
        response = (
            f"Reservation #{reservation.get('id')}\n"
            f"Status: {reservation.get('status', 'Unknown')}\n"
            f"Customer: {reservation.get('first_name')} {reservation.get('last_name')}\n"
            f"Phone: {reservation.get('phone', 'N/A')}\n"
            f"Date & Time: {reservation.get('reservation_datetime')}\n"
            f"Party Size: {reservation.get('party_size')} people\n"
        )

        special_requests = reservation.get('special_requests')
        if special_requests:
            response += f"Special Requests: {special_requests}\n"

        return response

    def _lookup_by_phone(self, phone: str, read_query_tool) -> str:
        """Look up all reservations for a customer by phone number."""
        if not phone:
            return "ReservationLookupTool error: phone is required for 'lookup_by_phone' action."

        # Get customer ID from phone
        customer_sql = f"SELECT id, first_name, last_name FROM customers WHERE phone = '{phone}'"

        try:
            customer_response = read_query_tool.run(query=customer_sql)
            customer_results = ast.literal_eval(customer_response)
        except Exception as e:
            return f"ReservationLookupTool error: Failed to lookup customer - {str(e)}"

        if not customer_results:
            return f"No customer found with phone number {phone}."

        customer = customer_results[0]
        customer_id = customer.get("id")

        # Get all reservations for this customer
        reservations_sql = (
            "SELECT id, reservation_datetime, party_size, status "
            "FROM reservations "
            f"WHERE customer_id = {customer_id} "
            "ORDER BY reservation_datetime DESC"
        )

        try:
            reservations_response = read_query_tool.run(query=reservations_sql)
            reservations_results = ast.literal_eval(reservations_response)
        except Exception as e:
            return f"ReservationLookupTool error: Failed to lookup reservations - {str(e)}"

        if not reservations_results:
            return f"No reservations found for {customer.get('first_name')} {customer.get('last_name')} (phone: {phone})."

        # Format response
        response = f"Reservations for {customer.get('first_name')} {customer.get('last_name')} (phone: {phone}):\n\n"

        for reservation in reservations_results:
            response += (
                f"Reservation #{reservation.get('id')} - {reservation.get('status')}\n"
                f"  Date & Time: {reservation.get('reservation_datetime')}\n"
                f"  Party Size: {reservation.get('party_size')} people\n\n"
            )

        return response.strip()
