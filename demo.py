"""
Demo script for restaurant flow - runs known-good scenarios.
Use this to verify the system works before presenting to your manager.
"""

from restaurant_flow.main import RestaurantFlow
from restaurant_flow.mcp_init import close_mcp_tools


def run_scenario(name: str, message: str):
    """Run a single scenario and print results."""
    print("\n" + "=" * 70)
    print(f"SCENARIO: {name}")
    print(f"INPUT: {message}")
    print("=" * 70)
    
    flow = RestaurantFlow()
    try:
        result = flow.kickoff(inputs={"customer_message": message})
        
        # Extract final response
        if hasattr(result, "final_response"):
            response = result.final_response
        elif isinstance(result, dict):
            response = result.get("final_response", str(result))
        else:
            response = str(result)
        
        print(f"\n✅ FINAL RESPONSE:\n{response}")
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


def main():
    """Run all demo scenarios."""
    # Use EXACT names from seed.sql to ensure customer lookup works
    # Available customers: Ava Thompson, Liam Patel, Emma Rodriguez, Noah Chen,
    # Mia Johnson, Ethan Garcia, Harper Davis, Lucas Nguyen, Isabella Martinez,
    # Oliver Wilson, Sophia Kim, James Brown, Grace Lopez, Henry Singh,
    # Charlotte Hernandez, Amelia Gonzalez, Benjamin Scott, Evelyn Rivera
    
    scenarios = [
        # Menu inquiry - should route to Menu Specialist
        (
            "Menu Inquiry",
            "What desserts do you have?"
        ),
        
        # Order with complete info - should route to Order Handler
        # Using exact menu item names and customer name from seed data
        (
            "Complete Order",
            "I want to order a Classic Caesar Salad and Pan-Seared Salmon for Noah Chen"
        ),
        
        # Reservation with complete info - should route to Reservation Agent
        # Using exact customer name from seed data
        (
            "Complete Reservation",
            "Book a table for 4 people for Harper Davis tomorrow at 7pm"
        ),
        
        # Complaint - should trigger escalation
        (
            "Complaint (Escalation)",
            "The food was cold and the waiter was rude. I want a refund."
        ),
        
        # Unclear intent - should route to fallback
        (
            "Unclear Intent (Fallback)",
            "Hello, is anyone there?"
        ),
    ]
    
    print("\n" + "#" * 70)
    print("# RESTAURANT FLOW DEMO - SINGLE-SHOT MODE")
    print("#" * 70)
    
    results = []
    for name, message in scenarios:
        success = run_scenario(name, message)
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} scenarios passed")
    
    close_mcp_tools()


if __name__ == "__main__":
    main()
