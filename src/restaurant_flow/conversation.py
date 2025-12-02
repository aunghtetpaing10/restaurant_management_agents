"""Conversation manager for multi-turn chat support."""

import uuid
from typing import Dict

from restaurant_flow.models import (
    ConversationState,
    ConversationMessage,
    ClarificationAnalysis,
    REQUIRED_INFO,
)
from restaurant_flow.agents import create_clarification_agent
from restaurant_flow.prompts import get_clarification_prompt
from restaurant_flow.main import RestaurantFlow


class ConversationManager:
    """Manages multi-turn conversations with clarification before flow execution."""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationState] = {}
    
    def get_or_create_session(self, session_id: str | None = None) -> ConversationState:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_id = session_id or str(uuid.uuid4())[:8]
        session = ConversationState(session_id=new_id)
        self.sessions[new_id] = session
        return session
    
    def _format_history(self, session: ConversationState) -> str:
        """Format conversation history for the prompt."""
        if not session.messages:
            return "(No previous messages)"
        
        lines = []
        for msg in session.messages:
            role = "Customer" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def _build_combined_message(
        self,
        session: ConversationState,
        intent: str,
        latest_user_message: str,
    ) -> str:
        """Build a combined message tailored to the detected intent."""

        # For simple intents, just use the latest user input
        simple_intents = {"menu_inquiry", "general_question", "complaint", "other"}
        if intent in simple_intents:
            if latest_user_message:
                return latest_user_message
            user_messages = [m.content for m in session.messages if m.role == "user"]
            return user_messages[-1] if user_messages else ""

        info = session.collected_info
        parts = []

        # Add items if ordering
        if info.get("items"):
            items_str = ", ".join(info["items"]) if isinstance(info["items"], list) else info["items"]
            parts.append(f"I want to order {items_str}")
        
        # Add customer name
        if info.get("customer_name"):
            parts.append(f"for {info['customer_name']}")
        
        # Add reservation details
        if info.get("party_size"):
            parts.append(f"for {info['party_size']} people")
        if info.get("date_time"):
            parts.append(f"at {info['date_time']}")
        
        # Fallback to last user message if no structured info
        if not parts and session.messages:
            user_messages = [m.content for m in session.messages if m.role == "user"]
            return " ".join(user_messages)
        
        return " ".join(parts)
    
    def _generate_clarification_question(self, intent: str, missing_fields: list[str]) -> str:
        """Generate a natural question for the required missing fields."""
        if not missing_fields:
            return ""

        friendly_names = {
            "customer_name": "your name",
            "items": "what you'd like to order",
            "party_size": "how many guests",
            "date_time": "the date and time",
        }

        parts = [friendly_names.get(field, field.replace("_", " ")) for field in missing_fields]

        if intent == "reservation_request":
            prefix = "To lock in your reservation"
        elif intent == "order_request":
            prefix = "To place your order"
        else:
            prefix = "To help you"

        if len(parts) == 1:
            request_str = parts[0]
        else:
            request_str = ", ".join(parts[:-1]) + f" and {parts[-1]}"

        return f"{prefix}, could you share {request_str}, please?"

    def chat(self, user_message: str, session_id: str | None = None) -> str:
        """
        Process a user message in a conversation.
        
        Returns either a clarifying question or the final flow response.
        """
        session = self.get_or_create_session(session_id)
        
        # Add user message to history
        session.messages.append(ConversationMessage(role="user", content=user_message))
        
        # Analyze conversation with clarification agent
        agent = create_clarification_agent()
        history = self._format_history(session)
        prompt = get_clarification_prompt(history, user_message)
        
        try:
            result = agent.kickoff(prompt, response_format=ClarificationAnalysis)
            analysis = result.pydantic
            
            print(f"[CONVERSATION] Intent: {analysis.intent} | Ready: {analysis.is_ready}")
            print(f"[CONVERSATION] Collected: {analysis.collected_info}")
            
            # Update session with extracted info
            session.current_intent = analysis.intent
            session.collected_info.update(analysis.collected_info)

            # Determine required info for detected intent
            required_fields = REQUIRED_INFO.get(analysis.intent, [])
            collected_snapshot = {
                **session.collected_info,
                **analysis.collected_info,
            }
            missing_required = [
                field
                for field in required_fields
                if not collected_snapshot.get(field)
            ]

            is_ready = not missing_required
            
            if is_ready:
                # Ready to process - run the flow
                session.status = "ready_to_process"
                
                # Build combined message from all collected info
                combined_message = self._build_combined_message(
                    session,
                    analysis.intent,
                    user_message,
                )
                print(f"[CONVERSATION] Proceeding with: '{combined_message}'")
                
                # Run the restaurant flow
                flow = RestaurantFlow()
                flow_result = flow.kickoff(
                    inputs={
                        "customer_message": combined_message,
                        "current_customer_id": session.customer_id,
                    }
                )

                # Flow kickoff returns the final step output (FinalResponse model in this flow)
                if isinstance(flow_result, dict):
                    final_response = flow_result.get("final_response")
                    if isinstance(final_response, dict):
                        response = final_response.get("final_response")
                    else:
                        response = str(final_response) if final_response is not None else ""
                else:
                    # Pydantic FinalResponse or similar object
                    inner = getattr(flow_result, "final_response", None)
                    if isinstance(inner, str):
                        response = inner
                    elif inner is not None and hasattr(inner, "final_response"):
                        # e.g. flow_result.final_response is another model with .final_response field
                        response = getattr(inner, "final_response", None) or str(inner)
                    else:
                        # Fallback: just stringify whatever we got back
                        response = str(flow_result)
                
                # Add assistant response to history
                session.messages.append(ConversationMessage(role="assistant", content=response))
                
                # Reset session for next interaction
                session.collected_info = {}
                session.status = "gathering_info"
                
                return response
            else:
                # Need more info - ask clarifying question
                question = (
                    self._generate_clarification_question(analysis.intent, missing_required)
                    or analysis.clarification_question
                    or "Could you share a bit more detail?"
                )
                session.messages.append(ConversationMessage(role="assistant", content=question))
                return question
                
        except Exception as e:
            print(f"[CONVERSATION] Error: {str(e)}")
            error_response = "I'm sorry, I'm having trouble understanding. Could you please rephrase your request?"
            session.messages.append(ConversationMessage(role="assistant", content=error_response))
            return error_response
    
    def reset_session(self, session_id: str):
        """Reset a session to start fresh."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_session_info(self, session_id: str) -> dict | None:
        """Get current session state for debugging."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "intent": session.current_intent,
            "collected_info": session.collected_info,
            "status": session.status,
            "message_count": len(session.messages),
        }


def run_chat_demo():
    """Demo function to test multi-turn conversation."""
    manager = ConversationManager()
    session_id = "demo"
    
    print("=" * 60)
    print("Restaurant Chat Demo (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        response = manager.chat(user_input, session_id)
        print(f"\nAssistant: {response}")
        
        # Show session state for debugging
        info = manager.get_session_info(session_id)
        if info:
            print(f"\n[Debug] {info}")


if __name__ == "__main__":
    run_chat_demo()
