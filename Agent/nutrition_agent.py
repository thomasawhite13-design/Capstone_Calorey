from __future__ import annotations

import os
import re
from typing import Optional

from dotenv import load_dotenv
from google.cloud import firestore
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from models import (
    AgentState,
    Meal,
    ProfileExtract,
    UserProfile,
    WeeklyMealPlan,
    WorkerState,
    planner_node_output,
)

load_dotenv()


# Logging

import logging
import json

# Configure once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Cloud Run expects raw JSON on stdout
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _scale_quantity_string(quantity_str: str, factor: float) -> str:
    """Scale the first numeric value found in a quantity string."""
    match = re.search(r"(\d*\.?\d+)", quantity_str)
    if not match:
        return quantity_str

    original_val = float(match.group(1))
    new_val = original_val * factor
    formatted_val = f"{new_val:.1f}" if new_val < 10 else f"{int(round(new_val))}"
    return quantity_str.replace(match.group(1), formatted_val, 1)


# ─────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────

class MetricsCalculator:
    """Pure calculation logic — no state, no I/O."""

    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "low": 1.375,
        "medium": 1.55,
        "high": 1.725,
        "athlete": 1.9,
    }

    GOAL_ADJUSTMENTS = {
        "weight_loss": -500,
        "muscle_gain": 300,
        "maintenance": 0,
        "athletic_performance": 200,
    }

    def calculate_targets(self, profile: UserProfile | dict) -> dict:
        if isinstance(profile, dict):
            profile = UserProfile.model_validate(profile)

        gender_bonus = 5 if profile.gender.lower() == "male" else -161
        bmr = (
            (10 * profile.weight)
            + (6.25 * profile.height)
            - (5 * profile.age)
            + gender_bonus
        )

        tdee = bmr * self.ACTIVITY_MULTIPLIERS[profile.activity_level]
        final_calories = int(tdee + self.GOAL_ADJUSTMENTS[profile.goal])

        protein_multiplier = (
            1.6 if profile.goal in ["muscle_gain", "athletic_performance"] else 0.8
        )
        final_protein = int(profile.weight * protein_multiplier)

        return {"calories": final_calories, "protein": final_protein}


# ─────────────────────────────────────────────
# Profile Service
# ─────────────────────────────────────────────

class ProfileService:
    """
    Handles profile extraction, merging, and persistence.
    Wraps Firestore so callers never touch the DB directly.
    """

    CHAT_MODEL = "openai:gpt-4o-mini"

    def __init__(self, db: firestore.Client):
        self._db = db
        self._metrics = MetricsCalculator()

    # ── Node helpers ──────────────────────────

    def extract_info(self, state: AgentState) -> dict:
        raw_profile = state["user_profile"]

        if isinstance(raw_profile, dict):
            raw_profile = UserProfile.model_validate(raw_profile)

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if not last_human or len(last_human.content.strip()) < 4:
            return {}  # Return nothing — don't overwrite profile with None

        model = init_chat_model(model=self.CHAT_MODEL, temperature=0)
        structured_llm = model.with_structured_output(ProfileExtract)

        system_msg = SystemMessage(content=(
            "Extract ONLY what the user has explicitly stated. "
            "If a field was not mentioned, leave it as None. "
            "Never guess or infer values. "
            "Only set allergies to ['No Allergies'] if the user explicitly states they have none, however once they have said they have no allergies you need to set it to ['No Allergies']. "
            "Only set dietary_type to 'unrestricted' if the user clearly states no dietary restrictions."
            "if a user mentions a height or weight without units, assume cm for height and kg for weight."
            "conversion for metres to cm is 1 metre = 100 cm"
            "conversion for feet to inches is 1 foot = 12 inches"
            "please record heights in either cm or inches"
            "conversion for weight in stones to pounds is 1 stone = 14 pounds"
            "all heights and weights should be values of one unit, not a combination (e.g., 5 feet 10 inches should be converted fully to inches)"
            "Finally if the user unambiguosly says they want a meal plan e.g.'generate meal plan', set wants_meal_plan to True, otherwise leave it as None."
        ))

        extracted: ProfileExtract = structured_llm.invoke(
            [system_msg] + state["messages"]
        )
        updates = {k: v for k, v in extracted.model_dump().items() if v is not None}

        updated_profile = raw_profile.model_copy(update=updates)

        if all([updated_profile.weight, updated_profile.height, updated_profile.age, updated_profile.gender, updated_profile.activity_level, updated_profile.goal]):
            targets = self._metrics.calculate_targets(updated_profile)
            updated_profile = updated_profile.model_copy(update={
                "daily_calorie_target": targets["calories"],
                "daily_protein_gram_target": targets["protein"],
            })

        return {"user_profile": updated_profile}

    def save_profile(self, state: AgentState) -> dict:
        profile = state["user_profile"]
        # Defensive check
        if isinstance(profile, dict):
            profile = UserProfile.model_validate(profile)

        user_id = profile.user_id
        if user_id:
            user_doc = self._db.collection("users").document(user_id)
            if user_doc.get().exists:
                user_doc.update({
                    "user_profile": profile.model_dump(),
                    "current_meal_plan": state["meal_plan"].model_dump() if isinstance(state.get("meal_plan"), WeeklyMealPlan) else state.get("meal_plan"),
                    "last_updated": firestore.SERVER_TIMESTAMP,
                })
            else:
                user_doc.set({
                    "user_profile": profile.model_dump(),
                    "current_meal_plan": state["meal_plan"].model_dump() if isinstance(state.get("meal_plan"), WeeklyMealPlan) else state.get("meal_plan"),
                    "created_at": firestore.SERVER_TIMESTAMP,
                })
        return state

    def is_profile_complete(self, profile: UserProfile) -> bool:
        if not isinstance(profile, UserProfile):
            profile = UserProfile.model_validate(profile)
        required = [
            profile.age, profile.weight, profile.height,
            profile.gender, profile.goal, profile.activity_level,
            profile.dietary_type, profile.meals_wanted, profile.daily_calorie_target, profile.daily_protein_gram_target
        ]
        return all(f not in [0, None, "", []] for f in required)


# ─────────────────────────────────────────────
# Orchestrator Service
# ─────────────────────────────────────────────

class OrchestratorService:
    """Drives the conversation to fill out a UserProfile."""

    CHAT_MODEL = "openai:gpt-4o-mini"

    SYSTEM_PROMPT_TEMPLATE = (
        "You are a Nutrition Orchestrator. Your goal is to fill out the user's profile.\n"
        "Review the current Profile and the conversation history.\n"
        "Don't over-confirm facts — take what the user says as truth and recite back "
        "the profile once, asking them to confirm before generating the meal plan.\n"
        "It is important we don't ask the same question too many times.\n\n"
        "Current Profile Status:\n{profile_data}\n\n"
        "INSTRUCTIONS:\n"
        "1. If any physical metrics (age, weight, height, gender) or goals are missing, ask politely.\n"
        "2. Only ask 1–2 questions at a time.\n"
        "3. Make sure you have prompted the user to fill out all required fields.\n"
        "REQUIRED FIELDS: age, weight, height, gender, goal, activity_level, dietary_type, meals_wanted.\n"
        "4. Don't ask for 'daily_calorie_target' or 'daily_protein_gram_target' — we calculate those."
    )

    def orchestrate(self, state: AgentState) -> dict:
        profile_data = (
            state["user_profile"].model_dump()
            if state.get("user_profile")
            else "No data yet"
        )
        model = init_chat_model(model=self.CHAT_MODEL, temperature=0.0)
        system_msg = SystemMessage(
            content=self.SYSTEM_PROMPT_TEMPLATE.format(profile_data=profile_data)
        )
        response = model.invoke([system_msg] + state["messages"])
        return {"messages": [response]}
    
    def finalise_plan(self, state: AgentState) -> dict:
        """
        Runs after all meal workers have completed.
        Updates the final message and sets a flag for the frontend.
        """
        profile = state["user_profile"]
        if isinstance(profile, dict):
            profile = UserProfile.model_validate(profile)

        updated_profile = profile.model_copy(
        update={"wants_meal_plan": False}
        )

        confirmation_msg = (
            "Success! I have generated your personalized 7-day meal plan. "
            "You can now view it by clicking the 'View Meal Plan' button below "
            "or by navigating to the Meal Plan section in your dashboard."
        )
    
        # Create the AI message for the chat history
        ai_message = AIMessage(content=confirmation_msg)
    
        return {
            "messages": [ai_message],
            "user_profile": updated_profile,
            "plan_generated_this_turn": True # Custom state flag
        }


# ─────────────────────────────────────────────
# Meal Planner Service
# ─────────────────────────────────────────────

class MealPlannerService:
    """Creates the high-level 7-day skeleton and routes workers."""

    PLANNER_PROMPT = (
        "You are a Master Menu Planner.\n"
        "For breakfast and lunch provide a high-level theme (e.g., 'sandwich, salad, pasta').\n"
        "For dinners provide a theme (e.g., Italian, Chinese, Mexican) and a primary protein.\n"
        "Create a 7-day high-level meal plan skeleton for:\n{user_profile}\n\n"
        "CRITICAL RULES:\n"
        "1. VARIETY: No two days should share the same cuisine or primary protein.\n"
        "2. TEXTURE: Vary meal types (soups, stir-fries, salads, roasts).\n"
        "3. EDAMAM READY: Use keywords that work well with a recipe API."
    )

    def plan(self, state: AgentState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        structured_llm = llm.with_structured_output(planner_node_output)
        response = structured_llm.invoke(
            self.PLANNER_PROMPT.format(user_profile=state["user_profile"])
        )
        return {"skeleton": response}

    # ── Fan-out router ─────────────────────────

    @staticmethod
    def route_after_planner(state: AgentState):
        profile = state.get("user_profile")
        plan = state.get("meal_plan")
        sends = []

        if isinstance(profile, dict):
            profile = UserProfile.model_validate(profile)
        if isinstance(plan, dict):
            plan = WeeklyMealPlan.model_validate(plan)
        
        wanted = profile.meals_wanted or ["breakfast", "lunch", "dinner"]

        if "breakfast" in wanted and not plan.breakfast:
            sends.append(Send("breakfast_worker", {
                "day": "Weekly",
                "meal_type": "breakfast",
                "constraint": state["skeleton"].breakfast_theme,
                "user_data": state["user_profile"],
                "day_meal_plan": None,
                "meal_plan": None,
                "iteration_count": 0,
            }))

        if "lunch" in wanted and not plan.lunch:
            sends.append(Send("lunch_worker", {
                "day": "Weekly",
                "meal_type": "lunch",
                "constraint": state["skeleton"].lunch_theme,
                "user_data": state["user_profile"],
                "day_meal_plan": None,
                "meal_plan": None,
                "iteration_count": 0,
            }))

        if "dinner" in wanted and not plan.dinners:
            sends.extend([
                Send("dinner_workers", {
                    "day": d.day,
                    "meal_type": "dinner",
                    "constraint": d,
                    "user_data": state["user_profile"],
                    "day_meal_plan": None,
                    "meal_plan": None,
                    "iteration_count": 0,
                })
                for d in state["skeleton"].dinner_themes
            ])

        return sends if sends else "orchestrator"


# ─────────────────────────────────────────────
# Worker Pipeline (search → format → validate → package)
# ─────────────────────────────────────────────

class MealWorkerPipeline:
    """
    Encapsulates the sub-graph used for every meal worker.
    Each instance builds and compiles the sub-graph once.
    """

    _FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a data extraction specialist. Convert unstructured meal research "
            "into a structured Meal.\n\nRESEARCH DATA:\n{search_data}\n\n"
            "USER PROFILE:\n{user_data}"
        )),
        ("user", "Extract the meal and nutritional information into the required schema."),
    ])

    _BUDGET_SPLIT = {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.40,
        "snack": 0.10,
    }

    def __init__(self, tavily_client: TavilyClient):
        self._tavily = tavily_client
        self._graph = self._build()

    # ── Public ────────────────────────────────

    @property
    def graph(self):
        return self._graph

    # ── Search ────────────────────────────────

    def _make_search_tool(self):
        tavily = self._tavily  # capture for closure

        @tool
        def meal_search_tool(query: str, search_depth: str = "advanced"):
            """Search the web for meal plans, recipes, and nutritional info."""
            try:
                response = tavily.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=3,
                    include_answer=True,
                )
                truncated = [
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "content": (r.get("content", "")[:800] + "...")
                        if len(r.get("content", "")) > 800
                        else r.get("content", ""),
                    }
                    for r in response.get("results", [])
                ]
                return {"ai_summary": response.get("answer"), "sources": truncated}
            except Exception as exc:
                return f"Search failed: {exc}"

        return meal_search_tool

    def _search_node(self, state: WorkerState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=800)
        instructions = (
            f"You are a research assistant finding meals for {state['meal_type']}. "
            f"User Profile: {state['user_data']}. "
            "Ensure the meal is realistic and matches dietary restrictions. "
            "Use the provided constraints to guide search keywords. "
            "If meals don't hit calorie targets, increase ingredient quantities consistently. "
            f"Constraints: {state['constraint']}"
        )
        if state.get("errors"):
            instructions += (
                f"\n\nCRITICAL: Previous attempt rejected for: {state['errors']}. "
                "Adjust your search to fix this."
            )

        agent = create_react_agent(
            llm,
            tools=[self._make_search_tool()],
            prompt=SystemMessage(content=instructions),
        )
        result = agent.invoke({
            "messages": [HumanMessage(
                content=f"Find meals for {state['meal_type']} based on the profile."
            )]
        })
        return {
            "search_data": result["messages"][-1].content,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ── Format ────────────────────────────────

    def _format_node(self, state: WorkerState) -> dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = self._FORMATTER_PROMPT | llm.with_structured_output(Meal)
        try:
            formatted = chain.invoke({
                "search_data": state["search_data"],
                "user_data": state["user_data"].model_dump() if state["user_data"] else {},
            })
            return {"day_meal_plan": formatted}
        except Exception as exc:
            return {"errors": [f"Formatting failed: {exc}"]}

    # ── Validate ──────────────────────────────

    def _validate_node(self, state: WorkerState) -> dict:
        user: UserProfile = state["user_data"]
        meal: Meal = state["day_meal_plan"]
        errors = []

        multiplier = self._BUDGET_SPLIT.get(meal.meal_type.lower(), 0.33)
        meal_target = user.daily_calorie_target * multiplier
        cal_diff = meal.calories - meal_target

        if abs(cal_diff) > 100:
            factor = round(meal_target / meal.calories, 1)
            for ing in meal.ingredients:
                ing.calories = int(ing.calories * factor)
                ing.protein = int(ing.protein * factor)
                ing.quantity = _scale_quantity_string(ing.quantity, factor)
            meal.calories = int(meal.calories * factor)
            meal.protein = int(meal.protein * factor)
            meal.carbs = int(meal.carbs * factor)
            meal.fat = int(meal.fat * factor)

        user_allergies = {
            a.lower().strip()
            for a in user.allergies
            if a.lower() != "no allergies"
        }
        user_dislikes = {d.lower().strip() for d in user.disliked_foods}

        meal_allergens: set[str] = set()
        is_vegan = is_vegetarian = True

        for ing in meal.ingredients:
            if not ing.vegan:
                is_vegan = False
            if not ing.vegetarian:
                is_vegetarian = False
            for allergen in ing.allergens:
                meal_allergens.add(allergen.lower().strip())
            if ing.name.lower() in user_dislikes:
                errors.append(f"PREFERENCE: User dislikes '{ing.name}', try to substitute.")

        forbidden = user_allergies & meal_allergens
        if forbidden:
            errors.append(f"ALLERGEN: Meal contains forbidden ingredients: {', '.join(forbidden)}.")
        if user.dietary_type == "vegan" and not is_vegan:
            errors.append("DIETARY: User is vegan but meal contains non-vegan items.")
        if user.dietary_type == "vegetarian" and not is_vegetarian:
            errors.append("DIETARY: User is vegetarian but meal contains meat or fish.")

        return {"errors": errors}

    # ── Package ───────────────────────────────

    @staticmethod
    def _package_node(state: WorkerState) -> dict:
        meal_type = state["meal_type"]
        meal_data = state["day_meal_plan"]

        if meal_type == "breakfast":
            return {"meal_plan": {"breakfast": meal_data}}
        if meal_type == "lunch":
            return {"meal_plan": {"lunch": meal_data}}
        return {"meal_plan": {"dinners": {state["day"].lower(): meal_data}}}

    # ── Graph assembly ────────────────────────

    def _build(self):
        sub = StateGraph(WorkerState)
        sub.add_node("search", self._search_node)
        sub.add_node("format", self._format_node)
        sub.add_node("validate", self._validate_node)
        sub.add_node("package_result", self._package_node)

        sub.set_entry_point("search")
        sub.add_edge("search", "format")
        sub.add_edge("format", "validate")
        sub.add_conditional_edges(
            "validate",
            lambda x: (
                "search"
                if x.get("errors") and x["iteration_count"] < 3
                else "package_result"
            ),
        )
        sub.add_edge("package_result", END)
        return sub.compile()


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

class GraphRouter:
    """
    Stateless routing functions used as conditional edges in the main graph.
    Kept separate so they can be unit-tested independently.
    """

    def __init__(self, profile_service: ProfileService):
        self._profile = profile_service

    def route_after_extract(self, state: AgentState) -> str:
        profile = state.get("user_profile")
        if isinstance(profile, dict):
            profile = UserProfile.model_validate(profile)

        if profile is None:
            return "orchestrator"

        # --- IN-LINE DEBUGGING CHECK ---
        checks = {
            "age": profile.age,
            "weight": profile.weight,
            "height": profile.height,
            "gender": profile.gender,
            "goal": profile.goal,
            "activity": profile.activity_level,
            "diet": profile.dietary_type,
            "meals": profile.meals_wanted,
            "calories": profile.daily_calorie_target,
            "protein": profile.daily_protein_gram_target
        }

        missing_fields = [k for k, v in checks.items() if v in [0, None, "", []]]
    
        if missing_fields:
            logger.info(json.dumps({
                "message": "ROUTER: Profile incomplete. ",
                "missing_fields": missing_fields,
                "profile_data": profile.model_dump() if profile else "NO PROFILE",
            }))
            return "orchestrator"

        if profile.wants_meal_plan:
            logger.info(json.dumps({
                "message": "ROUTER: All systems go. Routing to planner.",
                "profile_data": profile.model_dump() if profile else "NO PROFILE",
            }))
            return "planner"
    
        return "orchestrator"

# ─────────────────────────────────────────────
# NutritionAgent  ← main public class
# ─────────────────────────────────────────────

class NutritionAgent:
    """
    Top-level façade.

    Usage
    -----
        agent = NutritionAgent()

        # Standard chat turn (returns the assistant message text)
        reply = agent.chat(thread_id="user-123", user_message="Hi, I'm 30 years old…")

        # Retrieve current state
        profile   = agent.get_profile(thread_id="user-123")
        meal_plan = agent.get_meal_plan(thread_id="user-123")
    """

    def __init__(self, checkpointer=None, db: firestore.Client = None):
        load_dotenv()
        os.environ.setdefault(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "df-trial-487814-b3428a5d8bc5.json",
        )

        self._db = db or firestore.Client(database="nutrition-agent-store")
        self._tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Services
        self._profile_svc = ProfileService(self._db)
        self._orchestrator_svc = OrchestratorService()
        self._planner_svc = MealPlannerService()
        self._worker_pipeline = MealWorkerPipeline(self._tavily)
        self._router = GraphRouter(self._profile_svc)

        # Compiled graph
        self._graph = self._build_graph(checkpointer)

    # ── Public API ────────────────────────────

    def chat(self, thread_id: str, user_message: str, user_id: Optional[str] = None) -> str:
        """
        Process one user turn and return the assistant's reply text.

        Parameters
        ----------
        thread_id    : Unique conversation / session ID (required).
        user_message : The human's latest message.
        user_id      : Only needed on the *very first* turn for a new thread.
                       Ignored on subsequent turns once the profile is seeded.

        Raises ValueError if thread_id is empty, or if this is the first turn
        and no user_id was provided.
        """
        if not thread_id:
            raise ValueError("thread_id must not be empty")

        config = {"configurable": {"thread_id": thread_id}}

        # Check whether this thread already has a profile in the checkpointer.
        existing_state = self._graph.get_state(config)
        profile_exists = (
            existing_state
            and existing_state.values.get("user_profile") is not None
        )

        if not profile_exists:
            # First turn — we must seed a bare UserProfile so every downstream
            # node can safely do state["user_profile"] without a KeyError.
            if not user_id:
                raise ValueError(
                    "user_id is required on the first turn to initialise the profile."
                )
            seed_profile = UserProfile(user_id=user_id)
            input_state = {
                "messages": [HumanMessage(content=user_message)],
                "user_profile": seed_profile,
            }
        else:
            # Subsequent turns — profile already lives in the checkpointer.
            input_state = {"messages": [HumanMessage(content=user_message)]}

        result = self._graph.invoke(input_state, config=config)

        last_ai = next(
            (
                m
                for m in reversed(result.get("messages", []))
                if not isinstance(m, HumanMessage)
            ),
            None,
        )
        return last_ai.content if last_ai else ""

    def get_profile(self, thread_id: str) -> Optional[UserProfile]:
        """Return the current UserProfile for a thread, or None."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        return state.values.get("user_profile") if state else None

    def get_meal_plan(self, thread_id: str) -> Optional[WeeklyMealPlan]:
        """Return the current WeeklyMealPlan for a thread, or None."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        return state.values.get("meal_plan") if state else None
    
    def meal_generated(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        return state.values.get("plan_generated_this_turn", False) if state else False

    # ── Graph wiring ──────────────────────────

    def _build_graph(self, checkpointer=None):
        worker_graph = self._worker_pipeline.graph

        builder = StateGraph(AgentState)

        # Nodes
        builder.add_node("extract_info", self._profile_svc.extract_info)
        builder.add_node("orchestrator", self._orchestrator_svc.orchestrate)
        builder.add_node("planner", self._planner_svc.plan)
        builder.add_node("breakfast_worker", worker_graph)
        builder.add_node("lunch_worker", worker_graph)
        builder.add_node("dinner_workers", worker_graph)
        builder.add_node("finalise_plan", self._orchestrator_svc.finalise_plan)
        builder.add_node("save_profile", self._profile_svc.save_profile)

        # Edges
        builder.add_edge(START, "extract_info")
        builder.add_conditional_edges(
            "extract_info",
            self._router.route_after_extract,
            {
                "planner": "planner",
                "orchestrator": "orchestrator",
            },
        )

        # Fan-out after planner and each meal worker
        route = MealPlannerService.route_after_planner
        builder.add_conditional_edges(
            "planner",
            route,
            {
                "breakfast_worker": "breakfast_worker",
                "lunch_worker": "lunch_worker",
                "dinner_workers": "dinner_workers",
                "orchestrator": "orchestrator",
            },
        )
        builder.add_edge("breakfast_worker", "finalise_plan")
        builder.add_edge("lunch_worker", "finalise_plan")
        builder.add_edge("dinner_workers", "finalise_plan")
        builder.add_edge("finalise_plan", "save_profile")
        builder.add_edge("orchestrator", "save_profile")
        builder.add_edge("save_profile", END)

        return builder.compile(checkpointer=checkpointer)
