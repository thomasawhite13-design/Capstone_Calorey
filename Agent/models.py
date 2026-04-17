from typing import Annotated, List, Optional, Literal, Dict, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

# Define the allowed options as Literals
GoalLiteral = Literal["weight_loss", "muscle_gain", "maintenance", "athletic_performance"]
ActivityLiteral = Literal["sedentary", "low", "medium", "high", "athlete"]
DietLiteral = Literal["vegan", "vegetarian", "pescatarian", "paleo", "keto", "unrestricted"]
GenderLiteral = Literal["male", "female"]
DayLiteral = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday","Weekly"]
MealLiteral = Literal["breakfast", "lunch", "dinner", "snack"]

class UserProfile(BaseModel):
    """
    Comprehensive user profile for nutritional analysis and meal planning.
    """
    user_id: str = Field(..., description="Unique identifier for the user")
    
    # Physical Stats
    age: int = Field(default=0, ge=0, le=120)
    gender: GenderLiteral = Field(default=None)
    height: float = Field(default=0.0, description="Height in centimeters")
    weight: float = Field(default=0.0, description="Weight in kilograms")

    # Strict Categorical Data
    goal: Optional[GoalLiteral] = Field(default=None, description="The primary fitness objective")
    activity_level: Optional[ActivityLiteral] = Field(default=None, description="Daily physical activity level")
    dietary_type: Optional[DietLiteral] = Field(default=None, description="Specific dietary framework, if they specify no dietary requirements, this should be 'unrestricted'")

    # Nutritional Targets
    daily_calorie_target: int = Field(default=0, lt=10000)
    daily_protein_gram_target: int = Field(default=0, ge=0)
    
    # Preferences & Restrictions
    allergies: List[str] = Field(default_factory=list, description="List of food allergies (e.g., 'peanuts', 'shellfish'), if they specify no allergies, the list should have one element 'No Allergies'")
    disliked_foods: List[str] = Field(default_factory=list, description="Foods the user prefers to avoid but isn't allergic to")
    meals_wanted: List[MealLiteral] = Field(default_factory=list, description="Types of meals the user wants included in their plan (e.g., ['breakfast', 'dinner'])")

    # Metadata
    wants_meal_plan: bool = Field(False, description="User has explicitly confirmed they want the plan generated")

class ProfileExtract(BaseModel):
    """Only fields explicitly mentioned by the user."""
    age: Optional[int] = None
    gender: Optional[GenderLiteral] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    goal: Optional[GoalLiteral] = None
    activity_level: Optional[ActivityLiteral] = None
    dietary_type: Optional[DietLiteral] = None
    allergies: Optional[List[str]] = None
    disliked_foods: Optional[List[str]] = None
    meals_wanted: Optional[List[MealLiteral]] = None
    wants_meal_plan: Optional[bool] = None

class Ingredient(BaseModel): # Capitalized for PEP8
    name: str = Field(..., description="Name of the ingredient")
    calories: int = Field(..., description="Calories from this ingredient")
    protein: int = Field(0, description="Grams of protein")
    quantity: str = Field(..., description="E.g., '100g', '2 units'")
    allergens: List[str] = Field(default_factory=list)
    
    vegan: bool = Field(False)
    vegetarian: bool = Field(False)

class Meal(BaseModel):
    name: str = Field(..., description="Name of the meal (e.g., 'Berry Oatmeal')")
    meal_type: str = Field(..., description="Breakfast, Lunch, Dinner, or Snack")
    calories: int = Field(...)
    protein: int = Field(...)
    carbs: int = Field(0)
    fat: int = Field(0)
    
    ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)

class WeeklyMealPlan(BaseModel):
    breakfast: Optional[Meal] = Field(default=None, description="The breakfast meal plan for the week")
    lunch: Optional[Meal] = Field(default=None, description="The lunch meal plan for the week")
    dinners: Dict[str, Meal] = Field(
        default_factory=dict,
        description="A mapping of days of the week to their respective Dinner"
    )

def reduce_plan(old_plan: Optional[WeeklyMealPlan], update: dict) -> WeeklyMealPlan:
    if isinstance(old_plan, dict):
        old_plan = WeeklyMealPlan.model_validate(old_plan)
    
    new_plan = old_plan.model_copy() if old_plan else WeeklyMealPlan()

    if isinstance(update, WeeklyMealPlan):
        return update
    
    if isinstance(update, dict):
        if "breakfast" in update:
            new_plan.breakfast = update["breakfast"]
        if "lunch" in update:
            new_plan.lunch = update["lunch"]
        if "dinners" in update:
            # Merge new dinners into the existing dinners dict
            new_plan.dinners.update(update["dinners"])
        return new_plan

    return update

class DayConstraint(BaseModel):
    day: DayLiteral = Field(description="The day of the week")
    cuisine_theme: str = Field(description="A specific cuisine style (e.g., Mediterranean, Thai, Mexican)")
    primary_protein: str = Field(description="The main protein or base (e.g., Salmon, Lentils, Tofu)")
    search_keywords: str = Field(description="Specific keywords for the Edamam search tool")
    variety_logic: str = Field(description="Internal reasoning for why this day is different from others")

class planner_node_output(BaseModel):
    breakfast_theme: str = Field(description="The cuisine theme for breakfast")
    lunch_theme: str = Field(description="The cuisine theme for lunch")
    dinner_themes: List[DayConstraint] = Field(description="A list of 7 unique daily themes")

def latest(old: Optional[UserProfile], new: UserProfile) -> UserProfile:
    if isinstance(new, dict):
        return UserProfile.model_validate(new)
    return new

class AgentState(TypedDict):    
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: Annotated[UserProfile, latest]
    skeleton: planner_node_output =Field(description="The output from the planner node, which includes the cuisine themes for each meal type across the week")
    meal_plan: Annotated[WeeklyMealPlan, reduce_plan]
    plan_generated_this_turn: bool = False # Custom flag to indicate if the plan was generated in the current turn

class WorkerState(TypedDict):
    day: DayLiteral
    meal_type: Literal["breakfast", "lunch", "dinner"]
    user_data: dict
    constraint: DayConstraint
    search_data: Optional[str]
    day_meal_plan: Optional[Meal]
    meal_plan: Optional[WeeklyMealPlan]
    errors: Annotated[List[str], operator.add] # To track validation failures
    iteration_count: int = Field(default=0)

class SubAgentInput(TypedDict):
    day: str
    profile: UserProfile
