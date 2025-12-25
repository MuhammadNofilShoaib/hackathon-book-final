# Using Large Language Models for Cognitive Planning in Robots

## Concept

Large Language Models (LLMs) represent a paradigm shift in robotic cognitive planning by providing sophisticated reasoning capabilities that can decompose complex tasks into executable action sequences. Think of LLMs as the "reasoning engine" of a robot that can understand high-level goals, reason about the environment, and generate detailed step-by-step plans to achieve objectives.

In robotics, cognitive planning involves generating sequences of actions that transform the current state of the world to achieve a desired goal state. Traditional planning approaches required hand-coded rules and symbolic representations. LLMs bring natural language understanding to this process, allowing robots to interpret human instructions in natural language and generate appropriate action sequences without explicit programming for every possible scenario.

LLMs matter in Physical AI because they enable robots to perform complex reasoning tasks that were previously impossible without extensive manual programming. They can understand spatial relationships, temporal sequences, and causal connections, making them ideal for planning complex multi-step tasks in dynamic environments. For humanoid robots operating in human environments, LLMs can interpret natural language instructions and generate appropriate physical behaviors.

If you're familiar with how humans plan complex tasks by breaking them down into smaller steps, LLMs provide similar capabilities to robots. When a human receives an instruction like "Clean the kitchen," they naturally decompose this into subtasks like "find cleaning supplies," "wipe surfaces," and "dispose of waste." LLMs enable robots to perform similar decomposition and reasoning.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM ROBOT COGNITIVE PLANNING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌──────────────────────────────────────────┐   │
│  │   HUMAN         │    │                                          │   │
│  │   INSTRUCTION   │    │  ┌────────────────────────────────────┐  │   │
│  │                 │───▶│  │    LLM COGNITIVE PLANNER         │  │   │
│  │  "Clean the    │    │  │  ┌──────────────────────────────┐  │  │   │
│  │   kitchen and   │    │  │  │    LLM INFERENCE           │  │  │   │
│  │   then set the  │    │  │  │                              │  │  │   │
│  │   table"        │    │  │  │  • Prompt Engineering        │  │  │   │
│  │                 │    │  │  │  • Context Understanding     │  │  │   │
│  │                 │    │  │  │  • Task Decomposition       │  │  │   │
│  └─────────────────┘    │  │  └──────────────────────────────┘  │  │   │
│                         │  │                                    │  │   │
│  ┌─────────────────┐    │  │  ┌──────────────────────────────┐  │  │   │
│  │   ENVIRONMENT   │    │  │  │    TASK DECOMPOSITION      │  │  │   │
│  │   STATE         │    │  │  │                              │  │  │   │
│  │                 │───▶│  │  │  • Goal Analysis            │  │  │   │
│  │  • Object       │    │  │  │  • Subtask Generation      │  │  │   │
│  │    Locations    │    │  │  │  • Dependency Resolution   │  │  │   │
│  │  • Robot State  │    │  │  │  • Action Sequencing       │  │  │   │
│  │  • Constraints  │    │  │  └──────────────────────────────┘  │  │   │
│  └─────────────────┘    │  │                                    │  │   │
│                         │  │  ┌──────────────────────────────┐  │  │   │
│                         │  │  │    ACTION PLANNING          │  │  │   │
│                         │  │  │                              │  │  │   │
│                         │  │  │  • Path Planning            │  │  │   │
│                         │  │  │  • Manipulation Planning    │  │  │   │
│                         │  │  │  • Navigation Planning      │  │  │   │
│                         │  │  │  • Safety Constraints       │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  │                                    │  │   │
│                         │  │  ┌──────────────────────────────┐  │  │   │
│                         │  │  │    EXECUTION VERIFICATION  │  │  │   │
│                         │  │  │                              │  │  │   │
│                         │  │  │  • Feasibility Check        │  │  │   │
│                         │  │  │  • Resource Validation      │  │  │   │
│                         │  │  │  • Safety Validation       │  │  │   │
│                         │  │  │  • Plan Refinement          │  │  │   │
│                         │  │  └──────────────────────────────┘  │  │   │
│                         │  └──────────────────────────────────────────┘  │
│                         │         │                                   │   │
│                         │         ▼                                   │   │
│  ┌─────────────────┐    │  ┌──────────────────────────────────────────┐│   │
│  │   PHYSICAL      │    │  │         ROBOT EXECUTION                ││   │
│  │   ROBOT         │    │  │                                          ││   │
│  │                 │    │  │  ┌────────────────────────────────────┐  ││   │
│  │  • Manipulation │    │  │  │    EXECUTION PLANNING            │  ││   │
│  │  • Navigation  │    │  │  │                                    │  ││   │
│  │  • Locomotion  │    │  │  │  • Trajectory Generation          │  ││   │
│  │  • Interaction │    │  │  │  • Control Law Application        │  ││   │
│  │                 │    │  │  │  • Safety Verification           │  ││   │
│  │                 │    │  │  └────────────────────────────────────┘  ││   │
│  └─────────────────┘    │  │                                          ││   │
│         │                │  │  ┌────────────────────────────────────┐  ││   │
│         ▼                │  │  │    PHYSICAL EXECUTION             │  ││   │
│  ┌─────────────────┐    │  │  │                                    │  ││   │
│  │   ACTION        │    │  │  │  • Motor Control                  │  ││   │
│  │   EXECUTION     │    │  │  │  • Sensor Feedback               │  ││   │
│  │                 │    │  │  │  • Closed-loop Control           │  ││   │
│  │  • Grasping     │    │  │  │  • Compliance Control           │  ││   │
│  │  • Moving       │    │  │  │  • Balance Maintenance          │  ││   │
│  │  • Manipulating │    │  │  └────────────────────────────────────┘  ││   │
│  └─────────────────┘    │  └──────────────────────────────────────────┘│   │
│                         └──────────────────────────────────────────────────────┘
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                LLM PLANNING PROCESS FLOW                                │
│                                                                         │
│  INPUT: Natural Language Instruction ──▶ LLM Processing ──▶ Plan      │
│      │                                        │                      │   │
│      ▼                                        ▼                      ▼   │
│  [User Goal]    ┌─────────────────┐    ┌─────────────────┐          │   │
│                 │ 1. GOAL         │    │ 4. ACTION      │          │   │
│                 │    ANALYSIS     │    │    SEQUENCE    │          │   │
│                 └─────────────────┘    └─────────────────┘          │   │
│                        │                       │                     │   │
│                        ▼                       ▼                     │   │
│                 ┌─────────────────┐    ┌─────────────────┐          │   │
│                 │ 2. TASK        │    │ 5. EXECUTION   │          │   │
│                 │    DECOMPOSITION│    │    VALIDATION │          │   │
│                 └─────────────────┘    └─────────────────┘          │   │
│                        │                       │                     │   │
│                        ▼                       ▼                     │   │
│                 ┌─────────────────┐    ┌─────────────────┐          │   │
│                 │ 3. CONTEXT     │    │ 6. FEEDBACK    │          │   │
│                 │    ANALYSIS    │    │    LOOP        │          │   │
│                 └─────────────────┘    └─────────────────┘          │   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                LLM-ROBOT INTEGRATION ARCHITECTURE                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LLM INFERENCE ENGINE                                         │   │
│  │  (GPT, Claude, Llama, etc.)                                 │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  PROMPT ENGINEERING                                     │  │   │
│  │  │  • System Context                                      │  │   │
│  │  │  • Environment State                                   │  │   │
│  │  │  • Robot Capabilities                                  │  │   │
│  │  │  • Task Constraints                                    │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  REASONING MODULES                                      │  │   │
│  │  │  • Chain-of-Thought                                    │  │   │
│  │  │  • Tree-of-Thoughts                                    │  │   │
│  │  │  • Reflection & Self-Correction                        │  │   │
│  │  │  • Multi-step Planning                                 │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CUSTOM ROBOTICS INTERFACE                                   │   │
│  │  (Bridges LLM and Robot Systems)                            │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  PARSING & VALIDATION                                  │  │   │
│  │  │  • Natural Language → Robot Actions                   │  │   │
│  │  │  • Action Feasibility Check                           │  │   │
│  │  │  • Safety Constraint Verification                      │  │   │
│  │  │  • Plan Refinement                                     │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  EXECUTION BRIDGE                                       │  │   │
│  │  │  • Action Sequence Execution                          │  │   │
│  │  │  • Real-time Plan Adjustment                          │  │   │
│  │  │  • Failure Recovery                                    │  │   │
│  │  │  • Progress Monitoring                                │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This diagram illustrates the LLM cognitive planning architecture, showing how natural language instructions are processed by LLMs to generate executable robot action sequences.

## Real-world Analogy

Think of LLMs for robotic planning like a highly experienced project manager who can take high-level goals and break them down into detailed, executable tasks. Just as a project manager receives a goal like "organize a company event" and creates a detailed plan with subtasks like "book venue," "order catering," "send invitations," and "coordinate logistics," an LLM-powered robot can receive a high-level instruction and generate a detailed sequence of physical actions.

A human project manager needs to:
- Understand the high-level goal and requirements
- Break down the goal into manageable subtasks
- Consider constraints like budget, timeline, and resources
- Plan the sequence and dependencies of tasks
- Anticipate potential issues and create contingency plans

Similarly, an LLM for robot planning:
- Interprets natural language instructions from humans
- Decomposes high-level goals into specific robot actions
- Considers environmental constraints and robot capabilities
- Sequences actions in a logical order with proper dependencies
- Anticipates potential failures and plans recovery strategies

Just as a skilled project manager can adapt plans when circumstances change, LLMs can generate alternative plans when initial approaches fail or when the environment changes. The difference is that while project managers plan for abstract tasks, LLMs for robotics must plan for physical actions in real-world environments.

## Pseudo-code (LLM-Planning / Robotics style)

```python
# Large Language Model Cognitive Planning for Robotics
import openai
import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import time
from enum import Enum

class RobotActionType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    SENSING = "sensing"
    WAIT = "wait"

@dataclass
class RobotAction:
    """Data class for robot actions"""
    action_type: RobotActionType
    parameters: Dict[str, Any]
    description: str
    priority: int = 1

@dataclass
class PlanStep:
    """Data class for plan steps"""
    step_id: int
    action: RobotAction
    preconditions: List[str]
    effects: List[str]
    dependencies: List[int]  # List of step IDs this step depends on

class EnvironmentState:
    """Represents the current state of the robot's environment"""

    def __init__(self):
        self.objects = {}
        self.robot_position = (0.0, 0.0, 0.0)
        self.robot_orientation = (0.0, 0.0, 0.0, 1.0)
        self.robot_gripper_state = "open"  # "open" or "closed"
        self.rooms = {}
        self.surfaces = {}

    def update_object_location(self, obj_name: str, location: Tuple[float, float, float]):
        """Update the location of an object"""
        if obj_name not in self.objects:
            self.objects[obj_name] = {}
        self.objects[obj_name]['location'] = location

    def get_object_location(self, obj_name: str) -> Optional[Tuple[float, float, float]]:
        """Get the location of an object"""
        if obj_name in self.objects and 'location' in self.objects[obj_name]:
            return self.objects[obj_name]['location']
        return None

    def get_objects_in_room(self, room_name: str) -> List[str]:
        """Get all objects in a specific room"""
        room_objects = []
        for obj_name, obj_data in self.objects.items():
            if obj_data.get('room') == room_name:
                room_objects.append(obj_name)
        return room_objects

class RobotCapabilities:
    """Defines what the robot is capable of doing"""

    def __init__(self):
        self.navigation = True
        self.manipulation = True
        self.sensing = True
        self.max_reach = 1.5  # meters
        self.max_payload = 2.0  # kg
        self.gripper_range = (0.0, 0.1)  # meters
        self.navigation_speed = 0.5  # m/s
        self.manipulation_precision = "medium"

class LLMCognitivePlanner:
    """Large Language Model based cognitive planner for robots"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.environment_state = EnvironmentState()
        self.robot_capabilities = RobotCapabilities()

        # Initialize OpenAI client if API key provided
        if api_key:
            openai.api_key = api_key
            self.use_api = True
        else:
            # For demonstration, we'll use mock responses
            self.use_api = False
            print("Using mock LLM responses for demonstration")

    def generate_plan_prompt(self, goal: str) -> str:
        """Generate a prompt for the LLM to create a plan"""
        prompt = f"""
You are an expert robot cognitive planner. Your task is to create a detailed step-by-step plan for a robot to achieve the given goal.

Robot Capabilities:
- Navigation: The robot can move between locations
- Manipulation: The robot can grasp and move objects
- Sensing: The robot can perceive its environment
- Max reach: {self.robot_capabilities.max_reach} meters
- Max payload: {self.robot_capabilities.max_payload} kg

Current Environment State:
- Robot Position: {self.environment_state.robot_position}
- Robot Gripper State: {self.environment_state.robot_gripper_state}
- Available Objects: {list(self.environment_state.objects.keys())}
- Available Rooms: {list(self.environment_state.rooms.keys())}

Goal: {goal}

Please create a detailed plan with specific actions that the robot can execute. Each action should be one of the following types:
- NAVIGATION: Move to a specific location
- MANIPULATION: Grasp, release, or move an object
- INTERACTION: Communicate or interact with the environment
- SENSING: Perceive or scan the environment
- WAIT: Wait for a condition

Return the plan as a JSON object with the following structure:
{{
  "plan": [
    {{
      "step_id": 1,
      "action_type": "NAVIGATION|MANIPULATION|INTERACTION|SENSING|WAIT",
      "parameters": {{"target_location": "...", "object": "...", "other_params": "..."}},
      "description": "Human-readable description of the action",
      "preconditions": ["list of conditions that must be true before this action"],
      "effects": ["list of effects this action has on the environment"],
      "dependencies": [list of step IDs this step depends on]
    }}
  ]
}}

Be specific about locations, objects, and parameters. Consider the robot's capabilities and the current environment state.
"""
        return prompt

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM to generate a plan"""
        if self.use_api:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1  # Low temperature for consistency
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error calling LLM API: {e}")
                return self.mock_plan_generation(prompt)
        else:
            return self.mock_plan_generation(prompt)

    def mock_plan_generation(self, prompt: str) -> Dict[str, Any]:
        """Mock plan generation for demonstration"""
        print(f"Generating mock plan for goal in prompt...")

        # Extract goal from prompt (simplified)
        if "clean the kitchen" in prompt.lower():
            return {
                "plan": [
                    {
                        "step_id": 1,
                        "action_type": "NAVIGATION",
                        "parameters": {"target_location": "kitchen"},
                        "description": "Move to the kitchen",
                        "preconditions": [],
                        "effects": ["robot_in_kitchen"],
                        "dependencies": []
                    },
                    {
                        "step_id": 2,
                        "action_type": "SENSING",
                        "parameters": {"scan_area": "kitchen"},
                        "description": "Scan the kitchen to identify dirty items",
                        "preconditions": ["robot_in_kitchen"],
                        "effects": ["dirty_items_identified"],
                        "dependencies": [1]
                    },
                    {
                        "step_id": 3,
                        "action_type": "MANIPULATION",
                        "parameters": {"object": "dirty_plate", "action": "pick_up"},
                        "description": "Pick up the dirty plate",
                        "preconditions": ["dirty_items_identified", "dirty_plate_available"],
                        "effects": ["dirty_plate_grasped"],
                        "dependencies": [2]
                    },
                    {
                        "step_id": 4,
                        "action_type": "NAVIGATION",
                        "parameters": {"target_location": "sink"},
                        "description": "Move to the sink",
                        "preconditions": ["dirty_plate_grasped"],
                        "effects": ["robot_at_sink"],
                        "dependencies": [3]
                    },
                    {
                        "step_id": 5,
                        "action_type": "MANIPULATION",
                        "parameters": {"object": "dirty_plate", "action": "place_down"},
                        "description": "Place the dirty plate in the sink",
                        "preconditions": ["robot_at_sink", "dirty_plate_grasped"],
                        "effects": ["dirty_plate_in_sink"],
                        "dependencies": [4]
                    }
                ]
            }
        elif "set the table" in prompt.lower():
            return {
                "plan": [
                    {
                        "step_id": 1,
                        "action_type": "NAVIGATION",
                        "parameters": {"target_location": "kitchen"},
                        "description": "Move to the kitchen",
                        "preconditions": [],
                        "effects": ["robot_in_kitchen"],
                        "dependencies": []
                    },
                    {
                        "step_id": 2,
                        "action_type": "MANIPULATION",
                        "parameters": {"object": "plate", "action": "pick_up"},
                        "description": "Pick up a plate from the cabinet",
                        "preconditions": ["robot_in_kitchen", "plate_available"],
                        "effects": ["plate_grasped"],
                        "dependencies": [1]
                    },
                    {
                        "step_id": 3,
                        "action_type": "NAVIGATION",
                        "parameters": {"target_location": "dining_room"},
                        "description": "Move to the dining room",
                        "preconditions": ["plate_grasped"],
                        "effects": ["robot_in_dining_room"],
                        "dependencies": [2]
                    },
                    {
                        "step_id": 4,
                        "action_type": "MANIPULATION",
                        "parameters": {"object": "plate", "action": "place_down"},
                        "description": "Place the plate on the table",
                        "preconditions": ["robot_in_dining_room", "plate_grasped"],
                        "effects": ["plate_on_table"],
                        "dependencies": [3]
                    }
                ]
            }
        else:
            # Default plan for unknown goals
            return {
                "plan": [
                    {
                        "step_id": 1,
                        "action_type": "NAVIGATION",
                        "parameters": {"target_location": "default_location"},
                        "description": "Move to a default location",
                        "preconditions": [],
                        "effects": ["robot_moved"],
                        "dependencies": []
                    }
                ]
            }

    def create_plan(self, goal: str) -> List[PlanStep]:
        """Create a plan for the given goal using LLM"""
        # Generate prompt
        prompt = self.generate_plan_prompt(goal)

        # Get plan from LLM
        response = self.call_llm(prompt)

        # Parse the plan
        plan_steps = []
        if "plan" in response:
            for step_data in response["plan"]:
                action = RobotAction(
                    action_type=RobotActionType(step_data["action_type"]),
                    parameters=step_data["parameters"],
                    description=step_data["description"]
                )

                plan_step = PlanStep(
                    step_id=step_data["step_id"],
                    action=action,
                    preconditions=step_data["preconditions"],
                    effects=step_data["effects"],
                    dependencies=step_data["dependencies"]
                )

                plan_steps.append(plan_step)

        # Sort plan steps by dependencies
        plan_steps = self.sort_plan_by_dependencies(plan_steps)

        return plan_steps

    def sort_plan_by_dependencies(self, plan_steps: List[PlanStep]) -> List[PlanStep]:
        """Sort plan steps based on their dependencies"""
        # Create a dependency graph
        step_map = {step.step_id: step for step in plan_steps}
        sorted_steps = []
        visited = set()

        def visit(step_id):
            if step_id in visited:
                return

            step = step_map[step_id]
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    visit(dep_id)

            visited.add(step_id)
            sorted_steps.append(step)

        for step in plan_steps:
            visit(step.step_id)

        return sorted_steps

    def validate_plan(self, plan: List[PlanStep]) -> Tuple[bool, List[str]]:
        """Validate the plan for feasibility and safety"""
        issues = []

        for step in plan:
            # Check if robot has capability for action
            if step.action.action_type == RobotActionType.MANIPULATION:
                if not self.robot_capabilities.manipulation:
                    issues.append(f"Step {step.step_id}: Robot cannot perform manipulation")

            elif step.action.action_type == RobotActionType.NAVIGATION:
                if not self.robot_capabilities.navigation:
                    issues.append(f"Step {step.step_id}: Robot cannot navigate")

            # Check action-specific parameters
            if step.action.action_type == RobotActionType.MANIPULATION:
                obj_name = step.action.parameters.get("object")
                if obj_name:
                    obj_location = self.environment_state.get_object_location(obj_name)
                    if obj_location is None:
                        issues.append(f"Step {step.step_id}: Object '{obj_name}' location unknown")

        is_valid = len(issues) == 0
        return is_valid, issues

    def execute_plan(self, plan: List[PlanStep]) -> bool:
        """Execute the plan step by step"""
        print(f"Starting execution of plan with {len(plan)} steps...")

        for step in plan:
            print(f"Executing step {step.step_id}: {step.action.description}")

            success = self.execute_single_step(step)

            if not success:
                print(f"Step {step.step_id} failed. Stopping plan execution.")
                return False

            print(f"Step {step.step_id} completed successfully.")

        print("Plan execution completed successfully!")
        return True

    def execute_single_step(self, step: PlanStep) -> bool:
        """Execute a single plan step"""
        try:
            if step.action.action_type == RobotActionType.NAVIGATION:
                return self.execute_navigation(step.action.parameters)
            elif step.action.action_type == RobotActionType.MANIPULATION:
                return self.execute_manipulation(step.action.parameters)
            elif step.action.action_type == RobotActionType.INTERACTION:
                return self.execute_interaction(step.action.parameters)
            elif step.action.action_type == RobotActionType.SENSING:
                return self.execute_sensing(step.action.parameters)
            elif step.action.action_type == RobotActionType.WAIT:
                return self.execute_wait(step.action.parameters)
            else:
                print(f"Unknown action type: {step.action.action_type}")
                return False

        except Exception as e:
            print(f"Error executing step {step.step_id}: {e}")
            return False

    def execute_navigation(self, params: Dict) -> bool:
        """Execute navigation action"""
        target_location = params.get("target_location", "unknown")
        print(f"Navigating to {target_location}")

        # Simulate navigation
        time.sleep(1)  # Simulate time for navigation

        # Update robot position (simplified)
        if target_location == "kitchen":
            self.environment_state.robot_position = (3.0, 2.0, 0.0)
        elif target_location == "dining_room":
            self.environment_state.robot_position = (1.0, -1.0, 0.0)
        elif target_location == "living_room":
            self.environment_state.robot_position = (0.0, 0.0, 0.0)
        else:
            # Default movement
            self.environment_state.robot_position = (
                self.environment_state.robot_position[0] + 1.0,
                self.environment_state.robot_position[1],
                self.environment_state.robot_position[2]
            )

        print(f"Robot now at position: {self.environment_state.robot_position}")
        return True

    def execute_manipulation(self, params: Dict) -> bool:
        """Execute manipulation action"""
        obj_name = params.get("object", "unknown")
        action = params.get("action", "unknown")

        print(f"Manipulating {obj_name}, action: {action}")

        if action == "pick_up":
            print(f"Picking up {obj_name}")
            self.environment_state.robot_gripper_state = "closed"
        elif action == "place_down":
            print(f"Placing down {obj_name}")
            self.environment_state.robot_gripper_state = "open"
        elif action == "grasp":
            print(f"Grasping {obj_name}")
            self.environment_state.robot_gripper_state = "closed"
        elif action == "release":
            print(f"Releasing {obj_name}")
            self.environment_state.robot_gripper_state = "open"
        else:
            print(f"Unknown manipulation action: {action}")
            return False

        print(f"Gripper state: {self.environment_state.robot_gripper_state}")
        return True

    def execute_interaction(self, params: Dict) -> bool:
        """Execute interaction action"""
        interaction_type = params.get("type", "unknown")
        message = params.get("message", "Hello")

        print(f"Interaction: {interaction_type}, message: {message}")
        return True

    def execute_sensing(self, params: Dict) -> bool:
        """Execute sensing action"""
        scan_area = params.get("scan_area", "unknown")
        print(f"Sensing/Scanning area: {scan_area}")

        # Simulate sensing
        time.sleep(0.5)  # Simulate sensing time

        # Update environment state based on sensing
        if scan_area == "kitchen":
            # Add some objects to the environment
            self.environment_state.objects["dirty_plate"] = {
                "location": (3.1, 2.1, 0.0),
                "room": "kitchen"
            }
            print("Identified dirty_plate in kitchen")

        return True

    def execute_wait(self, params: Dict) -> bool:
        """Execute wait action"""
        duration = params.get("duration", 1.0)
        condition = params.get("condition", "time")

        print(f"Waiting for {duration} seconds or until {condition}")
        time.sleep(duration)
        return True

class LLMRobotInterface:
    """Interface between LLM planner and physical robot"""

    def __init__(self):
        self.planner = LLMCognitivePlanner()
        self.current_plan = None
        self.plan_execution_active = False

    def process_goal(self, goal: str) -> bool:
        """Process a high-level goal and execute the plan"""
        print(f"Processing goal: {goal}")

        # Create plan using LLM
        plan = self.planner.create_plan(goal)
        print(f"Generated plan with {len(plan)} steps")

        # Validate plan
        is_valid, issues = self.planner.validate_plan(plan)
        if not is_valid:
            print(f"Plan validation failed with issues: {issues}")
            return False

        # Execute plan
        success = self.planner.execute_plan(plan)
        return success

    def update_environment_state(self, state_updates: Dict):
        """Update the environment state with new information"""
        for key, value in state_updates.items():
            if hasattr(self.planner.environment_state, key):
                setattr(self.planner.environment_state, key, value)

    def get_robot_status(self) -> Dict:
        """Get current robot status"""
        return {
            "position": self.planner.environment_state.robot_position,
            "gripper_state": self.planner.environment_state.robot_gripper_state,
            "objects": list(self.planner.environment_state.objects.keys()),
            "plan_execution_active": self.plan_execution_active
        }

# Example usage of the LLM cognitive planner
def main():
    print("Initializing LLM Cognitive Planning System...")

    # Create robot interface
    robot_interface = LLMRobotInterface()

    # Example goals to process
    goals = [
        "Clean the kitchen",
        "Set the table for dinner",
        "Go to the living room and bring me the red book"
    ]

    for i, goal in enumerate(goals):
        print(f"\n{'='*50}")
        print(f"Processing Goal {i+1}: {goal}")
        print(f"{'='*50}")

        # Process the goal
        success = robot_interface.process_goal(goal)

        if success:
            print(f"✓ Goal '{goal}' completed successfully")
        else:
            print(f"✗ Goal '{goal}' failed")

        # Get status
        status = robot_interface.get_robot_status()
        print(f"Robot status: {status}")

        print(f"\nWaiting before next goal...")
        time.sleep(2)

    print(f"\n{'='*50}")
    print("LLM Cognitive Planning Demo Completed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
```

```python
# Advanced LLM Planning with Context and Learning
import openai
import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import time
import datetime
from collections import defaultdict

@dataclass
class PlanExecutionResult:
    """Result of plan execution"""
    success: bool
    steps_completed: int
    steps_failed: int
    execution_time: float
    issues: List[str]
    feedback: str

class LearningLLMPlanner:
    """LLM planner with learning capabilities"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.planner = LLMCognitivePlanner(api_key, model)
        self.execution_history = []
        self.failure_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        self.user_preferences = {}

    def learn_from_execution(self, goal: str, plan: List[PlanStep], result: PlanExecutionResult):
        """Learn from plan execution results"""
        if result.success:
            # Store successful patterns
            pattern_key = self.extract_pattern(goal, plan)
            self.success_patterns[pattern_key].append({
                "plan": [step.action.action_type.value for step in plan],
                "environment": str(self.planner.environment_state.__dict__)
            })
        else:
            # Store failure patterns
            pattern_key = self.extract_pattern(goal, plan)
            self.failure_patterns[pattern_key].append({
                "plan": [step.action.action_type.value for step in plan],
                "issues": result.issues,
                "environment": str(self.planner.environment_state.__dict__)
            })

        # Add to execution history
        self.execution_history.append({
            "goal": goal,
            "plan": plan,
            "result": result,
            "timestamp": datetime.datetime.now()
        })

        # Keep only recent history
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    def extract_pattern(self, goal: str, plan: List[PlanStep]) -> str:
        """Extract a pattern from goal and plan for learning"""
        # Simple pattern extraction - in practice, this would be more sophisticated
        goal_keywords = " ".join(goal.lower().split()[:3])  # First 3 words
        action_sequence = "-".join([step.action.action_type.value for step in plan[:3]])  # First 3 actions
        return f"{goal_keywords}_{action_sequence}"

    def adapt_plan_based_on_history(self, goal: str, plan: List[PlanStep]) -> List[PlanStep]:
        """Adapt plan based on execution history"""
        pattern_key = self.extract_pattern(goal, plan)

        # Check if we have failure patterns for similar goals
        if pattern_key in self.failure_patterns:
            print(f"Detected similar failure pattern for: {pattern_key}")
            # In a real implementation, we would modify the plan based on past failures
            # For this example, we'll just add a safety check

        # Check if we have success patterns for similar goals
        if pattern_key in self.success_patterns:
            print(f"Found successful pattern for: {pattern_key}")
            # In a real implementation, we might prioritize similar successful approaches

        return plan  # Return original plan for this example

    def create_contextual_plan(self, goal: str, context: Dict = None) -> List[PlanStep]:
        """Create a plan with contextual information"""
        if context is None:
            context = {}

        # Enhance environment state with context
        enhanced_context = {
            "current_time": datetime.datetime.now().strftime("%H:%M"),
            "day_of_week": datetime.datetime.now().strftime("%A"),
            "user_preferences": self.user_preferences,
            "recent_activities": [h["goal"] for h in self.execution_history[-5:]]
        }

        # Update planner environment with context
        for key, value in context.items():
            if hasattr(self.planner.environment_state, key):
                setattr(self.planner.environment_state, key, value)

        # Create plan
        plan = self.planner.create_plan(goal)

        # Adapt plan based on history
        adapted_plan = self.adapt_plan_based_on_history(goal, plan)

        return adapted_plan

    def execute_plan_with_monitoring(self, goal: str, plan: List[PlanStep]) -> PlanExecutionResult:
        """Execute plan with monitoring and feedback"""
        start_time = time.time()
        steps_completed = 0
        steps_failed = 0
        issues = []

        print(f"Starting monitored execution of plan for goal: {goal}")

        for step in plan:
            print(f"Executing step {step.step_id}: {step.action.description}")

            try:
                success = self.planner.execute_single_step(step)

                if success:
                    steps_completed += 1
                    print(f"✓ Step {step.step_id} completed")
                else:
                    steps_failed += 1
                    issue = f"Step {step.step_id} failed: {step.action.description}"
                    issues.append(issue)
                    print(f"✗ {issue}")

                    # In a real implementation, we might try alternative approaches
                    # or request human intervention

            except Exception as e:
                steps_failed += 1
                issue = f"Step {step.step_id} error: {str(e)}"
                issues.append(issue)
                print(f"✗ {issue}")

        execution_time = time.time() - start_time

        result = PlanExecutionResult(
            success=(steps_failed == 0),
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            execution_time=execution_time,
            issues=issues,
            feedback=f"Completed {steps_completed} steps, failed {steps_failed} steps in {execution_time:.2f}s"
        )

        # Learn from execution
        self.learn_from_execution(goal, plan, result)

        return result

class MultiModalLLMPlanner:
    """LLM planner that can incorporate visual and sensor information"""

    def __init__(self, api_key: str = None):
        self.learning_planner = LearningLLMPlanner(api_key)
        self.vision_system = None  # Would be connected to camera/sensors
        self.speech_system = None  # Would be connected to voice interface

    def integrate_sensor_data(self, sensor_data: Dict) -> Dict:
        """Integrate sensor data into environment state"""
        # Process sensor data and update environment state
        processed_data = {}

        # Example: Process camera data
        if 'camera' in sensor_data:
            # In a real implementation, this would use computer vision
            # to identify objects and update their locations
            processed_data['objects'] = sensor_data['camera'].get('detected_objects', [])

        # Example: Process LiDAR data
        if 'lidar' in sensor_data:
            # Process LiDAR to identify obstacles and free space
            processed_data['obstacles'] = sensor_data['lidar'].get('obstacles', [])
            processed_data['free_space'] = sensor_data['lidar'].get('free_space', [])

        # Example: Process IMU data
        if 'imu' in sensor_data:
            # Update robot orientation
            processed_data['orientation'] = sensor_data['imu'].get('orientation', (0,0,0,1))

        return processed_data

    def create_plan_with_sensor_input(self, goal: str, sensor_data: Dict = None) -> List[PlanStep]:
        """Create plan incorporating sensor data"""
        if sensor_data:
            # Integrate sensor data into environment state
            processed_data = self.integrate_sensor_data(sensor_data)

            # Update planner environment
            self.learning_planner.planner.environment_state.objects.update(
                {obj['name']: obj for obj in processed_data.get('objects', [])}
            )

        # Create contextual plan
        plan = self.learning_planner.create_contextual_plan(goal)

        return plan

    def execute_with_sensor_feedback(self, goal: str, sensor_data: Dict = None) -> PlanExecutionResult:
        """Execute plan with continuous sensor feedback"""
        # Create plan with initial sensor data
        plan = self.create_plan_with_sensor_input(goal, sensor_data)

        # Validate plan
        is_valid, issues = self.learning_planner.planner.validate_plan(plan)
        if not is_valid:
            return PlanExecutionResult(
                success=False,
                steps_completed=0,
                steps_failed=0,
                execution_time=0.0,
                issues=issues,
                feedback="Plan validation failed"
            )

        # Execute plan with monitoring
        result = self.learning_planner.execute_plan_with_monitoring(goal, plan)

        return result

# Example of using advanced LLM planning
def run_advanced_planning_example():
    print("Running Advanced LLM Planning Example...")

    # Create multi-modal planner
    multi_planner = MultiModalLLMPlanner()

    # Simulate sensor data
    sensor_data = {
        'camera': {
            'detected_objects': [
                {'name': 'red_book', 'location': (1.0, 0.5, 0.0), 'room': 'living_room'},
                {'name': 'blue_cup', 'location': (2.0, 1.0, 0.0), 'room': 'kitchen'}
            ]
        },
        'lidar': {
            'obstacles': [],
            'free_space': [(0, 0, 1), (1, 0, 1), (2, 0, 1)]
        }
    }

    # Example goals
    goals = [
        "Go to the living room and bring me the red book",
        "Clean the kitchen counter",
        "Set the table in the dining room"
    ]

    for i, goal in enumerate(goals):
        print(f"\n{'='*60}")
        print(f"Advanced Planning Example {i+1}: {goal}")
        print(f"{'='*60}")

        # Execute with sensor feedback
        result = multi_planner.execute_with_sensor_feedback(goal, sensor_data)

        print(f"Execution result: {result.success}")
        print(f"Feedback: {result.feedback}")

        if result.issues:
            print(f"Issues: {result.issues}")

        # Wait before next example
        time.sleep(1)

    # Show learning results
    print(f"\nLearning Summary:")
    print(f"Success patterns learned: {len(multi_planner.learning_planner.success_patterns)}")
    print(f"Failure patterns learned: {len(multi_planner.learning_planner.failure_patterns)}")
    print(f"Total executions: {len(multi_planner.learning_planner.execution_history)}")

def main():
    print("Starting LLM Cognitive Planning for Robots")

    # Run the advanced example
    run_advanced_planning_example()

    print("\nLLM cognitive planning enables robots to reason about complex tasks")
    print("and generate detailed action plans based on natural language instructions.")

if __name__ == "__main__":
    main()
```

## Summary

Large Language Models (LLMs) represent a transformative approach to cognitive planning in robotics, enabling robots to understand high-level goals in natural language and generate detailed, executable action sequences. These models provide sophisticated reasoning capabilities that can decompose complex tasks into manageable subtasks while considering environmental constraints and robot capabilities.

The key components of LLM-based cognitive planning include:
- **Natural Language Understanding**: Interpreting high-level goals from human instructions
- **Task Decomposition**: Breaking complex goals into specific, executable actions
- **Context Awareness**: Considering environmental state and robot capabilities
- **Plan Validation**: Ensuring plans are feasible and safe
- **Learning Capabilities**: Improving planning through experience and feedback

LLM planning is particularly valuable for Physical AI and humanoid robotics because it enables more natural human-robot interaction and allows robots to handle complex, multi-step tasks without extensive manual programming. The integration of sensor data and learning capabilities makes these systems adaptable to dynamic environments.

The combination of LLMs with traditional robotics systems creates cognitive robots that can reason about their environment, plan complex behaviors, and adapt to changing conditions in real-time.

## Exercises

1. **Basic Understanding**: Explain the difference between classical symbolic planning and LLM-based planning in robotics. What are the advantages and limitations of each approach?

2. **Application Exercise**: Design an LLM-based planning system for a household robot that can handle complex instructions like "Prepare dinner for two people." Include the specific components needed and how they would work together.

3. **Implementation Exercise**: Modify the LLM planning system to handle plan failures by generating alternative approaches. When a step fails, the system should reason about why it failed and create a new plan.

4. **Challenge Exercise**: Implement a multi-modal LLM planning system that incorporates visual perception, language understanding, and physical action planning. The system should be able to update its plans based on real-time sensor feedback and environmental changes.