"""
Classical planner implementation 

NOTE: core loop refactored from auto-d/classical_planning/planner.py
"""
from __future__ import annotations

import re
import sys 
import time 
import heapq
import argparse 
from collections import deque
from datetime import datetime
from typing import Any, Dict

from planner import Decision, Event

# Domain definition : Below schemas code for states that must be matched prior to fluent modification
# TODO: retrofit for lawn-watering planner!
actions = [
    {
        'name': 'Prep LaunchVehicle(Spacecraft)',
        'preconditions': {"On(Earth,LaunchVehicle)", "On(Earth,Spacecraft)", "Empty(LaunchVehicle)"},
        'add_list': {"On(LaunchVehicle,Spacecraft)"},
        'delete_list': {"On(Earth,Spacecraft)", "Empty(LaunchVehicle)"}
    },
    {
        'name': 'Prep LaunchVehicle(Lander)',
        'preconditions': {"On(Earth,LaunchVehicle)", "On(Earth,Lander)","Empty(LaunchVehicle)"},
        'add_list': {"On(LaunchVehicle,Lander)"},
        'delete_list': {"On(Earth,Lander)","Empty(LaunchVehicle)"}
    },
        {
        'name': 'Prep LaunchVehicle(FuelCell)',
        'preconditions': {"On(Earth,LaunchVehicle)", "On(Earth,FuelCell)", "Empty(LaunchVehicle)"},
        'add_list': {"On(LaunchVehicle,FuelCell)"},
        'delete_list': {"On(Earth,FuelCell)", "Empty(LaunchVehicle)"}
    },
        {
        'name': 'Embark SpaceCraft(Crew)',
        'preconditions': {"On(LaunchVehicle,Spacecraft)", "On(Earth,Crew)"},
        'add_list': {"On(Spacecraft,Crew)"},
        'delete_list': {"On(Earth,Crew)"}
    },
    {
        'name': 'Orbital Launch',
        'preconditions': {"On(Earth,LaunchVehicle)"},
        'add_list': {"Orbit(Earth,LaunchVehicle)"},
        'delete_list': {"On(Earth,LaunchVehicle)"}
    },
    {
        'name': 'Recover LaunchVehcle',
        'preconditions': {"Orbit(Earth,LaunchVehicle)"},
        'add_list': {"On(Earth,LaunchVehicle)"},
        'delete_list': {"Orbit(Earth,LaunchVehicle)"}
    },
    {
        'name': 'Dock Spacecraft',
        'preconditions': {"Orbit(Earth,LaunchVehicle)", "On(LaunchVehicle,Spacecraft)"},
        'add_list': {"Orbit(Earth,Spacecraft)", "Empty(LaunchVehicle)"},
        'delete_list': {"On(LaunchVehicle,Spacecraft)"}
    },
    {
        'name': 'Transload Lander',
        'preconditions': {"Orbit(Earth,LaunchVehicle)", "On(LaunchVehicle,Lander)"},
        'add_list': {"Orbit(Earth,Lander)", "Empty(LaunchVehicle)"},
        'delete_list': {"On(LaunchVehicle,Lander)"}
    },
    {
        'name': 'Transload Fuel Cell',
        'preconditions': {"Orbit(Earth,LaunchVehicle)", "On(LaunchVehicle,FuelCell)"},
        'add_list': {"Orbit(Earth,FuelCell)", "Empty(LaunchVehicle)"},
        'delete_list': {"On(LaunchVehicle,FuelCell)"}
    },
    {
        'name': 'Prep Spacecraft (Lander)',
        'preconditions': {"Orbit(Earth,Spacecraft)", "Orbit(Earth,Lander)"},
        'add_list': {"On(Spacecraft,Lander)"},
        'delete_list': {"Orbit(Earth,Lander)"}
    },
    {
        'name': 'Prep Spacecraft (Fuel)',
        'preconditions': {"Orbit(Earth,Spacecraft)", "Orbit(Earth,FuelCell)"},
        'add_list': {"On(Spacecraft,FuelCell)"},
        'delete_list': {"Orbit(Earth,FuelCell)"}
    },
    {
        'name': 'Translunar boost',
        'preconditions': {"Orbit(Earth,Spacecraft)", "On(Spacecraft,FuelCell)", "On(Spacecraft,Lander)", "On(Spacecraft,Crew)"},
        'add_list': {"Orbit(Moon,Spacecraft)"},
        'delete_list': {"Orbit(Earth,Spacecraft)"}
    },
    {
        'name': 'Lunar Landing',
        'preconditions': {"Orbit(Moon,Spacecraft)", "On(Spacecraft,Lander)", "On(Spacecraft,Crew)"},
        'add_list': {"On(Lander,Crew)", "On(Moon,Lander)"},
        'delete_list': {"On(Spacecraft,Lander)", "On(Spacecraft,Crew)"}
    },
]

# Planner!

def is_applicable(state, action):
    """
    Check if an action can be executed in the current state.
    
    Args:
        state: A set of strings representing true facts (fluents)
               e.g., {"Clear(A)", "OnTable(A)", "ArmEmpty"}
        action: A dictionary with keys:
               - 'name': Action name (e.g., "Stack(A,B)")
               - 'preconditions': Set of facts that must be true
               - 'add_list': Set of facts to add
               - 'delete_list': Set of facts to remove
    
    Returns:
        bool: True if ALL preconditions are in the state, False otherwise
    """
    return action['preconditions'].issubset(state)

def apply_action(state, action):
    """
    Apply an action to transform the state.
    
    Args:
        state: Current state (set of fluent strings)
        action: Action dictionary (same format as above)
    
    Returns:
        new_state: A NEW set representing the state after the action
                   (don't modify the original state!)
    
    The STRIPS assumption:
        new_state = (state - delete_list) ∪ add_list
    """
    new_state = state.copy()

    if is_applicable(state, action): 
        new_state = new_state.difference(action['delete_list'])
        new_state = new_state.union(action['add_list'])
        return new_state

def goal_satisfied(state, goal):
    """
    Check if the goal is satisfied in the current state.
    
    Args:
        state: Current state (set of fluents)
        goal: Set of fluents that must ALL be true
    
    Returns:
        bool: True if all goal fluents are in state
    """
    return goal.issubset(state)

def get_applicable_actions(state, actions):
    """
    Get all actions that can be applied in the current state.
    
    Args:
        state: Current state (set of fluents)
        actions: List of all action dictionaries
    
    Returns:
        list: Actions whose preconditions are satisfied
    """
    return [a for a in actions if is_applicable(state, a)]

def forward_search(initial_state, goal, actions, log=False):
    """
    Find a plan using BFS.
    
    Args:
        initial_state: Starting state (set of fluents)
        goal: Goal condition (set of fluents that must be true)
        actions: List of all possible action dictionaries
    
    Returns:
        tuple: (plan, explored_count)
               - plan: List of action names, or None if no plan exists
               - explored_count: Number of states explored
    """
    explored = 0
    
    # queue holds tuples of (state, list_of_action_names_so_far)
    queue = [ (initial_state, []) ]
    
    # Track visited states (using frozenset since regular sets aren't hashable)
    visited = {frozenset(initial_state)}
    
    # While queue is not empty:
    #   1. Dequeue the first item: state, plan = queue.pop(0)
    #   2. Increment explored counter
    #   3. Check if goal is satisfied - if so, return (plan, explored)
    #   4. Get all applicable actions using get_applicable_actions()
    #   5. For each applicable action:
    #      - Apply it to get the new state using apply_action()
    #      - If frozenset(new_state) not in visited:
    #        - Add frozenset(new_state) to visited
    #        - Enqueue (new_state, plan + [action['name']])
    
    while queue != []:         
        state, plan = queue.pop(0) 
        explored += 1
        
        if goal_satisfied(state, goal): 
            return plan, explored 

        if log: 
            print(f"@ {state}")

        applicable_actions = get_applicable_actions(state, actions)
        for action in applicable_actions: 
            
            if log: 
                print(f"--> {action['name']}\n   == {action['preconditions']}\n    + {action['add_list']}\n    - {action['delete_list']})")

            reachable_state = apply_action(state, action)
            
            if not frozenset(reachable_state) in visited: 
                visited.add(frozenset(reachable_state))
                queue.append((reachable_state, plan + [action['name']]))

    return None, explored  # No plan found

def goal_count_heuristic(state, goal):
    """
    Estimate distance to goal by counting unsatisfied goal facts.
    
    Args:
        state: Current state (set of fluents)
        goal: Goal condition (set of fluents)
    
    Returns:
        int: Number of goal facts NOT in the current state
    """
    return len(goal - state)

def heuristic_search(initial_state, goal, hint, actions, log=False):
    """
    Find a plan using A*-like search with goal-count heuristic.
    
    Args:
        initial_state: Starting state (set of fluents)
        goal: Goal condition (set of fluents)
        hint: State we should preferentially drift toward as part of our heuristic
        actions: List of all possible action dictionaries
    
    Returns:
        tuple: (plan, explored_count)
    
    Note: We use a counter as tie-breaker since sets aren't comparable.
    """
    explored = 0
    counter = 0  # Tie-breaker for priority queue
    
    h_initial = goal_count_heuristic(initial_state, hint)
    
    # Priority queue: (f_score, counter, g_score, state, plan)
    # counter is used to break ties (avoids comparing states directly)
    # NOTE: for the ignorant (me) heapq does a L-to-R comparison on the tuples here to determine ordering, 
    # this will result in the top-most item (returned by heappop(), e.g) being the maximum value of 
    # our heuristic, w/ aforementioned counter swinging into action to drive the lowest value when 
    # f-value is identical
    pq = []

    # NOTE: F = heuristic + path_cost, but the latter is 0 to start so we simply init f == h_initial 
    heapq.heappush(pq, (h_initial, counter, 0, initial_state, []))  
    counter += 1
    
    # Visited set for states we've fully processed
    visited = set()

    # While priority queue not empty:
    #   1. Pop (f, _, g, state, plan) = heapq.heappop(pq)
    #   2. Convert state to frozenset for hashing
    #   3. Skip if already in visited (continue to next iteration)
    #   4. Add to visited, increment explored
    #   5. If goal_satisfied(state, goal), return (plan, explored)
    #   6. For each action in get_applicable_actions(state, actions):
    #      - new_state = apply_action(state, action)
    #      - If frozenset(new_state) not in visited:
    #        - h = goal_count_heuristic(new_state, goal)
    #        - new_f = (g + 1) + h
    #        - heapq.heappush(pq, (new_f, counter, g+1, new_state, plan + [action['name']]))
    #        - counter += 1
    while len(pq) > 0: 
        
        # The total cost for the states in the queue is used by the heap to sort internally, the 
        # memorialized f-value has no further utility. Same for the counter value. 
        (_, _, g, state, plan) = heapq.heappop(pq)
        
        state = frozenset(state)
        if state in visited: 
            continue 

        visited.add(state)
        explored += 1

        if goal_satisfied(state, goal): 
            return (plan, explored)
        
        if log: 
            print(f"@ {list(state)}")

        applicable_actions = get_applicable_actions(state, actions)
        for action in applicable_actions: 
            if log: 
                print(f"--> {action['name']}\n   == {action['preconditions']}\n    + {action['add_list']}\n    - {action['delete_list']})")

            reachable_state = apply_action(state, action) 
            if not frozenset(reachable_state) in visited: 
                h = goal_count_heuristic(reachable_state, goal)
                f = (g + 1) + h 
                heapq.heappush(pq, (f, counter, g+1, reachable_state, plan + [action['name']]))
                counter +=1 
    
    return None, explored  # No plan found


class ClassicalPlanner:
    """Minimal rule-based planner for MVP CLI execution."""

    name = "classical"

    def __init__(self) -> None:
        self._events: deque[Event] = deque(maxlen=256)
        self._latest: Dict[tuple[str, str], Event] = {}

    def observe(self, event: Event) -> None:
        self._events.append(event)
        self._latest[(event.source, event.type)] = event

    def decide(self, now: datetime) -> Decision:
        irrigation = self._payload("irrigation", "irrigation_status")
        precipitation = self._payload("precipitation", "precipitation_summary")
        forecast = self._payload("weather", "forecast_summary")
        camera = self._payload("camera", "scene_activity")

        if irrigation.get("api_reports_on") or irrigation.get("expected_on"):
            return Decision(
                timestamp=now,
                planner=self.name,
                action="no_op",
                rationale="Irrigation appears to be running or expected to be running already.",
                metadata={"source": "irrigation"},
            )

        if precipitation.get("recent_rain") or precipitation.get("total_inches", 0.0) >= 0.1:
            return Decision(
                timestamp=now,
                planner=self.name,
                action="no_op",
                rationale="Recent precipitation exceeds the watering threshold.",
                metadata={"source": "precipitation"},
            )

        if (forecast.get("max_precip_probability") or 0) >= 50 or forecast.get("rain_expected_soon"):
            return Decision(
                timestamp=now,
                planner=self.name,
                action="no_op",
                rationale="Forecast indicates rain is likely soon.",
                metadata={"source": "weather"},
            )

        if camera.get("person_detected") or camera.get("animal_detected") or camera.get("lawn_mower_active"):
            return Decision(
                timestamp=now,
                planner=self.name,
                action="water_off",
                rationale="Scene appears occupied or unsafe for watering.",
                metadata={"source": "camera"},
            )

        return Decision(
            timestamp=now,
            planner=self.name,
            action="water_on",
            duration_seconds=300,
            rationale="No rain signal or occupancy conflict detected; emit a short watering action.",
            metadata={"source": "rule_heuristic"},
        )

    def _payload(self, source: str, event_type: str) -> Dict[str, Any]:
        event = self._latest.get((source, event_type))
        return event.payload if event is not None else {}

# Main 
def main(): 
    """
    CLI tool implementation which accepts a few flags and runs the lunar mission planner
    """

    # TODO: retrofit for lawn-watering planner!

    state = {"On(Earth,LaunchVehicle)", "Empty(LaunchVehicle)", "On(Earth,Spacecraft)", "On(Earth,Lander)", "On(Earth,Crew)", "Orbit(Earth,FuelCell)"}
    goal = {"On(Moon,Lander)", "On(Lander,Crew)"}

    parser = argparse.ArgumentParser(description="Toy lunar mission planner")
    parser.add_argument("--heuristic", action="store_true", help="Enable heuristic support for planning")
    parser.add_argument("--animate", action="store_true", help="Animate the planner states")
    parser.add_argument("--log", action="store_true", help="Log the planner states")

    args = parser.parse_args() 

    if not args.heuristic: 
        print("Initiating lunar mission planning with breadth-first search...")
        plan, explored = forward_search(state, goal, actions, log=args.log)
    else: 
        print("Iniating lunar mission planning with heuristic-supported A*-like search...")
        hint = {"Orbit(Earth,Spacecraft)", "Orbit(Earth,Crew)", "Orbit(Earth,Lander)"}
        plan, explored = heuristic_search(state, goal, hint, actions,log=args.log)
        
    print(f"\nSearch explored {explored} states.")
    if plan: 
        print(f"Discovered path to goal (length of {len(plan)}):")
        for i, step in enumerate(plan): 
            print(f" {i}. {step}")
    else: 
        print("No path to goal found!")
    
    print()

if __name__ == "__main__":
    main()
