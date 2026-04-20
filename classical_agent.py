"""Classical planner implementation using a small py_trees behavior tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import py_trees

from planner import BasePlanner, Decision, Event, PlannerProxy, PlannerRunResult


@dataclass(frozen=True)
class WorldState:
    """Condensed planner inputs for one behavior-tree tick."""

    irrigation: Dict[str, Any]
    precipitation: Dict[str, Any]
    forecast: Dict[str, Any]
    camera: Dict[str, Any]


@dataclass(frozen=True)
class TreeOutcome:
    """Behavior-tree leaf output before normalization to a Decision."""

    action: str
    rationale: str
    metadata: Dict[str, Any]
    duration_seconds: Optional[int] = None


class ClassicalPlanner(BasePlanner):
    """Minimal classical planner backed by a toy py_trees behavior tree."""

    name = "classical"

    class WorldStateCondition(py_trees.behaviour.Behaviour):
        """Leaf that checks one predicate against the current world state."""

        def __init__(
            self,
            *,
            name: str,
            blackboard: py_trees.blackboard.Client,
            predicate: Callable[[WorldState], bool],
            outcome: TreeOutcome,
        ) -> None:
            super().__init__(name=name)
            self.blackboard = blackboard
            self._predicate = predicate
            self._outcome = outcome

        def update(self) -> py_trees.common.Status:
            world_state = self.blackboard.world_state
            if self._predicate(world_state):
                self.blackboard.outcome = self._outcome
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE

    class DefaultOutcome(py_trees.behaviour.Behaviour):
        """Leaf that always selects the fallback outcome."""

        def __init__(self, *, blackboard: py_trees.blackboard.Client, outcome: TreeOutcome) -> None:
            super().__init__(name="DefaultWaterOn")
            self.blackboard = blackboard
            self._outcome = outcome

        def update(self) -> py_trees.common.Status:
            self.blackboard.outcome = self._outcome
            return py_trees.common.Status.SUCCESS

    def __init__(self) -> None:
        super().__init__()
        self._blackboard = py_trees.blackboard.Client(name="classical_planner")
        self._blackboard.register_key(key="world_state", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="outcome", access=py_trees.common.Access.WRITE)
        self._tree = self._build_tree()
        self._tree.setup()

    async def run(self, proxy: PlannerProxy, *, now: datetime) -> PlannerRunResult:
        # The classical planner is intentionally explicit here: fetch a small,
        # fixed set of perceptions, run the tree once, then emit at most one action.
        weather = await proxy.perceive("weather")
        precipitation = await proxy.perceive("precipitation")
        irrigation = await proxy.perceive("irrigation")
        camera = await proxy.perceive("camera")
        decision = self.decision_from_events(
            now,
            weather=weather,
            precipitation=precipitation,
            irrigation=irrigation,
            camera=camera,
        )

        action_count = 0
        if decision.action == "water_on":
            await proxy.act("irrigation", "water_on", duration_seconds=decision.duration_seconds or 300)
            action_count += 1
        elif decision.action == "water_off":
            await proxy.act("irrigation", "water_off")
            action_count += 1
        elif decision.action == "notify":
            await proxy.act("notification", "send", message=decision.rationale, metadata=decision.metadata)
            action_count += 1

        return PlannerRunResult(
            timestamp=now,
            planner=self.name,
            decision=decision,
            trace=self._consume_trace(),
            perception_count=4,
            action_count=action_count,
        )

    def decision_from_events(
        self,
        now: datetime,
        *,
        weather: Event,
        precipitation: Event,
        irrigation: Event,
        camera: Event,
    ) -> Decision:
        """Compute one decision from already-materialized perception events."""
        self._blackboard.world_state = WorldState(
            irrigation=irrigation.payload,
            precipitation=precipitation.payload,
            forecast=weather.payload,
            camera=camera.payload,
        )
        self._blackboard.outcome = None
        self._tree.tick_once()
        outcome = self._blackboard.outcome
        if outcome is None:
            raise RuntimeError("Behavior tree tick completed without producing an outcome.")
        return Decision(
            timestamp=now,
            planner=self.name,
            action=outcome.action,
            duration_seconds=outcome.duration_seconds,
            rationale=outcome.rationale,
            metadata=outcome.metadata,
        )

    def _build_tree(self) -> py_trees.behaviour.Behaviour:
        """Build the small policy tree used for the MVP planner."""
        root = py_trees.composites.Selector(name="IrrigationDecision", memory=False)
        root.add_children(
            [
                self.WorldStateCondition(
                    name="IrrigationRunning",
                    blackboard=self._blackboard,
                    predicate=self._irrigation_running,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Irrigation appears to be running or expected to be running already.",
                        metadata={"source": "irrigation", "tree_node": "irrigation_running"},
                    ),
                ),
                self.WorldStateCondition(
                    name="RecentRain",
                    blackboard=self._blackboard,
                    predicate=self._recent_rain,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Recent precipitation exceeds the watering threshold.",
                        metadata={"source": "precipitation", "tree_node": "recent_rain"},
                    ),
                ),
                self.WorldStateCondition(
                    name="ForecastRain",
                    blackboard=self._blackboard,
                    predicate=self._forecast_says_rain,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Forecast indicates rain is likely within the configured window.",
                        metadata={"source": "weather", "tree_node": "forecast_rain"},
                    ),
                ),
                self.WorldStateCondition(
                    name="SceneUnsafe",
                    blackboard=self._blackboard,
                    predicate=self._scene_unsafe,
                    outcome=TreeOutcome(
                        action="water_off",
                        rationale="Scene appears occupied or unsafe for watering.",
                        metadata={"source": "camera", "tree_node": "scene_unsafe"},
                    ),
                ),
                self.DefaultOutcome(
                    blackboard=self._blackboard,
                    outcome=TreeOutcome(
                        action="water_on",
                        duration_seconds=300,
                        rationale="No rain signal or occupancy conflict detected; emit a short watering action.",
                        metadata={"source": "rule_tree", "tree_node": "default_water_on"},
                    ),
                ),
            ]
        )
        return root

    @staticmethod
    def _irrigation_running(world_state: WorldState) -> bool:
        return bool(world_state.irrigation.get("api_reports_on") or world_state.irrigation.get("expected_on"))

    @staticmethod
    def _recent_rain(world_state: WorldState) -> bool:
        return bool(
            world_state.precipitation.get("recent_rain")
            or world_state.precipitation.get("total_inches", 0.0) >= 0.1
        )

    @staticmethod
    def _forecast_says_rain(world_state: WorldState) -> bool:
        return bool(
            world_state.forecast.get("rain_expected_in_window")
            or world_state.forecast.get("rain_expected_soon")
            or (world_state.forecast.get("max_precip_probability") or 0) >= 50
        )

    @staticmethod
    def _scene_unsafe(world_state: WorldState) -> bool:
        camera = world_state.camera
        return bool(
            camera.get("person_detected")
            or camera.get("animal_detected")
            or camera.get("lawn_mower_active")
        )
