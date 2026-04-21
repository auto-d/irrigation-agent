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
        irrigation = await proxy.perceive("irrigation")
        perception_count = 1

        if self._irrigation_abort(irrigation.payload):
            decision = self._decision(
                now,
                action="no_op",
                rationale="Irrigation is active now or the lawn has been watered within the last 24 hours.",
                metadata={"source": "irrigation", "policy_step": 0},
            )
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=perception_count,
                action_count=0,
            )

        precipitation = await proxy.perceive("precipitation")
        perception_count += 1
        if self._recent_rain_payload(precipitation.payload):
            decision = self._decision(
                now,
                action="no_op",
                rationale="Recent precipitation indicates the ground is already wet.",
                metadata={"source": "precipitation", "policy_step": 1},
            )
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=perception_count,
                action_count=0,
            )

        weather = await proxy.perceive("weather", forecast_hours=24)
        perception_count += 1
        if self._forecast_says_rain_payload(weather.payload):
            decision = self._decision(
                now,
                action="no_op",
                rationale="Rain is forecast within the next day.",
                metadata={"source": "weather", "policy_step": 2, "forecast_window_hours": 24},
            )
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=perception_count,
                action_count=0,
            )

        camera = await proxy.perceive("camera")
        perception_count += 1
        if self._scene_unsafe_payload(camera.payload):
            decision = self._decision(
                now,
                action="no_op",
                rationale="A human or obstacle appears to be on the lawn.",
                metadata={"source": "camera", "policy_step": 3},
            )
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=perception_count,
                action_count=0,
            )

        decision = self._decision(
            now,
            action="water_on",
            duration_seconds=3600,
            rationale="No recent watering, rain, or obstacle detected; water for 60 minutes.",
            metadata={"source": "rule_tree", "policy_step": 4},
        )

        action_count = 0
        if decision.action == "water_on":
            await proxy.act("irrigation", "water_on", duration_seconds=decision.duration_seconds or 3600)
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
            perception_count=perception_count,
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
        return self._decision(
            now,
            action=outcome.action,
            duration_seconds=outcome.duration_seconds,
            rationale=outcome.rationale,
            metadata=outcome.metadata,
        )

    def _decision(
        self,
        now: datetime,
        *,
        action: str,
        rationale: str,
        metadata: Dict[str, Any],
        duration_seconds: Optional[int] = None,
    ) -> Decision:
        return Decision(
            timestamp=now,
            planner=self.name,
            action=action,
            duration_seconds=duration_seconds,
            rationale=rationale,
            metadata=metadata,
        )

    def _build_tree(self) -> py_trees.behaviour.Behaviour:
        """Build the small policy tree used for the MVP planner."""
        root = py_trees.composites.Selector(name="IrrigationDecision", memory=False)
        root.add_children(
            [
                self.WorldStateCondition(
                    name="IrrigationRecentOrRunning",
                    blackboard=self._blackboard,
                    predicate=self._irrigation_abort,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Irrigation is active now or the lawn has been watered within the last 24 hours.",
                        metadata={"source": "irrigation", "tree_node": "irrigation_recent_or_running", "policy_step": 0},
                    ),
                ),
                self.WorldStateCondition(
                    name="RecentRain",
                    blackboard=self._blackboard,
                    predicate=self._recent_rain,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Recent precipitation indicates the ground is already wet.",
                        metadata={"source": "precipitation", "tree_node": "recent_rain", "policy_step": 1},
                    ),
                ),
                self.WorldStateCondition(
                    name="ForecastRain",
                    blackboard=self._blackboard,
                    predicate=self._forecast_says_rain,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="Rain is forecast within the next day.",
                        metadata={"source": "weather", "tree_node": "forecast_rain", "policy_step": 2},
                    ),
                ),
                self.WorldStateCondition(
                    name="SceneUnsafe",
                    blackboard=self._blackboard,
                    predicate=self._scene_unsafe,
                    outcome=TreeOutcome(
                        action="no_op",
                        rationale="A human or obstacle appears to be on the lawn.",
                        metadata={"source": "camera", "tree_node": "scene_unsafe", "policy_step": 3},
                    ),
                ),
                self.DefaultOutcome(
                    blackboard=self._blackboard,
                    outcome=TreeOutcome(
                        action="water_on",
                        duration_seconds=3600,
                        rationale="No recent watering, rain, or obstacle detected; water for 60 minutes.",
                        metadata={"source": "rule_tree", "tree_node": "default_water_on", "policy_step": 4},
                    ),
                ),
            ]
        )
        return root

    @staticmethod
    def _irrigation_abort(world_state: WorldState | Dict[str, Any]) -> bool:
        irrigation = world_state.irrigation if isinstance(world_state, WorldState) else world_state
        return bool(
            irrigation.get("api_reports_on")
            or irrigation.get("expected_on")
            or irrigation.get("watered_within_24h")
        )

    @staticmethod
    def _recent_rain(world_state: WorldState) -> bool:
        return ClassicalPlanner._recent_rain_payload(world_state.precipitation)

    @staticmethod
    def _recent_rain_payload(precipitation: Dict[str, Any]) -> bool:
        return bool(
            precipitation.get("recent_rain")
            or precipitation.get("total_inches", 0.0) >= 0.1
        )

    @staticmethod
    def _forecast_says_rain(world_state: WorldState) -> bool:
        return ClassicalPlanner._forecast_says_rain_payload(world_state.forecast)

    @staticmethod
    def _forecast_says_rain_payload(forecast: Dict[str, Any]) -> bool:
        return bool(
            forecast.get("rain_expected_in_window")
            or forecast.get("rain_expected_soon")
            or (forecast.get("max_precip_probability") or 0) >= 50
        )

    @staticmethod
    def _scene_unsafe(world_state: WorldState) -> bool:
        return ClassicalPlanner._scene_unsafe_payload(world_state.camera)

    @staticmethod
    def _scene_unsafe_payload(camera: Dict[str, Any]) -> bool:
        return bool(
            camera.get("person_detected")
            or camera.get("animal_detected")
            or camera.get("lawn_mower_active")
        )
