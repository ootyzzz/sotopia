#!/usr/bin/env python3
"""
Temporary test script for Phase 1a.1: Test GoalVisibilitySystem integration
"""
import asyncio
from sotopia.visibility.goal_visibility import GoalVisibilitySystem

async def test_visibility():
    # Test GoalVisibilitySystem
    system = GoalVisibilitySystem("both")
    all_goals = {"agent1": "Goal 1", "agent2": "Goal 2"}

    visible_for_agent1 = system.get_visible_goals("agent1", all_goals)
    visible_for_agent2 = system.get_visible_goals("agent2", all_goals)

    print("=== GoalVisibilitySystem Test ===")
    print(f"Mode: both")
    print(f"Agent1 visible goals: {visible_for_agent1}")
    print(f"Agent2 visible goals: {visible_for_agent2}")

    # Test "none" mode
    system_none = GoalVisibilitySystem("none")
    visible_for_agent1_none = system_none.get_visible_goals("agent1", all_goals)
    visible_for_agent2_none = system_none.get_visible_goals("agent2", all_goals)

    print(f"\nMode: none")
    print(f"Agent1 visible goals: {visible_for_agent1_none}")
    print(f"Agent2 visible goals: {visible_for_agent2_none}")

    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    asyncio.run(test_visibility())