#!/usr/bin/env python3
"""
Test script for vis_goal modes with SToM logic
"""
from sotopia.visibility.goal_visibility import GoalVisibilitySystem

def test_stom_logic():
    # Mock goals for testing
    all_goals = {
        "Benjamin Jackson": "Win the game (Extra information: you just need five more minutes to accomplish your goal in the game)",
        "Ethan Smith": "Stop the game because it's too late (Extra information: you feel tired and need to go to bed early but don't want to upset your friend)"
    }
    
    print("Testing SToM-based vis_goal modes:")
    print("=" * 60)
    
    # Test agent2 mode
    print("\n1. Testing vis_goal='agent2' (SToM logic):")
    system_agent2 = GoalVisibilitySystem("agent2")
    for agent_name in all_goals.keys():
        own_goal, stom_section = system_agent2.get_agent_goals_and_stom(agent_name, all_goals)
        print(f"\n--- {agent_name} ---")
        print(f"Own goal: {own_goal}")
        print(f"SToM section: {stom_section}")
    
    # Test none mode
    print("\n2. Testing vis_goal='none' (no SToM):")
    system_none = GoalVisibilitySystem("none")
    for agent_name in all_goals.keys():
        own_goal, stom_section = system_none.get_agent_goals_and_stom(agent_name, all_goals)
        print(f"\n--- {agent_name} ---")
        print(f"Own goal: {own_goal}")
        print(f"SToM section: {stom_section}")

if __name__ == "__main__":
    test_stom_logic()