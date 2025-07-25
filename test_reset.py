#!/usr/bin/env python3
"""
Test script to demonstrate the reset functionality
"""

import numpy as np
from environment_peg_insertion import PegInsertionEnvironment

def test_reset_functionality():
    print("=== Testing Reset Functionality ===")
    
    # Create environment
    env = PegInsertionEnvironment(gui=True, hz=60)
    
    print("Initial position:", env.eef_pos)
    
    # Move to some position
    env.move_smooth(
        target=np.array([0.45, 0.1, env.table_height + 0.2]), 
        duration=5.0, 
        relative=False
    )
    print("Position after movement:", env.eef_pos)
    print("Position after movement:", env.get_current_position())

    
    # Reset the environment
    print("\n2. Resetting environment...")
    reset_pos = env.reset()
    print("Position after reset, reset_pos:", reset_pos)
    print("Position after reset, env.eef_pos:", env.eef_pos)
    print("Position after reset, get_current_position:", env.get_current_position())

    # Test another movement to verify reset worked
    # env.move_smooth(
    #     target=np.array([0.40, -0.1, env.table_height + 0.2]), 
    #     duration=2.0, 
    #     relative=False
    # )
    
    # Cleanup
    env.disconnect()
    
    print("\n=== Reset Test Complete ===")
    print("Check images/ directory for:")
    print("- wrist_before_reset.png / side_before_reset.png")
    print("- wrist_after_reset.png / side_after_reset.png")
    print("- wrist_final_test.png / side_final_test.png")

if __name__ == "__main__":
    test_reset_functionality()
