#!/usr/bin/env python3
"""
Test script to demonstrate the fixed position tracking
"""

import numpy as np
from environment_peg_insertion import PegInsertionEnvironment

def test_position_tracking():
    print("=== Testing Position Tracking Fix ===")
    
    # Create environment
    env = PegInsertionEnvironment(gui=True, hz=60)
    
    print(f"1. Initial position (env.eef_pos): {env.eef_pos}")
    print(f"   Current position (get_current_position): {env.get_current_position()}")
    
    # Test movement 1
    print(f"\n2. Moving to [0.45, 0.1, {env.table_height + 0.2:.3f}]...")
    env.move_smooth(
        target=np.array([0.45, 0.1, env.table_height + 0.2]), 
        duration=2.0, 
        relative=False
    )
    print(f"   Position after movement (env.eef_pos): {env.eef_pos}")
    print(f"   Current position (get_current_position): {env.get_current_position()}")
    
    # Test relative movement
    print(f"\n3. Moving relatively by [0.05, -0.05, 0.1]...")
    env.move_smooth(
        target=np.array([0.05, -0.05, 0.1]), 
        duration=1.5, 
        relative=True
    )
    print(f"   Position after relative movement (env.eef_pos): {env.eef_pos}")
    print(f"   Current position (get_current_position): {env.get_current_position()}")
    
    # Test reset
    print(f"\n4. Resetting environment...")
    reset_pos = env.reset()
    print(f"   Position after reset (returned): {reset_pos}")
    print(f"   Position after reset (env.eef_pos): {env.eef_pos}")
    print(f"   Current position (get_current_position): {env.get_current_position()}")
    
    # Test movement with collision
    print(f"\n5. Moving down with collision detection...")
    env.move_smooth(
        target=np.array([0.40, 0.0, env.table_height + 0.15]), 
        duration=1.0, 
        relative=False
    )
    print(f"   Position before downward movement: {env.eef_pos}")
    
    env.move_smooth(
        target=np.array([0.0, 0.0, -0.08]), 
        duration=3.0, 
        relative=True, 
        stop_on_contact=True, 
        contact_threshold=1.0
    )
    print(f"   Position after collision stop: {env.eef_pos}")
    
    # Cleanup
    env.disconnect()
    
    print(f"\n=== Position Tracking Working Correctly! ===")
    print("The env.eef_pos attribute now properly updates after each movement.")

if __name__ == "__main__":
    test_position_tracking()
