#!/usr/bin/env python3
"""
Test script to demonstrate labeled camera views for VLM
"""

import numpy as np
from environment_peg_insertion import PegInsertionEnvironment

def test_labeled_views():
    print("=== Testing Labeled Camera Views for VLM ===")
    
    # Create environment
    env = PegInsertionEnvironment(gui=False, hz=60)
    
    # Test 1: Initial position with labels
    print("1. Rendering initial position with view labels...")
    wrist_rgb, side_rgb = env.render_views(save_images=True, image_postfix="labeled_initial")
    print(f"   Wrist image shape: {wrist_rgb.shape}")
    print(f"   Side image shape: {side_rgb.shape}")
    
    # Test 2: Move and render with labels
    print("\n2. Moving robot and rendering with labels...")
    env.move_smooth(
        target=np.array([0.40, 0.0, env.table_height + 0.15]), 
        duration=2.0, 
        relative=False
    )
    wrist_rgb, side_rgb = env.render_views(save_images=True, image_postfix="labeled_moved")
    
    # Test 3: Test with collision and labels
    print("\n3. Testing downward movement with collision and labels...")
    env.move_smooth(
        target=np.array([0.0, 0.0, -0.08]), 
        duration=3.0, 
        relative=True, 
        stop_on_contact=True, 
        contact_threshold=1.0
    )
    wrist_rgb, side_rgb = env.render_views(save_images=True, image_postfix="labeled_collision")
    
    # Cleanup
    env.close()
    
    print("\n=== Labeled View Test Complete ===")
    print("Check images/ directory for labeled images:")
    print("- wrist_labeled_initial.png (with 'WRIST VIEW' label)")
    print("- side_labeled_initial.png (with 'SIDE VIEW' label)")
    print("- wrist_labeled_moved.png")
    print("- side_labeled_moved.png")
    print("- wrist_labeled_collision.png")
    print("- side_labeled_collision.png")
    print("\nThe VLM can now clearly see which view is which!")

if __name__ == "__main__":
    test_labeled_views()
