#!/usr/bin/env python3
"""
Test script to demonstrate VLM-optimized rendering improvements
"""

import numpy as np
from environment_peg_insertion import PegInsertionEnvironment

def main():
    print("=== VLM-Optimized Peg Insertion Environment Test ===")
    
    # Create environment
    env = PegInsertionEnvironment(gui=False, hz=60)  # No GUI for faster testing
    
    # Move to position above the hole
    print("Moving to position above insertion hole...")
    env.move_smooth(
        target=np.array([0.40, 0.0, env.table_height + 0.15]), 
        duration=2.0, 
        relative=False
    )
    
    # Render initial view
    print("Rendering initial camera views...")
    wrist_rgb, side_rgb = env.render_views(save_images=True, image_prefix="initial_position")
    
    # Move down with collision detection
    print("Moving down towards insertion hole with collision detection...")
    env.move_smooth(
        target=np.array([0.0, 0.0, -0.08]), 
        duration=4.0, 
        relative=True, 
        stop_on_contact=True, 
        contact_threshold=0.8
    )
    
    # Render final view
    print("Rendering final camera views...")
    wrist_rgb_final, side_rgb_final = env.render_views(save_images=True, image_prefix="final_position")
    
    print(f"Initial wrist image shape: {wrist_rgb.shape}")
    print(f"Final wrist image shape: {wrist_rgb_final.shape}")
    print(f"Image resolution: {wrist_rgb.shape[1]}x{wrist_rgb.shape[0]} (high resolution for VLM)")
    
    # Cleanup
    env.disconnect()
    
    print("\n=== Key VLM Optimizations Applied ===")
    print("1. Wrist camera rotated 90 degrees for better orientation")
    print("2. Higher resolution (1280x960) for better detail")
    print("3. Optimized lighting with reduced glare and better contrast")
    print("4. Gamma correction and contrast enhancement for VLM processing")
    print("5. Improved ambient lighting for shadow detail")
    print("\nCheck the images/ directory for the rendered views!")

if __name__ == "__main__":
    main()
