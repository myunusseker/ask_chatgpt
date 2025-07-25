#!/usr/bin/env python3
"""
Demonstration: How PyBullet connection mode affects lighting and shadows
"""

import numpy as np
from environment_peg_insertion import PegInsertionEnvironment

def test_lighting_differences():
    print("=== LIGHTING AND SHADOWS COMPARISON ===")
    print("Testing how GUI vs DIRECT mode affects rendering quality...\n")
    
    # Test 1: GUI Mode (like environment_peg_insertion.py)
    print("1. Testing GUI Mode (p.GUI)...")
    env_gui = PegInsertionEnvironment(gui=True, hz=60)
    
    # Move to same position
    env_gui.move_smooth(
        target=np.array([0.40, 0.0, env_gui.table_height + 0.15]), 
        duration=1.0, 
        relative=False
    )
    
    # Render with GUI mode
    wrist_gui, side_gui = env_gui.render_views(save_images=True, image_prefix="gui_mode")
    print("   GUI mode images saved with prefix 'gui_mode'")
    env_gui.disconnect()
    
    # Test 2: DIRECT Mode (like test_vlm_rendering.py)
    print("\n2. Testing DIRECT Mode (p.DIRECT)...")
    env_direct = PegInsertionEnvironment(gui=False, hz=60)
    
    # Move to same position
    env_direct.move_smooth(
        target=np.array([0.40, 0.0, env_direct.table_height + 0.15]), 
        duration=1.0, 
        relative=False
    )
    
    # Render with DIRECT mode
    wrist_direct, side_direct = env_direct.render_views(save_images=True, image_prefix="direct_mode")
    print("   DIRECT mode images saved with prefix 'direct_mode'")
    env_direct.disconnect()
    
    print("\n=== WHY LIGHTING IS DIFFERENT ===")
    print("PyBullet handles lighting differently in GUI vs DIRECT mode:")
    print("• GUI Mode (p.GUI):")
    print("  - Uses OpenGL rendering pipeline optimized for interactive display")
    print("  - May have different default lighting and shadow calculations")
    print("  - Lighting can be affected by the GUI window's OpenGL context")
    print("  - Sometimes has more realistic shadows and reflections")
    print("")
    print("• DIRECT Mode (p.DIRECT):")
    print("  - Uses headless rendering optimized for batch processing")
    print("  - More consistent lighting across different machines")
    print("  - May have simpler shadow calculations for performance")
    print("  - Better for reproducible results in scientific applications")
    print("")
    print("Even though both use the SAME lighting parameters in getCameraImage():")
    print("  - light_direction = [-0.5, -0.5, -1]")
    print("  - light_color = [1.2, 1.2, 1.2]")
    print("  - light_ambient_coeff = 0.4")
    print("  - light_diffuse_coeff = 0.8")
    print("  - light_specular_coeff = 0.3")
    print("")
    print("The underlying OpenGL context affects how these are rendered!")
    
    print(f"\nCompare the images:")
    print(f"• gui_mode_wrist_rgb.png vs direct_mode_wrist_rgb.png")
    print(f"• gui_mode_side_rgb.png vs direct_mode_side_rgb.png")

if __name__ == "__main__":
    test_lighting_differences()
