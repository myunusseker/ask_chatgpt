"""
Analysis: Why environment_peg_insertion.py and test_vlm_rendering.py produce different images

Both scripts use the SAME rendering method (render_views) with identical camera parameters,
but they produce different images because the ROBOT POSITIONS are different when the images are captured.

=== MOVEMENT SEQUENCE COMPARISON ===

environment_peg_insertion.py:
1. Move to: [0.39, 0.0, table_height+0.12] = [0.39, 0.0, 0.74] (absolute)
2. Move by: [0.0, 0.0, -0.05] (relative, downward 5cm)
3. Final position: ~[0.39, 0.0, 0.69] 
4. Render images at FINAL position

test_vlm_rendering.py:
1. Move to: [0.40, 0.0, table_height+0.15] = [0.40, 0.0, 0.77] (absolute)
2. Render "initial_position" images at this HIGH position
3. Move by: [0.0, 0.0, -0.08] (relative, downward 8cm)  
4. Final position: ~[0.40, 0.0, 0.69]
5. Render "final_position" images at FINAL position

=== KEY DIFFERENCES ===

1. X-COORDINATE:
   - environment_peg_insertion.py: x = 0.39
   - test_vlm_rendering.py: x = 0.40 (1cm difference)

2. STARTING HEIGHT:
   - environment_peg_insertion.py: starts at 0.74m
   - test_vlm_rendering.py: starts at 0.77m (3cm higher)

3. DOWNWARD MOVEMENT:
   - environment_peg_insertion.py: moves down 5cm
   - test_vlm_rendering.py: moves down 8cm

4. NUMBER OF IMAGES:
   - environment_peg_insertion.py: 1 set of images (after_insertion)
   - test_vlm_rendering.py: 2 sets of images (initial_position + final_position)

5. GUI MODE:
   - environment_peg_insertion.py: GUI=True (shows visual simulation)
   - test_vlm_rendering.py: GUI=False (headless mode)

=== RESULT ===
The camera rendering system is IDENTICAL, but the robot arm is in different positions
when the photos are taken, which changes:
- The viewing angle from the wrist camera
- What objects are visible in the frame
- The distance relationships between objects
- The perspective and composition of both wrist and side cameras

This explains why the images look different - it's like taking photos of the same scene
from slightly different positions and angles!
"""

print(__doc__)

# Let's also demonstrate this with actual coordinates
if __name__ == "__main__":
    import numpy as np
    
    table_height = 0.62
    
    print("=== COORDINATE ANALYSIS ===")
    print("\nenvironment_peg_insertion.py:")
    pos1 = np.array([0.39, 0.0, table_height + 0.12])
    pos1_final = pos1 + np.array([0.0, 0.0, -0.05])
    print(f"  Start position: {pos1}")
    print(f"  Final position: {pos1_final}")
    print(f"  Image captured at: {pos1_final}")
    
    print("\ntest_vlm_rendering.py:")
    pos2 = np.array([0.40, 0.0, table_height + 0.15])
    pos2_final = pos2 + np.array([0.0, 0.0, -0.08])
    print(f"  Start position: {pos2}")
    print(f"  Final position: {pos2_final}")
    print(f"  Images captured at: {pos2} AND {pos2_final}")
    
    print(f"\n=== POSITION DIFFERENCES ===")
    print(f"X-difference: {abs(pos1_final[0] - pos2_final[0]):.3f}m = {abs(pos1_final[0] - pos2_final[0])*100:.1f}cm")
    print(f"Z-difference: {abs(pos1_final[2] - pos2_final[2]):.3f}m = {abs(pos1_final[2] - pos2_final[2])*100:.1f}cm")
    print(f"Total distance difference: {np.linalg.norm(pos1_final - pos2_final):.3f}m")
