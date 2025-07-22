import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load ground and table
plane_id = p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0])
table_height = 0.62
hz=60

# Load Franka Panda
start_pos = [0, 0, table_height]  # table height is about 0.62m
start_ori = p.getQuaternionFromEuler([0, 0, 0])
franka_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_ori, useFixedBase=True)

# Step once to let things settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1/hz)

# Joint indices for the 7 DOF arm (not including fingers)
joint_indices = [0, 1, 2, 3, 4, 5, 6]

# Default initial joint positions
initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

# Move to initial joint positions
for i, pos in zip(joint_indices, initial_joint_positions):
    p.resetJointState(franka_id, i, pos)

# Open the gripper
gripper_indices = [9, 10]  # Finger joint indices for Franka Panda
gripper_open_positions = [0.00, 0.00]  # Open positions for fingers

for i, pos in zip(gripper_indices, gripper_open_positions):
    p.resetJointState(franka_id, i, pos)

# Step again
for _ in range(100):
    p.stepSimulation()
    time.sleep(1/hz)

# Get current EEF pose (EEF link index = 11 for Panda)
eef_index = 11
eef_state = p.getLinkState(franka_id, eef_index)
eef_pos = np.array(eef_state[4])  # position
eef_ori = eef_state[5]            # orientation (quaternion)

# Create and attach peg to end effector
peg_radius = 0.012  # 1.2cm radius
peg_height = 0.08   # 8cm height

# Create peg visual shape only (no collision to avoid IK interference)
peg_visual = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=peg_radius,
    length=peg_height,
    rgbaColor=[0, 1, 0, 1]  # Green peg
)

# Create peg body at end effector position (visual only, no collision)
peg_start_pos = [eef_pos[0], eef_pos[1], eef_pos[2]- peg_height / 2]
peg_id = p.createMultiBody(
    baseMass=0,  # No mass
    baseCollisionShapeIndex=-1,  # No collision shape
    baseVisualShapeIndex=peg_visual,
    basePosition=peg_start_pos
)

print(f"Peg created at position: {peg_start_pos}")

# Raise EEF by 0.1 meters in Z
target_pos = eef_pos + np.array([0, 0, 0.1])

# Inverse kinematics to get new joint values
new_joint_positions = p.calculateInverseKinematics(franka_id, eef_index, target_pos, eef_ori)

# Apply new joint positions
for i, j in zip(joint_indices, new_joint_positions):
    p.setJointMotorControl2(franka_id, i, p.POSITION_CONTROL, j, force=200)

# Simulate to see the effect
for step in range(500):
    p.stepSimulation()
    time.sleep(1/hz)
    current_eef_state = p.getLinkState(franka_id, eef_index)
    current_eef_pos = np.array(current_eef_state[4])
    current_eef_ori = current_eef_state[5]
    
    # Update peg position to be below the end effector
    new_peg_pos = [current_eef_pos[0], current_eef_pos[1], current_eef_pos[2] - peg_height / 2]
    p.resetBasePositionAndOrientation(peg_id, new_peg_pos, current_eef_ori)

p.disconnect()
