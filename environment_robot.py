import pybullet as p
import pybullet_data
import time
import numpy as np


def move_to_xy(robot_id, target_xy, table_height, down_offset=0.05):
    joint_indices = [0, 1, 2, 3, 4, 5, 6]
    num_joints = p.getNumJoints(robot_id)
    damping = [0.1] * num_joints

    # Target position above hole
    target_pos = [target_xy[0], target_xy[1], table_height + 0.2]

    # ✅ Gripper pointing down: rotate 180° around X
    target_ori = p.getQuaternionFromEuler([0, np.pi / 2, np.pi / 2])

    # Move above the hole
    joint_positions = p.calculateInverseKinematics(
        robot_id, 11, target_pos, target_ori, jointDamping=damping
    )
    for i, j in zip(joint_indices, joint_positions):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=j, force=500)

    for _ in range(300):
        p.stepSimulation()
        time.sleep(1/60)


    # Move downward to insert
    target_pos[2] -= down_offset
    joint_positions = p.calculateInverseKinematics(
        robot_id, 11, target_pos, target_ori, jointDamping=damping
    )
    for i, j in zip(joint_indices, joint_positions):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=j, force=500)

    for _ in range(300):
        p.stepSimulation()
        time.sleep(1/60)


    # Debug EE frame (XYZ axes)
    ee_state = p.getLinkState(robot_id, 11)
    ee_pos = ee_state[0]
    ee_ori = ee_state[1]
    rot_matrix = p.getMatrixFromQuaternion(ee_ori)
    x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
    z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

    p.addUserDebugLine(ee_pos, [ee_pos[i] + 0.1 * x_axis[i] for i in range(3)], [1, 0, 0], 3, lifeTime=10)  # Red = X
    p.addUserDebugLine(ee_pos, [ee_pos[i] + 0.1 * y_axis[i] for i in range(3)], [0, 1, 0], 3, lifeTime=10)  # Green = Y
    p.addUserDebugLine(ee_pos, [ee_pos[i] + 0.1 * z_axis[i] for i in range(3)], [0, 0, 1], 3, lifeTime=10)  # Blue = Z


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    # Load table
    table_pos = [0.5, 0, 0]
    table_id = p.loadURDF("table/table.urdf", basePosition=table_pos)
    table_height = 0.62

    # ✅ Load robot raised above table
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, table_height + 0.01],
        useFixedBase=True
    )

    # Joint indices for the 7 DOF arm (not including fingers)
    joint_indices = [0, 1, 2, 3, 4, 5, 6]

    # Default initial joint positions
    initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

    # Move to initial joint positions
    # for i, pos in zip(joint_indices, initial_joint_positions):
    #     p.resetJointState(robot_id, i, pos)
    #     time.sleep(1/60)


    # Add fake hole (just for visuals)
    hole_pos = [0.6, 0.0, table_height + 0.015]
    p.loadURDF("cube_small.urdf", basePosition=hole_pos, globalScaling=0.03)

    for _ in range(200):
        p.stepSimulation()
        time.sleep(1/60)


    # Command: move to hole and insert
    move_to_xy(robot_id, [0.6, 0.0], table_height)

    time.sleep(10)
    p.disconnect()


if __name__ == "__main__":
    main()
