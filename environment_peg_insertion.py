import os
import time
from typing import Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import imageio
from PIL import Image


class PegInsertionEnv:
    """Environment where a Franka robot performs peg insertion task."""

    def __init__(self, gui: bool = True, speed: int = 30):
        self.speed = speed
        self.gui = gui
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = None
        self.robot_id = None
        self.peg_id = None
        self.hole_id = None
        
        # Task parameters
        self.table_height = 0.625
        self.peg_start_height = self.table_height + 0.05
        self.hole_position = np.array([0.5, 0.0, self.table_height + 0.001])  # Fixed hole position on table
        self.hole_radius = 0.02  # 1.5cm radius
        self.peg_radius = 0.012   # 1.2cm radius (slightly smaller than hole)
        
        # Robot parameters
        self.robot_base_pos = [0, 0, self.table_height]  # On top of table
        self.end_effector_link = 11  # Franka end effector link
        
        # Camera parameters
        self.diagonal_camera_pos = [0.8, -0.8, 0.8]
        self.diagonal_camera_target = [0.5, 0.0, self.table_height]
        self.wrist_camera_offset = [0, 0, 0.05]  # Smaller offset from end effector

    def reset(self) -> np.ndarray:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table
        table_pos = [0.5, 0, 0]
        self.table_id = p.loadURDF("table/table.urdf", table_pos)
        
        # Load Franka robot
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", self.robot_base_pos, useFixedBase=True)
        
        # Set robot to initial pose (arm extended towards table)
        self._set_robot_initial_pose()
        
        # Run simulation steps to stabilize
        for _ in range(100):
            p.stepSimulation()
        
        # Create peg (visual only, no collision for smooth insertion)
        peg_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.peg_radius,
            length=0.08,
            rgbaColor=[0, 1, 0, 1]  # Green peg
        )
        
        # Start peg at end effector
        ee_pos = self._get_end_effector_position()
        peg_start_pos = [ee_pos[0], ee_pos[1], ee_pos[2]-0.1]
        
        self.peg_id = p.createMultiBody(
            baseMass=0,  # No mass since it's visual only
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=peg_visual,
            basePosition=peg_start_pos
        )
        
        # Create hole as a recessed area on the table
        # First create a visual ring around the hole
        hole_ring_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.hole_radius + 0.005,  # Slightly larger ring
            length=0.002,
            rgbaColor=[0.5, 0.5, 0.5, 1]  # Gray ring
        )
        
        # Create the actual hole (visual depression)
        hole_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.hole_radius,
            length=0.01,
            rgbaColor=[0.2, 0.2, 0.2, 1]  # Dark hole
        )
        
        # Ring around hole (visual only)
        self.hole_ring_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=hole_ring_visual,
            basePosition=[self.hole_position[0], self.hole_position[1], self.hole_position[2]]
        )
        
        # Hole itself (visual only, no collision to allow insertion)
        self.hole_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=hole_visual,
            basePosition=[self.hole_position[0], self.hole_position[1], self.hole_position[2] - 0.005]
        )
        
        # Attach peg to robot end effector initially
        self._attach_peg_to_robot()
        
        return self.get_state()

    def _set_robot_initial_pose(self):
        """Set robot to initial pose with arm extended towards table."""
        # Joint positions for a proper initial pose - arm reaching forward and down
        joint_positions = [0, 0.3, 0, -2.0, 0, 2.3, 0.785]  # Proper downward-reaching pose
        
        for i in range(len(joint_positions)):
            p.resetJointState(self.robot_id, i, joint_positions[i])
            
        # Set gripper to closed position
        p.resetJointState(self.robot_id, 9, 0.04)   # Finger 1
        p.resetJointState(self.robot_id, 10, 0.04)  # Finger 2
        
        # Enable position control for all joints
        joint_indices = list(range(7))  # First 7 joints
        p.setJointMotorControlArray(
            self.robot_id,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[87] * len(joint_indices)  # Joint torque limits
        )

    def _get_end_effector_position(self) -> np.ndarray:
        """Get current end effector position."""
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
        return np.array(ee_state[0])

    def _get_end_effector_orientation(self) -> np.ndarray:
        """Get current end effector orientation."""
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
        return np.array(ee_state[1])  # Returns quaternion [x, y, z, w]

    def _attach_peg_to_robot(self):
        """Attach peg to robot end effector."""
        self.peg_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_link,
            childBodyUniqueId=self.peg_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -0.08],
            childFramePosition=[0, 0, 0.04]
        )

    def get_state(self) -> np.ndarray:
        """Get current peg position."""
        pos, _ = p.getBasePositionAndOrientation(self.peg_id)
        return np.array(pos)

    def apply_insertion_action(self, target_xy: list, record: bool = False, video_path: str = "peg_insertion.mp4") -> np.ndarray:
        """
        Apply peg insertion action.
        Args:
            target_xy: [x, y] target position for peg insertion
            record: Whether to record video
            video_path: Path to save video
        """
        if record:
            os.makedirs("videos", exist_ok=True)
            self.video_path = os.path.join("videos", video_path)
            self.frames = []
            
        print(f"Inserting peg at position: {target_xy}")
        
        target_x, target_y = target_xy
        
        # Phase 1: Move above target position (approach)
        print("Phase 1: Approaching target position...")
        self._move_to_position([target_x, target_y, self.table_height + 0.1], record)
        
        # Phase 2: Insert peg (slow insertion)
        print("Phase 2: Inserting peg...")
        self._move_to_position([target_x, target_y, self.table_height - 0.02], record, slow=True)
        
        if record and self.frames:
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"Video saved to {self.video_path}")
            
        return self.get_state()
    

    def move_to_xy(robot_id, target_xy, table_height, down_offset=0.1):
        # Get current joint positions
        joint_indices = [0, 1, 2, 3, 4, 5, 6]
        current_joint_positions = [p.getJointState(robot_id, i)[0] for i in joint_indices]

        # Define the target pose (x, y, z)
        target_pos = [target_xy[0], target_xy[1], table_height + 0.15]
        target_ori = p.getQuaternionFromEuler([np.pi, 0, 0])  # Facing down

        # Move to above target
        joint_positions = p.calculateInverseKinematics(robot_id, 11, target_pos, target_ori)
        for i, j in zip(joint_indices, joint_positions):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j)

        for _ in range(100): p.stepSimulation()

        # Insertion down
        target_pos[2] -= down_offset
        joint_positions = p.calculateInverseKinematics(robot_id, 11, target_pos, target_ori)
        for i, j in zip(joint_indices, joint_positions):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j)

        for _ in range(100): p.stepSimulation()

    def _move_to_position(self, target_pos: list, record: bool = False, slow: bool = False):
        """Move robot end effector to target position using inverse kinematics with proper orientation."""
        import math
        
        target_pos = np.array(target_pos)
        
        # Define target orientation - pointing straight down for peg insertion
        target_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])  # 180 degrees around X-axis
        
        # Use more steps for slower, more precise insertion
        steps = 200 if slow else 120
        
        # Get current joint positions as starting point
        current_joint_positions = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        for step in range(steps):
            # Calculate inverse kinematics with both position and orientation constraints
            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.end_effector_link,
                targetPosition=target_pos,
                targetOrientation=target_orientation,
                lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                jointRanges=[5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8],
                restPoses=current_joint_positions,
                maxNumIterations=100,
                residualThreshold=0.01
            )
            
            # Apply joint positions with proper control
            joint_indices = list(range(7))  # First 7 joints
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_poses[:7],
                forces=[87] * 7,  # Joint torque limits
                positionGains=[0.1] * 7,  # Position gains for smoother movement
                velocityGains=[1.0] * 7   # Velocity gains for damping
            )
            
            p.stepSimulation()
            
            if self.gui:
                sleep_time = (1 / self.speed) * (2 if slow else 1)  # Slower for insertion
                time.sleep(sleep_time)
                
            if record:
                # Capture frame for video
                diagonal_view = self._get_diagonal_camera_image()
                frame_bgr = cv2.cvtColor(np.array(diagonal_view), cv2.COLOR_RGB2BGR)
                self.frames.append(frame_bgr)
            
            # Check if close enough to target
            current_ee_pos = self._get_end_effector_position()
            distance = np.linalg.norm(target_pos - current_ee_pos)
            threshold = 0.01 if slow else 0.02  # More precise for slow insertion
            if distance < threshold:
                break

    def _get_diagonal_camera_image(self) -> np.ndarray:
        """Get diagonal view camera image."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.diagonal_camera_pos,
            cameraTargetPosition=self.diagonal_camera_target,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
        )
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        return rgb

    def _get_wrist_camera_image(self) -> np.ndarray:
        """Get wrist camera image."""
        ee_pos = self._get_end_effector_position()
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
        ee_orn = ee_state[1]
        
        # Calculate camera position relative to end effector
        camera_pos = ee_pos + self.wrist_camera_offset
        
        # Camera looks down towards table
        target_pos = [camera_pos[0], camera_pos[1], self.table_height]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
        )
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        return rgb

    def render_views(self, diagonal_path="diagonal_view.png", wrist_path="wrist_view.png"):
        """Render both camera views and save them."""
        # Diagonal view
        diagonal_rgb = self._get_diagonal_camera_image()
        Image.fromarray(diagonal_rgb).save(diagonal_path)
        
        # Wrist view
        wrist_rgb = self._get_wrist_camera_image()
        Image.fromarray(wrist_rgb).save(wrist_path)

    def get_insertion_success(self) -> bool:
        """Check if peg insertion was successful."""
        peg_pos = self.get_state()
        hole_pos = self.hole_position
        
        # Check if peg is close to hole position (xy plane) - must be very precise
        xy_distance = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        
        # Check if peg is at correct insertion depth
        insertion_depth = hole_pos[2] - peg_pos[2]
        proper_insertion = insertion_depth >= 0.02  # At least 2cm inserted
        
        # Success requires both precise alignment and proper insertion depth
        return xy_distance < (self.hole_radius * 0.8) and proper_insertion

    def get_reward(self) -> float:
        """Calculate reward based on peg insertion quality."""
        peg_pos = self.get_state()
        hole_pos = self.hole_position
        
        # Distance in XY plane (alignment penalty)
        xy_distance = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        alignment_reward = max(0, (self.hole_radius - xy_distance) / self.hole_radius)
        
        # Insertion depth reward
        insertion_depth = max(0, hole_pos[2] - peg_pos[2])
        depth_reward = min(1.0, insertion_depth / 0.03)  # Reward for inserting up to 3cm
        
        # Precision bonus for being very close to hole center
        precision_bonus = 2.0 if xy_distance < (self.hole_radius * 0.5) else 0.0
        
        # Insertion success bonus
        success_bonus = 10.0 if self.get_insertion_success() else 0.0
        
        # Combined reward
        reward = alignment_reward + depth_reward + precision_bonus + success_bonus - xy_distance
        
        return reward

    def convert_video_to_gif(self, gif_path: str = "peg_insertion.gif") -> None:
        """Convert recorded video to GIF."""
        if hasattr(self, "video_path") and os.path.exists(self.video_path):
            gif_path_full = os.path.join("videos", gif_path)
            reader = imageio.get_reader(self.video_path)
            fps = reader.get_meta_data()["fps"]
            writer = imageio.get_writer(gif_path_full, fps=fps, loop=0)
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            print(f"GIF saved to {gif_path_full}")
        else:
            print("Video not found or not recorded yet.")

    def close(self) -> None:
        """Close the environment."""
        p.disconnect()


if __name__ == "__main__":
    env = PegInsertionEnv(gui=False)  # Use headless mode for testing
    env.reset()
    
    # Test insertion at hole location
    hole_xy = env.hole_position[:2]
    print(f"Applying insertion action at hole position: {hole_xy}")
    #env.apply_insertion_action(hole_xy, record=False, video_path="peg_insertion_demo.mp4")
    env.move_to_xy(env.robot_id, hole_xy, env.table_height)
    
    # Render views
    env.render_views("diagonal_view.png", "wrist_view.png")
    
    # Check success
    success = env.get_insertion_success()
    reward = env.get_reward()
    print(f"Insertion successful: {success}")
    print(f"Final reward: {reward:.3f}")
    
    env.close()
