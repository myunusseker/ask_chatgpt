import os
import time
from typing import Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import imageio
from PIL import Image


class SlideToGoalEnv:
    """Simple environment where a cube is pushed towards a goal."""

    def __init__(self, gui: bool = True, speed: int = 30):
        self.speed = speed
        self.gui = gui
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")
        self.block_id = None
        self.goal_position = np.array([0.5, 0.5, 0])

        self.last_camera_target = None
        self.last_camera_distance = 0.7 #0.7
        self.last_camera_yaw = -45
        self.last_camera_pitch = -45

    def reset(self) -> np.ndarray:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.02]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.block_id = p.loadURDF("cube_small.urdf", start_pos, start_orientation)
        p.changeVisualShape(self.block_id, -1, rgbaColor=[0, 0, 1, 1])

        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.1,
            length=0.001,
            rgbaColor=[1, 0, 0, 0.6],
            visualFramePosition=[0, 0, 0],
        )

        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_position,
        )

        return self.get_state()

    def get_state(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.block_id)
        return np.array(pos)

    def apply_push(self, force_vector, record: bool = False, video_path: str = "slide_demo.mp4") -> np.ndarray:
        if record:
            os.makedirs("videos", exist_ok=True)
            self.video_path = os.path.join("videos", video_path)
            self.frames = []
        print(force_vector)
        p.applyExternalForce(
            objectUniqueId=self.block_id,
            linkIndex=-1,
            forceObj=force_vector,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
        )

        for _ in range(240):
            pos, _ = p.getBasePositionAndOrientation(self.block_id)
            self.last_camera_target = pos
            p.resetDebugVisualizerCamera(
                cameraDistance=self.last_camera_distance,
                cameraYaw=self.last_camera_yaw,
                cameraPitch=self.last_camera_pitch,
                cameraTargetPosition=pos,
            )
            p.stepSimulation()

            if self.gui:
                time.sleep(1 / self.speed)

            if record:
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=self.last_camera_target,
                    distance=self.last_camera_distance,
                    yaw=self.last_camera_yaw,
                    pitch=self.last_camera_pitch,
                    roll=0,
                    upAxisIndex=2,
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
                )
                _, _, rgb, _, _ = p.getCameraImage(
                    width=512,
                    height=512,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                )
                frame_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                self.frames.append(frame_bgr)

        if record and self.frames:
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"Video saved to {self.video_path}")

        return self.get_state()

    def render_views(self, topdown_path="topdown.png", side_path="side.png", use_static_side=False):
        # Get block position for tracker view
        block_pos, _ = p.getBasePositionAndOrientation(self.block_id)

        # ---- Top-down camera ----
        view_matrix_top = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0.5, 1],
            cameraTargetPosition=[0.5, 0.5, 0],
            cameraUpVector=[0, 1, 0]
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
        )

        _, _, rgb_top, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix_top,
            projectionMatrix=proj_matrix
        )

        # ---- Side camera ----
        if use_static_side:
            # Static side view from initial position
            view_matrix_side = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.2, 0.2, 0],  # Always look at center
                distance=self.last_camera_distance,
                yaw=self.last_camera_yaw,
                pitch=self.last_camera_pitch,
                roll=0,
                upAxisIndex=2
            )
        else:
            # Tracker camera (match GUI debug camera) - follows the cube
            if self.last_camera_target is None:
                self.last_camera_target = block_pos

            view_matrix_side = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.last_camera_target,
                distance=self.last_camera_distance,
                yaw=self.last_camera_yaw,
                pitch=self.last_camera_pitch,
                roll=0,
                upAxisIndex=2
            )

        _, _, rgb_side, _, _ = p.getCameraImage(
            width=512,
            height=512,
            viewMatrix=view_matrix_side,
            projectionMatrix=proj_matrix,
        )

        Image.fromarray(rgb_top).save(topdown_path)
        Image.fromarray(rgb_side).save(side_path)

    def convert_video_to_gif(self, gif_path: str = "slide_demo.gif") -> None:
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
        p.disconnect()


if __name__ == "__main__":
    env = SlideToGoalEnv(gui=True)
    env.reset()
    env.apply_push([60, 60, 0], record=True, video_path="slide_demo.mp4")
    env.convert_video_to_gif("slide_demo.gif")
    env.close()