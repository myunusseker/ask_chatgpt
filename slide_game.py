import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import cv2
import imageio

class SlideToGoalEnv:
    def __init__(self, gui=True):
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
        self.last_camera_distance = 0.7
        self.last_camera_yaw = -45
        self.last_camera_pitch = -45

    def reset(self):
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
            visualFramePosition=[0, 0, 0]
        )

        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_position
        )

        return self.get_state()

    def get_state(self):
        pos, _ = p.getBasePositionAndOrientation(self.block_id)
        return np.array(pos)

    def apply_push(self, force_vector, record=False, video_path="slide_demo.mp4"):
        if record:
            os.makedirs("videos", exist_ok=True)
            self.video_path = os.path.join("videos", video_path)
            self.frames = []

        p.applyExternalForce(
            objectUniqueId=self.block_id,
            linkIndex=-1,
            forceObj=force_vector,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME
        )

        for _ in range(240):
            pos, _ = p.getBasePositionAndOrientation(self.block_id)
            self.last_camera_target = pos
            p.resetDebugVisualizerCamera(
                cameraDistance=self.last_camera_distance,
                cameraYaw=self.last_camera_yaw,
                cameraPitch=self.last_camera_pitch,
                cameraTargetPosition=pos
            )
            p.stepSimulation()

            if self.gui:
                time.sleep(1 / 30)

            if record:
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=self.last_camera_target,
                    distance=self.last_camera_distance,
                    yaw=self.last_camera_yaw,
                    pitch=self.last_camera_pitch,
                    roll=0,
                    upAxisIndex=2
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
                frame_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                self.frames.append(frame_bgr)

        if record:
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"Video saved to {self.video_path}")

        return self.get_state()

    def convert_video_to_gif(self, gif_path="slide_demo.gif"):
        if hasattr(self, 'video_path') and os.path.exists(self.video_path):
            gif_path_full = os.path.join("videos", gif_path)
            reader = imageio.get_reader(self.video_path)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(gif_path_full, fps=fps, loop=0)  # <- loop=0 makes it loop forever
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            print(f"GIF saved to {gif_path_full}")
        else:
            print("Video not found or not recorded yet.")

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    env = SlideToGoalEnv(gui=True)
    env.reset()
    env.apply_push([60, 60, 0], record=True, video_path="slide_demo.mp4")
    env.convert_video_to_gif("slide_demo.gif")
    env.close()
