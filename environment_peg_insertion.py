import pybullet as p
import pybullet_data
import time
import numpy as np


class PegInsertionEnvironment:
    def __init__(self, gui=True, hz=60):
        """Initialize the peg insertion environment with robot and peg setup."""
        self.hz = hz
        self.gui = gui

        # Connect to simulation
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground and table
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0])
        self.table_height = 0.62

        # Load Franka Panda
        start_pos = [0, 0, self.table_height]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.franka_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_ori, useFixedBase=True)

        # Joint indices for the 7 DOF arm (not including fingers)
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [9, 10]  # Finger joint indices for Franka Panda
        self.eef_index = 11  # End effector link index

        # Default initial joint positions
        initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

        # Move to initial joint positions
        for i, pos in zip(self.joint_indices, initial_joint_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Open the gripper
        gripper_open_positions = [0.04, 0.04]
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Get current EEF pose
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(eef_state[4])  # position
        self.eef_ori = eef_state[5]            # orientation (quaternion)

        # Create and attach peg
        self._create_and_attach_peg()

        # Create insertion hole on table
        self._create_insertion_hole()

        # Close the gripper to hold the peg
        self._close_gripper()

        print("Peg insertion environment initialized successfully!")

    def _create_and_attach_peg(self):
        """Create the peg and attach it to the end effector."""
        # Peg dimensions (rectangular prism)
        self.peg_width = 0.024   # 2.4cm width (x-axis)
        self.peg_depth = 0.024   # 1.2cm depth (y-axis)
        self.peg_height = 0.08   # 8cm height (z-axis)

        # Create peg collision and visual shapes
        peg_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.peg_width/2, self.peg_depth/2, self.peg_height/2]
        )
        peg_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.peg_width/2, self.peg_depth/2, self.peg_height/2],
            rgbaColor=[0, 1, 0, 1]  # Green peg
        )

        # Calculate proper peg position
        peg_start_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] - self.peg_height / 2 + 0.02]

        # Create peg body
        self.peg_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=peg_collision,
            baseVisualShapeIndex=peg_visual,
            basePosition=peg_start_pos,
            baseOrientation=self.eef_ori
        )

        # Create constraint to attach peg to end effector
        self.peg_joint = p.createConstraint(
            parentBodyUniqueId=self.franka_id,
            parentLinkIndex=self.eef_index,
            childBodyUniqueId=self.peg_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, self.peg_height/2 - 0.02],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )

        # Configure constraint for maximum rigidity
        p.changeConstraint(self.peg_joint, maxForce=50000)

        print(f"Rectangular peg properly attached as fixed joint at position: {peg_start_pos}")

    def _create_insertion_hole(self):
        """Create a hollow box (hole) on the table for peg insertion."""
        # Hole dimensions - slightly larger than peg
        clearance = 0.002  # 2mm clearance on each side
        hole_width = self.peg_width + 2 * clearance   # Slightly wider than peg
        hole_depth = self.peg_depth + 2 * clearance   # Slightly deeper than peg
        hole_height = 0.05  # 5cm deep hole
        wall_thickness = 0.02  # 2cm thick walls

        # Position the hole on the table (you can adjust this position)
        hole_pos = [0.4, 0.0, self.table_height + hole_height/2]

        # Create the outer box (walls)
        outer_width = hole_width + 2 * wall_thickness
        outer_depth = hole_depth + 2 * wall_thickness
        
        outer_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[outer_width/2, outer_depth/2, hole_height/2]
        )
        outer_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[outer_width/2, outer_depth/2, hole_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1.0]  # Light gray
        )

        # Create the inner box (hole - this will be removed via compound shape)
        inner_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[hole_width/2, hole_depth/2, hole_height/2 + 0.001]  # Slightly taller to ensure clean cut
        )

        # Create compound shape: outer box minus inner box
        # We'll create 4 walls instead of using compound shapes for better physics
        wall_positions = [
            # Front wall
            [hole_pos[0] + (hole_width/2 + wall_thickness/2), hole_pos[1], hole_pos[2]],
            # Back wall  
            [hole_pos[0] - (hole_width/2 + wall_thickness/2), hole_pos[1], hole_pos[2]],
            # Left wall
            [hole_pos[0], hole_pos[1] + (hole_depth/2 + wall_thickness/2), hole_pos[2]],
            # Right wall
            [hole_pos[0], hole_pos[1] - (hole_depth/2 + wall_thickness/2), hole_pos[2]]
        ]
        
        wall_shapes = [
            # Front and back walls (thin in x, full in y)
            [wall_thickness/2, outer_depth/2, hole_height/2],
            [wall_thickness/2, outer_depth/2, hole_height/2],
            # Left and right walls (full in x, thin in y) 
            [hole_width/2, wall_thickness/2, hole_height/2],
            [hole_width/2, wall_thickness/2, hole_height/2]
        ]

        self.hole_walls = []
        for i, (pos, shape) in enumerate(zip(wall_positions, wall_shapes)):
            wall_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=shape
            )
            wall_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=shape,
                rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Red color
            )
            
            wall_id = p.createMultiBody(
                baseMass=0,  # Static walls
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos
            )
            self.hole_walls.append(wall_id)

        # Store hole information for reference
        self.hole_position = hole_pos
        self.hole_width = hole_width
        self.hole_depth = hole_depth
        self.hole_height = hole_height

    def _close_gripper(self):
        """Close the gripper to hold the peg."""
        gripper_closed_positions = [0.01, 0.01]
        for i, pos in zip(self.gripper_indices, gripper_closed_positions):
            p.setJointMotorControl2(
                self.franka_id,
                i,
                p.POSITION_CONTROL,
                pos,
                force=10,
                positionGain=0.3,
                velocityGain=1.0
            )

        # Let the gripper settle
        for _ in range(10):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

    def move_smooth(self, target, duration=3.0, relative=True, stop_on_contact=False, contact_threshold=0.5):
        """Move the robot with peg using smooth continuous linear interpolation.
        
        Args:
            target: 3D numpy array or list with target coordinates
            duration: Duration of the movement in seconds
            relative: If True, target is offset from current position. If False, target is absolute global coordinates.
            stop_on_contact: If True, stop movement when peg contacts the table
            contact_threshold: Force threshold (N) to detect contact
        """
        # Get current end effector position
        current_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        start_pos = np.array(current_eef_state[4])
        
        # Calculate target position based on relative flag
        if relative:
            target_pos = start_pos + np.array(target)
        else:
            target_pos = np.array(target)

        # Calculate total number of steps for the duration
        total_steps = int(duration * self.hz)
        
        # Check if GUI is enabled for proper sleep timing
        gui_enabled = p.getConnectionInfo()['connectionMethod'] == p.GUI
        
        # Smooth movement with continuous interpolation
        for step in range(total_steps):
            # Check for contact if enabled
            if stop_on_contact:
                total_force = 0
                
                # Check contact with table
                contact_points = p.getContactPoints(bodyA=self.peg_id, bodyB=self.table_id)
                for contact in contact_points:
                    normal_force = contact[9]  # Normal force magnitude
                    total_force += abs(normal_force)
                
                # Check contact with all hole walls
                for wall_id in self.hole_walls:
                    wall_contacts = p.getContactPoints(bodyA=self.peg_id, bodyB=wall_id)
                    for contact in wall_contacts:
                        normal_force = contact[9]  # Normal force magnitude
                        total_force += abs(normal_force)
                
                if total_force > contact_threshold:
                    print(f"Contact detected! Total force: {total_force:.2f}N - Stopping movement")
                    break
            
            # Calculate interpolation factor (0 to 1) with smooth acceleration/deceleration
            t = step / (total_steps - 1) if total_steps > 1 else 1.0
            
            # Apply smooth S-curve (ease-in-out) for more natural motion
            # Using smoothstep function: 3t² - 2t³
            alpha = 3 * t**2 - 2 * t**3
            
            # Calculate current target position along the linear path
            current_target = start_pos + alpha * (target_pos - start_pos)
            
            # Calculate IK for current target position
            joint_positions = p.calculateInverseKinematics(
                self.franka_id, 
                self.eef_index, 
                current_target, 
                self.eef_ori,
                maxNumIterations=100,
                residualThreshold=1e-5
            )

            # Apply joint positions with motor control
            for j, joint_pos in zip(self.joint_indices, joint_positions):
                p.setJointMotorControl2(
                    self.franka_id, 
                    j, 
                    p.POSITION_CONTROL, 
                    joint_pos, 
                    force=200,
                    positionGain=0.1,
                    velocityGain=1.0
                )

            # Step simulation
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        # Update stored end effector position after movement
        final_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(final_eef_state[4])
        self.eef_ori = final_eef_state[5]

    def get_current_position(self):
        """Get the current end effector position (always up-to-date)."""
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        current_pos = np.array(eef_state[4])
        current_ori = eef_state[5]
        
        # Update stored values
        self.eef_pos = current_pos
        self.eef_ori = current_ori
        
        return current_pos

    def render_views(self, save_images=True, image_postfix="peg_insertion"):
        """Render camera views from wrist camera and diagonal side view.
        
        Args:
            save_images: Whether to save images to disk
            image_postfix: Postfix for saved image files
            
        Returns:
            tuple: (wrist_rgb, wrist_depth, side_rgb, side_depth) as numpy arrays
        """
        # Get current end effector state for wrist camera
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        eef_pos = np.array(eef_state[4])
        eef_ori = eef_state[5]  # quaternion
        
        # Convert quaternion to rotation matrix for camera orientation
        eef_rot_matrix = p.getMatrixFromQuaternion(eef_ori)
        eef_rot_matrix = np.array(eef_rot_matrix).reshape(3, 3)
        
        # Wrist camera setup (looking down from end effector, rotated 90 degrees)
        wrist_cam_pos = eef_pos + np.array([-0.1, 0., 0.1])  # 10cm above end effector
        wrist_target = eef_pos + np.array([0, 0, -0.1])   # Looking down
        wrist_up = [1, 0, 0]  # Rotated 90 degrees from [0, 1, 0] to [1, 0, 0]
        
        # Diagonal side camera setup
        side_cam_pos = np.array([0.5, 0.1, self.table_height + 0.2])  # Positioned diagonally
        side_target = np.array([0.4, 0.0, self.table_height])  # Looking at hole position
        side_up = [0, 0, 1]  # Up direction
        
        # Camera parameters for VLM-optimized quality
        width, height = 512, 512  # High resolution 4:3 aspect ratio for better detail
        fov = 55  # Slightly narrower field of view for better focus
        aspect = width / height
        near = 0.01
        far = 10.0
        
        # Compute view and projection matrices for wrist camera
        wrist_view_matrix = p.computeViewMatrix(
            cameraEyePosition=wrist_cam_pos,
            cameraTargetPosition=wrist_target,
            cameraUpVector=wrist_up
        )
        
        # Compute view and projection matrices for side camera
        side_view_matrix = p.computeViewMatrix(
            cameraEyePosition=side_cam_pos,
            cameraTargetPosition=side_target,
            cameraUpVector=side_up
        )
        
        # Projection matrix (same for both cameras)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Enhanced rendering parameters for VLM-optimized quality
        renderer = p.ER_BULLET_HARDWARE_OPENGL
        flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        
        # Optimized lighting setup for VLM processing
        light_direction = [-0.5, -0.5, -1]  # Softer angle from top
        light_color = [1.2, 1.2, 1.2]  # Slightly brighter white light
        light_distance = 1.5
        light_ambient_coeff = 0.4  # Higher ambient for better shadow detail
        light_diffuse_coeff = 0.8  # Strong diffuse lighting
        light_specular_coeff = 0.3  # Reduced specular to avoid glare
        
        # Render wrist camera view with enhanced quality
        wrist_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=wrist_view_matrix,
            projectionMatrix=projection_matrix,
            lightDirection=light_direction,
            lightColor=light_color,
            lightDistance=light_distance,
            lightAmbientCoeff=light_ambient_coeff,
            lightDiffuseCoeff=light_diffuse_coeff,
            lightSpecularCoeff=light_specular_coeff,
            renderer=renderer,
            flags=flags
        )
        
        # Render side camera view with enhanced quality
        side_img = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=side_view_matrix,
            projectionMatrix=projection_matrix,
            lightDirection=light_direction,
            lightColor=light_color,
            lightDistance=light_distance,
            lightAmbientCoeff=light_ambient_coeff,
            lightDiffuseCoeff=light_diffuse_coeff,
            lightSpecularCoeff=light_specular_coeff,
            renderer=renderer,
            flags=flags
        )
        
        # Extract RGB and apply VLM-optimized post-processing
        wrist_rgb = np.array(wrist_img[2]).reshape(height, width, 4)[:, :, :3]  # Remove alpha
        side_rgb = np.array(side_img[2]).reshape(height, width, 4)[:, :, :3]  # Remove alpha
        
        # Apply contrast enhancement and gamma correction for VLM processing
        def enhance_for_vlm(image):
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Apply slight gamma correction for better contrast
            gamma = 1.1
            img_float = np.power(img_float, 1.0/gamma)
            
            # Enhance contrast slightly
            img_float = np.clip((img_float - 0.5) * 1.15 + 0.5, 0.0, 1.0)
            
            # Convert back to uint8
            return (img_float * 255).astype(np.uint8)
        
        wrist_rgb = enhance_for_vlm(wrist_rgb)
        side_rgb = enhance_for_vlm(side_rgb)
        
        # Add view labels to images for VLM clarity
        def add_view_label(image, label_text):
            from PIL import Image, ImageDraw, ImageFont
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            # Calculate text position (top-left corner with padding)
            text_position = (20, 472)
            text_color = (255, 255, 255)  # White text
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=32)
            # Draw main text
            draw.text(text_position, label_text, fill=text_color, font=font)

            # Convert back to numpy array
            return np.array(pil_image)
        
        # Add labels to the images
        wrist_rgb_labeled = add_view_label(wrist_rgb, "WRIST VIEW")
        side_rgb_labeled = add_view_label(side_rgb, "SIDE VIEW")
        
        # Save images if requested
        if save_images:
            import os
            from PIL import Image
            
            # Create directory if it doesn't exist
            os.makedirs("images", exist_ok=True)
            
            # Save labeled RGB images
            Image.fromarray(wrist_rgb_labeled).save(f"images/wrist_{image_postfix}.png")
            Image.fromarray(side_rgb_labeled).save(f"images/side_{image_postfix}.png")

            print(f"Camera views with labels saved to images/ with postfix '{image_postfix}'")

        return wrist_rgb_labeled, side_rgb_labeled

    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot to initial joint positions
        initial_joint_positions = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]
        
        # First, reset joint states
        for i, pos in zip(self.joint_indices, initial_joint_positions):
            p.resetJointState(self.franka_id, i, pos)

        # Reset gripper to open position
        gripper_open_positions = [0.04, 0.04]
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.resetJointState(self.franka_id, i, pos)


        # Apply joint motor control to maintain reset positions
        for j, joint_pos in zip(self.joint_indices, initial_joint_positions):
            p.setJointMotorControl2(
                self.franka_id, 
                j, 
                p.POSITION_CONTROL, 
                joint_pos, 
                force=200,
                positionGain=0.3,
                velocityGain=1.0
            )
        
        # Apply gripper motor control
        for i, pos in zip(self.gripper_indices, gripper_open_positions):
            p.setJointMotorControl2(
                self.franka_id,
                i,
                p.POSITION_CONTROL,
                pos,
                force=10,
                positionGain=0.3,
                velocityGain=1.0
            )
        # Apply motor control to ensure robot actually moves to reset positions
        for _ in range(1):  # Give more time for the robot to settle
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        # Update current EEF pose after robot has actually moved
        eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(eef_state[4])  # position
        self.eef_ori = eef_state[5]            # orientation (quaternion)

        # Reset peg position to be attached to end effector
        if hasattr(self, 'peg_id'):
            peg_start_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] - self.peg_height / 2 + 0.02]
            p.resetBasePositionAndOrientation(self.peg_id, peg_start_pos, self.eef_ori)

        # Close the gripper to hold the peg
        self._close_gripper()

        # Let the simulation settle more
        for _ in range(50):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/self.hz)

        # Final position update to ensure accuracy
        final_eef_state = p.getLinkState(self.franka_id, self.eef_index)
        self.eef_pos = np.array(final_eef_state[4])
        self.eef_ori = final_eef_state[5]

        print("Environment reset to initial state")
        return self.eef_pos

    def disconnect(self):
        """Disconnect from the simulation."""
        p.disconnect()
    
    def close(self):
        """Close the environment (alias for disconnect)."""
        self.disconnect()

    def apply_action(self, offset, initial_env=False):
        self.move_smooth(target=np.array([0.41, 0.01, self.table_height+0.12]), duration=3.0, relative=False)
        if initial_env:
            self.move_smooth(target=np.array([0.0, 0.0, 0.04]), duration=3.0, relative=True)
        else:
            self.move_smooth(target=np.array([offset[0], offset[1], offset[2]]), duration=1.0, relative=True)
            self.move_smooth(
                target=np.array([0.0, 0.0, -0.08]), 
                duration=3.0, 
                relative=True, 
                stop_on_contact=True, 
                contact_threshold=1.0
            )

# Example usage
if __name__ == "__main__":
    # Create environment
    env = PegInsertionEnvironment(gui=True, hz=60)

    env.apply_action(offset=np.array([0.01, 0.0, 0.0]))    
    env.reset()
    env.apply_action(offset=np.array([-0.01, 0.0, 0.0]))    
    # Render camera views after movement
    print("Rendering camera views...")
    wrist_rgb, side_rgb = env.render_views(save_images=True, image_postfix="after_insertion")
    
    # Disconnect
    env.disconnect()
