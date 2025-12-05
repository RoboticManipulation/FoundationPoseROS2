import sys
import os

# CRITICAL: Set before any torch/CUDA imports to help with memory fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
import argparse
import os
from scipy.spatial.transform import Rotation as R
from cam_2_base_transform import *
import glob
import torch
import time

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register

# Modify `__init__` to add `is_register` attribute
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False  # Initialize as False
    self.last_pose = None  # Track the last valid pose

# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)

    # Check if pose is valid: non-None, not identity, and has non-zero translation
    if pose is not None:
        translation_norm = np.linalg.norm(pose[:3, 3])
        # A valid pose should have non-zero translation and the pose should be different from identity
        if translation_norm > 0.01:  # At least 1cm from origin
            self.is_register = True
            self.last_pose = pose.copy()
        else:
            self.is_register = False
    else:
        self.is_register = False

    return pose

# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# Argument Parser
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=1)  # Minimal refinement to save memory
parser.add_argument('--track_refine_iter', type=int, default=1)  # Minimal refinement to save memory
args = parser.parse_args()

class PoseEstimationNode(Node):
    def __init__(self, mesh_path):
        super().__init__('pose_estimation_node')

        # ROS subscriptions and publishers
        # Simulation camera topics (Isaac Sim)
        # self.image_sub = self.create_subscription(Image, '/sim_camera_rgb', self.image_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/sim_camera_depth', self.depth_callback, 10)
        # self.info_sub = self.create_subscription(CameraInfo, '/sim_camera_info', self.camera_info_callback, 10)
        
        # Real camera topics (RealSense)
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_rect_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None  # Initialize cam_K as None until we receive the camera info

        # Load single mesh for mustard0
        self.mesh = trimesh.load(mesh_path)
        self.get_logger().info(f"Loaded mesh: {mesh_path}")

        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = to_origin
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.pose_est = None  # Single pose estimator
        self.pose_pub = self.create_publisher(PoseStamped, '/mustard0_pose', 10)
        self.is_initialized = False
        self.is_processing = False  # Lock to prevent concurrent processing
        self.frame_count = 0  # Frame counter for skipping
        self.last_oom_time = 0  # Track last OOM error time

    def camera_info_callback(self, msg):
        if self.cam_K is None:  # Update cam_K only once to avoid redundant updates
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_callback(self, msg):
        try:
            # Convert depth image
            raw_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Log raw depth info on first callback
            if not hasattr(self, '_depth_logged'):
                self.get_logger().info(f"Raw depth encoding: {msg.encoding}")
                self.get_logger().info(f"Raw depth shape: {raw_depth.shape}")
                self.get_logger().info(f"Raw depth dtype: {raw_depth.dtype}")
                self.get_logger().info(f"Raw depth stats: min={raw_depth.min()}, max={raw_depth.max()}, mean={raw_depth.mean()}")
                self._depth_logged = True

            # Ensure depth is single-channel (some encodings return 3 channels)
            if len(raw_depth.shape) == 3:
                self.get_logger().info(f"Depth has {raw_depth.shape[2]} channels, extracting first channel")
                raw_depth = raw_depth[:, :, 0]

            # Convert based on encoding
            if msg.encoding == "32FC1":
                # Already in meters (float32)
                self.depth_image = raw_depth
            elif msg.encoding == "16UC1":
                # Millimeters (uint16), convert to meters
                self.depth_image = raw_depth.astype(np.float32) / 1000.0
            else:
                # Try generic conversion
                self.depth_image = raw_depth.astype(np.float32) / 1000.0

            self.process_images()
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def process_images(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        # Prevent concurrent processing
        if self.is_processing:
            return

        # During initialization, only process every 20th frame to reduce memory pressure
        self.frame_count += 1
        if not self.is_initialized and self.frame_count % 20 != 0:
            return

        self.is_processing = True

        try:
            H, W = self.color_image.shape[:2]
            color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.1) | (depth >= np.inf)] = 0

            # Initialize pose estimator on first frame
            if not self.is_initialized:
                # If we recently had an OOM error, wait longer before retrying
                time_since_oom = time.time() - self.last_oom_time
                if self.last_oom_time > 0 and time_since_oom < 5.0:
                    self.get_logger().info(f"Waiting {5.0 - time_since_oom:.1f}s after OOM before retry...")
                    return

                # Aggressively clear CUDA cache before starting heavy initialization
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.5)  # Give GPU time to clean up

                self.get_logger().info("Creating large mask covering most of the image...")

                # Create a large mask covering 80% of the image (avoiding only edges)
                # This lets FoundationPose search the entire frame for the mustard bottle
                margin_h = H // 10  # 10% margin on top/bottom
                margin_w = W // 10  # 10% margin on left/right

                mask = np.zeros((H, W), dtype=np.uint8)
                mask[margin_h:H-margin_h, margin_w:W-margin_w] = 255

                # Count valid depth points in the mask
                valid_depth_count = np.sum((depth > 0.1) & (depth < 5.0) & (mask == 255))
                self.get_logger().info(f"Valid depth points in mask: {valid_depth_count}")

                if valid_depth_count < 100:
                    self.get_logger().warn("Not enough valid depth points. Retrying...")
                    return

                # Initialize FoundationPose with the mesh (only once)
                if self.pose_est is None:
                    self.pose_est = FoundationPose(
                        model_pts=self.mesh.vertices,
                        model_normals=self.mesh.vertex_normals,
                        mesh=self.mesh,
                        scorer=self.scorer,
                        refiner=self.refiner,
                        glctx=self.glctx
                    )

                    # Reduce the number of candidate poses to save memory
                    # Default generates 252 candidates, we'll use fewer viewpoints
                    try:
                        self.pose_est.make_rotation_grid(min_n_views=16, inplane_step=120)
                        self.get_logger().info("Reduced rotation grid to save GPU memory")
                    except Exception as e:
                        self.get_logger().warn(f"Could not reduce rotation grid: {e}")

                try:
                    # Register the initial pose with center mask
                    pose = self.pose_est.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

                    # Check if registration actually succeeded using the is_register flag
                    if self.pose_est.is_register:
                        self.is_initialized = True
                        self.get_logger().info("✓ Pose initialized successfully! Now tracking...")

                        # Save the successful mask for debugging
                        cv2.imwrite("initialization_mask.png", mask)
                    else:
                        self.get_logger().warn("✗ Registration failed. Will retry on next frame...")
                        return
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.last_oom_time = time.time()
                        self.get_logger().error("OOM during initialization. Will retry after 5s cooldown...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        return
                    else:
                        raise
                finally:
                    # Always clear CUDA cache after attempt
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # Track the pose
            if self.pose_est.is_register:
                pose = self.pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(self.to_origin)

                # Publish the pose
                self.publish_pose_stamped(center_pose)

                # Visualize the pose
                visualization_image = self.visualize_pose(color, center_pose)
                cv2.imshow('Mustard0 Pose Estimation', visualization_image[..., ::-1])
                cv2.waitKey(1)

                # Clear CUDA cache periodically during tracking
                if self.frame_count % 30 == 0:
                    torch.cuda.empty_cache()

        finally:
            # Always release the processing lock
            self.is_processing = False

    def visualize_pose(self, image, center_pose):
        vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

    def publish_pose_stamped(self, center_pose):
        # Convert the center_pose matrix to a PoseStamped message
        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = "mustard0_frame"

        # Convert center_pose to the pose format
        position = center_pose[:3, 3]
        rotation_matrix = center_pose[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Combine position and quaternion into a single array
        pose_array = np.concatenate((position, quaternion))

        # Apply transformation to convert from camera to base frame
        transformed_pose = transformation(pose_array)

        # Populate PoseStamped message with transformed pose
        pose_stamped_msg.pose.position.x = transformed_pose[0]
        pose_stamped_msg.pose.position.y = transformed_pose[1]
        pose_stamped_msg.pose.position.z = transformed_pose[2]

        pose_stamped_msg.pose.orientation.w = transformed_pose[3]
        pose_stamped_msg.pose.orientation.x = transformed_pose[4]
        pose_stamped_msg.pose.orientation.y = transformed_pose[5]
        pose_stamped_msg.pose.orientation.z = transformed_pose[6]

        # Publish the transformed pose
        self.pose_pub.publish(pose_stamped_msg)

def main(args=None):
    source_directory = "demo_data"

    # Find the mustard0 mesh file
    file_paths = glob.glob(os.path.join(source_directory, '**', '*mustard*.obj'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*mustard*.stl'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*mustard*.STL'), recursive=True)

    if not file_paths:
        print("Error: No mustard mesh file found in demo_data directory")
        return

    # Use the first mustard mesh found
    mustard_mesh_path = file_paths[0]
    print(f"Using mesh: {mustard_mesh_path}")

    rclpy.init(args=args)
    node = PoseEstimationNode(mustard_mesh_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
