#!/usr/bin/env python
"""
Starter Script for C106B Grasp Planning Lab
Authors: Chris Correa, Riddhi Bagadiaa, Jay Monga
"""
from cv2 import approxPolyDP
import numpy as np
import cv2
import argparse
from utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, triangulate, detect_face_3, COLORS, apply_transform
import trimesh
from policies import GraspingPolicy, AdaptiveGraspingPolicy
from utils.utils import look_at_general
import matplotlib.pyplot as plt
import vedo

try:
    import rospy
    import tf
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Pose
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, CameraInfo
    from baxter_interface import gripper as baxter_gripper
    from intera_interface import gripper as sawyer_gripper
    from path_planner import PathPlanner
    ros_enabled = True
except:
    print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
    ros_enabled = False

def lookup_transform(to_frame, from_frame='base'):
    """
    Returns the AR tag position in world coordinates 

    Parameters
    ----------
    to_frame : string
        examples are: ar_marker_7, nozzle, pawn, ar_marker_3, etc
    from_frame : string
        lets be real, you're probably only going to use 'base'

    Returns
    -------
    :4x4 :obj:`numpy.ndarray` relative pose between frames
    """
    if not ros_enabled:
        print('I am the lookup transform function!  ' \
            + 'You\'re not using ROS, so I\'m returning the Identity Matrix.')
        return np.identity(4)
    listener = tf.TransformListener()
    attempts, max_attempts, rate = 0, 500, rospy.Rate(10)
    tag_rot=[]
    print("entering")
    while attempts < max_attempts:
        try:
            t = listener.getLatestCommonTime(from_frame, to_frame)
            tag_pos, tag_rot = listener.lookupTransform(from_frame, to_frame, t)
            attempts = max_attempts
        except Exception as e:
            print("exception!:", e)
            rate.sleep()
            attempts += 1
    print("exiting")
    rot = rotation_from_quaternion(tag_rot)
    return create_transform_matrix(rot, tag_pos)


def execute_grasp(T_world_grasp, planner, gripper):
    """
    Perform a pick and place procedure for the object. One strategy (which we have
    provided some starter code for) is to
    1. Move the gripper from its starting pose to some distance behind the object
    2. Move the gripper to the grasping pose
    3. Close the gripper
    4. Move up
    5. Place the object somewhere on the table
    6. Open the gripper. 

    As long as your procedure ends up picking up and placing the object somewhere
    else on the table, we consider this a success!

    HINT: We don't require anything fancy for path planning, so using the MoveIt
    API should suffice. Take a look at path_planner.py. The `plan_to_pose` and
    `execute_plan` functions should be useful. If you would like to be fancy,
    you can also explore the `compute_cartesian_path` functionality described in
    http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    
    Parameters
    ----------
    T_world_grasp : 4x4 :obj:`numpy.ndarray`
        pose of gripper relative to world frame when grasping object
    """
    def close_gripper():
        """closes the gripper"""
        gripper.close(block=True)
        rospy.sleep(1.0)

    def open_gripper():
        """opens the gripper"""
        gripper.open(block=True)
        rospy.sleep(1.0)

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return

    # Start with gripper open
    open_gripper()

    R, t = T_world_grasp[:3, :3], T_world_grasp[:3, -1]
    x, y, z = t
    vert_offset = 0.05
    horizontal_offset = 0.08

    def execute_motion(x, y, z, R):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(R)
        poseStamped = PoseStamped()
        poseStamped.pose = pose
        poseStamped.header.frame_id = 'base'
        plan = planner.plan_to_pose(poseStamped)
        planner.execute_plan(plan)

    # Go behind object
    execute_motion(x, y, z + vert_offset, R)
    
    # Then swoop in
    execute_motion(x, y, z, R)

    # Close gripper
    close_gripper()

    # Bring the object up
    execute_motion(x, y, z + vert_offset, R)

    # And over
    execute_motion(x, y + horizontal_offset, z + vert_offset, R)

    # And now place it
    execute_motion(x, y + horizontal_offset, z, R)
    open_gripper()

def correlation_coefficient(window1, window2):
    """
    https://stackoverflow.com/questions/53463087/python-opencv-search-correspondences-of-2-images-with-harris-corner-detection
    """
    product = np.mean((window1 - window1.mean()) * (window2 - window2.mean()))
    stds = window1.std() * window2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def locate_cube(camera_image_topic, camera_info_topic, camera_frame):
    """
    Finds size and pose of cube in field of view.
    We are leaving this very open ended! Feel free to be creative!
    OpenCV will probably be useful. You may want to look for the
    corners of the cube, or its edges. From that, try your best to reconstruct
    the actual size and pose of the cube!

    Parameters
    ----------
    camera_image_topic : string
        ROS topic for camera image
    camera_info_topic : string
        ROS topic for camera info
    camera_frame : string
        Name of TF frame for camera

    Returns
    -------
    :obj:`trimesh.primatives.Box` : trimesh object for reconstructed cube
    """
    # Idea:
    # Find cube corners / face (figure this out)
    # Reconstruct 3d locations of keypoins
    # w/ cube face, compute normal, find 'up' direction in camera frame, and cross to get
    # last axis for cube rotation

    # Idea:
    # Optimization based-approach (similar to eq. 2 on https://arxiv.org/pdf/2004.03691.pdf)
    # - Obj fn: Sum of min distances between observed corners and corners of transformed cube model

    bridge = CvBridge()
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    K = np.array(info.K).reshape((3, 3))

    image = rospy.wait_for_message(camera_image_topic, Image)
    cv_image_1 = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    T_world_camera_1 = lookup_transform(camera_frame)
    T_camera1 = np.linalg.inv(T_world_camera_1)

    raw_input('Move left arm camera and press <enter> when done:')

    image = rospy.wait_for_message(camera_image_topic, Image)
    cv_image_2 = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    T_world_camera_2 = lookup_transform(camera_frame)

    plt.imsave('cv_image_1.png', cv_image_1)
    plt.imsave('cv_image_2.png', cv_image_2)

    # (R, t) of camera 2 w/ camera 1 having (I, 0)
    g_c1c2 = np.matmul(T_camera1, T_world_camera_2)
    R, t = g_c1c2[:3, :3], g_c1c2[:3, 3]

    # Get the largest face in the image
    for c in COLORS:
        kp1, des1 = detect_face_3(cv_image_1, c, vis=True)
        kp2, des2 = detect_face_3(cv_image_2, c, vis=True)
        
        # Match correspondences
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        if len(matches) > 4:
            matches_img = cv2.drawMatches(
                cv_image_1, kp1,
                cv_image_2, kp2,
                matches[:10],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            pts1 = np.array([kp1[mat.queryIdx].pt for mat in matches])
            pts2 = np.array([kp2[mat.trainIdx].pt for mat in matches])
            pts3d = triangulate(pts1, pts2, K, R, t)

            print("Num matches:", len(matches))

            n_iters = 20000
            n_pts = 20
            th = 0.001
            best_err = float('inf')
            best_n = None
            inliers = None
            for _ in range(n_iters):
                idx = np.random.choice(range(len(pts3d)), size=n_pts, replace=False)
                sample = pts3d[idx]

                # Get plane
                plane_center = np.mean(sample, axis=0)
                pts_centered = sample - plane_center
                # Use SVD to solve (pts - center) @ n = 0
                _, _, vh = np.linalg.svd(pts_centered)
                n = vh[-1]

                # Metric
                errs = np.abs(np.matmul(pts_centered, n))
                err = np.mean(errs)
                if err < best_err:
                    best_err = err
                    best_n = n
                    inliers = sample[errs < th]
            plane_center = np.mean(inliers, axis=0)
            pts_centered = inliers - plane_center
            n = best_n

            print("Best error:", best_err)
            print("Num Inliers:", len(inliers))

            # Reverse n so it points away from camera
            if n[-1] < 0:
                n *= -1

            # Length of one side of cube - use mean L1 dist from center
            side_length = np.linalg.norm(pts_centered, ord=1, axis=0).mean()

            # Translate to world coordinates
            n, plane_center = apply_transform(T_world_camera_1, np.array([n, plane_center]))

            # To get pose, assume cube is lying on flat surface
            # Then we can use look_at_general
            init_z = n
            cube_center = plane_center + init_z * side_length / 2
            pose = look_at_general(cube_center, init_z)
            mesh = trimesh.primitives.Box((side_length, side_length, side_length))
            mesh.apply_transform(pose)
            mesh.fix_normals()
            print(pose)
            print(side_length)
            vedo.show([mesh, vedo.Point(pos=(0, 0, 0), r=30)], new=True)
            return mesh
    raise Exception('Cube pose not found. Please retry.')

def parse_args():
    """
    Parses arguments from the user. Read comments for more details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='pawn', help=
        """Which Object you\'re trying to pick up.  Options: nozzle, pawn, cube.  
        Default: pawn"""
    )
    parser.add_argument('-n_vert', type=int, default=1000, help=
        'How many vertices you want to sample on the object surface.  Default: 1000'
    )
    parser.add_argument('-n_facets', type=int, default=32, help=
        """You will approximate the friction cone as a set of n_facets vectors along 
        the surface.  This way, to check if a vector is within the friction cone, all 
        you have to do is check if that vector can be represented by a POSITIVE 
        linear combination of the n_facets vectors.  Default: 32"""
    )
    parser.add_argument('-n_grasps', type=int, default=500, help=
        'How many grasps you want to sample.  Default: 500')
    parser.add_argument('-n_execute', type=int, default=5, help=
        'How many grasps you want to execute.  Default: 5')
    parser.add_argument('-metric', '-m', type=str, default='compute_force_closure', help=
        """Which grasp metric in grasp_metrics.py to use.  
        Options: compute_force_closure, compute_gravity_resistance, compute_robust_force_closure"""
    )
    parser.add_argument('-arm', '-a', type=str, default='right', help=
        'Options: left, right.  Default: right'
    )
    parser.add_argument('-robot', type=str, default='baxter', help=
        """Which robot you're using.  Options: baxter, sawyer.  
        Default: baxter"""
    )
    parser.add_argument('--adaptive_policy', action='store_true', help=
        """Whether or not to use the policy for the adaptive gripper rather
        then the one for the parallel gripper."""
    )
    parser.add_argument('--sim', action='store_true', help=
        """If you don\'t use this flag, you will only visualize the grasps.  This is 
        so you can run this outside of hte lab"""
    )
    parser.add_argument('--debug', action='store_true', help=
        'Whether or not to use a random seed'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        np.random.seed(0)

    if not args.sim:
        # Init rospy node (so we can use ROS commands)
        rospy.init_node('dummy_tf_node')

    if args.obj != 'cube':
        # Mesh loading and pre-processing
        mesh = trimesh.load_mesh("objects/{}.obj".format(args.obj))
        # Transform object mesh to world frame
        T_world_obj = lookup_transform(args.obj) 
        mesh.apply_transform(T_world_obj)
        mesh.fix_normals()
    else:
        camera_frame = ''
        robot = args.robot
        if robot == 'baxter':
            camera_topic = '/cameras/left_hand_camera/image'
            camera_info = '/cameras/left_hand_camera/camera_info'
            camera_frame = '/left_hand_camera'
        elif robot == 'sawyer':
            camera_topic = '/usb_cam/image_raw'
            camera_info = '/usb_cam/camera_info'
            camera_frame = '/usb_cam'
        else:
            print("Unknown robot type!")
            rospy.shutdown()
        mesh = locate_cube(camera_topic, camera_info, camera_frame)

    # This policy takes a mesh and returns the best actions to execute on the robot
    if args.adaptive_policy:
        grasping_policy = AdaptiveGraspingPolicy(
            args.n_vert, 
            args.n_grasps, 
            args.n_execute, 
            args.n_facets, 
            args.metric,
            l_palm=0.015,
            l_proximal=0.05,
            l_tip=0.035
        )
    else:
        grasping_policy = GraspingPolicy(
            args.n_vert, 
            args.n_grasps, 
            args.n_execute, 
            args.n_facets, 
            args.metric
        )

    # Each grasp is represented by T_grasp_world, a RigidTransform defining the 
    # position of the end effector
    if args.adaptive_policy:
        grasp_vertices_total, grasp_poses, grasp_indices, grasp_angles = grasping_policy.top_n_actions(mesh, args.obj)
    else:
        grasp_vertices_total, grasp_poses = grasping_policy.top_n_actions(mesh, args.obj)

    if not args.sim:
        # Execute each grasp on the baxter / sawyer
        if args.robot == "baxter":
            gripper = baxter_gripper.Gripper(args.arm)
            planner = PathPlanner('{}_arm'.format(args.arm))
        elif args.robot == "sawyer":
            gripper = sawyer_gripper.Gripper("right")
            planner = PathPlanner('{}_arm'.format("right"))
        else:
            print("Unknown robot type!")
            rospy.shutdown()

    if not args.adaptive_policy:
        for grasp_vertices, grasp_pose in zip(grasp_vertices_total, grasp_poses):
            grasping_policy.visualize_grasp(mesh, grasp_vertices, grasp_pose)
            if not args.sim:
                repeat = True
                while repeat:
                    execute_grasp(grasp_pose, planner, gripper)
                    repeat = raw_input("repeat? [y|n] ") == 'y'
    else:
        for grasp_vertices, grasp_pose, indices, angles in zip(grasp_vertices_total, grasp_poses, grasp_indices, grasp_angles):
            grasping_policy.visualize_grasp(mesh, grasp_vertices, indices, grasp_pose, angles)
            if not args.sim:
                repeat = True
                while repeat:
                    execute_grasp(grasp_pose, planner, gripper)
                    repeat = raw_input("repeat? [y|n] ") == 'y'
