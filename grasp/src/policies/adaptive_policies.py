#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasping Policy for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
import trimesh

# 106B lab imports
from metrics import (
    compute_force_closure, 
    compute_gravity_resistance,
    compute_robust_force_closure,
    compute_ferrari_canny
)
from utils import length, normalize, find_intersections, find_contacts, find_grasp_vertices, look_at_general, look_at_rotated, create_transform_matrix
from utils import homog_3d, xi_screw, pose_from_homog_3d
from scipy.spatial.transform import Rotation
import vedo

from casadi import Opti, sin, cos, tan, vertcat, mtimes, sumsqr, sum1, dot

# These have not been measured, but should still work
ARM_LENGTH = 0.2
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1
OBJECT_MASS = {'nozzle': .25, 'pawn': .25, 'cube': .25, 'e1': .25}

class AdaptiveGripper():
    def __init__(self, g0, mesh, l_palm=0.075, l_proximal=0.06, l_tip=0.045):
        self.g0 = g0
        self.mesh = mesh
        self.l_palm = l_palm
        self.l_proximal = l_proximal
        self.l_tip = l_tip

    def valid(self, table_height=0, arm_length=ARM_LENGTH):
        """
        Returns if this pose should be accepted.
        """
        # Check collision of table with gripper base and arm
        l_palm_half = self.l_palm / 2
        kpts = np.matmul(self.g0, np.array([
            [0, l_palm_half, 0, 1],
            [0, -l_palm_half, 0, 1],
            [0, 0, -arm_length, 1]
        ]).T)
        heights = kpts[2, :]
        if np.any(np.isnan(heights)):
            return False
        if not np.all(heights > table_height):
            return False
        return True
    
    def joint_positions(self, joint_angles):
        """
        Return a dictionary of the joint positions
        for the given configuration.
        """
        # Z points forward, Y points into left finger, X points down
        # Positive joint angles close the gripper
        # Joint angles of 0 has both fingers pointing straight into Z
        l_palm_half = self.l_palm / 2
        thetas_left, thetas_right = joint_angles[:2], joint_angles[2:]

        # Initial keypoint positions (homogenous)
        left_base = np.array([0, l_palm_half, 0, 1])
        left_joint = np.array([0, l_palm_half, self.l_proximal, 1])
        left_tip = np.array([0, l_palm_half, self.l_proximal + self.l_tip, 1])
        right_base = np.array([0, -l_palm_half, 0, 1])
        right_joint = np.array([0, -l_palm_half, self.l_proximal, 1])
        right_tip = np.array([0, -l_palm_half, self.l_proximal + self.l_tip, 1])

        # Left finger FK
        theta1, theta2 = thetas_left
        g_left_joint1 = homog_3d(xi_screw([1, 0, 0], [0, l_palm_half, 0], 0), theta1)
        g_left_joint2 = homog_3d(xi_screw([1, 0, 0], [0, l_palm_half, self.l_proximal], 0), theta2)

        g01_l = np.matmul(self.g0, g_left_joint1)
        g02_l = np.matmul(g01_l, g_left_joint2)

        # Right finger FK
        theta1, theta2 = thetas_right
        g_right_joint1 = homog_3d(xi_screw([-1, 0, 0], [0, -l_palm_half, 0], 0), theta1)
        g_right_joint2 = homog_3d(xi_screw([-1, 0, 0], [0, -l_palm_half, self.l_proximal], 0), theta2)

        g01_r = np.matmul(self.g0, g_right_joint1)
        g02_r = np.matmul(g01_r, g_right_joint2)

        return {
            'l_tip': np.matmul(g02_l, left_tip)[:3],
            'l_joint': np.matmul(g01_l, left_joint)[:3],
            'l_base': np.matmul(self.g0, left_base)[:3],
            'r_tip': np.matmul(g02_r, right_tip)[:3],
            'r_joint': np.matmul(g01_r, right_joint)[:3],
            'r_base': np.matmul(self.g0, right_base)[:3],
        }

    def check_collisions(self, joint_angles, indices=[0, 1, 2, 3, 4]):
        """
        Find collisions between the mesh and the gripper
        with the given joint angles.
        """
        positions = self.joint_positions(joint_angles)
        ret = []
        if 0 in indices:
            ret.append(find_contacts(self.mesh, positions['l_base'], positions['r_base']))
        if 1 in indices:
            ret.append(find_contacts(self.mesh, positions['l_base'], positions['l_joint']))
        if 2 in indices:
            ret.append(find_contacts(self.mesh, positions['l_joint'], positions['l_tip']))
        if 3 in indices:
            ret.append(find_contacts(self.mesh, positions['r_base'], positions['r_joint']))
        if 4 in indices:
            ret.append(find_contacts(self.mesh, positions['r_joint'], positions['r_tip']))
        return ret

    def contact_points(self, theta_min=None, theta_max=None, n_samples=32, pt_th=1e-4):
        """
        Find the contact points between the mesh and
        this gripper as it closes. Assumes object is concave.
        """
        # TODO: Fix issue where if initial pose intersects with mesh, will return an invalid grasp
        if theta_min is None:
            theta_min = [-np.pi / 2., -np.pi / 2.]
        if theta_max is None:
            theta_max = [np.pi / 2., 3. / 4. * np.pi]
        
        th1_min, th2_min = theta_min
        th1_max, th2_max = theta_max

        # Check base intersections
        collisions = self.check_collisions([th1_min, th2_min, th1_min, th2_min])
        (base, proximal_left, tip_left, proximal_right, tip_right) = collisions
        if (base[-1]
            or proximal_left[-1]
            or proximal_right[-1]
            or tip_left[-1]
            or tip_right[-1]):
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Move joint 1:
        indices = [1, 2, 3, 4]
        for th1 in np.linspace(th1_min, th1_max, num=n_samples):
            collisions = self.check_collisions([th1, th2_min, th1, th2_min], indices=indices)
            (proximal_left, tip_left, proximal_right, tip_right) = collisions
            # Check limb intersections
            if (proximal_left[-1]
                or proximal_right[-1]
                or tip_left[-1]
                or tip_right[-1]):
                break
        
        # Move left joint 2
        indices = [2]
        for th2_left in np.linspace(th2_min, th2_max, num=n_samples):
            collisions = self.check_collisions([th1, th2_left, th1, th2_min], indices=indices)
            tip_left = collisions[0]
            if tip_left[-1]:
                break
        
        # Move right joint 2
        indices = [4]
        for th2_right in np.linspace(th2_min, th2_max, num=n_samples):
            collisions = self.check_collisions([th1, th2_left, th1, th2_right], indices=indices)
            tip_right = collisions[0]
            if tip_right[-1]:
                break
        
        joint_angles = np.array([th1, th2_left, th1, th2_right])
        collisions = self.check_collisions([th1, th2_left, th1, th2_right])
        # Skip 2 close collisions (due to discrete steps of theta)
        contact_points = []
        contact_faces = []
        indices = []
        for i, (pts, faces, intersects) in enumerate(collisions):
            # Will return None if no collisions
            if faces is None:
                faces = []
            for pt, face in zip(pts, faces):
                if len(contact_points) > 0:
                    dists = np.linalg.norm(np.array(contact_points) - pt, axis=1)
                    if np.any(dists < pt_th):
                        continue
                contact_points.append(pt)
                contact_faces.append(face)
                indices.append(i)
        return np.array(contact_points), np.array(contact_faces), np.array(indices), joint_angles


class AdaptiveGraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name, **gripper_args):
        """
        Parameters
        ----------
        n_vert : int
            We are sampling vertices on the surface of the object.
        n_grasps : int
            how many grasps to sample.
        n_execute : int
            how many grasps to return in policy.action()
        n_facets : int
            how many facets should be used to approximate the friction cone between the 
            finger and the object
        metric_name : string
            name of one of the function in src/metrics/metrics.py
        """
        self.n_vert = n_vert
        self.n_grasps = n_grasps
        self.n_execute = n_execute
        self.n_facets = n_facets
        # This is a function, one of the functions in src/metrics/metrics.py
        self.metric = eval(metric_name)
        self.gripper_args = gripper_args

    def sample_grasps(self, vertices, normals, mesh):
        """
        Samples a bunch of candidate grasps points.

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals

        Returns
        -------
        n_graspsx4x4 :obj:`numpy.ndarray`
            grasps poses. Each grasp is the 4x4 homogenous transformation matrix of
            the center of the gripper base.
        """
        grasp_poses = []
        table_height = np.min(vertices[:, -1])
        n_verts = len(vertices)
        while len(grasp_poses) < self.n_grasps:
            # Sample random index
            idx = np.random.randint(0, high=n_verts)
            p, n = vertices[idx], normals[idx]

            # Look at normal, with random orientation
            th = np.random.random_sample() * 2. * np.pi
            g = look_at_rotated(p, -n, th)

            # If good, add to list of grasps
            gripper = AdaptiveGripper(g, mesh, **self.gripper_args)
            if gripper.valid(table_height=table_height):
                grasp_poses.append(g)
        return np.array(grasp_poses)

    def score_grasps(self, grasp_poses, object_mass, mesh):
        """
        Takes mesh and returns pairs of contacts and the quality of grasp between the contacts, sorted by quality
        
        Parameters
        ----------
        grasp_poses : n_graspsx4x4 :obj:`numpy.ndarray`
            grasps. Each grasp is the 4x4 homogenous transformation matrix of
            the center of the gripper base.

        Returns
        -------
        scores : `list` of int
            grasp quality for each 
        verts : contact points for each grasp
        """
        scores, verts, vert_indices, grasp_angles = [], [], [], []
        for g in grasp_poses:
            gripper = AdaptiveGripper(g, mesh, **self.gripper_args)
            contact_points, contact_faces, indices, joint_angles = gripper.contact_points()
            verts.append(contact_points)
            vert_indices.append(indices)
            grasp_angles.append(joint_angles)
            if len(contact_points) < 2:
                scores.append(-float('inf'))
            else:
                scores.append(self.metric(contact_points, mesh.face_normals[contact_faces], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, mesh))
        scores = np.array(scores).astype(np.float64)
        return scores, np.array(verts), np.array(vert_indices), np.array(grasp_angles)
        
    def visualize_grasp(self, mesh, vertices, indices, pose, angles):
        """Visualizes a grasp on an object. Object specified by a mesh, as
        loaded by trimesh. vertices is a set of (x, y, z) contact points.
        pose is the pose of the gripper base.
        Parameters
        ----------
        mesh (trimesh.base.Trimesh): mesh of the object
        vertices (np.ndarray): 2x3 matrix, coordinates of the 2 contact points
        pose (np.ndarray): 4x4 homogenous transform matrix
        """
        center = pose[:3, 3]
        approach = pose[:3, 2]
        tail = center - ARM_LENGTH * approach

        contact_points = []
        colors = ['r', 'g', 'b', 'p', 'y']
        for v, i in zip(vertices, indices):
            contact_points.append(vedo.Point(pos=v, r=30, c=colors[i]))

        print("Joint Angles:", angles)
        if len(angles) <= 0:
            print("Not a valid grasp!")
            return
        gripper = AdaptiveGripper(pose, mesh, **self.gripper_args)
        joint_positions = gripper.joint_positions(angles)
        base = vedo.shapes.Tube([joint_positions['l_base'], joint_positions['r_base']], r=0.001, c='r')
        l_proximal = vedo.shapes.Tube([joint_positions['l_base'], joint_positions['l_joint']], r=0.001, c='g')
        l_tip = vedo.shapes.Tube([joint_positions['l_joint'], joint_positions['l_tip']], r=0.001, c='b')
        r_proximal = vedo.shapes.Tube([joint_positions['r_base'], joint_positions['r_joint']], r=0.001, c='p')
        r_tip = vedo.shapes.Tube([joint_positions['r_joint'], joint_positions['r_tip']], r=0.001, c='y')

        approach = vedo.shapes.Tube([center, tail], r=0.001, c='g')
        vedo.show([mesh, approach, base, l_proximal, l_tip, r_proximal, r_tip] + contact_points, new=True)

    def top_n_actions(self, mesh, obj_name):
        """
        Takes in a mesh, samples a bunch of grasps on the mesh, evaluates them using the 
        metric given in the constructor, and returns the best grasps for the mesh.  SHOULD
        RETURN GRASPS IN ORDER OF THEIR GRASP QUALITY.

        Parameters
        ----------
        mesh : :obj:`Trimesh`

        Returns
        -------
        :obj:`list` of :obj:Pose
            the matrices T_world_grasp, which represents the hand poses of the baxter / sawyer
            which would result in the fingers being placed at the vertices of the best grasps

        RETURNS LIST OF LISTS
        """
        # Some objects have vertices in odd places, so you should sample evenly across 
        # the mesh to get nicer candidate grasp points using trimesh.sample.sample_surface_even()
        all_vertices, all_poses, all_indices, all_angles, all_scores = [], [], [], [], []

        assert len(all_vertices) == len(all_poses)
        vertices, face_ind = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        normals = mesh.face_normals[face_ind]

        grasp_poses = self.sample_grasps(vertices, normals, mesh)
        mass = OBJECT_MASS[obj_name]
        grasp_qualities, grasp_vertices, grasp_indices, grasp_angles = self.score_grasps(grasp_poses, mass, mesh)

        # This is the poses of the grasps with the highest grasp qualities. Should be shape:
        # n_executex4x4
        top_grasp_ind = np.argsort(-grasp_qualities)[:self.n_execute]
        top_n_grasps = grasp_poses[top_grasp_ind]
        top_n_grasp_verts = grasp_vertices[top_grasp_ind]
        top_n_grasp_indices = grasp_indices[top_grasp_ind]
        top_n_grasp_angles = grasp_angles[top_grasp_ind]
        top_n_scores = grasp_qualities[top_grasp_ind]

        all_vertices.extend(top_n_grasp_verts)
        all_poses.extend(top_n_grasps)
        all_indices.extend(top_n_grasp_indices)
        all_angles.extend(top_n_grasp_angles)
        all_scores.extend(top_n_scores)

        return all_vertices, all_poses, all_indices, all_angles, all_scores
