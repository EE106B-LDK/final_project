#!/usr/bin/env python
"""
Starter Script for C106B Grasp Planning Lab
Authors: Chris Correa, Riddhi Bagadiaa, Jay Monga
"""
import numpy as np
import argparse
from utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, triangulate, detect_face_3, COLORS, apply_transform
import trimesh
from policies import AdaptiveGraspingPolicy
from utils.utils import look_at_general
import matplotlib.pyplot as plt
import vedo
from scipy.optimize import differential_evolution, rosen
import time

def parse_args():
    """
    Parses arguments from the user. Read comments for more details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='pawn', help=
        """Which Object you\'re trying to pick up.  Options: nozzle, pawn.  
        Default: pawn"""
    )
    parser.add_argument('-n_vert', type=int, default=1000, help=
        'How many vertices you want to sample on the object surface.  Default: 1000'
    )
    parser.add_argument('-n_iter', type=int, default=100, help=
        'Maximum number of DE iterations to run.  Default: 200'
    )
    parser.add_argument('-n_facets', type=int, default=32, help=
        """You will approximate the friction cone as a set of n_facets vectors along 
        the surface.  This way, to check if a vector is within the friction cone, all 
        you have to do is check if that vector can be represented by a POSITIVE 
        linear combination of the n_facets vectors.  Default: 32"""
    )
    parser.add_argument('-n_grasps', type=int, default=200, help=
        'How many grasps you want to sample.  Default: 500')
    parser.add_argument('-metric', '-m', type=str, default='compute_gravity_resistance', help=
        """Which grasp metric in grasp_metrics.py to use.  
        Options: compute_force_closure, compute_gravity_resistance, compute_robust_force_closure"""
    )
    parser.add_argument('--debug', action='store_true', help=
        'Whether or not to use a random seed'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        np.random.seed(0)

    # Mesh loading and pre-processing
    mesh = trimesh.load_mesh("objects/{}.obj".format(args.obj))
    # Transform object mesh to world frame
    T_world_obj = np.identity(4)
    mesh.apply_transform(T_world_obj)
    mesh.fix_normals()

    # Function to minimize with DE
    def gripper_eval(params):
        """
        A score for the given set of gripper parameters.
        Lower is better.
        """
        l_palm, l_proximal, l_tip = params
        policy = AdaptiveGraspingPolicy(
            args.n_vert, 
            args.n_grasps, 
            1, 
            args.n_facets, 
            args.metric,
            l_palm=l_palm,
            l_proximal=l_proximal,
            l_tip=l_tip
        )
        _, _, _, _, scores = policy.top_n_actions(mesh, args.obj)
        return -scores[0]

    f = open('de_results.txt', 'a')
    def de_callback(xk, convergence):
        f.write('%s, %d\n' % (str(xk), convergence))
    
    bounds = [(1e-3, 0.1), (1e-3, 0.1), (1e-3, 0.1)]
    start = time.time()
    result = differential_evolution(
        gripper_eval,
        bounds,
        maxiter=args.n_iter,
        popsize=1,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=False,
        disp=True,
        callback=de_callback,
        workers=-1,
        updating='deferred')
    print(result)
    print("Runtime (s):", time.time() - start)
    f.close()
