# !/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for C106B Grasp Planning Lab
Author: Chris Correa
"""
import numpy as np
from scipy.linalg import expm
from utils import vec, adj, normalize, look_at_general, find_grasp_vertices, normal_at_point
from casadi import Opti, sin, cos, tan, vertcat, mtimes, sumsqr, sum1, dot
from trimesh.proximity import ProximityQuery

# Can edit to make grasp point selection more/less restrictive
MAX_GRIPPER_DIST = .075
MIN_GRIPPER_DIST = .03
GRAVITY = 9.80665

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Compute the force closure of some object at contacts, with normal vectors 
    stored in normals. You can use the line method described in the project document.
    If you do, you will not need num_facets. This is the most basic (and probably least useful)
    grasp metric.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float : 1 or 0 if the grasp is/isn't force closure for the object
    """
    normal0 = -1.0 * normals[0] / (1.0 * np.linalg.norm(normals[0]))
    normal1 = -1.0 * normals[1] / (1.0 * np.linalg.norm(normals[1]))

    alpha = np.arctan(mu)
    line = vertices[0] - vertices[1]
    line = line / (1.0 * np.linalg.norm(line))
    angle1 = np.arccos(normal1.dot(line))

    line = -1 * line
    angle2 = np.arccos(normal0.dot(line))

    if angle1 > alpha or angle2 > alpha:
        return 0
    if gamma == 0:
        return 0
    return 1

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """ 
    Defined in the book on page 219. Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient

    Returns
    -------
    6x8 :obj:`numpy.ndarray` : grasp map
    """
    # raise NotImplementedError
    B = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]])
    G = []
    for v, n in zip(vertices, normals):
        g_oc = look_at_general(v, -n)
        adj_g = adj(np.linalg.inv(g_oc))
        G.append(np.matmul(adj_g.T, B))
    return np.hstack(G)

def find_contact_forces(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute that contact forces needed to produce the desired wrench

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : 6x :obj:`numpy.ndarray` potential wrench to be produced

    Returns
    -------
    bool: whether contact forces can produce the desired_wrench on the object
    """
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    f = np.matmul(np.linalg.pinv(G), desired_wrench)
    # Check if f in cone
    for (f1, f2, f3, f4) in [f[:4], f[4:]]:
        if not (
            f3 > 0 and
            abs(f4) <= gamma * f3 and
            np.linalg.norm([f1, f2]) < mu * f3
        ):
            return False
    return True

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Gravity produces some wrench on your object. Computes how much normal force is required
    to resist the wrench produced by gravity.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    n_verts = len(vertices)
    f_dim = 4 * n_verts
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    wrench = np.array([0, 0, -GRAVITY * object_mass, 0, 0, 0])
    opti = Opti()

    # Define vars
    f = opti.variable(f_dim, 1)

    # Define objective func
    obj = mtimes(f.T, f)
    opti.minimize(obj)

    # Define contraints
    # Rotation matrix about z
    def R(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    # Find facet edges
    v = np.array([mu, 0, 1])
    thetas = np.linspace(0, 2 * np.pi, num=num_facets, endpoint=False)
    vs = [np.matmul(R(theta), v) for theta in thetas]
    
    # Find facet normals
    facet_normals = [normalize(np.cross(vs[-1], vs[0]))]
    for i in range(num_facets - 1):
        v1, v2 = vs[i], vs[i + 1]
        facet_normals.append(normalize(np.cross(v1, v2)))

    constraints = []
    for i in range(n_verts):
        constraints.extend([dot(n, f[4*i:4*i+3]) > 0 for n in facet_normals])
        constraints.append(f[4*i+3] <= gamma * f[4*i+2])
    constraints.append(mtimes(G, f) == -wrench)
    opti.subject_to(constraints)

    # Set initial conditions
    opti.set_initial(f, np.zeros((f_dim, 1)))
    
    # Construct and solve
    opti.solver('ipopt')
    p_opts = {"expand": False, "print_time": False, "verbose": False}
    s_opts = {"max_iter": 1e4, "print_level": 0}

    opti.solver('ipopt', p_opts, s_opts)
    try:
        sol = opti.solve()
        f_sol = sol.value(f)
        # TODO: Double-check - should this be sum or mean?
        return -sum([f_sol[4*i+2] for i in range(n_verts)])
    except:
        return -float('inf')


def compute_robust_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the robust force closure metric.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    num_samples = 10
    var = 0.01 #TODO tune variance

    plane_bases = []
    for n in normals:
        up = np.array([0, 0, 1])
        a = normalize(np.cross(up, n))
        b = normalize(np.cross(a, n))
        plane_bases.append(np.array([a, b]).T)

    proximityQuery = ProximityQuery(mesh)
    def sample():
        deltas = np.random.normal(loc=0, scale=(np.ones((2, 2)) * var))
        # Perturb contacts with points from normal plane
        verts_perturbed = []
        for v, d, b in zip(vertices, deltas, plane_bases):
            verts_perturbed.append(v + np.matmul(b, d))
        closest, _, _ = proximityQuery.on_surface(verts_perturbed)
        normals = normal_at_point(mesh, closest[0]), normal_at_point(mesh, closest[1])
        # Force closure at perturbed point
        return compute_force_closure(closest, normals, num_facets, mu, gamma, object_mass, mesh)

    samples = [sample() for _ in range(num_samples)]
    return np.mean(samples)


def compute_ferrari_canny(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the Ferrari Canny metric.
    Use your favourite python convex optimization package. We suggest casadi.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object

    Returns
    -------
    float: quality of the grasp
    """
    num_samples = 100
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)

    opti = Opti()

    # Define vars
    f = opti.variable(8, 1)

    # Define objective func
    obj = mtimes(f.T, f)
    opti.minimize(obj)

    # Define contraints
    # Rotation matrix about z
    def R(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    # Find facet edges
    v = np.array([mu, 0, 1])
    thetas = np.linspace(0, 2 * np.pi, num=num_facets, endpoint=False)
    vs = [np.matmul(R(theta), v) for theta in thetas]

    # Find facet normals
    facet_normals = [np.cross(vs[-1], vs[0])]
    for i in range(num_facets - 1):
        v1, v2 = vs[i], vs[i + 1]
        facet_normals.append(normalize(np.cross(v1, v2)))

    constraints = []
    constraints.extend([dot(n, f[0:3]) > 0 for n in facet_normals])
    constraints.extend([dot(n, f[4:7]) > 0 for n in facet_normals])
    constraints.append(f[3] <= gamma * f[2])
    constraints.append(f[7] <= gamma * f[6])
    opti.subject_to(constraints)

    # Set initial conditions
    opti.set_initial(f, np.zeros((8, 1)))

    # Construct and solve
    opti.solver('ipopt')
    p_opts = {"expand": False}
    s_opts = {"max_iter": 1e4}

    opti.solver('ipopt', p_opts, s_opts)

    def LQ(omega):
        if not find_contact_forces(vertices, normals, num_facets, mu, gamma, omega):
            return float('inf')

        lq_opti = opti.copy()
        lq_opti.subject_to(mtimes(G, f) == omega)

        try:
            sol = lq_opti.solve()
            f_sol = sol.value(f)
            return np.linalg.norm(f_sol)
        except:
            return float('inf')
    
    omegas = np.random.random_sample(size=(num_samples, 6))
    lqs = np.array([LQ(normalize(omega)) for omega in omegas])
    lqs = lqs[np.isfinite(lqs)]
    if len(lqs) > 0:
        lq_max = np.max(lqs)
        return 1 / lq_max
    return -float('inf')
