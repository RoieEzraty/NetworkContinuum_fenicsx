import dolfinx.fem.petsc
import numpy as np

from scipy.interpolate import make_interp_spline
from ufl import TrialFunction, TestFunction
from dolfinx.fem import form, Function
from ufl import dot, grad, dx, inner

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass

# --- Projection, Smoothing and Spline functions ---


def project_scalar_func(func, Funcspace: "FuncspaceClass"):
    """
    Project a scalar UFL expression onto the scalar function space.

    Inputs:
    func - ufl or dolfinx functions

    Outputs:
    proj_func - dolfinx.fem.function.Function
    """
    u = TrialFunction(Funcspace.ScalarFuncSpace)
    v = TestFunction(Funcspace.ScalarFuncSpace)
    a_proj = form(dot(u, v) * dx)
    L_proj = form(dot(func, v) * dx)
    proj_func = Function(Funcspace.ScalarFuncSpace)
    proj_problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=proj_func)
    proj_func = proj_problem.solve()
    return proj_func


def project_vector_func(func, Funcspace: "FuncspaceClass"):
    """
    Project a vector UFL expression onto the vector function space.
    """
    U = TrialFunction(Funcspace.VectorFuncSpace)
    V = TestFunction(Funcspace.VectorFuncSpace)
    a_proj = form(dot(U, V) * dx)
    L_proj = form(dot(func, V) * dx)
    proj_func = Function(Funcspace.VectorFuncSpace)
    # proj_problem = fem.petsc.LinearProblem(a_proj, L_proj, u=VectorFuncSpace)
    proj_problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=proj_func)
    proj_func = proj_problem.solve()
    return proj_func


def project_tensor_func(func, Funcspace: "FuncspaceClass"):
    """
    Project a vector UFL expression onto the vector function space.
    """
    U = TrialFunction(Funcspace.TensorFuncSpace)
    V = TestFunction(Funcspace.TensorFuncSpace)
    a_proj = form(inner(U, V) * dx)
    L_proj = form(inner(func, V) * dx)
    proj_func = Function(Funcspace.TensorFuncSpace)
    # proj_problem = fem.petsc.LinearProblem(a_proj, L_proj, u=VectorFuncSpace)
    proj_problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=proj_func)
    proj_func = proj_problem.solve()
    return proj_func

    V_tensor = FunctionSpace(mesh, ("CG", 1, (2, 2)))  # Tensor space
    Q = TrialFunction(V_tensor)
    V = TestFunction(V_tensor)
    
    a = inner(Q, V) * dx
    L = inner(custom_off_diag_tensor_expr(SpatialCoordinate(mesh)), V) * dx
    
    c_func = Function(V_tensor)
    problem = dolfinx.fem.petsc.LinearProblem(a, L, u=c_func)
    c_func = problem.solve()



def smooth_field(field, Funcspace: "FuncspaceClass", alpha=0.01, space='vector'):
    """
    Helmholtz smoother:  (I – α∇²) ũ  =  u   ⇒
    Solve  (ũ·v + α ∇ũ:∇v) dx  =  (u·v) dx

    Inputs:
    field - dolfinx.fem.function.Function to smooth

    Outputs:
    dolfinx.fem.function.Function smoothed function
    """
    Vh = field.function_space
    u = TrialFunction(Vh)
    v = TestFunction(Vh)

    if space == 'vector':
        u_smooth = Function(Funcspace.VectorFuncSpace)
        a = form(dot(u, v) * dx + alpha * inner(grad(u), grad(v)) * dx)
        L = form(dot(field, v) * dx)
    elif space == 'tensor':
        u_smooth = Function(Funcspace.TensorFuncSpace)
        a = form(inner(u, v) * dx + alpha * inner(grad(u), grad(v)) * dx)
        L = form(inner(field, v) * dx)
    else:
        u_smooth = Function(Funcspace.ScalarFuncSpace)
        a = form(dot(u, v) * dx + alpha * inner(grad(u), grad(v)) * dx)
        L = form(dot(field, v) * dx)
    problem = dolfinx.fem.petsc.LinearProblem(a, L, u=u_smooth)
    return problem.solve()


def spline_over_coords(arr, coords):
    """
    Fit a cubic spline to data over y-coordinates for boundary interpolation.

    Inputs:
    arr    - np.array to approximate by spline
    coords - np.array to approximate arr over

    Outputs:
    Adalike_bc_fn - spline function
    """
    # Build a spline model over y-values
    spline_y = coords[:, 1]
    spline_vals = arr
    spline_model = make_interp_spline(spline_y, spline_vals, k=3)  # cubic spline
    # Create callable boundary condition for interpolation
    # adalike_bc_fn = lambda x: spline_model(x[1])

    def spline_fn(y: np.ndarray) -> float:
        """Return interpolated value from spline model."""
        return spline_model(y[1])

    # adalike_bc_fn = lambda y: spline_model(y)
    return spline_fn


def mean_funcs(func1, func2):
    def func_together(y: np.ndarray) -> float:
        return (func1(y) + func2(y)) / 2
    return func_together