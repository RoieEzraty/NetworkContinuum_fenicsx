import dolfinx.fem.petsc
import numpy as np
import copy

from ufl import TrialFunction, TestFunction
from ufl import dot, grad, div, dx, inner
from ufl import as_matrix, as_tensor, Identity
from dolfinx.fem import form
from dolfinx.fem import Function, dirichletbc, Constant
from dolfinx.fem import functionspace, locate_dofs_geometrical

import funcs_proj_smooth

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass

# --- calculation functions ---


def Poisson(inputs, outputs, c, Mesh, Funcspace: "FuncspaceClass"):
    """
    Solve the Poisson equation ∇·(c∇p) = 0 on a 2D domain with given boundary conditions.

    Parameters:
    - left_bc  - anonymous function defining Dirichlet BC at the left boundary (x = x_min)
    - right_bc - anonymous function defining Dirichlet BC at the right boundary (x = x_max)
    - c        - dolfinx Function representing the scalar conductivity field c(x, y)

    Returns:
    - p_sol - dolfinx Function, the computed pressure field satisfying the Poisson equation
    """
    # Weak form: a = (c * grad(u), grad(v)) * dx
    # c_tensor = c * Identity(2)
    if len(c.ufl_shape) == 2:  # c is tensor, use inner product
        # a = form(inner(dot(c, grad(Funcspace.u)), grad(Funcspace.v)) * dx)
        a = form(dot(dot(c, grad(Funcspace.u)), grad(Funcspace.v)) * dx)
    else:  # c is scalar, use *
        a = form(dot(c * grad(Funcspace.u), grad(Funcspace.v)) * dx)
    L = form(Constant(Mesh.domain, 0.0) * Funcspace.v * dx)

    # Input BC (left)
    bc_left = dirichletbc(inputs, Mesh.left_dofs)

    # Output BC (right) – zero
    # outval_fn = Function(Funcspace.ScalarFuncSpace)
    # outval_fn.interpolate(right_bc)
    bc_right = dirichletbc(outputs, Mesh.right_dofs)

    # both bcs
    bcs = [bc_left, bc_right]

    # Solve Poisson equation
    p_sol = Function(Funcspace.ScalarFuncSpace)
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, u=p_sol)
    p_sol = problem.solve()
    return p_sol


def Q(c, p, Funcspace):
    """
    Compute flux vector field Q = -c∇p from pressure field.

    Parameters:
    - p: dolfinx function, scalar pressure field p(x, y)

    Returns:
    - Q: dolfinx function, flux vector field Q = -c∇p
    """
    a_proj = form(dot(Funcspace.U, Funcspace.V) * dx)
    if len(c.ufl_shape) == 2:  # c is tensor, use dot
        L_proj = form(dot(dot(-grad(p), c), Funcspace.V) * dx)
    else:  # c is scalar, use *
        L_proj = form(-dot(grad(p) * c, Funcspace.V) * dx)
    Q = Function(Funcspace.VectorFuncSpace)
    problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=Q)
    Q = problem.solve()
    return Q


def gradc(c, Funcspace):
    """
    Compute grad c
    """
    a_proj = form(dot(Funcspace.U, Funcspace.V) * dx)
    L_proj = form(dot(grad(c), Funcspace.V) * dx)
    grad_c = Function(Funcspace.VectorFuncSpace)
    problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=grad_c)
    grad_c = problem.solve()
    return grad_c


def laplacianc(c, Funcspace):
    """
    Compute laplacian of conductivity field for smoothing

    Parameters:
    - c - dolfinx function, conductivity field c(x, y)

    Returns:
    - laplacian_c - dolfinx function, laplacian of conductivity field
    """
    if len(c.ufl_shape) == 2:
        # Compute component-wise Laplacian tensor
        laplace_components = [[div(grad(c[i, j])) for j in range(2)] for i in range(2)]
        laplace_expr = as_tensor(laplace_components)
        # laplace_expr = div(grad(c[0, 0])) + div(grad(c[1, 1]))
        a_proj = form(inner(Funcspace.U_tensor, Funcspace.V_tensor) * dx)
        L_proj = form(inner(laplace_expr, Funcspace.V_tensor) * dx)
        laplacian_c = Function(Funcspace.TensorFuncSpace)
    else:
        laplace_expr = div(grad(c))
        a_proj = form(dot(Funcspace.u, Funcspace.v) * dx)
        L_proj = form(dot(laplace_expr, Funcspace.v) * dx)
        laplacian_c = Function(Funcspace.ScalarFuncSpace)
    problem = dolfinx.fem.petsc.LinearProblem(a_proj, L_proj, u=laplacian_c)
    laplacian_c = problem.solve()
    return laplacian_c


def Q_at_dofs(Q, dofs):
    """
    Compute flux Q = -c∇p at specific locations from full Q(x,y)

    Parameters:
    Q    - dolfinx function, flux field Q(x, y)
    dofs - np.array of coordinates to calculate Q at

    Returns:
    - Q_x_dofs - np.array of calculated Q at specific coordinates
    """
    # Get full array of values of the vector field Q
    Q_array = Q.x.array.reshape((-1, 2))  # shape (num_dofs, 2)

    # Extract components at DOFs
    Q_x_dofs = Q_array[dofs, 0]  # only the first component (x-direction)
    # Q_y_dofs = Q_array[dofs, 1]  # second component is Q_y
    return Q_x_dofs


def loss(Q, target, dofs, Funcspace: "FuncspaceClass"):
    """
    Compute the loss as an array (measured - desired)

    Inputs:
    Q        - dolfinx function of flux
    target   - anonymous function of desired (target) values
    coords - np.array

    Outputs:
    np.array of the loss
    """
    # Project Q_x
    Q_x = funcs_proj_smooth.project_scalar_func(Q.sub(0), Funcspace)

    # Subtract target from Q_x on the boundary and store in a new function
    # loss_func = Function(Funcspace.ScalarFuncSpace)
    # loss_func.x.array[:] = 0.0
    # loss_func.x.array[dofs] = Q_x.x.array[dofs] - target
    # return loss_func
    return Q_x.x.array[dofs] - target


def Adalike_update(Loss, update_type, Mesh: "MeshClass", BCs=[0, 0]):
    """
    Compute Adalike update values (arrays and spline interpolations)

    Inputs:
    Loss           - np.array of loss at right coords
    update_type    - str, update method:
                     'Loss'          - bc_left = -Loss without negative values
                                       bc_right = Loss without positive values
                     'BCs times Loss - (BC_left - BCs_right)*Loss
                     'Loss integral' - integrate loss from all points on opposite boundary,
                                       weight by distance from point on boundary
    BCs            - lst of np.array of left and right BCs
    """
    L = Mesh.L
    H = Mesh.H
    Ngrid = Mesh.Ngrid

    if update_type == 'Loss':
        Loss_ceil = -ceil_loss(Loss)  # invert and don't allow negative values
        update_l = Loss_ceil
        update_r = -update_l
    elif update_type == 'BCs times Loss':
        Loss_ceil = ceil_loss(Loss)
        update_l = (BCs[1] - BCs[0]) * Loss
        update_r = - update_l
    elif update_type == 'Loss integral':
        bc_l = BCs[0]
        bc_r = BCs[1]
        length = len(BCs[0])
        update_l = np.zeros(length)
        update_r = np.zeros(length)

        for i, bc_l_val in enumerate(bc_l):
            integrand_l = np.zeros(length)
            integrand_r = np.zeros(length)
            normalize = np.zeros(length)
            for j, bc_r_val in enumerate(bc_r):
                # integrand_ij = (1/H**2)*(bc_r_val - bc_l_val)*(Loss[j])/np.sqrt(L**2 + (i-j)**2*H/Ngrid**2)
                integrand_l_ij = -(Loss[j]) / np.sqrt(L ** 2 + (i - j) ** 2 * (H / Ngrid) ** 2)
                integrand_r_ij = (Loss[i]) / np.sqrt(L ** 2 + (i - j) ** 2 * (H / Ngrid) ** 2)
                normalize_ij = 1 / np.sqrt(L ** 2 + (i - j) ** 2 * (H / Ngrid) ** 2)
                # normalize_ij = 1/H**2
                integrand_l[j] = (integrand_l_ij)
                integrand_r[j] = (integrand_r_ij)
                normalize[j] = (normalize_ij)
            update_l[i] = (sum(integrand_l) / sum(normalize))
            update_r[i] = (sum(integrand_r) / sum(normalize))

        update_l = update_l + np.mean(-Loss)
        update_r = update_r + np.mean(Loss)

        # correct for wrong direction delta_p
        update_r[update_r > update_l] = update_l[update_r > update_l]
    return update_l, update_r


def ceil_loss(Loss):
    L = copy.copy(Loss)
    L[L > 0] = 0
    return L


def floor_loss(Loss):
    L = copy.copy(Loss)
    L[L < 0] = 0
    return L


def MSE(y1, y2):
    return np.square(np.subtract(y1, y2)).mean()