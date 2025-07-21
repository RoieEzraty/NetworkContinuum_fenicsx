import numpy as np
from dolfinx.fem import Function
from ufl import dot, sqrt, outer, Identity, as_tensor

import calc, funcs_proj_smooth

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass
    from SupervisorClass import SupervisorClass

# ===================================================
# Class - Material state
# ===================================================


class StateClass:
    """
    Class of material state variables
    """
    def __init__(self, gamma_Q, gamma_c, gamma_decay, c_ss, c_type):
        self.gamma_Q = gamma_Q
        self.gamma_c = gamma_c
        self.gamma_decay = gamma_decay
        self.c_type = c_type
        if c_type == 'tensor':
            # self.c_ss = c_ss * Identity(2)
            self.c_ss = c_ss * as_tensor([[0.01, 0.01], [0.01, 0.01]])
        else:
            self.c_ss = c_ss

    def init_c(self, c0, c_type, Funcspace: "FuncspaceClass", upload_from_file=False):
        if upload_from_file:
            self.c = Function(Funcspace.ScalarFuncSpace)
            try:
                c_array = np.load("c_latest.npy")
                self.c.x.array[:] = c_array
                print("Loaded previous c field from c_latest.npy")
            except FileNotFoundError:
                print("No previous c found. Starting fresh.")
                self.c.interpolate(lambda x: np.full(x.shape[1], c0))
        else:
            # Initialize conductivity field c
            self.c = Function(Funcspace.ScalarFuncSpace)
            self.c.interpolate(lambda x: np.full(x.shape[1], c0))
            if c_type == 'tensor':
                self.c = self.c * Identity(2)
                # self.c = as_tensor([[1.8*self.c, 1*self.c], [1*self.c, 1.8*self.c]])

    def calc_Poisson(self, Supervisor: "SupervisorClass", Mesh: "MeshClass", Funcspace: "FuncspaceClass", update=False):
        if not update:
            self.p = calc.Poisson(Supervisor.inputs.dolfx_1d, Supervisor.outputs.dolfx_1d, self.c, Mesh, Funcspace)
        else:
            self.p_update = calc.Poisson(Supervisor.update.l_dolfx_1d, Supervisor.update.r_dolfx_1d, self.c, Mesh, Funcspace)

    def calc_Q(self, Funcspace: "FuncspaceClass", Mesh: "MeshClass", update=False, smooth=True):
        if not update:
            self.Q = calc.Q(self.c, self.p, Funcspace)
            if smooth:
                self.Q = funcs_proj_smooth.smooth_field(self.Q, Funcspace, space='vector')
            self.absQ = funcs_proj_smooth.project_scalar_func(sqrt(dot(self.Q, self.Q)), Funcspace)
            self.Q_x_right = calc.Q_at_dofs(self.Q, Mesh.right_dofs)
        else:
            self.Q_update = calc.Q(self.c, self.p_update, Funcspace)
            if smooth:
                self.Q_update = funcs_proj_smooth.smooth_field(self.Q_update, Funcspace, space='vector')
            if self.c_type == 'tensor':
                # absQ_tensor_expr = as_tensor([[sqrt(self.Q_update[0] * self.Q_update[0]), sqrt(self.Q_update[0] * self.Q_update[1])],
                #                               [sqrt(self.Q_update[1] * self.Q_update[0]), sqrt(self.Q_update[1] * self.Q_update[1])]])
                # absQ_tensor_expr = outer(self.Q_update, self.Q_update)
                w = 1.0  # example: double the off-diagonal weight
                absQ_tensor_expr = as_tensor([
                    [1 / w * self.Q_update[0] * self.Q_update[0], w * self.Q_update[0] * self.Q_update[1]],
                    [w * self.Q_update[1] * self.Q_update[0], 1 / w * self.Q_update[1] * self.Q_update[1]]
                ])
                self.absQ_update = funcs_proj_smooth.project_tensor_func(absQ_tensor_expr, Funcspace)
            else:
                self.absQ_update = funcs_proj_smooth.project_scalar_func(dot(self.Q_update, self.Q_update), Funcspace)
            self.Q_update_x_right = calc.Q_at_dofs(self.Q_update, Mesh.right_dofs)

    def evolve_c(self, Funcspace: "FuncspaceClass", smooth=True):
        self.calc_laplacian_c(Funcspace, smooth=smooth)
        if self.c_type == 'tensor':
            c_new = funcs_proj_smooth.project_tensor_func(self.gamma_Q * self.absQ_update + self.gamma_c * self.laplacian_c +
                                                          self.gamma_decay * (self.c - self.c_ss) + self.c, Funcspace)
        else:
            c_new = funcs_proj_smooth.project_scalar_func(self.gamma_Q * self.absQ_update + self.gamma_c * self.laplacian_c +
                                                          self.gamma_decay * (self.c - self.c_ss) + self.c, Funcspace)
            # c_new = project_scalar_func(gamma * absQ - beta * abs_grad_c_pow + 1.0)
            # c_new = project_scalar_func(gamma_Q * absQ + gamma_c * abs_grad_c_pow + gamma_decay * (c-c_ss) + c)
            # c_new = smooth_scalar_field(c_new)
        self.c = c_new

    def calc_laplacian_c(self, Funcspace: "FuncspaceClass", smooth=True):
        # calculate
        self.laplacian_c = calc.laplacianc(self.c, Funcspace)
        if self.c_type == 'tensor':  # if c is tensor, project as tensor
            self.laplacian_c = funcs_proj_smooth.project_tensor_func(self.laplacian_c, Funcspace)
            space = 'tensor'
        else:
            self.laplacian_c = funcs_proj_smooth.project_scalar_func(self.laplacian_c, Funcspace)
            space = 'scalar'
        if smooth:
            self.laplacian_c = funcs_proj_smooth.smooth_field(self.laplacian_c, Funcspace, space=space)