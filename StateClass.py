import numpy as np
from dolfinx.fem import Function
from ufl import dot, sqrt

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
    def __init__(self, gamma_Q, gamma_c, gamma_decay, c_ss):
        self.gamma_Q = gamma_Q
        self.gamma_c = gamma_c
        self.gamma_decay = gamma_decay
        self.c_ss = c_ss

    def init_c(self, c0, Funcspace: "FuncspaceClass", upload_from_file=False):
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

    def calc_Poisson(self, Supervisor: "SupervisorClass", Mesh: "MeshClass", Funcspace: "FuncspaceClass", update=False):
        if not update:
            self.p = calc.Poisson(Supervisor.input_val_2d, Supervisor.output_val_2d, self.c, Mesh, Funcspace)
        else:
            self.p_update = calc.Poisson(Supervisor.update_bc_fn_l, Supervisor.update_bc_fn_r, self.c, Mesh, Funcspace)

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
            self.absQ_update = funcs_proj_smooth.project_scalar_func(sqrt(dot(self.Q_update, self.Q_update)), Funcspace)
            self.Q_update_x_right = calc.Q_at_dofs(self.Q_update, Mesh.right_dofs)

    def evolve_c(self, Funcspace: "FuncspaceClass", smooth=True):
        self.calc_laplacian_c(Funcspace, smooth=smooth)
        c_new = funcs_proj_smooth.project_scalar_func(self.gamma_Q * self.absQ_update + self.gamma_c * self.laplacian_c +
                                                      self.gamma_decay * (self.c - self.c_ss) + self.c, Funcspace)
        # c_new = project_scalar_func(gamma * absQ - beta * abs_grad_c_pow + 1.0)
        # c_new = project_scalar_func(gamma_Q * absQ + gamma_c * abs_grad_c_pow + gamma_decay * (c-c_ss) + c)
        # c_new = smooth_scalar_field(c_new)
        self.c = c_new

    def calc_laplacian_c(self, Funcspace: "FuncspaceClass", smooth=True):
        # calculate
        self.laplacian_c = calc.laplacianc(self.c, Funcspace)
        self.laplacian_c = funcs_proj_smooth.project_scalar_func(self.laplacian_c, Funcspace)
        if smooth:
            self.laplacian_c = funcs_proj_smooth.smooth_field(self.laplacian_c, Funcspace, space='scalar')