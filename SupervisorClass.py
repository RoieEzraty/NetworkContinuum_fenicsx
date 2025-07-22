import copy
import numpy as np
from dolfinx.fem import Function

import bcs, calc, funcs_proj_smooth

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass
    from StateClass import StateClass

# ===================================================
# Class - Variables of the supervisor
# ===================================================


class SupervisorClass:
    """
    Class of variables relevant to the supervisor, the one that operates the material
    """
    def __init__(self):
        pass

    def init_loss(self):
        self.Loss_vec = []

    def pose_input_val(self, y0, sigma, offset, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.inputs = Inputs(y0, sigma, offset, Mesh, Funcspace)

    def pose_output_val(self, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.outputs = Outputs(Mesh, Funcspace)

    def pose_target_val(self, y0, sigma, offset, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.target = Target(y0, sigma, offset, Mesh, Funcspace)

    def calc_loss(self, State: "StateClass", Mesh: "MeshClass", Funcspace: "FuncspaceClass", num=1):
        if num == 1:
            self.Loss = calc.loss(State.Q, self.target.array, Mesh.right_dofs, Funcspace)
            self.Loss_ceil = calc.ceil_loss(self.Loss)
        elif num == 2:
            self.Loss_2 = calc.loss(State.Q, self.target.array, Mesh.right_dofs, Funcspace)
            self.Loss_ceil_2 = calc.ceil_loss(self.Loss_2)
            self.Loss_tot = np.sqrt(np.square(self.Loss) + np.square(self.Loss_2))

    def append_loss(self, shared_loss):
        if shared_loss == 'True':
            self.Loss_vec.append(np.mean(self.Loss**2 + self.Loss_2**2))
        else:
            self.Loss_vec.append(np.mean(self.Loss**2))

    def calc_update(self, update_type, Mesh: "MeshClass", Funcspace: "FuncspaceClass", num=1):
        if num == 1:
            if update_type == 'dumb inputs':
                self.update = Update(self.Loss, update_type, Mesh, self.inputs, self.target, Funcspace)
            else:
                self.update = Update(self.Loss, update_type, Mesh, self.inputs, self.outputs, Funcspace)
        elif num == 2:
            self.update_2 = Update(self.Loss_2, update_type, Mesh, self.inputs, self.outputs, Funcspace)

    def combine_updates(self, Funcspace):
        self.update.l_fn = funcs_proj_smooth.mean_funcs(self.update.l_fn, self.update_2.l_fn)
        self.update.r_fn = funcs_proj_smooth.mean_funcs(self.update.r_fn, self.update_2.r_fn)
        self.update.l_array = self.update.l_array + self.update_2.l_array
        self.update.r_array = self.update.r_array + self.update_2.r_array
        self.update.l_dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.update.l_dolfx_1d.interpolate(self.update.l_fn)
        self.update.r_dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.update.r_dolfx_1d.interpolate(self.update.r_fn)


class Inputs:
    """
    Class of variables relevant to the inputs
    """
    def __init__(self, y0, sigma, offset, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.f1d, self.f2d = bcs.input_val(y0, sigma, offset)
        self.array = np.array([self.f1d(y) for y in Mesh.coords_left[:, 1]])
        self.dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.dolfx_1d.interpolate(self.f2d)


class Outputs:
    """
    Class of variables relevant to the outputs
    """
    def __init__(self, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.f1d, self.f2d = bcs.output_val()
        self.array = np.array([self.f1d(y) for y in Mesh.coords_right[:, 1]])
        self.dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.dolfx_1d.interpolate(self.f2d)


class Target:
    """
    Class of variables relevant to the target values
    """
    def __init__(self, y0, sigma, offset, Mesh: "MeshClass", Funcspace: "FuncspaceClass"):
        self.f1d, self.f2d = bcs.target_val(y0, sigma, offset)
        self.array = np.array([self.f1d(y) for y in Mesh.coords_left[:, 1]])
        self.dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.dolfx_1d.interpolate(self.f2d)


class Update:
    """
    Class of variables relevant to the update modality values
    """
    def __init__(self, Loss, update_type, Mesh: "MeshClass", inputs, outputs, Funcspace: "FuncspaceClass"):
        self.l_array, self.r_array = calc.update(Loss, update_type, Mesh, [inputs.array, outputs.array])
        self.l_fn = funcs_proj_smooth.spline_over_coords(self.l_array, Mesh.coords_left)
        self.r_fn = funcs_proj_smooth.spline_over_coords(self.r_array, Mesh.coords_right)
        self.l_dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.l_dolfx_1d.interpolate(self.l_fn)
        self.r_dolfx_1d = Function(Funcspace.ScalarFuncSpace)
        self.r_dolfx_1d.interpolate(self.r_fn)