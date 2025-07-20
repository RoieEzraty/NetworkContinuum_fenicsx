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
        self.Loss_vec = []

    def pose_input_val(self, y0_in, sigma_in, offset_in, Mesh: "MeshClass"):
        self.input_val_1d, self.input_val_2d, self.input_val_array = bcs.input_val(y0_in, sigma_in, offset_in,
                                                                                   Mesh.coords_left)

    def pose_output_val(self, Mesh: "MeshClass"):
        self.output_val_1d, self.output_val_2d, self.output_val_array = bcs.output_val(Mesh.coords_right)

    def pose_target_val(self, y1_target, sigma_target, offset_target, Mesh: "MeshClass"):
        self.target_val_1d, self.target_val_array = bcs.target_val(y1_target, sigma_target, offset_target,
                                                                   Mesh.coords_right)

    def calc_loss(self, State: "StateClass", Mesh: "MeshClass", Funcspace: "FuncspaceClass", num=1):
        if num == 1:
            self.Loss = calc.loss(State.Q, self.target_val_array, Mesh.right_dofs, Funcspace)
            self.Loss_ceil = calc.ceil_loss(self.Loss)
        elif num == 2:
            self.Loss_2 = calc.loss(State.Q, self.target_val_array, Mesh.right_dofs, Funcspace)
            self.Loss_ceil_2 = calc.ceil_loss(self.Loss_2)
            self.Loss_tot = np.sqrt(np.square(self.Loss) + np.square(self.Loss_2))

    def append_loss(self, shared_loss):
        if shared_loss == 'True':
            self.Loss_vec.append(np.mean(self.Loss**2 + self.Loss_2**2))
        else:
            self.Loss_vec.append(np.mean(self.Loss**2))

    def calc_update(self, update_type, Mesh: "MeshClass", num=1):
        if num == 1:
            self.update_l, self.update_r, \
                self.update_bc_fn_l, self.update_bc_fn_r = calc.Adalike_update(self.Loss, update_type, Mesh,
                                                                               [self.input_val_array,
                                                                                self.output_val_array])
        elif num == 2:
            self.update_l_2, self.update_r_2, \
                self.update_bc_fn_l_2, self.update_bc_fn_r_2 = calc.Adalike_update(self.Loss_2, update_type, Mesh,
                                                                                   [self.input_val_array,
                                                                                    self.output_val_array])

            self.update_bc_fn_l = funcs_proj_smooth.mean_funcs(self.update_bc_fn_l, self.update_bc_fn_l_2)
            self.update_bc_fn_r = funcs_proj_smooth.mean_funcs(self.update_bc_fn_r, self.update_bc_fn_r_2)