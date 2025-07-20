from dolfinx import mesh
from mpi4py import MPI
from dolfinx.fem import functionspace, locate_dofs_geometrical
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass

# ===================================================
# Class - dolfinx mesh variables
# ===================================================


class MeshClass:
    """
    Class with mesh variables
    """
    def __init__(self, x_min, x_max, y_min, y_max, Ngrid):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                            [[x_min, y_min], [x_max, y_max]],
                                            [Ngrid, Ngrid],
                                            mesh.CellType.triangle)
        
        self.L = x_max - x_min
        self.H = y_max - y_min
        self.Ngrid = Ngrid

    def build_dofs(self, Funcspace: "FuncspaceClass"):
        # DOFs on the left/right boundary (x = x_min/x_max)
        self.right_dofs = locate_dofs_geometrical(Funcspace.VectorFuncSpace, lambda x: np.isclose(x[0], self.x_max))
        self.left_dofs = locate_dofs_geometrical(Funcspace.VectorFuncSpace, lambda x: np.isclose(x[0], self.x_min))
        self.coords = Funcspace.VectorFuncSpace.tabulate_dof_coordinates()
        self.coords_right = self.coords[self.right_dofs]
        self.coords_left = self.coords[self.left_dofs]
        self.y_array = np.linspace(self.y_min, self.y_max, len(self.coords_right))