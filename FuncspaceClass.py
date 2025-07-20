from dolfinx.fem import functionspace, locate_dofs_geometrical, Function
from ufl import TrialFunction, TestFunction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass

# ===================================================
# Class - Function space and Trial functions
# ===================================================


class FuncspaceClass:
    """
    Class with functionspace, dofs, coords, etc.
    """
    def __init__(self, Mesh: "MeshClass"):
        # --- Function space ---
        self.ScalarFuncSpace = functionspace(Mesh.domain, ("Lagrange", 1))  # scalar function space, 1d over 2d domain
        self.VectorFuncSpace = functionspace(Mesh.domain, ("CG", 1, (2,)))  # vector function space, 2d over 2d domain
        self.TensorFuncSpace = functionspace(Mesh.domain, ("CG", 1, (2, 2)))  # tensor function space, 2x2 over 2d domain

        # --- Trial and Test Functions ---
        self.u = TrialFunction(self.ScalarFuncSpace)
        self.v = TestFunction(self.ScalarFuncSpace)
        self.U = TrialFunction(self.VectorFuncSpace)
        self.V = TestFunction(self.VectorFuncSpace)
        self.U_tensor = TrialFunction(self.TensorFuncSpace)
        self.V_tensor = TestFunction(self.TensorFuncSpace)