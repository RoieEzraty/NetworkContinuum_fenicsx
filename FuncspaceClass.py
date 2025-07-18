from dolfinx.fem import functionspace, locate_dofs_geometrical
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
        self.ScalarFuncSpace = functionspace(Mesh.domain, ("Lagrange", 1))
        self.VectorFuncSpace = functionspace(Mesh.domain, ("CG", 1, (2,)))

        # --- Trial and Test Functions ---
        self.u = TrialFunction(self.ScalarFuncSpace)
        self.v = TestFunction(self.ScalarFuncSpace)
        self.U = TrialFunction(self.VectorFuncSpace)
        self.V = TestFunction(self.VectorFuncSpace)