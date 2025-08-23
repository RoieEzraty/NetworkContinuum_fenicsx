from dolfinx import mesh
from mpi4py import MPI
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.io import gmshio
import numpy as np
import gmsh

class MeshClass:
    """
    Class with mesh variables
    """
    def __init__(self, x_min, x_max, y_min, y_max, Ngrid, shape='full',
                 nx_holes=4, ny_holes=4, hole_size=0.08, edge_pad=0.05, lc=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.shape = shape

        self.L = x_max - x_min
        self.H = y_max - y_min
        self.Ngrid = Ngrid

        # init tags to None ONCE, before branching
        self.cell_tags = None
        self.facet_tags = None

        if shape == 'full':
            self.domain = mesh.create_rectangle(
                MPI.COMM_WORLD,
                [[x_min, y_min], [x_max, y_max]],
                [Ngrid, Ngrid],
                mesh.CellType.triangle
            )
        elif shape == 'grid':
            self.domain, self.cell_tags, self.facet_tags = self.rect_w_square_holes(
                x_min, x_max, y_min, y_max,
                nx_holes=nx_holes, ny_holes=ny_holes,
                hole_size=hole_size, edge_pad=edge_pad, lc=lc
            )
        else:
            raise ValueError("shape must be 'full' or 'grid'")

    def build_dofs(self, Funcspace: "FuncspaceClass"):
        # DOFs on the left/right boundary (x = x_min/x_max)
        self.right_dofs = locate_dofs_geometrical(
            Funcspace.VectorFuncSpace,
            lambda x: np.isclose(x[0], self.x_max)
        )
        self.left_dofs = locate_dofs_geometrical(
            Funcspace.VectorFuncSpace,
            lambda x: np.isclose(x[0], self.x_min)
        )
        self.coords = Funcspace.VectorFuncSpace.tabulate_dof_coordinates()
        self.coords_right = self.coords[self.right_dofs]
        self.coords_left  = self.coords[self.left_dofs]
        self.y_array = np.linspace(self.y_min, self.y_max, len(self.coords_right))

        if self.shape == 'grid' and self.facet_tags is not None:
            # store facet sets on the instance for later BC use
            self.outer_facets = self.facet_tags.find(1)  # outer boundary
            self.hole_facets  = self.facet_tags.find(2)  # hole walls

    def rect_w_square_holes(
        self, xmin, xmax, ymin, ymax,
        nx_holes=4, ny_holes=4,
        hole_size=0.08,
        edge_pad=0.05,
        lc=None,
    ):
        """
        Build [xmin,xmax]×[ymin,ymax] with an nx×ny grid of square holes.
        Returns (msh, cell_tags, facet_tags). facet_tags: 1=outer, 2=holes.
        """
        Lx = xmax - xmin
        Ly = ymax - ymin
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain must have positive size.")

        free_x = Lx - 2*edge_pad - nx_holes*hole_size
        free_y = Ly - 2*edge_pad - ny_holes*hole_size
        if free_x < 0 or free_y < 0:
            raise ValueError("edge_pad + hole_size too large for the domain size")

        gap_x = free_x / (nx_holes + 1)
        gap_y = free_y / (ny_holes + 1)

        comm = MPI.COMM_WORLD
        rank = comm.rank

        if rank == 0:
            gmsh.initialize()
            gmsh.model.add("perforated_rect")

            # Outer rectangle
            outer = gmsh.model.occ.addRectangle(xmin, ymin, 0.0, Lx, Ly)

            # Holes
            holes = []
            for i in range(nx_holes):
                for j in range(ny_holes):
                    x0 = xmin + edge_pad + gap_x*(i+1) + hole_size*i
                    y0 = ymin + edge_pad + gap_y*(j+1) + hole_size*j
                    r = gmsh.model.occ.addRectangle(x0, y0, 0.0, hole_size, hole_size)
                    holes.append(r)

            gmsh.model.occ.synchronize()
            if holes:
                cut = gmsh.model.occ.cut([(2, outer)], [(2, h) for h in holes])
                surfaces = [e[1] for e in cut[0] if e[0] == 2]
                assert len(surfaces) == 1
                domain = (2, surfaces[0])
            else:
                domain = (2, outer)

            # Tag boundaries
            gmsh.model.occ.synchronize()
            
            # After cut and synchronize
            gmsh.model.occ.synchronize()
            
            # Get curve loops for the new domain surface
            loops = gmsh.model.getBoundary([domain], oriented=False, recursive=False)
            curve_loops = gmsh.model.getCurveLoops([l for l in loops if l[0] == 1])
            # curve_loops = (loop_tags, [list_of_curves_per_loop])
            
            outer_curve_tags = curve_loops[1][0]                    # first loop = outer
            hole_curve_tags  = [c for tags in curve_loops[1][1:] for c in tags]  # flatten rest
            
            # Tag physical groups
            pg_outer = gmsh.model.addPhysicalGroup(1, outer_curve_tags, tag=1)
            gmsh.model.setPhysicalName(1, pg_outer, "outer")
            
            if hole_curve_tags:
                pg_holes = gmsh.model.addPhysicalGroup(1, hole_curve_tags, tag=2)
                gmsh.model.setPhysicalName(1, pg_holes, "holes")


            if lc is not None:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)

        # Build distributed dolfinx mesh from rank 0's model
        msh, cell_tags, facet_tags = gmshio.model_to_mesh(
            gmsh.model if rank == 0 else None, comm, 0, gdim=2
        )

        if rank == 0:
            gmsh.finalize()

        return msh, cell_tags, facet_tags
