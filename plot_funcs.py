import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import pyvista as pv

from dolfinx.plot import vtk_mesh
from matplotlib import gridspec

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MeshClass import MeshClass
    from FuncspaceClass import FuncspaceClass
    from StateClass import StateClass
    from SupervisorClass import SupervisorClass

# --- plot functions ---

# from matplotlib import rcParams
# rcParams['text.usetex'] = True


def scalar_field(field, string, coords):
    plt.figure(figsize=(6, 4))
    plt.tricontourf(coords[:, 0], coords[:, 1], field.x.array, levels=100, cmap="plasma")
    plt.title(r"${}$ at iteration ${}$, cycle ${}$".format(string, i + 1, j + 1))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar(label=fr"${string}$")
    plt.tight_layout()
    plt.show()


def vector_field(field, string):
    coords = field.function_space.tabulate_dof_coordinates()
    values = field.x.array.reshape((-1, 2))  # reshape into (n_points, 2)

    plt.figure(figsize=(6, 4))
    plt.quiver(coords[:, 0], coords[:, 1], values[:, 0], values[:, 1], scale=50)
    plt.ylabel(r"{}".format(string))
    plt.xlabel(r"$x$")
    plt.tight_layout()
    plt.show()


def over_line(lambda_func, line, ylabel='', xlabel=''):
    vals = np.array([lambda_func([0.0, y]) for y in line])
    plt.plot(line, vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def inputs_and_target(Supervisor: "SupervisorClass", Mesh: "MeshClass"):
    plt.plot(Mesh.y_array, Supervisor.inputs.array, 'b', label='Input')
    plt.plot(Mesh.y_array, Supervisor.target.array, 'r', label='Target')
    plt.title(r"Input and Target Iteration 1")
    plt.xlabel(r"$y$")
    plt.ylabel(r"Value")
    plt.legend()
    plt.show()

def measurement_fields(Mesh: "MeshClass", State: "StateClass", Supervisor: "SupervisorClass", iteration=1, cycle=1, num=1, stack=False):
    """
    4 panels:
      [0] scalar field p (cividis)
      [1] scalar field |Q| (plasma)
      [2] line plots: Q_x at right boundary, target, loss
      [3] BEASTAL L/R
    Holes are empty because we render the true 2D cell mesh.
    """

    length = 1000
    height = 380
    linewidth = 3.0
    zoom = 1.5
    font = 16

    # choose layout
    shape = (2, 2) if stack else (1, 4)
    pl = pv.Plotter(shape=shape, window_size=(length, height if not stack else length), border=False)
    grid = Mesh.pv_grid().copy() # build once

    sbar0 = dict(title="p", vertical=True,    # vertical orientation
                 position_x=Mesh.x_max+0.15,  # move rightward
                 position_y=0.05,   # lower edge
                 width=0.08,
                 height=0.9)

    sbar1 = dict(title="|Q|", vertical=True,    # vertical orientation
                 position_x=Mesh.x_max+0.15,  # move rightward
                 position_y=0.05,   # lower edge
                 width=0.08,
                 height=0.9)   
    
    # ---------- panel 0: p ----------
    
    grid_p = grid.copy()
    grid_p.point_data["p"] = State.p.x.array
    pl.subplot(0, 0)
    pl.add_mesh(grid_p, scalars="p", cmap="plasma",
                show_edges=False, lighting=False, scalar_bar_args=sbar0, show_scalar_bar=True)
    outline = _domain_outline_polyline(Mesh)
    pl.add_mesh(outline, color="black", line_width=3.0)

    pl.view_xy()
    pl.hide_axes()   # hide axes indicator
    pl.camera.zoom(zoom)   # ← same zoom
    pl.add_text(f"measure \niteration {iteration+1}\n cycle {cycle+1}", font_size=10, position="upper_left") 

    # ---------- panel 1: |Q| ----------
    grid_q = grid.copy()
    grid_q.point_data["absQ"] = State.absQ.x.array
    pl.subplot(0, 1)
    pl.add_mesh(grid_q, scalars="absQ", cmap="plasma",
                show_edges=False, lighting=False, scalar_bar_args=sbar1, show_scalar_bar=True)
    outline = _domain_outline_polyline(Mesh)
    pl.add_mesh(outline, color="black", line_width=linewidth)
    pl.view_xy()
    pl.hide_axes()   # hide axes indicator
    pl.camera.zoom(zoom)   # ← same zoom
    
    # ---------- panel 2: input ----------

    # data
    y_l = Mesh.coords_left[:, 1]
    input_l = Supervisor.inputs.array
    
    if stack:
        pl.subplot(1, 0)
    else:
        pl.subplot(0, 2)
    ch = pv.Chart2D()
    ch.line(input_l, y_l, color="black", width=linewidth, label="input_x")
    ch.y_label = "y"
    ch.x_label = ""
    ch.x_axis.tick_label_size = font
    ch.x_axis.label_size = font
    ch.y_axis.tick_label_size = font
    ch.y_axis.label_size = font
    ch.legend_visible = True
    ch.grid = False
    pl.add_chart(ch)

    # ---------- panel 3: right-boundary lines ----------
    # data
    y_r = Mesh.coords_right[:, 1]
    Qx_r = State.Q_x_right
    target = Supervisor.target.array
    Loss  = Supervisor.Loss if num == 1 else Supervisor.Loss_2

    # chart
    if stack:
        pl.subplot(1, 1)
    else:
        pl.subplot(0, 3)
    ch = pv.Chart2D()
    ch.line(Qx_r, y_r, color="black", width=linewidth, label="Q_x")
    ch.line(target, Mesh.y_array, color="blue", width=linewidth, label="Target")
    ch.line(Loss, y_r, color="tab:red", width=linewidth, label="Loss")
    ch.x_label = ""
    ch.y_label = "y"
    ch.x_axis.tick_label_size = font
    ch.x_axis.label_size = font
    ch.y_axis.tick_label_size = font
    ch.y_axis.label_size = font
    ch.legend_visible = True
    ch.grid = False
    pl.add_chart(ch)

    # axes + show
    for rc in [(0,1),(1,0)] if stack else [(0,1),(0,2)]:
        pl.subplot(*rc); pl.show_axes()
    pl.link_views()  # link camera across subplots with meshes
    pl.show()


def update_fields(Mesh, State, Supervisor, iteration=1, cycle=1, stack=False):
    """
    4 panels:
      [0] scalar field p_update (cividis)
      [1] scalar field |Q|_update (plasma)
      [2] scalar field c (plasma or your choice)
      [3] BEASTAL L/R line plots
    """

    # layout + styling
    length, height = 1000, 380
    linewidth = 3.0
    zoom = 1.5
    font = 16

    shape = (2, 2) if stack else (1, 4)
    pl = pv.Plotter(shape=shape, window_size=(length, height if not stack else length), border=False)

    # build once, reuse via copies
    grid_base = Mesh.pv_grid().copy()
    outline = _domain_outline_polyline(Mesh)

    # vertical scalar bars (screen coords 0..1)
    sbar_common = dict(vertical=True, position_x=Mesh.x_max+0.15, position_y=0.05, width=0.08, height=0.9)
    
    # ---------- panel 0: p_update ----------
    pl.subplot(0, 0)
    grid_pu = grid_base.copy()
    grid_pu.point_data["p_update"] = np.asarray(State.p_update.x.array)
    # independent color limits for this field
    clim_pu = (float(grid_pu.point_data["p_update"].min()), float(grid_pu.point_data["p_update"].max()))
    sbar0 = {**sbar_common, "title": "p"}
    pl.add_mesh(grid_pu, scalars="p_update", cmap="plasma",
                clim=clim_pu, show_edges=False, lighting=False,
                scalar_bar_args=sbar0, show_scalar_bar=True)
    pl.add_mesh(outline, color="black", line_width=linewidth)
    pl.view_xy(); pl.hide_axes(); pl.camera.zoom(zoom)
    pl.add_text(f"update \niteration {iteration+1}\ncycle {cycle+1}", font_size=10, position="upper_left")

    # ---------- panel 1: |Q|_update ----------
    # tensor vs scalar handling (match your earlier pattern)
    if State.c_type == "tensor":
        absQ_update_array = State.absQ_update.x.array[::4] + State.absQ_update.x.array[3::4]
    else:
        absQ_update_array = State.absQ_update.x.array

    pl.subplot(0, 1)
    grid_qu = grid_base.copy()
    grid_qu.point_data["absQ_update"] = np.asarray(absQ_update_array)
    clim_qu = (float(grid_qu.point_data["absQ_update"].min()), float(grid_qu.point_data["absQ_update"].max()))
    sbar1 = {**sbar_common, "title": "|Q|"}
    pl.add_mesh(grid_qu, scalars="absQ_update", cmap="plasma",
                clim=clim_qu, show_edges=False, lighting=False,
                scalar_bar_args=sbar1, show_scalar_bar=True)
    pl.add_mesh(outline, color="black", line_width=linewidth)
    pl.view_xy(); pl.hide_axes(); pl.camera.zoom(zoom)

    # ---------- panel 2: c field ----------
    if stack:
        pl.subplot(1, 0)
    else:
        pl.subplot(0, 2)

    if State.c_type == "tensor":
        c_array = State.c.x.array[::4] + State.c.x.array[3::4]
    else:
        c_array = State.c.x.array    # <- fix: use c, not absQ_update

    y_r = Mesh.coords_right[:, 1]
    ch = pv.Chart2D()
    ch.line(y_r, Supervisor.update.l_array, color="black", width=linewidth, label="BEASTAL left")
    ch.line(y_r, Supervisor.update.r_array, color="black", width=linewidth, style="--", label="BEASTAL right")
    ch.x_label = "y"; ch.y_label = ""
    ch.x_axis.tick_label_size = font; ch.x_axis.label_size = font
    ch.y_axis.tick_label_size = font; ch.y_axis.label_size = font
    ch.legend_visible = True
    ch.grid = False
    pl.add_chart(ch)

    # ---------- panel 3: BEASTAL L/R (lines, no grid) ----------
    if stack:
        pl.subplot(1, 1)
    else:
        pl.subplot(0, 3)

    
    grid_c = grid_base.copy()
    grid_c.point_data["c"] = np.asarray(c_array)
    clim_c = (float(grid_c.point_data["c"].min()), float(grid_c.point_data["c"].max()))
    sbar2 = {**sbar_common, "title": "c"}
    pl.add_mesh(grid_c, scalars="c", cmap="cividis",
                clim=clim_c, show_edges=False, lighting=False,
                scalar_bar_args=sbar2, show_scalar_bar=True)
    pl.add_mesh(outline, color="black", line_width=linewidth)
    pl.view_xy(); pl.hide_axes(); pl.camera.zoom(zoom)
    
    # show
    pl.link_views()
    pl.show()


def Loss_vec(Supervisor: "SupervisorClass"):
    plt.figure(figsize=(3, 3))
    plt.plot(Supervisor.Loss_vec, 'k')
    plt.ylabel('Loss')
    plt.xlabel('iteration')

    plt.tight_layout()
    plt.show()


def Q(State: "StateClass", Mesh: "MeshClass", update=False, iteration=1, cycle=1, include_p=False):
    """
    Plot each component of the 2x2 tensor c over the domain in 4 subplots.
    """

    if update:
        Q = State.Q_update
    else:
        Q = State.Q

    singlefig = 3.5

    if include_p:
        fig = plt.figure(figsize=(3*singlefig, singlefig))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.45)
        ax0u = fig.add_subplot(gs[0, 0])
        ax1u = fig.add_subplot(gs[0, 1])
        ax2u = fig.add_subplot(gs[0, 2])
        axes = [ax0u, ax1u, ax2u]
    else:
        fig = plt.figure(figsize=(2*singlefig, singlefig))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.45)
        ax0u = fig.add_subplot(gs[0, 0])
        ax1u = fig.add_subplot(gs[0, 1])
        axes = [ax0u, ax1u]

    global_min = min(Q.x.array)
    global_max = max(Q.x.array)
    
    labels = ['x', 'y']

    for i, ax in enumerate(axes):
        if i<2:
            tcf = ax.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], Q.x.array[i::2], levels=100, cmap="plasma", vmin=global_min, vmax=global_max)
            ax.set_title(r"$Q_{{{}}}$ at iteration {}, cycle {}".format(labels[i], iteration + 1, cycle + 1))
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            # fig.colorbar(tcf, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
            fig.colorbar(tcf, ax=ax, orientation="vertical", fraction=0.1, pad=0.02)
        else:
            tcf = ax.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], State.p.x.array, levels=100, cmap="plasma")
            ax.set_title(r"$p$ at iteration {}, cycle {}".format(iteration + 1, cycle + 1))
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")    

    plt.tight_layout()
    plt.show()


# def c_tensor(State: "StateClass", Mesh: "MeshClass", iteration=1, cycle=1):

#     # Instantiate figure and subplots
#     fig_update = plt.figure(figsize=(7, 7))
#     gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], wspace=0.45)
#     axes = [fig_update.add_subplot(gs[j, i]) for i in range(2) for j in range(2)]

#     # Extract tensor components
#     c_components = [State.c.x.array[i::4] for i in range(4)]

#     # Global min/max for color normalization
#     global_min = min(comp.min() for comp in c_components)
#     global_max = max(comp.max() for comp in c_components)
#     # global_min = 0
#     # global_max = 2

#     # Titles for each component
#     titles = [r"$c_{00}$", r"$c_{10}$", r"$c_{01}$", r"$c_{11}$"]

#     # Plot all components with shared color scale
#     for i, ax in enumerate(axes):
#         tcf = ax.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], c_components[i], levels=100,
#                              cmap="cividis", vmin=global_min, vmax=global_max)
#         ax.set_title(f"{titles[i]} at iter {iteration + 1}, cycle {cycle + 1}")
#         if i < 2:
#             ax.set_xlabel(r"$x$")
#         if i == 0 or i == 2:
#             ax.set_ylabel(r"$y$")

#     # Add one shared colorbar
#     fig_update.colorbar(tcf, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, boundaries=[global_min, global_max])

#     plt.tight_layout()
#     plt.show()

def c_tensor(State: "StateClass", Mesh: "MeshClass", iteration=1, cycle=1, stack=True):
    """
    Plot the 4 tensor components c00, c10, c01, c11 on the mesh using PyVista.
    Equivalent to the matplotlib tricontourf version, but interactive.
    """

    # Prepare PyVista grid
    grid = Mesh.pv_grid().copy()

    # Extract tensor components
    c_components = [State.c.x.array[i::4] for i in range(4)]

    # Global min/max for consistent color normalization
    global_min = min(comp.min() for comp in c_components)
    global_max = max(comp.max() for comp in c_components)

    # Titles for each component
    titles = [r"c00", r"c10", r"c01", r"c11"]

    # Choose layout
    shape = (2, 2) if stack else (1, 4)
    pl = pv.Plotter(shape=shape, window_size=(500, 500), border=False)

    # vertical scalar bars (screen coords 0..1)
    sbar_common = dict(vertical=True, position_x=Mesh.x_max+0.1, position_y=0.05, width=0.08, height=0.9)

    # Plot each tensor component
    for i, comp in enumerate(c_components):
        if stack:
            row, col = divmod(i, 2)
        else:
            row, col = (0, i)

        pl.subplot(row, col)
        grid_comp = grid.copy()
        grid_comp.point_data[titles[i]] = comp

        pl.add_mesh(grid_comp,
                    scalars=titles[i],
                    cmap="cividis",
                    clim=[global_min, global_max],
                    show_edges=False,
                    lighting=False,
                    scalar_bar_args=sbar_common)
        outline = _domain_outline_polyline(Mesh)
        pl.add_mesh(outline, color="black", line_width=2.0)

        pl.view_xy()
        pl.hide_axes()
        # pl.add_text(f"{titles[i]}", font_size=10)

    # Link camera views so you can zoom/pan all at once
    pl.link_views()
    pl.show()



def plot_mesh(Mesh):
    """
    Plot a dolfinx 2D mesh using matplotlib's Triangulation.

    Parameters
    ----------
    Mesh : MeshClass
        An object with attribute `.domain` being a dolfinx Mesh.
    """
    def triangulation_from_connectivity(dmesh):
        """
        Convert a dolfinx mesh into (x, y, triangles) arrays suitable for
        matplotlib.tri.Triangulation.

        Steps:
        - Access mesh topology (cell → vertex connectivity).
        - Loop over cells and build triangles (splitting quads/polygons if needed).
        - Collect node coordinates.
        """
        tdim = dmesh.topology.dim
        assert tdim == 2, f"Need 2D mesh, got {tdim}"

        # Build connectivity: cells (tdim) → vertices (dim 0)
        dmesh.topology.create_connectivity(tdim, 0)
        conn = dmesh.topology.connectivity(tdim, 0)
        idx, off = conn.array, conn.offsets  # CSR-like storage

        tris = []
        for i in range(len(off) - 1):
            # Vertex indices of cell i
            verts = idx[off[i]:off[i+1]]
            nv = len(verts)

            if nv == 3:
                # Already a triangle
                tris.append(verts)
            elif nv == 4:
                # Split quad into 2 triangles (0-1-2 and 0-2-3)
                tris.append([verts[0], verts[1], verts[2]])
                tris.append([verts[0], verts[2], verts[3]])
            elif nv > 4:
                # Generic fan triangulation for polygons with >4 vertices
                for k in range(1, nv - 1):
                    tris.append([verts[0], verts[k], verts[k+1]])

        tris = np.asarray(tris, dtype=np.int32)

        # Mesh node coordinates (geometry)
        X = dmesh.geometry.x
        return X[:, 0], X[:, 1], tris

    # Extract coordinates and triangle connectivity
    x, y, triangles = triangulation_from_connectivity(Mesh.domain)

    # Build matplotlib triangulation object
    tri = mtri.Triangulation(x, y, triangles)

    # Plot
    plt.figure(figsize=(6, 12))
    plt.triplot(tri, lw=0.3)       # edges only (wireframe)
    plt.gca().set_aspect('equal')  # square aspect ratio
    plt.title("Cell mesh")
    plt.show()


def _domain_outline_polyline(Mesh: "MeshClass"):
    # counter-clockwise + close the loop
    pts = np.array([
        [Mesh.x_min, Mesh.y_min, 0.0],
        [Mesh.x_max, Mesh.y_min, 0.0],
        [Mesh.x_max, Mesh.y_max, 0.0],
        [Mesh.x_min, Mesh.y_max, 0.0],
        [Mesh.x_min, Mesh.y_min, 0.0],
    ])
    # one polyline with 5 points -> "5, 0,1,2,3,4"
    lines = np.array([5, 0, 1, 2, 3, 4], dtype=np.int64)
    return pv.PolyData(pts, lines=lines)


def check_spline(Mesh: "MeshClass", spline, array_comparison=None):
    # Evaluate the spline at each y-value
    spline_vals = np.array([spline(np.array([0.0, y])) for y in Mesh.y_array])  # dummy x=0.0

    # Plot
    plt.figure(figsize=(6, 4))
    if array_comparison is not None and len(array_comparison) > 0:
        plt.plot(Mesh.y_array, array_comparison, '.k', label=r"array")
    plt.plot(Mesh.y_array, spline_vals, label=r"spline", color='k')
    plt.xlabel(r"$y$")
    plt.ylabel(r"$\ell(y)$")
    plt.title("Cubic spline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()