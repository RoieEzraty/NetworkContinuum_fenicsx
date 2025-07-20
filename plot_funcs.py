import matplotlib.pyplot as plt
import numpy as np

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


def measurement_fields(State: "StateClass", Supervisor: "SupervisorClass", Mesh, iteration=1, cycle=1, num=1):
    if num == 1:
        Loss = Supervisor.Loss
    elif num == 2:
        Loss = Supervisor.Loss_2

    # instantiate figure during measurement and grid
    fig_measure = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], wspace=0.45)
    ax0 = fig_measure.add_subplot(gs[0, 0])
    ax1 = fig_measure.add_subplot(gs[0, 1])
    ax2 = fig_measure.add_subplot(gs[1, 0])
    ax3 = fig_measure.add_subplot(gs[1, 1])

    # p
    ax0.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], State.p.x.array, levels=100, cmap="viridis")
    ax0.set_title(r"$p$ at iteration ${}$, cycle ${}$".format(iteration + 1, cycle + 1))
    ax0.set_xlabel(r"$x$")
    ax0.set_ylabel(r"$y$")
    # ax0.colorbar(label="Pressure")

    # Q
    absQ = State.absQ.x.array
    ax1.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], absQ, levels=100, cmap="plasma")
    ax1.set_title(r"$\|Q\|$ at iteration ${}$, cycle ${}$".format(iteration + 1, cycle + 1))
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")

    # Q_x at right boundary
    ax2.plot(Mesh.coords_right[:, 1], State.Q_x_right, 'b', label='Q_x')
    ax2.plot(Mesh.y_array, Supervisor.target.array, 'k', label='Target')
    ax2.plot(Mesh.coords_right[:, 1], Loss, 'r', label='Loss')
    ax2.set_xlabel(r"$y$")
    ax2.legend()

    if num == 1:
        # BADALINE for BCs
        ax3.plot(Mesh.coords_right[:, 1], Supervisor.update.l_array, 'k', label='BADALINE left')
        ax3.plot(Mesh.coords_right[:, 1], Supervisor.update.r_array, 'k--', label='BADALINE right')
        ax3.set_xlabel(r"$y$")
        ax3.legend()
    elif num == 2:
        # BADALINE for BCs
        ax3.plot(Mesh.coords_right[:, 1], Supervisor.update_2.l_array, 'k', label='BADALINE left')
        ax3.plot(Mesh.coords_right[:, 1], Supervisor.update_2.r_array, 'k--', label='BADALINE right')
        ax3.set_xlabel(r"$y$")
        ax3.legend()

    plt.tight_layout()
    plt.show()


def update_fields(State: "StateClass", Supervisor: "SupervisorClass", Mesh: "MeshClass", iteration=1, cycle=1):
    # instantiate figure during measurement and grid
    fig_update = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], wspace=0.45)
    ax0u = fig_update.add_subplot(gs[0, 0])
    ax1u = fig_update.add_subplot(gs[0, 1])
    ax2u = fig_update.add_subplot(gs[1, 0])
    ax3u = fig_update.add_subplot(gs[1, 1])

    # p_sol
    ax0u.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], State.p_update.x.array, levels=100, cmap="viridis")
    ax0u.set_title(r"$p^!$ at iteration ${}$, cycle ${}$".format(iteration + 1, cycle + 1))
    ax0u.set_xlabel(r"$x$")
    ax0u.set_ylabel(r"$y$")

    # |Q|
    if State.c_type == 'tensor':
        absQ_update = State.absQ_update.x.array[:len(Mesh.coords[:, 0])] + State.absQ_update.x.array[-len(Mesh.coords[:, 0]):]
    else:
        absQ_update = State.absQ_update.x.array
    ax1u.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], absQ_update, levels=100, cmap="plasma")
    # ax1u.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], State.absQ_update.x.array, levels=100, cmap="plasma")
    ax1u.set_title(r"$\|Q^!\|$ at iteration ${}$, cycle ${}$".format(iteration + 1, cycle + 1))
    ax1u.set_xlabel(r"$x$")
    ax1u.set_ylabel(r"$y$")

    # c
    ax2u.tricontourf(Mesh.coords[:, 0], Mesh.coords[:, 1], State.c.x.array, levels=100, cmap="plasma")
    ax2u.set_title(r"$c$")
    ax2u.set_xlabel(r"$x$")
    ax2u.set_ylabel(r"$y$")

    # Update modality values
    update_bc_fn_l = np.array([Supervisor.update_bc_fn_l([0.0, y]) for y in Mesh.y_array])
    update_bc_fn_r = np.array([Supervisor.update_bc_fn_r([0.0, y]) for y in Mesh.y_array])
    ax3u.plot(Mesh.coords_right[:, 1], update_bc_fn_l, 'k', label='BADALINE left')
    ax3u.plot(Mesh.coords_right[:, 1], update_bc_fn_r, 'k--', label='BADALINE right')
    ax3u.set_xlabel(r"$y$")
    ax3u.legend()

    plt.tight_layout()
    plt.show()


def Loss_vec(Supervisor: "SupervisorClass"):
    plt.figure(figsize=(3, 3))
    plt.plot(Supervisor.Loss_vec, 'k')
    plt.ylabel('Loss')
    plt.xlabel('iteration')

    plt.tight_layout()
    plt.show()