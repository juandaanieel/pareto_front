# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

from pymoo.util.ref_dirs import get_reference_directions

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultMultiObjectiveTermination


# %% [markdown]
# # Pareto optimization
#
# Multi-objective optimization (MOO) tries to optimise problems where there is not a single optimal solution, but instead a set of optimal trade-offs.
# Here, I prepare a small tutorial on how to use the python packagae [pymoo](https://pymoo.org) to perform Pareto optimisation.
#
# The elements of a MOO problem are:
#
# 1. Well-defined optimisation functions with bounds and constraints if any.
# 2. Algorithm used to solve the MOO problem. List of algorithms is available in `pymoo` website.
# 3. Termination conditions. This can be tricky since there are many criteria for what termination means. By default I just let the algorithm run for a long time.
#
# ## Basic Pareto optimization
#
# ### Problem definition
#
# We will consider two functions $f_1(x, y)$ and $f_2(x, y)$ defined over a range $-1<x, y<6$ with no constrains.

# %%
class pareto_problem_1(Problem):
    """
    A simple bi-objective problem with two decision variables.

    Parameters:
    ---------
    f1 : callable
        The first objective function.
    f2 : callable
        The second objective function.
    """

    def __init__(self, f1, f2):

        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=[-1, -1],
            xu=[6, 6],
        )

        self.f1 = f1
        self.f2 = f2

    def _evaluate(self, x, out):
        """
        Evaluate the objective functions for the given variables.
        """
        # These functions required to be vectorised
        _f1 = self.f1(x)
        _f2 = self.f2(x)
        y = np.column_stack([_f1, _f2])
        out["F"] = np.array(y)


# %%
shift = 5
# Example functions to optimize
def f1(xyz, f_noise=None):
    x, y = xyz[:, 0], xyz[:, 1]
    if f_noise is not None:
        noise = f_noise(xyz)
        return (x**2 + y**4 + noise)
    else:
        return (x**2 + y**4)

def f2(xyz, f_noise=None):
    x, y = xyz[:, 0], xyz[:, 1]
    if f_noise is not None:
        noise = f_noise(xyz)
        return ((x - shift)**2 + (y - shift/2)**4 + noise)
    else:
        return ((x-shift)**2+(y - shift/2)**4)


# %%
problem = pareto_problem_1(f1=f1, f2=f2)

n_pop = 20
algorithm = NSGA2(
    pop_size=n_pop,
    n_offsprings=n_pop,
    sampling=FloatRandomSampling(),
    eliminate_duplicates=True,
)

n_generations = [10, 25, 50, 100]
results = []

for n_gen in n_generations:
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        save_history=True,
        verbose=True,
    )
    results.append(res)

# %%
_X = np.linspace(-1, 6, 100)
_Y = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(_X, _Y)
Z1 = f1(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
Z2 = f2(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)


# %%
colors = ['C0', 'C1', 'C2', 'C3']
levels = [0, 1, 10, 20, 40, 60, 80, 100]
fig, ax = plt.subplots(ncols=3, figsize=(10, 3), layout='constrained')

ax[0].scatter(Z2, Z1, c="k", s=1, alpha=0.4)

for i, result in enumerate(results):
    result_1 = result.F
    order = np.argsort(result_1[:, 0])
    ax[0].plot(
        result_1[order, 1], result_1[order, 0], 'o',
        color=colors[i],alpha=0.8, label=fr"$n_{{gen}}$ {n_generations[i]}", markersize=3
        )
    ax[0].plot(result_1[order, 1], result_1[order, 0], color=colors[i])

ax[0].legend(frameon=True, title=r"Pareto Fronts:")
ax[0].set_xlabel(r"$f_1$")
ax[0].set_ylabel(r"$f_2$")
ax[0].set_title(fr"$n_{{population}} = {n_pop}$")



ax[1].pcolormesh(X, Y, Z1, shading="auto", cmap="inferno", vmin=0, vmax=110)
ax[1].contour(X, Y, Z1, levels=levels, colors='white', linewidths=0.5)
ax[1].set_title(r"$f_1(x,y)$")
im = ax[2].pcolormesh(X, Y, Z2, shading="auto", cmap="inferno", vmin=0, vmax=110)
ax[2].contour(X, Y, Z2, levels=levels, colors='white', linewidths=0.5)
ax[2].set_title(r"$f_2(x,y)$")

for i, result in enumerate(results):
    x, y = result.X.T        
    ax[1].scatter(x, y, c=colors[i], s=20)
    ax[2].scatter(x, y, c=colors[i], s=20)
cbar = fig.colorbar(im, ax=ax[2], orientation="vertical")
cbar.set_label(r"$f_1, f_2$")

ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[2].set_xlabel("x")
ax[2].set_yticklabels([])

fig.suptitle("Pareto Fronts and Objective Functions")

ax[0].set_xlim(-5, 60)
ax[0].set_ylim(-5, 70)

# %% [markdown]
# ### Analysing stored data
#
# During the optimisation procedure, we can store intermediate data by setting `history=True` in the arguments of `minimize`.
# Then, we can access the history of the optimization as `result.history` which is a list of the executions of the algorithm.
#
# The elements that are relevant from this object are:
# - `result.history[0].pop`: all points computed in the first iteration of the algorithm
# - `result.history[0].opt`: all optimal points found in the first iteration

# %%
# There are as many elements in the history as there are generations
print("Number of elements in history:", len(results[0].history))
# The population size is the same as n_pop
print("Population size of one generation:", results[0].history[0].pop_size)
# The number of optimal points varies per generation
print("Number of optimal points in the last generation:", len(results[-1].history[0].opt))


# %%
# We can extract the optimal points from the last generation
opt = results[-1].history[0].opt
evaluated_points = opt.get("X")
optimal_values = opt.get("F")

# %%
index = 1
# By using all this data, we can take a look at the evolution of the optimization
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), layout='constrained')
colors_history = plt.cm.viridis(np.linspace(0, 1, len(results[index].history)))

for i, _element in enumerate(results[index].history):
    opt = _element.opt
    evaluated_points = opt.get("X")
    optimal_values = opt.get("F")

    ax[0].scatter(evaluated_points[:, 0], evaluated_points[:, 1], s=20, alpha=0.5, color=colors_history[i])
    ax[1].scatter(optimal_values[:, 0], optimal_values[:, 1], s=20, alpha=0.5, color=colors_history[i])

ax[0].set_title("Evaluated Points")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[1].set_title("Optimal Values")
ax[1].set_xlabel(r"$f_1$")
ax[1].set_ylabel(r"$f_2$")

# Add a colorbar for the generation colors
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(results[index].history)-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Generation')

# %% [markdown]
# ## Pareto optimisation with noise
#
# We will assume that there is quasi-static noise in the system and we will study the convergence of the Pareto front.
#
# To do so we will generate a mesh of random numbers, and we will interpolate for the values outside of the mesh.

# %%
from scipy.interpolate import RegularGridInterpolator

def f_noise(X, Y, noise_mesh, noise_norm=1):
    """
    Function to return the noise value at given X, Y coordinates.
    """
        
    interpolator = RegularGridInterpolator(
        (Y, X),
        noise_mesh * noise_norm, 
        bounds_error=False,
        fill_value=0.0
        )
    
    return interpolator


# %%
_X = np.linspace(-1, 6, 100)
_Y = np.linspace(-1, 6, 100)

# Generate mesh of random numbers in the range of _X and _Y
np.random.seed(42)  # For reproducible results
noise_mesh = np.random.uniform(-1, 1, size=X.shape)

n_pop = 20
algorithm = NSGA2(
    pop_size=n_pop,
    n_offsprings=n_pop,
    sampling=FloatRandomSampling(),
    eliminate_duplicates=True,
)

n_generations = 25
noise_strengths = [0.1, 1.0, 5.0, 10.0]

data_noise = []

for noise_strength in noise_strengths:
    # Redefine the functions to include noise
    f_noise_func = f_noise(_Y, _X, noise_mesh=noise_mesh, noise_norm=noise_strength)
    _f1 = lambda xyz: f1(xyz, f_noise=f_noise_func)
    _f2 = lambda xyz: f2(xyz, f_noise=f_noise_func)

    problem = pareto_problem_1(f1=_f1, f2=_f2)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        save_history=False,
        verbose=False,
    )

    data_noise.append(res)

# %%
_X = np.linspace(-1, 6, 100)
_Y = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(_X, _Y)
Z1 = f1(np.column_stack([X.ravel(), Y.ravel()]), f_noise=f_noise_func).reshape(X.shape)
Z2 = f2(np.column_stack([X.ravel(), Y.ravel()]), f_noise=f_noise_func).reshape(X.shape)


# %%
colors = ['C0', 'C1', 'C2', 'C3']
fig, ax = plt.subplots(ncols=3, figsize=(10, 3), layout='constrained')

ax[0].scatter(Z2, Z1, c="k", s=1, alpha=0.4)

i = -1
result_1 = data_noise[-1].F
order = np.argsort(result_1[:, 0])
ax[0].plot(result_1[order, 1], result_1[order, 0], 'o', color=colors[i], alpha=0.8, label=f"{noise_strengths[i]}", markersize=3)
ax[0].plot(result_1[order, 1], result_1[order, 0], color=colors[i])

ax[0].legend(frameon=True, title=r"Pareto Fronts for $A_{noise}$:")
ax[0].set_xlabel(r"$f_1$")
ax[0].set_ylabel(r"$f_2$")
ax[0].set_title(fr"$n_{{population}} = {n_pop}$, $n_{{gen}} = {n_generations}$")

i = -1
res = results[i]
x, y = res.X.T

ax[1].pcolormesh(X, Y, Z1, shading="auto", cmap="inferno", vmin=0, vmax=100)
ax[1].contour(X, Y, Z1, levels=[0, 1, 10, 20, 100], colors='white', linewidths=0.5)
ax[1].set_title(r"$f_1(x,y) + A_{noise}(x, y)$")
im = ax[2].pcolormesh(X, Y, Z2, shading="auto", cmap="inferno", vmin=0, vmax=100)
ax[2].contour(X, Y, Z2, levels=[0, 1, 10, 20, 100], colors='white', linewidths=0.5)
ax[2].set_title(r"$f_2(x,y) + A_{noise}(x, y)$")
ax[1].scatter(x, y, c=colors[i], s=20)
ax[2].scatter(x, y, c=colors[i], s=20)
cbar = fig.colorbar(im, ax=ax[2], orientation="vertical")
cbar.set_label(r"$f_1, f_2$")

ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[2].set_xlabel("x")
ax[2].set_yticklabels([])

fig.suptitle("Pareto Fronts and Objective Functions")

ax[0].set_xlim(-20, 60)
ax[0].set_ylim(-20, 70)

# %% [markdown]
# ## Second example: Variables of different type

# %%
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


# %%
class MixedVariableProblem(Problem):

    def __init__(self, f1, f2):
        vars = {
            "int_var": Integer(bounds=(-1, 6)),
            "cont_var": Real(bounds=(-1, 6)),
        }
        super().__init__(vars=vars, n_obj=2)
        self.f1 = f1
        self.f2 = f2

    def _evaluate(self, X, out, *args, **kwargs):
        
        xs, ys= [], []
        for _dict in X:
            xs.append(_dict["int_var"])
            ys.append(_dict["cont_var"])

        z = np.vstack([xs, ys]).T
        _f1 = self.f1(z)
        _f2 = self.f2(z)
        y = np.column_stack([_f1, _f2])
        
        out["F"] = np.array(y)


# %%
n_pop = 20
n_gen = 25

problem = MixedVariableProblem(f1=f1, f2=f2)

algorithm = MixedVariableGA(pop_size=n_pop, survival=RankAndCrowdingSurvival())

res = minimize(problem,
               algorithm,
               ('n_gen', n_gen),
               seed=1,
               verbose=False)

# %%
xs, ys = [], []
for _dict in res.X:
    xs.append(_dict["int_var"])
    ys.append(_dict["cont_var"])

xs = np.array(xs)
ys = np.array(ys)

# %%
_X = np.linspace(-1, 6, 100)
_Y = np.linspace(-1, 6, 100)
X, Y = np.meshgrid(_X, _Y)
Z1 = f1(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
Z2 = f2(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)


# %%
colors = ['C0', 'C1', 'C2', 'C3']
fig, ax = plt.subplots(ncols=3, figsize=(10, 3), layout='constrained')

ax[0].scatter(Z2, Z1, c="k", s=1, alpha=0.4)

i = -1
result_1 = res.F
order = np.argsort(result_1[:, 0])
ax[0].plot(result_1[order, 1], result_1[order, 0], 'o', color=colors[i], alpha=0.8, label=f"{noise_strengths[i]}", markersize=3)
ax[0].plot(result_1[order, 1], result_1[order, 0], color=colors[i])

ax[0].legend(frameon=True, title=r"Pareto Fronts for $A_{noise}$:")
ax[0].set_xlabel(r"$f_1$")
ax[0].set_ylabel(r"$f_2$")
ax[0].set_title(fr"$n_{{population}} = {n_pop}$, $n_{{gen}} = {n_generations}$")

i = -1
res = results[i]
x, y = res.X.T

ax[1].pcolormesh(X, Y, Z1, shading="auto", cmap="inferno", vmin=0, vmax=100)
ax[1].contour(X, Y, Z1, levels=levels, colors='white', linewidths=0.5)
ax[1].set_title(r"$f_1(x,y) + A_{noise}(x, y)$")
im = ax[2].pcolormesh(X, Y, Z2, shading="auto", cmap="inferno", vmin=0, vmax=100)
ax[2].contour(X, Y, Z2, levels=levels, colors='white', linewidths=0.5)
ax[2].set_title(r"$f_2(x,y) + A_{noise}(x, y)$")
ax[1].scatter(xs, ys, c=colors[i], s=20)
ax[2].scatter(xs, ys, c=colors[i], s=20)
cbar = fig.colorbar(im, ax=ax[2], orientation="vertical")
cbar.set_label(r"$f_1, f_2$")

ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[2].set_xlabel("x")
ax[2].set_yticklabels([])

fig.suptitle("Pareto Fronts and Objective Functions")

ax[0].set_xlim(-5, 60)
ax[0].set_ylim(-5, 70)

# %% [markdown]
# ## Other possible settings
#
# There are other possible settings that are relevant for the Pareto optimization, but in our case (limited by the number of function calls) they are not super important.
# Anyway, here are some other configuration settings if you are curious.
#
# ### Termination condition
#
# ```python
# termination = DefaultMultiObjectiveTermination(
#     xtol=1e-12,
#     cvtol=1e-12,
#     ftol=1e-12,
#     period=30,
#     n_max_gen=1000,
#     n_max_evals=500000
# )
# ```
#
# ### Other algorithms
#
# We use the algorithm `NSGA2`, which is good for an initial approach, but there are other relevant algorithms.
# They can be implement in the following way, for example:
#
# ```python
#
# ref_dirs = np.linspace([0, 1], [1, 0], 100)
#
# algorithm = MOEAD(
#     ref_dirs,
#     n_neighbors=25,
#     prob_neighbor_mating=0.4,
# )
#
# res_MOED = minimize(
#             problem,
#             algorithm,
#             ("n_gen", 2000),
#             save_history=False,
#             verbose=False,
#         )
#
# ```
