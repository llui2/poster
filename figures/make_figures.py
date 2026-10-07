from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

OUT = Path(__file__).resolve().parent

RHO_STAR = 0.5
THETA_A = 0.0
THETA_B = 1.0
THETA_T = -0.5
U_B = 1.0 / 6.0

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 160,
    }
)


def U(theta):
    return theta**2 / 2.0 - theta**3 / 3.0


def Up(theta):
    return theta - theta**2


def Upp(theta):
    return 1.0 - 2.0 * theta


def gp(theta):
    return 2.0 * theta


def rhs(_s, state, rho):
    theta, y = state
    root = np.sqrt(rho)
    return [
        root * y,
        (rho * gp(theta) - 1.0) * y - root * Up(theta),
    ]


def jacobian(theta, rho):
    root = np.sqrt(rho)
    return np.array(
        [
            [0.0, root],
            [-root * Upp(theta), rho * gp(theta) - 1.0],
        ]
    )


def eigendirection(theta, rho, stable=True):
    values, vectors = np.linalg.eig(jacobian(theta, rho))
    index = np.argmin(values.real) if stable else np.argmax(values.real)
    vector = np.real(vectors[:, index])
    return vector / np.linalg.norm(vector)


def manifold_branch(rho, stable=True, sign=1.0, horizon=100.0):
    state0 = (
        np.array([THETA_B, 0.0])
        + sign * 1e-7 * eigendirection(THETA_B, rho, stable)
    )
    span = (0.0, -horizon) if stable else (0.0, horizon)

    def boundary(_s, state):
        theta, y = state
        return min(theta + 0.95, 1.8 - theta, y + 1.0, 1.0 - y)

    boundary.terminal = True
    boundary.direction = 0

    solution = solve_ivp(
        lambda s, state: rhs(s, state, rho),
        span,
        state0,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.025,
        events=boundary,
    )
    return solution.y


# Landscape
theta = np.linspace(-0.72, 1.55, 900)

fig, ax = plt.subplots(figsize=(5.0, 3.45))
ax.plot(theta, U(theta), color="0.1", linewidth=2.2)
ax.axhline(U_B, color="0.55", linestyle="--", linewidth=1.2)
ax.plot(THETA_A, U(THETA_A), "ko", markersize=7)
ax.plot(
    THETA_B,
    U_B,
    marker="o",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.7,
    markersize=8,
)
ax.plot(THETA_T, U_B, "o", color="0.4", markersize=5)
ax.text(THETA_A, -0.018, r"$\theta_a$", ha="center", va="top")
ax.text(THETA_B, U_B + 0.018, r"$\theta_b$", ha="center", va="bottom")
ax.text(THETA_T + 0.03, U_B + 0.018, r"$\theta_t$", ha="left", va="bottom")
ax.set_xlim(-0.72, 1.55)
ax.set_ylim(-0.11, 0.40)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$U(\theta)$")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "landscape.pdf", bbox_inches="tight")
plt.close(fig)


# Main phase portrait
rho = 1.0
theta = np.linspace(-0.82, 1.66, 240)
y = np.linspace(-0.82, 0.82, 200)
THETA, Y = np.meshgrid(theta, y)
root = np.sqrt(rho)
DTHETA = root * Y
DY = (rho * gp(THETA) - 1.0) * Y - root * Up(THETA)

fig, ax = plt.subplots(figsize=(6.8, 5.15))
anti_damping = rho * gp(THETA) - 1.0 > 0.0
ax.contourf(
    THETA,
    Y,
    anti_damping.astype(float),
    levels=[0.5, 1.5],
    colors=["#dcecf7"],
    alpha=0.9,
)
ax.streamplot(
    theta,
    y,
    DTHETA,
    DY,
    density=1.15,
    color="0.80",
    linewidth=0.9,
    arrowsize=0.9,
)

theta_ref = np.linspace(THETA_T, THETA_B, 900)
y_ref = np.sqrt(np.maximum(2.0 * (U_B - U(theta_ref)), 0.0))
ax.plot(theta_ref, y_ref, "k--", linewidth=2.4)
ax.plot(theta_ref, -y_ref, "k--", linewidth=2.4)

for sign in (1.0, -1.0):
    branch = manifold_branch(rho, stable=True, sign=sign)
    ax.plot(branch[0], branch[1], linewidth=3.0, color="#292a90")

ax.plot(THETA_A, 0.0, "ko", markersize=8)
ax.plot(
    THETA_B,
    0.0,
    marker="o",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.7,
    markersize=9,
)
ax.annotate(
    "lost safe region",
    xy=(0.75, 0.23),
    xytext=(0.18, 0.58),
    fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=1.5, color="0.25"),
    color="0.25",
)
ax.text(
    0.97,
    0.96,
    r"$\rho=1>\rho^*$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=16,
)
ax.set_xlim(-0.82, 1.66)
ax.set_ylim(-0.82, 0.82)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$y$")
ax.set_xticks([THETA_T, THETA_A, THETA_B])
ax.set_xticklabels([r"$\theta_t$", r"$\theta_a$", r"$\theta_b$"])
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "basin_geometry.pdf", bbox_inches="tight")
plt.close(fig)


# Stable-manifold displacement onset
def stable_profile(rho, n=320):
    a = rho * gp(THETA_B) - 1.0
    lambda_s = 0.5 * (a - np.sqrt(a * a + 4.0 * rho))
    slope = lambda_s / np.sqrt(rho)

    theta0 = THETA_B - 1e-6
    y0 = slope * (theta0 - THETA_B)

    def dydtheta(theta, y):
        return [
            (rho * gp(theta) - 1.0) / np.sqrt(rho)
            - Up(theta) / y[0]
        ]

    solution = solve_ivp(
        dydtheta,
        (theta0, 0.0),
        [y0],
        rtol=1e-8,
        atol=1e-10,
        max_step=0.002,
        dense_output=True,
    )

    theta = np.linspace(0.0, theta0, n)
    y = solution.sol(theta)[0]
    theta = np.r_[theta, THETA_B]
    y = np.r_[y, 0.0]

    displacement = 0.5 * y**2 + U(theta) - U_B
    return theta, displacement


def delta_profile(rho, theta):
    integrand = (rho * gp(theta) - 1.0) * np.sqrt(
        np.maximum(U_B - U(theta), 0.0)
    )
    cumulative = cumulative_trapezoid(integrand, theta, initial=0.0)
    return np.sqrt(2.0 / rho) * (cumulative - cumulative[-1])


rho_values = np.linspace(0.12, 1.6, 56)
minimum_exact = []
minimum_first_order = []

for rho in rho_values:
    theta, exact = stable_profile(float(rho))
    minimum_exact.append(exact.min())
    minimum_first_order.append(delta_profile(float(rho), theta).min())

fig, ax = plt.subplots(figsize=(5.5, 3.7))
ax.plot(
    rho_values,
    minimum_exact,
    color="#292a90",
    linewidth=2.5,
    label=r"$D_{\min}$",
)
ax.plot(
    rho_values,
    minimum_first_order,
    color="0.20",
    linestyle="--",
    linewidth=2.0,
    label=r"$\Delta_{\min}$",
)
ax.axhline(0.0, color="0.7", linewidth=1.0)
ax.axvline(RHO_STAR, color="0.35", linestyle=":", linewidth=1.7)
ax.text(
    RHO_STAR + 0.025,
    0.95,
    r"$\rho^*=1/2$",
    transform=ax.get_xaxis_transform(),
    ha="left",
    va="top",
    fontsize=14,
)
ax.set_xlim(rho_values[0], rho_values[-1])
ax.set_xlabel(r"$\rho=\eta/\gamma$")
ax.set_ylabel("boundary displacement")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "threshold.pdf", bbox_inches="tight")
plt.close(fig)


# FitzHugh--Nagumo realization
b = 2.0
rho = 2.0
theta_a = np.sqrt(3.0 * (b - 1.0) / b)
theta_turn = np.sqrt(6.0 * (b - 1.0) / b)


def U_fhn(theta):
    return (1.0 - b) * theta**2 / 2.0 + b * theta**4 / 12.0


def Up_fhn(theta):
    return (1.0 - b) * theta + b * theta**3 / 3.0


def gp_fhn(theta):
    return b * (1.0 - theta**2)


def Upp_fhn(theta):
    return 1.0 - b + b * theta**2


def rhs_fhn(_s, state):
    theta, y = state
    root = np.sqrt(rho)
    return [
        root * y,
        (rho * gp_fhn(theta) - 1.0) * y - root * Up_fhn(theta),
    ]


def fhn_direction(stable=True):
    root = np.sqrt(rho)
    jac = np.array(
        [
            [0.0, root],
            [-root * Upp_fhn(0.0), rho * gp_fhn(0.0) - 1.0],
        ]
    )
    values, vectors = np.linalg.eig(jac)
    index = np.argmin(values.real) if stable else np.argmax(values.real)
    vector = np.real(vectors[:, index])
    return vector / np.linalg.norm(vector)


def fhn_branch(sign, stable=True):
    state0 = sign * 1e-7 * fhn_direction(stable)
    span = (0.0, -80.0) if stable else (0.0, 80.0)

    def boundary(_s, state):
        theta, y = state
        return min(theta + 2.0, 2.0 - theta, y + 1.25, 1.25 - y)

    boundary.terminal = True
    boundary.direction = 0

    return solve_ivp(
        rhs_fhn,
        span,
        state0,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
        events=boundary,
    ).y


fig, ax = plt.subplots(figsize=(5.5, 4.1))

for left, right in [(-theta_turn, 0.0), (0.0, theta_turn)]:
    theta = np.linspace(left, right, 500)
    y = np.sqrt(np.maximum(-2.0 * U_fhn(theta), 0.0))
    ax.plot(theta, y, "k--", linewidth=2.0)
    ax.plot(theta, -y, "k--", linewidth=2.0)

for sign in (1.0, -1.0):
    branch = fhn_branch(sign, stable=True)
    ax.plot(branch[0], branch[1], color="#292a90", linewidth=2.7)

ax.plot([-theta_a, theta_a], [0.0, 0.0], "ko", markersize=7)
ax.plot(
    0.0,
    0.0,
    marker="o",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.5,
    markersize=8,
)
ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.15, 1.15)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$y$")
ax.text(
    0.97,
    0.95,
    r"FitzHugh--Nagumo: $\rho^*=1/b$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=13,
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fhn.pdf", bbox_inches="tight")
plt.close(fig)

print("generated poster figures")
