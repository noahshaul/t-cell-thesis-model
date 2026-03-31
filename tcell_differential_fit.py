#!/usr/bin/env python3
"""
Two-phase differential evolution fitting for T cell dynamics.

Phase 1 (100 iterations): Fit 7 population-dynamics parameters
    r_V, clearanceRate, K0_ns, K0_nl, K0_sl, gamma_S, sigma_S
    Loss: viral load + total cell counts + cell proportions
    Gene expression parameters held at homeostatic defaults.

Phase 2 (100 iterations): Fit 13 gene-expression (GRN) parameters
    alpha0t, alpha0p, alpha_tn, alpha_ts, alpha_tl,
    alpha_pn, alpha_ps, alpha_pl, delta_t, delta_p, R_t, R_p, R_v
    Loss: per-cell gene expression only
    Population parameters frozen at Phase 1 result.
    Gene expression only evaluated when cell type > 1 % of total (avoids 0/eps artifacts).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Fixed model constants  (DO NOT CHANGE)
# =============================================================================
r_N = 0.00
C_N = 1000
C_S = 320000
C_V = 1e17
epsilon = 1e-9
n = 3

# Fixed initial conditions for populations  (DO NOT CHANGE)
N0 = 1000.0
S0 = 0.0
L0 = 0.0
V0 = 2e5

# Baseline per-cell gene expression at day 0
BASELINE_TCF7 = 3.1171
BASELINE_PRDM1 = 0.0117

# Initial transcript totals derived from baseline per-cell values
Tn0 = N0 * BASELINE_TCF7    # 3117.1
Pn0 = N0 * BASELINE_PRDM1   # 11.7
Ts0 = 0.0
Tl0 = 0.0
Ps0 = 0.0
Pl0 = 0.0

X0_state = [N0, S0, L0, V0, Tn0, Ts0, Tl0, Pn0, Ps0, Pl0]

# =============================================================================
# Training data
# =============================================================================
days_counts = np.array([0, 3, 4, 5, 6, 7, 10, 14, 21, 32, 60, 90], dtype=float)
cell_counts_total = np.array([
    1000, 1682.40, 1822.97, 6148.03, 20282.86, 29493.89,
    37057.11, 20414.37, 11273.47, 6043.75, 3041.94, 1468.86
])

days_proportions = np.array([0, 3, 4, 5, 6, 7, 10, 14, 21, 32, 60, 90], dtype=float)
naive_prop = np.array([
    1.000000, 0.723500, 0.254647, 0.101551, 0.122122, 0.069267,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
])
effector_prop = np.array([
    0.000000, 0.276500, 0.745353, 0.898449, 0.877878, 0.930733,
    0.860250, 0.798608, 0.565457, 0.713000, 0.612484, 0.198500
])
memory_prop = np.array([
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.139750, 0.201392, 0.434543, 0.287000, 0.387516, 0.801500
])

gene_expression_data = {
    0:  {'naive':    {'Tcf7': 3.1171, 'Prdm1': 0.0117}},
    3:  {'naive':    {'Tcf7': 3.7951, 'Prdm1': 0.3818},
         'effector': {'Tcf7': 1.5086, 'Prdm1': 1.4508}},
    4:  {'naive':    {'Tcf7': 2.5115, 'Prdm1': 0.8490},
         'effector': {'Tcf7': 0.4272, 'Prdm1': 1.2132}},
    5:  {'naive':    {'Tcf7': 3.7661, 'Prdm1': 0.2416},
         'effector': {'Tcf7': 0.3717, 'Prdm1': 0.8421}},
    6:  {'naive':    {'Tcf7': 4.9245, 'Prdm1': 0.0896},
         'effector': {'Tcf7': 0.9784, 'Prdm1': 0.3624}},
    7:  {'naive':    {'Tcf7': 3.2337, 'Prdm1': 0.0595},
         'effector': {'Tcf7': 0.3856, 'Prdm1': 0.3532}},
    10: {'memory':   {'Tcf7': 3.0793, 'Prdm1': 0.1100},
         'effector': {'Tcf7': 0.7475, 'Prdm1': 0.3850}},
    14: {'memory':   {'Tcf7': 2.8308, 'Prdm1': 0.1139},
         'effector': {'Tcf7': 0.8378, 'Prdm1': 0.3911}},
    21: {'memory':   {'Tcf7': 2.9174, 'Prdm1': 0.0834},
         'effector': {'Tcf7': 1.3140, 'Prdm1': 0.2743}},
    32: {'memory':   {'Tcf7': 2.9119, 'Prdm1': 0.0866},
         'effector': {'Tcf7': 1.3309, 'Prdm1': 0.3093}},
    60: {'memory':   {'Tcf7': 2.7647, 'Prdm1': 0.1561},
         'effector': {'Tcf7': 1.2744, 'Prdm1': 0.3678}},
    90: {'memory':   {'Tcf7': 3.1751, 'Prdm1': 0.1842},
         'effector': {'Tcf7': 1.5410, 'Prdm1': 0.5562}},
}

# Days 10-90 have no naive data → use baseline as pseudo-target
_naive_late_days = [d for d in gene_expression_data if 'naive' not in gene_expression_data[d]]

# Viral-load target function (Phase 1 only)
def get_target_viral_load(day):
    if day <= 4:
        return V0 + (1e7 - V0) * (day / 4.0)
    elif day <= 14:
        return 1e7 * np.exp(-0.8 * (day - 4))
    else:
        return 1e-3

# =============================================================================
# Gene expression normalisation ranges
# =============================================================================
_all_tcf7  = [ct['Tcf7']  for d in gene_expression_data.values() for ct in d.values()]
_all_prdm1 = [ct['Prdm1'] for d in gene_expression_data.values() for ct in d.values()]
TCF7_RANGE  = max(_all_tcf7)  - min(_all_tcf7)
PRDM1_RANGE = max(_all_prdm1) - min(_all_prdm1)

# =============================================================================
# Homeostatic gene expression defaults used in Phase 1
# Derived by solving dTn/dt = 0 and dPn/dt = 0 at t=0 with S=L=0.
# This gives:
#   alpha0t = delta_t * BASELINE_TCF7  - alpha_tn * H_t_on(tn0) * H_p_off(pn0)
#   alpha0p = delta_p * BASELINE_PRDM1 - alpha_pn * H_p_on(pn0) * H_t_off(tn0) * v_act0
# =============================================================================
DELTA_T_DEFAULT  = 0.5
DELTA_P_DEFAULT  = 0.1
R_T_DEFAULT      = 1.5
R_P_DEFAULT      = 0.5
R_V_DEFAULT      = 1e5
ALPHA_TN_DEFAULT = 0.5
ALPHA_TS_DEFAULT = 0.3
ALPHA_TL_DEFAULT = 0.2
ALPHA_PN_DEFAULT = 0.3
ALPHA_PS_DEFAULT = 0.5
ALPHA_PL_DEFAULT = 0.2

_tn0      = BASELINE_TCF7
_pn0      = BASELINE_PRDM1
_H_t_on0  = _tn0**n  / (R_T_DEFAULT**n  + _tn0**n)
_H_p_off0 = R_P_DEFAULT**n / (R_P_DEFAULT**n + _pn0**n)
_H_p_on0  = _pn0**n  / (R_P_DEFAULT**n  + _pn0**n)
_H_t_off0 = R_T_DEFAULT**n / (R_T_DEFAULT**n + _tn0**n)
_v_act0   = V0**n    / (R_V_DEFAULT**n   + V0**n)

ALPHA0T_DEFAULT = max(
    DELTA_T_DEFAULT * BASELINE_TCF7  - ALPHA_TN_DEFAULT * _H_t_on0  * _H_p_off0,
    1e-6
)
ALPHA0P_DEFAULT = max(
    DELTA_P_DEFAULT * BASELINE_PRDM1 - ALPHA_PN_DEFAULT * _H_p_on0  * _H_t_off0 * _v_act0,
    1e-8
)

# =============================================================================
# ODE model  (DO NOT MODIFY)
# =============================================================================
def tcell_model(t, X, free_params):
    """
    States: N, S, L, V, Tn, Ts, Tl, Pn, Ps, Pl
    """
    N, S, L, V, Tn, Ts, Tl, Pn, Ps, Pl = X

    N  = max(N,  0)
    S  = max(S,  0)
    L  = max(L,  0)
    V  = max(V,  epsilon)
    Tn = max(Tn, 0)
    Ts = max(Ts, 0)
    Tl = max(Tl, 0)
    Pn = max(Pn, 0)
    Ps = max(Ps, 0)
    Pl = max(Pl, 0)

    (
        r_V, clearanceRate,
        K0_ns, K0_nl, K0_sl,
        alpha0t, alpha0p,
        alpha_tn, alpha_ts, alpha_tl,
        alpha_pn, alpha_ps, alpha_pl,
        delta_t, delta_p,
        R_t, R_p, R_v, gamma_S, sigma_S
    ) = free_params

    tn = Tn / (N + epsilon)
    ts = Ts / (S + epsilon)
    tl = Tl / (L + epsilon)

    pn = Pn / (N + epsilon)
    ps = Ps / (S + epsilon)
    pl = Pl / (L + epsilon)

    v_act = V**n / (R_v**n + V**n)

    K_ns = K0_ns * (pn**n / (R_p**n + pn**n)) * (R_t**n / (R_t**n + tn**n))
    K_nl = K0_nl * (tn**n / (R_t**n + tn**n)) * (R_p**n / (R_p**n + pn**n)) * v_act
    K_sl = K0_sl * (ts**n / (R_t**n + ts**n)) * (R_p**n / (R_p**n + ps**n))

    dN_dt = r_N * N * (1 - N / (C_N + epsilon)) - (K_ns + K_nl) * N
    dS_dt = (gamma_S * v_act - sigma_S) * S * (1 - S / (C_S + epsilon)) + K_ns * N - K_sl * S
    dL_dt = K_nl * N + K_sl * S

    dV_dt = r_V * V * (1 - V / (C_V + epsilon)) - clearanceRate * V * S

    dTn_dt = (
        alpha0t * N
        + alpha_tn * N * (tn**n / (R_t**n + tn**n)) * (R_p**n / (R_p**n + pn**n))
        - delta_t * Tn
        - (K_ns + K_nl) * Tn
    )

    dTs_dt = (
        alpha0t * S
        + alpha_ts * S * (ts**n / (R_t**n + ts**n)) * (R_p**n / (R_p**n + ps**n))
        - delta_t * Ts
        + K_ns * Tn
        - K_sl * Ts
    )

    dTl_dt = (
        alpha0t * L
        + alpha_tl * L * (tl**n / (R_t**n + tl**n)) * (R_p**n / (R_p**n + pl**n))
        - delta_t * Tl
        + K_nl * Tn
        + K_sl * Ts
    )

    dPn_dt = (
        alpha0p * N
        + alpha_pn * N * (pn**n / (R_p**n + pn**n)) * (R_t**n / (R_t**n + tn**n)) * v_act
        - delta_p * Pn
        - (K_ns + K_nl) * Pn
    )

    dPs_dt = (
        alpha0p * S
        + alpha_ps * S * (ps**n / (R_p**n + ps**n)) * (R_t**n / (R_t**n + ts**n)) * v_act
        - delta_p * Ps
        + K_ns * Pn
        - K_sl * Ps
    )

    dPl_dt = (
        alpha0p * L
        + alpha_pl * L * (pl**n / (R_p**n + pl**n)) * (R_t**n / (R_t**n + tl**n)) * v_act
        - delta_p * Pl
        + K_nl * Pn
        + K_sl * Ps
    )

    return [dN_dt, dS_dt, dL_dt, dV_dt, dTn_dt, dTs_dt, dTl_dt, dPn_dt, dPs_dt, dPl_dt]

# =============================================================================
# ODE runner helper
# =============================================================================
def run_ode(free_params, t_eval, max_step=0.5):
    """Integrate the model and return the solution object, or None on failure."""
    t_eval = np.asarray(t_eval, dtype=float)
    t_span = (t_eval[0], t_eval[-1])
    try:
        sol = solve_ivp(
            tcell_model,
            t_span,
            X0_state,
            args=(free_params,),
            method='RK45',
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-3,
            atol=1e-5,
            dense_output=False,
        )
        if sol.success and np.all(np.isfinite(sol.y)):
            return sol
        return None
    except Exception:
        return None


def _find_idx(t_sol, day, tol=0.05):
    """Return index of the nearest point in t_sol within tolerance, else None."""
    diffs = np.abs(t_sol - day)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] <= tol else None

# =============================================================================
# Full parameter vector builders
# =============================================================================
def _build_p1(pop_params):
    """20-param tuple for Phase 1 (gene expression at homeostatic defaults)."""
    r_V, clearanceRate, K0_ns, K0_nl, K0_sl, gamma_S, sigma_S = pop_params
    return (
        r_V, clearanceRate,
        K0_ns, K0_nl, K0_sl,
        ALPHA0T_DEFAULT, ALPHA0P_DEFAULT,
        ALPHA_TN_DEFAULT, ALPHA_TS_DEFAULT, ALPHA_TL_DEFAULT,
        ALPHA_PN_DEFAULT, ALPHA_PS_DEFAULT, ALPHA_PL_DEFAULT,
        DELTA_T_DEFAULT,  DELTA_P_DEFAULT,
        R_T_DEFAULT, R_P_DEFAULT, R_V_DEFAULT,
        gamma_S, sigma_S,
    )


def _build_p2(ge_params, pop_frozen):
    """20-param tuple for Phase 2 (population params frozen)."""
    r_V, clearanceRate, K0_ns, K0_nl, K0_sl, gamma_S, sigma_S = pop_frozen
    (alpha0t, alpha0p,
     alpha_tn, alpha_ts, alpha_tl,
     alpha_pn, alpha_ps, alpha_pl,
     delta_t, delta_p, R_t, R_p, R_v) = ge_params
    return (
        r_V, clearanceRate,
        K0_ns, K0_nl, K0_sl,
        alpha0t, alpha0p,
        alpha_tn, alpha_ts, alpha_tl,
        alpha_pn, alpha_ps, alpha_pl,
        delta_t, delta_p,
        R_t, R_p, R_v,
        gamma_S, sigma_S,
    )

# =============================================================================
# Phase 1 evaluation grid and loss
# =============================================================================
_P1_T = np.unique(np.concatenate([
    days_counts,
    np.arange(0.0, 15.0, 1.0),   # viral-load evaluation points
]))

def phase1_loss(pop_params):
    full = _build_p1(pop_params)
    sol  = run_ode(full, _P1_T)
    if sol is None:
        return 1e10

    t   = sol.t
    N_s = sol.y[0]
    S_s = sol.y[1]
    L_s = sol.y[2]
    V_s = sol.y[3]
    tot = N_s + S_s + L_s

    loss = 0.0

    # --- Viral load (log10 scale, days 0-14) ---
    for day in _P1_T:
        if day > 14.0:
            continue
        i = _find_idx(t, day)
        if i is None:
            continue
        V_pred   = max(float(V_s[i]), 1.0)
        V_target = get_target_viral_load(float(day))
        loss += (np.log10(V_pred) - np.log10(V_target)) ** 2

    # --- Total cell counts (normalised by peak) ---
    count_norm = float(np.max(cell_counts_total))
    for k, day in enumerate(days_counts):
        i = _find_idx(t, day)
        if i is None:
            continue
        loss += ((float(tot[i]) - cell_counts_total[k]) / count_norm) ** 2

    # --- Cell proportions ---
    for k, day in enumerate(days_proportions):
        i = _find_idx(t, day)
        if i is None:
            continue
        tot_t = float(tot[i])
        if tot_t < epsilon:
            continue
        n_pred = float(N_s[i]) / tot_t
        s_pred = float(S_s[i]) / tot_t
        l_pred = float(L_s[i]) / tot_t
        loss += (n_pred - naive_prop[k])    ** 2
        loss += (s_pred - effector_prop[k]) ** 2
        loss += (l_pred - memory_prop[k])   ** 2

    return loss if np.isfinite(loss) else 1e10

# Phase 1 parameter bounds
# Order: r_V, clearanceRate, K0_ns, K0_nl, K0_sl, gamma_S, sigma_S
phase1_bounds = [
    (0.1,    5.0),     # r_V
    (0.0001, 0.05),    # clearanceRate
    (0.01,   5.0),     # K0_ns
    (0.01,   5.0),     # K0_nl
    (0.01,   5.0),     # K0_sl
    (0.5,    10.0),    # gamma_S
    (0.1,    5.0),     # sigma_S
]

# =============================================================================
# Phase 2 evaluation grid and loss
# =============================================================================
_P2_T = np.unique(np.concatenate([
    np.array(sorted(gene_expression_data.keys()), dtype=float),
    days_counts,
]))

def phase2_loss(ge_params, pop_frozen):
    full = _build_p2(ge_params, pop_frozen)
    sol  = run_ode(full, _P2_T)
    if sol is None:
        return 1e10

    t    = sol.t
    N_s  = sol.y[0]
    S_s  = sol.y[1]
    L_s  = sol.y[2]
    Tn_s = sol.y[4]
    Ts_s = sol.y[5]
    Tl_s = sol.y[6]
    Pn_s = sol.y[7]
    Ps_s = sol.y[8]
    Pl_s = sol.y[9]
    tot  = N_s + S_s + L_s

    loss = 0.0

    for day, cell_types in gene_expression_data.items():
        i = _find_idx(t, float(day))
        if i is None:
            continue

        N_t   = float(N_s[i])
        S_t   = float(S_s[i])
        L_t   = float(L_s[i])
        tot_t = max(float(tot[i]), epsilon)

        # Naive cells (data available days 0-7)
        if 'naive' in cell_types and N_t / tot_t > 0.01:
            tn = float(Tn_s[i]) / max(N_t, epsilon)
            pn = float(Pn_s[i]) / max(N_t, epsilon)
            loss += ((tn - cell_types['naive']['Tcf7'])  / TCF7_RANGE)  ** 2
            loss += ((pn - cell_types['naive']['Prdm1']) / PRDM1_RANGE) ** 2

        # Effector cells
        if 'effector' in cell_types and S_t / tot_t > 0.01:
            ts = float(Ts_s[i]) / max(S_t, epsilon)
            ps = float(Ps_s[i]) / max(S_t, epsilon)
            loss += ((ts - cell_types['effector']['Tcf7'])  / TCF7_RANGE)  ** 2
            loss += ((ps - cell_types['effector']['Prdm1']) / PRDM1_RANGE) ** 2

        # Memory cells
        if 'memory' in cell_types and L_t / tot_t > 0.01:
            tl = float(Tl_s[i]) / max(L_t, epsilon)
            pl = float(Pl_s[i]) / max(L_t, epsilon)
            loss += ((tl - cell_types['memory']['Tcf7'])  / TCF7_RANGE)  ** 2
            loss += ((pl - cell_types['memory']['Prdm1']) / PRDM1_RANGE) ** 2

    # Naive-baseline penalty for days 10+ (no naive data → penalise deviation from day-0)
    for day in _naive_late_days:
        i = _find_idx(t, float(day))
        if i is None:
            continue
        N_t   = float(N_s[i])
        tot_t = max(float(tot[i]), epsilon)
        if N_t / tot_t > 0.01:
            tn = float(Tn_s[i]) / max(N_t, epsilon)
            pn = float(Pn_s[i]) / max(N_t, epsilon)
            loss += ((tn - BASELINE_TCF7)  / TCF7_RANGE)  ** 2
            loss += ((pn - BASELINE_PRDM1) / PRDM1_RANGE) ** 2

    return loss if np.isfinite(loss) else 1e10

# Phase 2 parameter bounds
# Order: alpha0t, alpha0p, alpha_tn, alpha_ts, alpha_tl,
#        alpha_pn, alpha_ps, alpha_pl, delta_t, delta_p, R_t, R_p, R_v
phase2_bounds = [
    (0.01, 3.0),    # alpha0t
    (0.01, 3.0),    # alpha0p
    (0.01, 3.0),    # alpha_tn
    (0.01, 3.0),    # alpha_ts
    (0.01, 3.0),    # alpha_tl
    (0.01, 3.0),    # alpha_pn
    (0.01, 3.0),    # alpha_ps
    (0.01, 3.0),    # alpha_pl
    (0.01, 1.5),    # delta_t
    (0.01, 1.5),    # delta_p
    (0.1,  5.0),    # R_t
    (0.01, 2.0),    # R_p
    (1e3,  1e7),    # R_v
]

# =============================================================================
# Plotting
# =============================================================================
def plot_results(final_params, save_prefix='tcell_fit'):
    """Generate and save all result figures."""
    t_plot = np.linspace(0, 90, 901)
    sol = run_ode(final_params, t_plot, max_step=0.1)
    if sol is None:
        print("WARNING: final ODE run failed – cannot plot.")
        return

    t   = sol.t
    N_s = sol.y[0]
    S_s = sol.y[1]
    L_s = sol.y[2]
    V_s = sol.y[3]
    Tn_s = sol.y[4]
    Ts_s = sol.y[5]
    Tl_s = sol.y[6]
    Pn_s = sol.y[7]
    Ps_s = sol.y[8]
    Pl_s = sol.y[9]
    tot  = N_s + S_s + L_s

    # Per-cell expression (safe division)
    tn_pc = np.where(N_s > epsilon, Tn_s / N_s, np.nan)
    ts_pc = np.where(S_s > epsilon, Ts_s / S_s, np.nan)
    tl_pc = np.where(L_s > epsilon, Tl_s / L_s, np.nan)
    pn_pc = np.where(N_s > epsilon, Pn_s / N_s, np.nan)
    ps_pc = np.where(S_s > epsilon, Ps_s / S_s, np.nan)
    pl_pc = np.where(L_s > epsilon, Pl_s / L_s, np.nan)

    # ---- Figure 1: Cell counts -------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, N_s,  label='Naive (N)',         color='steelblue',   lw=2)
    ax1.plot(t, S_s,  label='Effector (S)',       color='firebrick',   lw=2)
    ax1.plot(t, L_s,  label='Memory (L)',         color='forestgreen', lw=2)
    ax1.plot(t, tot,  label='Total',              color='black',       lw=2, ls='--')
    ax1.scatter(days_counts, cell_counts_total,
                label='Observed total', color='black', zorder=5, s=60)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Cell count')
    ax1.set_title('T cell population dynamics')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(bottom=0)
    fig1.tight_layout()
    fig1.savefig(f'{save_prefix}_counts.png', dpi=150)
    plt.close(fig1)

    # ---- Figure 2: Viral load (predicted only) ---------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(t, V_s, color='darkorange', lw=2, label='Predicted')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Viral load')
    ax2.set_title('Viral load dynamics')
    ax2.legend()
    ax2.set_xlim(0, 30)
    fig2.tight_layout()
    fig2.savefig(f'{save_prefix}_viral.png', dpi=150)
    plt.close(fig2)

    # ---- Figure 3: Per-cell Tcf7 ----------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(t, tn_pc, label='Naive Tcf7',   color='steelblue',   lw=2)
    ax3.plot(t, ts_pc, label='Effector Tcf7', color='firebrick',   lw=2)
    ax3.plot(t, tl_pc, label='Memory Tcf7',  color='forestgreen', lw=2)

    # Data scatter
    naive_days_ge    = [d for d in gene_expression_data if 'naive'    in gene_expression_data[d]]
    effector_days_ge = [d for d in gene_expression_data if 'effector' in gene_expression_data[d]]
    memory_days_ge   = [d for d in gene_expression_data if 'memory'   in gene_expression_data[d]]

    ax3.scatter(naive_days_ge,
                [gene_expression_data[d]['naive']['Tcf7']    for d in naive_days_ge],
                color='steelblue',   zorder=5, s=60, label='Naive data')
    ax3.scatter(effector_days_ge,
                [gene_expression_data[d]['effector']['Tcf7'] for d in effector_days_ge],
                color='firebrick',   zorder=5, s=60, label='Effector data')
    ax3.scatter(memory_days_ge,
                [gene_expression_data[d]['memory']['Tcf7']   for d in memory_days_ge],
                color='forestgreen', zorder=5, s=60, label='Memory data')

    ax3.set_xlabel('Day')
    ax3.set_ylabel('Per-cell Tcf7 expression')
    ax3.set_title('Per-cell Tcf7 expression by cell type')
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 90)
    fig3.tight_layout()
    fig3.savefig(f'{save_prefix}_tcf7.png', dpi=150)
    plt.close(fig3)

    # ---- Figure 4: Per-cell Prdm1 ---------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.plot(t, pn_pc, label='Naive Prdm1',    color='steelblue',   lw=2)
    ax4.plot(t, ps_pc, label='Effector Prdm1', color='firebrick',   lw=2)
    ax4.plot(t, pl_pc, label='Memory Prdm1',   color='forestgreen', lw=2)

    ax4.scatter(naive_days_ge,
                [gene_expression_data[d]['naive']['Prdm1']    for d in naive_days_ge],
                color='steelblue',   zorder=5, s=60, label='Naive data')
    ax4.scatter(effector_days_ge,
                [gene_expression_data[d]['effector']['Prdm1'] for d in effector_days_ge],
                color='firebrick',   zorder=5, s=60, label='Effector data')
    ax4.scatter(memory_days_ge,
                [gene_expression_data[d]['memory']['Prdm1']   for d in memory_days_ge],
                color='forestgreen', zorder=5, s=60, label='Memory data')

    ax4.set_xlabel('Day')
    ax4.set_ylabel('Per-cell Prdm1 expression')
    ax4.set_title('Per-cell Prdm1 expression by cell type')
    ax4.legend(fontsize=9)
    ax4.set_xlim(0, 90)
    fig4.tight_layout()
    fig4.savefig(f'{save_prefix}_prdm1.png', dpi=150)
    plt.close(fig4)

    print(f"Plots saved: {save_prefix}_counts.png, {save_prefix}_viral.png, "
          f"{save_prefix}_tcf7.png, {save_prefix}_prdm1.png")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Phase 1: population dynamics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: fitting population parameters (100 iterations)")
    print("=" * 60)

    result1 = differential_evolution(
        phase1_loss,
        bounds=phase1_bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        workers=1,
        disp=True,
        tol=1e-7,
    )
    best_pop = result1.x
    print(f"\nPhase 1 done.  Loss = {result1.fun:.6f}")
    print("  r_V          =", best_pop[0])
    print("  clearanceRate=", best_pop[1])
    print("  K0_ns        =", best_pop[2])
    print("  K0_nl        =", best_pop[3])
    print("  K0_sl        =", best_pop[4])
    print("  gamma_S      =", best_pop[5])
    print("  sigma_S      =", best_pop[6])

    # ------------------------------------------------------------------
    # Phase 2: gene expression
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 2: fitting gene expression parameters (100 iterations)")
    print("=" * 60)

    def _p2_loss(ge_params):
        return phase2_loss(ge_params, best_pop)

    result2 = differential_evolution(
        _p2_loss,
        bounds=phase2_bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        workers=1,
        disp=True,
        tol=1e-7,
    )
    best_ge = result2.x
    print(f"\nPhase 2 done.  Loss = {result2.fun:.6f}")
    labels_ge = [
        'alpha0t', 'alpha0p', 'alpha_tn', 'alpha_ts', 'alpha_tl',
        'alpha_pn', 'alpha_ps', 'alpha_pl', 'delta_t', 'delta_p',
        'R_t', 'R_p', 'R_v',
    ]
    for lbl, val in zip(labels_ge, best_ge):
        print(f"  {lbl:12s} = {val:.6g}")

    # ------------------------------------------------------------------
    # Build final parameter vector and plot
    # ------------------------------------------------------------------
    final_params = _build_p2(best_ge, best_pop)
    plot_results(final_params, save_prefix='tcell_fit')
