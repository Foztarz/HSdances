"""Create HS_circGLMM_states_joint_kfold_fits.ipynb (refit every fold and model, saving the posteriors) and
HS_circGLMM_states_joint_kfold_lpd.ipynb (predictive density of the left-out individuals from the saved posteriors,
with all posterior draws and a bootstrap Monte Carlo error), reusing cells of HS_circGLMM_states_joint_kfold.ipynb."""
import json

src = json.load(open('HS_circGLMM_states_joint_kfold.ipynb', encoding='utf8'))
kernel_meta = src['metadata']
src_code = [''.join(c['source']) for c in src['cells'] if c['cell_type'] == 'code']


def src_cell(marker):
    # code cell of the k-fold notebook containing `marker`
    found = [s for s in src_code if marker in s]
    assert len(found) == 1, marker
    return found[0]


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': s.strip('\n').splitlines(keepends=True)}


def write(name, cells):
    nb = {'cells': cells, 'metadata': kernel_meta, 'nbformat': 4, 'nbformat_minor': 5}
    with open(name, 'w', encoding='utf8', newline='\r\n') as f:
        f.write(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f'created {name}')


setup_loader = src_cell("setup_notebook = 'HS_circGLMM_states_joint.ipynb'")
setup_rest = src_cell('for src in setup_cells[2:]:')
fold_cell = src_cell('def keeps_all_cells(fold_inds):')
recode_cell = src_cell('def recode(df_sub):')
lpd_cell = src_cell('def heldout_lpd(idata, key, df_te, rng):')
helpers = lpd_cell[:lpd_cell.index('def heldout_lpd(')] # imports, np_lognorm, bounded_np, thin

common_settings = r"""
# settings cell of HS_circGLMM_states_joint.ipynb, then overrides
run_setup(setup_cells[0]) # PYTENSOR_FLAGS
run_setup(setup_cells[1]) # settings

models_to_fit = [] # build only the cross-validation models
cv_models = ['unimodal', 'states_cond_band', 'states_cond_x_band', 'states_cond_x_band_fixed']
n_folds = 10 # None: leave one individual out
cv_seed = 20261022
test_run = False
post_dir = f'2Results/state_models/kfold_posteriors_{fit_tag}'
fits_file = f'2Results/state_models/HS_states_joint_{fit_tag}_kfold_fits.csv'
"""

# ---------------------------------------------------------------- fits notebook
fits_cells = [
md(r"""
# Cross-validation of the joint dance state models: refits to the training individuals
---

## Details

&ensp;AUTHOR:&emsp;   James Foster &emsp; 2026 11 03

MODIFIED:&emsp; James Foster &emsp; 2026 11 03

DESCRIPTION:&emsp; Fits every model of [HS_circGLMM_states_joint_kfold.ipynb](HS_circGLMM_states_joint_kfold.ipynb) to the training individuals of every fold, as there, and saves the population-level posterior of each fit, so that the predictive density of the left-out individuals can be calculated separately ([HS_circGLMM_states_joint_kfold_lpd.ipynb](HS_circGLMM_states_joint_kfold_lpd.ipynb)) and recalculated without refitting.
In the first k-fold run, some fits had chains that disagreed (one chain stuck with a much lower log density), so each fit uses 6 chains and longer tuning, and a fit whose chains disagree is repeated with a new seed.

EXPECTED RUN TIME:&emsp; about 30–60 min per state model and 5 min per unimodal model on the MacBook under the watchdog, i.e. about 1–1.5 days for 10 folds. Results are saved after each fit, so the run can be stopped and resumed.

INPUTS:&emsp;    `0Data/hs_dance_reorg.csv`; the model code in `HS_circGLMM_states_joint.ipynb` (cells tagged `kfold-setup`)

OUTPUTS:&emsp;  `2Results/state_models/kfold_posteriors_<fit_tag>/fold<k>_<model>_s<seed offset>.nc` (posterior of the population-level parameters); `2Results/state_models/HS_states_joint_<fit_tag>_kfold_fits.csv` (sampler diagnostics of each fit)
"""),
md(r"""
## Settings
As in `HS_circGLMM_states_joint_kfold.ipynb` (keep `n_folds` and `cv_seed`, so that the folds are the same), and:
- `seed_offsets`: sampler seeds `cv_seed + offset + fold`, tried in turn until the chains agree (mean log density of the chains within `max_lp_range`); the fit with the smallest range is used
- `cv_chains`, `cv_draws`, `cv_tune`: sampler settings for each fit
"""),
code(setup_loader),
code(common_settings + r"""
cv_chains = 6
cv_draws = 1000
cv_tune = 2000
seed_offsets = [1000, 2000, 3000]
max_lp_range = 20
if test_run:
    cv_chains, cv_draws, cv_tune = 2, 100, 100
    fits_file = fits_file.replace('.csv', '_test.csv')
    post_dir += '_test'
print(f'fit_tag: {fit_tag}, models: {cv_models}, folds: {n_folds}, test run: {test_run}')
"""),
code(setup_rest),
md(r"""
# Folds
As in `HS_circGLMM_states_joint_kfold.ipynb`.
"""),
code(fold_cell),
md(r"""
# Refitting to the training individuals
As in `HS_circGLMM_states_joint_kfold.ipynb`.
Only the population-level variables (no dimension longer than 20) are saved: the effects of the training individuals and their dances are not needed to predict new individuals.
"""),
code(recode_cell + r"""


def save_posterior(idata, path):
    post = idata.posterior
    keep = [v for v in post.data_vars if all(post[v].sizes[d] <= 20 for d in post[v].dims if d not in ('chain', 'draw'))]
    post[keep].to_netcdf(path)
"""),
md(r"""
# Run the fits
Each fit is saved as it finishes, so an interrupted run can be resumed (saved fits are skipped).
Fits from the refit run (`kfold_refit_20261103`, same settings and first seed) are used instead of being repeated.
"""),
code(r"""
import os, time, gc, shutil
import logging
logging.getLogger('pymc').setLevel(logging.WARNING)

os.makedirs(post_dir, exist_ok=True)
fits = pd.read_csv(fits_file) if os.path.exists(fits_file) else pd.DataFrame(columns=['fold', 'model', 'seed_offset'])
# fits of the refit run: same sampler settings and seed as the first attempt here
refit_file = f'2Results/state_models/HS_states_joint_{fit_tag}_kfold_refit.csv'
if os.path.exists(refit_file) and not test_run:
    for r in pd.read_csv(refit_file).drop_duplicates(['fold', 'model']).itertuples():
        old, new = f'{post_dir}/fold{r.fold}_{r.model}_refit.nc', f'{post_dir}/fold{r.fold}_{r.model}_s{seed_offsets[0]}.nc'
        done = ((fits['fold'] == r.fold) & (fits['model'] == r.model) & (fits['seed_offset'] == seed_offsets[0])).any()
        if os.path.exists(old) and not done:
            shutil.copy(old, new)
            fits = pd.concat([fits, pd.DataFrame([{
                'fold': r.fold, 'model': r.model, 'seed_offset': seed_offsets[0], 'chains': cv_chains,
                'tune': cv_tune, 'draws': cv_draws, 'divergences': r.divergences, 'chain_logp': r.chain_logp,
                'chain_logp_range': r.chain_logp_range, 'fit_minutes': r.fit_minutes, 'file': new}])], ignore_index=True)
    fits.to_csv(fits_file, index=False)

for k, test_inds in enumerate(folds):
    df_tr, dances_tr = recode(df_full[~df_full['Individual_label'].isin(test_inds)])
    for key in cv_models:
        for off in seed_offsets:
            prev = fits[(fits['fold'] == k) & (fits['model'] == key)]
            if (prev['chain_logp_range'] <= max_lp_range).any():
                break # chains agreed in a saved fit
            if (prev['seed_offset'] == off).any():
                continue
            t0 = time.time()
            model = build_cv_model(key, df_tr, dances_tr)
            idata_k = sample_cv(model, key, seed=cv_seed + off + k)
            path = f'{post_dir}/fold{k}_{key}_s{off}.nc'
            save_posterior(idata_k, path)
            st = idata_k.sample_stats
            lp = st['logp' if 'logp' in st else 'lp'].mean('draw').values
            row = {'fold': k, 'model': key, 'seed_offset': off, 'chains': cv_chains, 'tune': cv_tune, 'draws': cv_draws,
                   'divergences': int(st['diverging'].sum()), 'chain_logp': ' '.join(f'{x:.1f}' for x in lp),
                   'chain_logp_range': float(lp.max() - lp.min()), 'fit_minutes': (time.time() - t0) / 60, 'file': path}
            if 'n_steps' in st:
                row['mean_steps'] = float(st['n_steps'].mean())
            fits = pd.concat([fits, pd.DataFrame([row])], ignore_index=True)
            fits.to_csv(fits_file, index=False)
            print(f'fold {k} ({", ".join(test_inds)}), {key}, seed offset {off}: divergences {row["divergences"]}, '
                  f'chain logp {row["chain_logp"]}, fit {row["fit_minutes"]:.1f} min', flush=True)
            del model, idata_k
            gc.collect()
df, dances, n_dance = df_full, dances_full, len(dances_full) # restore the full data
"""),
md(r"""
# Fits used
The fit with the smallest range of chain log densities, for each fold and model.
"""),
code(r"""
fits = pd.read_csv(fits_file)
best = fits.loc[fits.groupby(['fold', 'model'])['chain_logp_range'].idxmin()]
print(best[['fold', 'model', 'seed_offset', 'divergences', 'chain_logp_range', 'fit_minutes']].to_string(index=False))
print('\nFits whose chains still disagree:')
print(best[best['chain_logp_range'] > max_lp_range][['fold', 'model', 'chain_logp']].to_string(index=False))
"""),
]

# ---------------------------------------------------------------- lpd notebook
lpd_functions = helpers + r'''

def thin_dict(post, S, names):
    # S evenly spaced draws (chains concatenated) of each variable; S = None: all draws
    n_all = post.sizes['chain'] * post.sizes['draw']
    idx = np.arange(n_all) if S is None else np.linspace(0, n_all - 1, S).astype(int)
    return {v: thin(post, v, idx) for v in names if v in post}


def model_info(key):
    spec = model_specs[key]
    states = list(spec['states'])
    Xk_c, _ = design(cells_tab, spec['kappa'])
    Xs_c, _ = design(cells_tab, spec['state'])
    return {'states': states, 'direction': spec['direction'], 'Xk_c': Xk_c, 'Xs_c': Xs_c,
            'main': [s for s in states if s not in OUTLIER_STATES], 'out': [s for s in states if s in OUTLIER_STATES]}


def ind_loglik(P, info, d_ind, u, zk, zs):
    # log likelihood of an individual's waggle runs for draws P (each S x ...) and individual effects
    # u (direction offset), zk (concentration effect, standardised), zs (state-logit effects, standardised): S x M (x states)
    S, M = u.shape
    states, direction, main, out = info['states'], info['direction'], info['main'], info['out']
    mu_c, k0 = P['mu_cell'], P['kappa_Intercept']
    keff = P['kappa_effect'] if 'kappa_effect' in P else np.zeros((S, 0))
    sk_i, sk_d = P['kappa_Individual_sigma'], P['kappa_Dance_sigma']
    # dance direction offsets: grid and log weights (S x G)
    if direction == 'vonmises':
        v = np.broadcast_to(np.linspace(-np.pi, np.pi, G_grid, endpoint=False), (S, G_grid))
        lw_v = P['mu_Dance_kappa'][:, None] * np.cos(v)
    else:
        t = np.linspace(-5, 5, G_grid)
        v = P['mu_Dance_sigma'][:, None] * t[None, :]
        lw_v = np.broadcast_to(-0.5 * t ** 2, (S, G_grid))
    lw_v = lw_v - logsumexp(lw_v, axis=1, keepdims=True)
    # dance concentration effects: Gauss-Hermite nodes and log weights
    xq, wq = np.polynomial.hermite_e.hermegauss(Q_nodes)
    lw_q = np.log(wq / wq.sum())
    w_ax = P['w_axial'] if 'axial' in states else None
    ll = np.zeros((S, M))
    for dance, d_d in d_ind.groupby('Dance_label'):
        th = d_d['angle'].values
        c = int(d_d['cell_idx'].iloc[0])
        xk, xs = info['Xk_c'][c], info['Xs_c'][c]
        n, C, Sn = len(th), np.cos(th).sum(), np.sin(th).sum()
        # direction: S x M x G
        off = u[:, :, None] + v[:, None, :]
        off = off if direction == 'vonmises' else bounded_np(off, MAX_OFFSET)
        mu = mu_c[:, c][:, None, None] + off
        # concentration: S x M x Q
        eta_k = (k0 + keff @ xk)[:, None, None] + (sk_i[:, None] * zk)[:, :, None] + sk_d[:, None, None] * xq
        kap = np.log1p(np.exp(-np.abs(eta_k))) + np.maximum(eta_k, 0) # softplus
        norm = n * np_lognorm(kap) # S x M x Q
        proj = (C * np.cos(mu) + Sn * np.sin(mu))[..., None] # S x M x G x 1
        kq = kap[:, :, None, :] # S x M x 1 x Q
        lw = lw_v[:, None, :, None] + lw_q # S x 1 x G x Q
        L = {}
        L['forward'] = logsumexp(kq * proj - norm[:, :, None, :] + lw, axis=(2, 3))
        if 'reversed' in states:
            L['reversed'] = logsumexp(-kq * proj - norm[:, :, None, :] + lw, axis=(2, 3))
        if 'axial' in states:
            acc = 0.0
            lwa, lwb = np.log(w_ax)[:, None, None, None], np.log1p(-w_ax)[:, None, None, None]
            for th_j in th:
                cj = kq * np.cos(th_j - mu)[..., None]
                acc = acc + np.logaddexp(lwa + cj, lwb - cj)
            L['axial'] = logsumexp(acc - norm[:, :, None, :] + lw, axis=(2, 3))
        if 'unrelated' in states:
            R = np.hypot(C, Sn)
            L['unrelated'] = logsumexp(kap * R + np.log(i0e(kap * R)) - norm + lw_q, axis=2)
        if 'disoriented' in states:
            L['disoriented'] = np.full((S, M), -n * LOG_2PI)
        # state probabilities: S x M x states
        etas = [np.zeros((S, M))]
        for i, s in enumerate(main[1:]):
            e = P[f'{s}_Intercept'][:, None] + np.zeros((1, M))
            if f'{s}_effect' in P:
                e = e + (P[f'{s}_effect'] @ xs)[:, None]
            e = e + P[f'{s}_Individual_sigma'][:, None] * zs[:, :, i]
            etas.append(e)
        log_pi = log_softmax(np.stack(etas, axis=-1), axis=-1)
        if out:
            e_out = P[f'{out[0]}_Intercept'] + (P[f'{out[0]}_effect'] @ xs if f'{out[0]}_effect' in P else 0)
            e_out = e_out[:, None, None]
            log_pi = np.concatenate([log_pi - np.logaddexp(0, e_out),
                                     np.broadcast_to(-np.logaddexp(0, -e_out), (S, M, 1))], axis=-1)
        L_s = np.stack([L[s] for s in main + out], axis=-1)
        ll += logsumexp(log_pi + L_s, axis=-1)
    return ll
'''

is_functions = r'''
from scipy import stats
import arviz as az


def a1inv(R):
    # concentration of a von Mises distribution with mean resultant length R (Best & Fisher 1981)
    R = min(R, 0.9999)
    if R < 0.53:
        return 2 * R + R ** 3 + 5 * R ** 5 / 6
    if R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    return 1 / (R ** 3 - 4 * R ** 2 + 3 * R)


def log_prior(P, u, Z):
    # log density of the individual effects in their population distribution: S x M
    k_u = P['mu_Individual_kappa'][:, None]
    return k_u * np.cos(u) - np_lognorm(k_u) - 0.5 * (Z ** 2).sum(-1) - 0.5 * Z.shape[-1] * LOG_2PI


class Proposal:
    # von Mises for the direction offset u, multivariate t for the standardised effects Z, mixed with their
    # population distribution (weight `defensive`), so that the importance weights stay bounded
    def __init__(self, mu_u, kappa_u, mean, cov):
        self.mu_u, self.kappa_u, self.mean, self.cov = mu_u, kappa_u, mean, cov
        self.t = stats.multivariate_t(loc=mean, shape=cov, df=is_df)

    def sample(self, P, M, rng):
        S, D = len(P['mu_Individual_kappa']), len(self.mean)
        from_prior = rng.random((S, M)) < is_defensive
        u = np.where(from_prior, rng.vonmises(0.0, np.broadcast_to(P['mu_Individual_kappa'][:, None], (S, M))),
                     rng.vonmises(self.mu_u, self.kappa_u, (S, M)))
        Z = np.where(from_prior[..., None], rng.standard_normal((S, M, D)),
                     self.t.rvs(S * M, random_state=rng).reshape(S, M, D))
        return u, Z

    def log_q(self, P, u, Z):
        lp_prop = (self.kappa_u * np.cos(u - self.mu_u) - np_lognorm(self.kappa_u)
                   + self.t.logpdf(Z.reshape(-1, Z.shape[-1])).reshape(u.shape))
        return np.logaddexp(np.log(is_defensive) + log_prior(P, u, Z), np.log1p(-is_defensive) + lp_prop)

    @classmethod
    def fit(cls, u, Z, log_w, prev):
        # weighted moments of the samples, shrunk towards the previous proposal when the effective sample size is small
        w = np.exp(log_w - log_w.max()).ravel()
        w /= w.sum()
        ess = 1 / (w ** 2).sum()
        u, Z = u.ravel(), Z.reshape(-1, Z.shape[-1])
        mu_u = np.arctan2((w * np.sin(u)).sum(), (w * np.cos(u)).sum())
        kappa_u = a1inv(np.hypot((w * np.sin(u)).sum(), (w * np.cos(u)).sum())) / is_inflate
        mean = w @ Z
        cov = (w[:, None] * (Z - mean)).T @ (Z - mean)
        lam = ess / (ess + Z.shape[1] + 2)
        cov = is_inflate * (lam * cov + (1 - lam) * prev.cov) + 1e-6 * np.eye(Z.shape[1])
        return cls(mu_u, kappa_u, mean, cov), ess


def ess(log_w, axis=None):
    w = np.exp(log_w - log_w.max(axis=axis, keepdims=axis is not None))
    return w.sum(axis) ** 2 / (w ** 2).sum(axis)


def summarise(log_w):
    # log mean of the weights (S draws x M samples), effective sample sizes and Pareto k-hat:
    # ess of all weights; ess_draws of the per-draw estimates p(y | draw), over the S draws;
    # ess_within: median over draws of the effective sample size of the M samples;
    # se_mc: bootstrap standard error of the lpd, resampling the draws
    flat = log_w.ravel()
    _, k_hat = az.psislw(flat.copy())
    S, M = log_w.shape
    per_draw = logsumexp(log_w, axis=1) - np.log(M)
    boot = np.random.default_rng(0).integers(S, size=(n_boot, S))
    se_mc = float(np.std(logsumexp(per_draw[boot], axis=1) - np.log(S)))
    return {'lpd': logsumexp(flat) - np.log(flat.size), 'se_mc': se_mc, 'ess': ess(flat), 'ess_draws': ess(per_draw),
            'ess_within': float(np.median(ess(log_w, axis=1))), 'pareto_k': float(k_hat), 'n_draws': S}


def loglik_chunked(P, info, d_ind, u, Z, chunk=500):
    # ind_loglik in chunks of draws, to limit memory
    out = []
    for s0 in range(0, len(u), chunk):
        sl = slice(s0, s0 + chunk)
        out.append(ind_loglik({v: x[sl] for v, x in P.items()}, info, d_ind, u[sl], Z[sl, :, 0], Z[sl, :, 1:]))
    return np.concatenate(out)


def is_lpd(post, key, d_ind, rng):
    # predictive density of a new individual by adaptive importance sampling of its effects
    info = model_info(key)
    P_ad = thin_dict(post, is_S_adapt, post.data_vars)
    P = thin_dict(post, S_draws, post.data_vars)
    D = len(info['main'])  # concentration effect and the state-logit effects of the main states after the first
    prop = Proposal(0.0, float(np.median(P_ad['mu_Individual_kappa'])), np.zeros(D), np.eye(D))
    ess_adapt = []
    for r in range(is_rounds):
        if r == 0: # first round: the population distribution, as in the simple Monte Carlo estimate
            u = rng.vonmises(0.0, np.broadcast_to(P_ad['mu_Individual_kappa'][:, None], (is_S_adapt, is_M_adapt)))
            Z = rng.standard_normal((is_S_adapt, is_M_adapt, D))
            log_q = log_prior(P_ad, u, Z)
        else:
            u, Z = prop.sample(P_ad, is_M_adapt, rng)
            log_q = prop.log_q(P_ad, u, Z)
        log_w = loglik_chunked(P_ad, info, d_ind, u, Z) + log_prior(P_ad, u, Z) - log_q
        prop, ess = Proposal.fit(u, Z, log_w, prop)
        ess_adapt.append(ess)
    u, Z = prop.sample(P, is_M, rng)
    log_w = loglik_chunked(P, info, d_ind, u, Z) + log_prior(P, u, Z) - prop.log_q(P, u, Z)
    res = summarise(log_w)
    res['ess_adapt'] = ' '.join(f'{e:.0f}' for e in ess_adapt)
    return res


def mc_lpd(post, key, d_ind, rng, M, chunk=32):
    # simple Monte Carlo estimate, sampling the individual's effects from their population distribution,
    # in chunks of `chunk` samples per draw to limit memory
    info = model_info(key)
    P = thin_dict(post, S_draws, post.data_vars)
    S, D = len(P['mu_Individual_kappa']), len(info['main'])
    parts = []
    for m0 in range(0, M, chunk):
        m = min(chunk, M - m0)
        u = rng.vonmises(0.0, np.broadcast_to(P['mu_Individual_kappa'][:, None], (S, m)))
        Z = rng.standard_normal((S, m, D))
        parts.append(loglik_chunked(P, info, d_ind, u, Z))
    return summarise(np.concatenate(parts, axis=1))
'''

lpd_cells = [
md(r"""
# Cross-validation of the joint dance state models: predictive density of new individuals
---

## Details

&ensp;AUTHOR:&emsp;   James Foster &emsp; 2026 11 03

MODIFIED:&emsp; James Foster &emsp; 2026 11 03

DESCRIPTION:&emsp; Calculates the log predictive density of the left-out individuals of each fold from the posteriors saved by [HS_circGLMM_states_joint_kfold_fits.ipynb](HS_circGLMM_states_joint_kfold_fits.ipynb), and compares the models.
The density is that of [HS_circGLMM_states_joint_kfold.ipynb](HS_circGLMM_states_joint_kfold.ipynb), with all posterior draws instead of 500, and a Monte Carlo standard error for each individual.
For individuals with many waggle runs, $p(\boldsymbol{y}_i \mid \phi)$ varies by tens of log units between posterior draws (90r in fold 2: SD 4.7 over draws), so a few draws dominate the estimate: the effective number of draws can be below 10, and the estimate is then noisy and probably too low (about 1 lpd for 90r with 500 draws).
More samples of the individual's effects per draw (`M_inner`) or importance sampling of them (`lpd_method = 'is'`, tested on 90r: no better) do not help, since the limit is the number of posterior draws.
The same method is used for every individual.

EXPECTED RUN TIME:&emsp; about 3 s per waggle run of the left-out individuals for each state model (about 15–30 min per fold and state model, 8 h in all), much less for the unimodal model; a single core.

INPUTS:&emsp;    `0Data/hs_dance_reorg.csv`; the model code in `HS_circGLMM_states_joint.ipynb` (cells tagged `kfold-setup`); `2Results/state_models/HS_states_joint_<fit_tag>_kfold_fits.csv` and the posteriors it lists

OUTPUTS:&emsp;  `2Results/state_models/HS_states_joint_<fit_tag>_kfold_lpd.csv` (log predictive density of each left-out individual for each model, with its Monte Carlo standard error and effective sample sizes; `_lpd_is.csv` for importance sampling)
"""),
md(r"""
## Settings
- `S_draws`: posterior draws used (`None`: all). For individuals with many runs, $p(\boldsymbol{y}_i \mid \phi)$ varies by orders of magnitude between posterior draws, so a few draws dominate the estimate, and all draws are needed
- `G_grid`, `Q_nodes`: as in `HS_circGLMM_states_joint_kfold.ipynb`
- `lpd_method`: `'mc'` (samples of the individual's effects from their population distribution, `M_inner` per draw, as in `HS_circGLMM_states_joint_kfold.ipynb`) or `'is'` (adaptive importance sampling of the individual's effects, below)
- `n_boot`: bootstrap samples (over posterior draws) for the Monte Carlo standard error `se_mc` of each individual's lpd; it misses draws that were never sampled, so it is too small when `ess_draws` is small
- `is_rounds`, `is_S_adapt`, `is_M_adapt`: rounds of adaptation of the proposal, with `is_M_adapt` samples for each of `is_S_adapt` draws
- `is_M`: samples per draw for the estimate; `is_defensive`: weight of the population distribution in the proposal; `is_df`: degrees of freedom of the t proposal; `is_inflate`: widening of the fitted proposal
- `validate`: compare the importance sampling with a long simple Monte Carlo run (`M_validate` samples per draw) for some individuals, before the main loop
"""),
code(setup_loader),
code(common_settings + r"""
S_draws = None # posterior draws used for the predictive density (None: all)
G_grid = 121 # grid points for each dance's direction offset
Q_nodes = 8 # Gauss-Hermite nodes for each dance's concentration effect
lpd_method = 'mc'
M_inner = 16 # samples per draw for lpd_method = 'mc'
n_boot = 200
is_rounds, is_S_adapt, is_M_adapt = 4, 200, 32
is_M = 8
is_defensive, is_df, is_inflate = 0.1, 5, 1.5
validate = False
validate_cases = [] # (fold, model, individual)
M_validate = 1024
out_file = f'2Results/state_models/HS_states_joint_{fit_tag}_kfold_lpd{"_is" if lpd_method == "is" else ""}.csv'
print(f'fit_tag: {fit_tag}, method: {lpd_method}, output: {out_file}')
"""),
code(setup_rest),
md(r"""
# Folds
As in `HS_circGLMM_states_joint_kfold.ipynb`.
"""),
code(fold_cell),
md(r"""
# Predictive density of a new individual
As in `HS_circGLMM_states_joint_kfold.ipynb`, the log predictive density of a left-out individual $i$ is
$$
\log p(\boldsymbol{y}_i \mid \text{training data}) = \log \mathbb{E}_{\phi \mid \text{train}} \int p(\boldsymbol{y}_i \mid b_i, \phi) \, p(b_i \mid \phi) \, db_i
$$
with the dance effects integrated on a grid (direction) and by Gauss–Hermite quadrature (concentration), and the dance states summed.
`ind_loglik` calculates $\log p(\boldsymbol{y}_i \mid b_i, \phi)$ for arrays of draws and individual effects (the code of `heldout_lpd` there).
"""),
code(lpd_functions),
md(r"""
## Importance sampling of the individual effects
The individual's effects are $b_i = (u_i, \boldsymbol{z}_i)$: the direction offset $u_i$ and the standardised concentration and state-logit effects $\boldsymbol{z}_i$ (population distributions von Mises$(0, \kappa_u)$ and standard normal).
With samples $b^{(s,m)}$ from a proposal $q$ for each posterior draw $\phi^{(s)}$,
$$
\hat p(\boldsymbol{y}_i) = \frac{1}{SM} \sum_{s,m} \frac{p(\boldsymbol{y}_i \mid b^{(s,m)}, \phi^{(s)}) \, p(b^{(s,m)} \mid \phi^{(s)})}{q(b^{(s,m)} \mid \phi^{(s)})}
$$
The proposal is fitted to the individual's data in `is_rounds` rounds: the first samples from the population distribution (the simple estimate), and each round fits a von Mises distribution for $u_i$ and a multivariate t distribution for $\boldsymbol{z}_i$ to the weighted samples of the previous round (widened by `is_inflate`, with the covariance shrunk towards the previous one when the effective sample size is small).
The final proposal is mixed with the population distribution (weight `is_defensive`), which bounds the weights by $p(\boldsymbol{y}_i \mid b, \phi)/\text{is\_defensive}$, so the estimate is never much worse than the simple one.
One proposal is used for all posterior draws.
For each individual, `ess` is the effective sample size of the final weights (of $S \times$ `is_M`), `ess_draws` that of the per-draw estimates over the $S$ draws (the limit set by the number of posterior draws), `ess_within` the median over draws of that of the `is_M` samples, and `pareto_k` the Pareto $\hat{k}$ of the weights (above 0.7: unreliable).
"""),
code(is_functions),
md(r"""
# Validation
For the individuals in `validate_cases`, the importance sampling estimate is compared with a long simple Monte Carlo run.
"""),
code(r"""
import time
import xarray as xr

posts = {}


def load_post(k, key):
    if (k, key) not in posts:
        fits = pd.read_csv(fits_file)
        best = fits.loc[fits[(fits['fold'] == k) & (fits['model'] == key)]['chain_logp_range'].idxmin()]
        with xr.open_dataset(best['file']) as ds:
            posts[(k, key)] = ds.load()
    return posts[(k, key)]


if validate:
    rows = []
    for k, key, ind in validate_cases:
        d_ind = df_full[df_full['Individual_label'] == ind]
        t0 = time.time()
        r_is = is_lpd(load_post(k, key), key, d_ind, np.random.default_rng([cv_seed, k, 101]))
        t_is = time.time() - t0
        r_mc = mc_lpd(load_post(k, key), key, d_ind, np.random.default_rng([cv_seed, k, 102]), M_validate)
        rows.append({'fold': k, 'model': key, 'individual': ind,
                     **{f'{n}_is': r_is[n] for n in ['lpd', 'se_mc', 'ess', 'ess_draws', 'ess_within', 'pareto_k']},
                     'ess_adapt': r_is['ess_adapt'], 'seconds_is': t_is,
                     **{f'{n}_mc': r_mc[n] for n in ['lpd', 'se_mc', 'ess', 'ess_draws', 'ess_within', 'pareto_k']},
                     'seconds_mc': time.time() - t0 - t_is})
        print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
"""),
md(r"""
# Predictive density of every left-out individual
Results are saved after each fold and model, so an interrupted run can be resumed (completed pairs are skipped).
Each fold and model has its own random numbers, so the results do not depend on the order of the calculations.
"""),
code(r"""
import os, time

fits = pd.read_csv(fits_file)
done = pd.read_csv(out_file) if os.path.exists(out_file) else pd.DataFrame(columns=['fold', 'model'])
rows = [done]
for k, test_inds in enumerate(folds):
    df_te = df_full[df_full['Individual_label'].isin(test_inds)]
    for j, key in enumerate(cv_models):
        if ((done['fold'] == k) & (done['model'] == key)).any():
            continue
        if not ((fits['fold'] == k) & (fits['model'] == key)).any():
            print(f'fold {k}, {key}: not fitted yet')
            continue
        t0 = time.time()
        rng = np.random.default_rng([cv_seed, k, j])
        post = load_post(k, key)
        res = {}
        for ind, d_ind in df_te.groupby('Individual_label'):
            if lpd_method == 'is':
                res[ind] = is_lpd(post, key, d_ind, rng)
            else:
                res[ind] = mc_lpd(post, key, d_ind, rng, M_inner)
            res[ind].update({'n_dances': d_ind['Dance_label'].nunique(), 'n_runs': len(d_ind)})
        posts.pop((k, key))
        res = (pd.DataFrame(res).T.assign(fold=k, model=key, minutes=(time.time() - t0) / 60)
               .rename_axis('individual').reset_index())
        rows.append(res)
        pd.concat(rows, ignore_index=True).to_csv(out_file, index=False)
        print(f'fold {k} ({", ".join(test_inds)}), {key}: lpd {res["lpd"].sum():.1f} '
              f'(MC SE {np.sqrt((res["se_mc"] ** 2).sum()):.1f}), min ESS {res["ess"].min():.0f} '
              f'(over draws {res["ess_draws"].min():.0f}), '
              f'max Pareto k {res["pareto_k"].max():.2f}, {(time.time() - t0) / 60:.1f} min', flush=True)
"""),
md(r"""
# Comparison
As in `HS_circGLMM_states_joint_kfold.ipynb`: the elpd of each model is the sum over individuals, and the standard error of the difference is $\sqrt{n \, \text{Var}(\Delta_i)}$, over the $n$ individuals with results for every model.
`se_mc_diff` is the Monte Carlo standard error of the difference, $\sqrt{\sum_i (\text{se}_{\text{mc},i,m}^2 + \text{se}_{\text{mc},i,\text{best}}^2)}$ (the fits are independent); it is a lower bound when `ess_draws` is small.
Check `ess_draws` and `se_mc` for each individual.
"""),
code(r"""
cv = pd.read_csv(out_file)
tab = cv.pivot_table(index='individual', columns='model', values='lpd').dropna()
summary = pd.DataFrame({'elpd': tab.sum(), 'n_individuals': tab.count()})
best = summary['elpd'].idxmax()
diff = tab.sub(tab[best], axis=0)
summary['elpd_diff'] = diff.sum()
summary['se_diff'] = np.sqrt(len(tab) * diff.var(ddof=1)) if len(tab) > 1 else np.nan
se_mc = cv.pivot_table(index='individual', columns='model', values='se_mc').loc[tab.index]
summary['se_mc_diff'] = np.sqrt((se_mc ** 2).add(se_mc[best] ** 2, axis=0).sum())
summary.loc[best, 'se_mc_diff'] = 0.0
print(summary.sort_values('elpd', ascending=False).round(1))
print('\nPer individual (lpd), with the difference from the best model:')
print(tab.assign(n_runs=cv.groupby('individual')['n_runs'].first(),
                 max_se_mc=cv.groupby('individual')['se_mc'].max().round(1),
                 min_ess_draws=cv.groupby('individual')['ess_draws'].min().round(0),
                 **{f'diff_{m}': diff[m] for m in tab.columns if m != best}).round(1).sort_values('n_runs', ascending=False))
"""),
]

write('HS_circGLMM_states_joint_kfold_fits.ipynb', fits_cells)
write('HS_circGLMM_states_joint_kfold_lpd.ipynb', lpd_cells)
