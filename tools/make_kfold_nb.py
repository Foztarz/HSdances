"""Create HS_circGLMM_states_kfold.ipynb and tag the reusable setup cells of HS_circGLMM_states.ipynb."""
import json

# --- tag setup cells in the states notebook ---
p = 'HS_circGLMM_states.ipynb'
nb = json.load(open(p, encoding='utf8'))
setup_markers = ["%set_env PYTENSOR_FLAGS", "elevation = 'low' # 'low', 'medium' or 'high'",
                 "df_all = pd.read_csv('0Data/hs_dance_reorg.csv'", "def dance_moments(a):",
                 "def build_state_model("]
for c in nb['cells']:
    if c['cell_type'] == 'code' and any(m in ''.join(c['source']) for m in setup_markers):
        tags = c.setdefault('metadata', {}).setdefault('tags', [])
        if 'kfold-setup' not in tags:
            tags.append('kfold-setup')
n_tagged = sum('kfold-setup' in c.get('metadata', {}).get('tags', []) for c in nb['cells'])
assert n_tagged == 5, n_tagged
with open(p, 'w', encoding='utf8', newline='\r\n') as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
kernel_meta = nb['metadata']


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': s.strip('\n').splitlines(keepends=True)}


cells = [
md(r"""
# Cross-validation of the dance state models, leaving out individuals
---

## Details

&ensp;AUTHOR:&emsp;   James Foster &emsp; 2026 10 21

MODIFIED:&emsp; James Foster &emsp; 2026 10 21

DESCRIPTION:&emsp; Compares the state-switching model (with condition directions estimated, or fixed at the feeder) and the unimodal null model from [HS_circGLMM_states.ipynb](HS_circGLMM_states.ipynb) by how well they predict the dances of a **new individual**: each individual is left out in turn, the models are refitted to the remaining individuals, and the log predictive density of the left-out individual's waggle runs is calculated.
PSIS-LOO was unreliable for these models (most dances with Pareto $\hat{k} > 0.7$), and grouping by individual tests the effects shared across individuals, which are the main interest of the study.

INPUTS:&emsp;    `0Data/hs_dance_reorg.csv`; the model code in `HS_circGLMM_states.ipynb` (cells tagged `kfold-setup`)

OUTPUTS:&emsp;  `2Results/state_models/HS_states_<elevation>_<fit_tag>_kfold.csv` (log predictive density of each individual for each model)
"""),
md(r"""
## Settings
- `elevation` and the model settings are as in `HS_circGLMM_states.ipynb` (its settings cell is run below, then overridden here)
- `cv_models`: models to compare, keys of `build_state_model` calls below
- `n_folds`: `None` for leave-one-individual-out, or a number of folds (individuals assigned to balance the number of waggle runs)
- `cv_chains`, `cv_draws`, `cv_tune`: sampler settings for each refit (fewer chains than the full fits, since only the predictive density is needed)
- `S_draws`, `M_inner`, `G_grid`, `Q_nodes`: numbers of posterior draws, Monte Carlo samples of individual effects per draw, grid points for dance direction offsets and Gauss–Hermite nodes for dance concentration effects (see *Predictive density* below)
- `setup_notebook` (first code cell): the notebook whose `kfold-setup` cells are run; point it at a frozen copy to resume a run after the model code has changed
- `test_run`: a quick check of the code (one fold, short chains)
"""),
code(r"""
import json
from IPython import get_ipython

# run the data preparation and model code from HS_circGLMM_states.ipynb (cells tagged 'kfold-setup')
setup_notebook = 'HS_circGLMM_states.ipynb' # notebook with the model code (a frozen copy keeps a resumed run consistent)
states_nb = json.load(open(setup_notebook, encoding='utf8'))
setup_cells = [''.join(c['source']) for c in states_nb['cells']
               if 'kfold-setup' in c.get('metadata', {}).get('tags', [])]
print(f'{len(setup_cells)} setup cells found')


def run_setup(src):
    # run a setup cell in this namespace, stopping on errors
    get_ipython().run_cell(src).raise_error()
"""),
code(r"""
# settings cell of HS_circGLMM_states.ipynb, then overrides
run_setup(setup_cells[0]) # PYTENSOR_FLAGS
run_setup(setup_cells[1]) # settings

elevation = 'high' # 'low', 'medium' or 'high'
cv_models = ['unimodal', 'three_states_condition', 'three_states_fixed']
n_folds = None # None: leave one individual out
cv_chains = 4
cv_draws = 500
cv_tune = 1000
S_draws = 500 # posterior draws used for the predictive density
M_inner = 16 # samples of a new individual's effects per draw
G_grid = 121 # grid points for each dance's direction offset
Q_nodes = 8 # Gauss-Hermite nodes for each dance's concentration effect
cv_seed = 20261021
test_run = False
if test_run:
    cv_draws, cv_tune, S_draws, M_inner = 100, 200, 100, 4
print(f'elevation: {elevation}, fit_tag: {fit_tag}, models: {cv_models}')
"""),
code(r"""
import matplotlib.pyplot as plt
for src in setup_cells[2:]: # data, dance moments, model code
    run_setup(src)
plt.close('all')
df_full, dances_full = df.copy(), dances.copy()
print(f'{df_full["Individual"].nunique()} individuals, {len(dances_full)} dances, {len(df_full)} runs')
"""),
md(r"""
# Folds
With `n_folds = None`, each individual is a fold.
Otherwise individuals are assigned, from most to fewest waggle runs, to the fold with fewest runs so far.
Every condition must still be present in each training set.
"""),
code(r"""
ind_runs = df_full.groupby('Individual_label').size().sort_values(ascending=False)
if n_folds is None:
    folds = [[ind] for ind in ind_runs.index]
else:
    folds, load = [[] for _ in range(n_folds)], np.zeros(n_folds)
    for ind, n in ind_runs.items():
        k = int(np.argmin(load))
        folds[k].append(ind)
        load[k] += n
for k, f in enumerate(folds):
    train_conds = set(df_full.loc[~df_full['Individual_label'].isin(f), 'condition'])
    assert train_conds == set(conditions_incl), f'fold {k}: a condition is missing from the training set'
if test_run:
    folds = folds[-1:]
pd.DataFrame({'fold': range(len(folds)), 'individuals': [', '.join(f) for f in folds],
              'runs': [int(ind_runs[f].sum()) for f in folds]})
"""),
md(r"""
# Refitting to the training individuals
`build_state_model` and `init_mean` read the data from the global `df` and `dances`, so these are replaced by the training subset (with dances and individuals recoded from 0) for each fit.
"""),
code(r"""
def recode(df_sub):
    # data and dance table for a subset of individuals, coded as in HS_circGLMM_states.ipynb
    d = df_sub.drop(columns=['Dance', 'Individual']).copy()
    d = d.sort_values(['Dance_label', 'waggle_run']).reset_index(drop=True)
    d['Dance'] = d['Dance_label'].astype('category').cat.codes
    d['Individual'] = d['Individual_label'].astype('category').cat.codes
    tab = (d.groupby('Dance')
             .agg(Dance_label=('Dance_label', 'first'), Individual=('Individual', 'first'),
                  Individual_label=('Individual_label', 'first'), condition=('condition', 'first'),
                  cond_idx=('cond_idx', 'first'), n_runs=('angle', 'size'), sun_el=('sun_el', 'mean'))
             .reset_index())
    tab = tab.join(d.groupby('Dance')['angle'].apply(dance_moments).unstack(), on='Dance')
    return d, tab


def build_cv_model(key, df_tr, dances_tr):
    global df, dances, n_dance
    df, dances, n_dance = df_tr, dances_tr, len(dances_tr)
    if key == 'unimodal':
        return build_state_model(df, dances, states=['forward'], state_effects=False, direction='vonmises')
    if key == 'two_states':
        return build_state_model(df, dances, states=['forward', 'reversed'] + STATES[3:], state_effects=True)
    if key == 'three_states_same':
        return build_state_model(df, dances, states=STATES, state_effects=False)
    if key == 'three_states_fixed':
        return build_state_model(df, dances, states=STATES, state_effects=True, fixed_directions=True)
    return build_state_model(df, dances, states=STATES, state_effects=True)


def model_spec(key):
    # states and direction prior of each model, as in build_cv_model
    if key == 'unimodal':
        return ['forward'], 'vonmises'
    if key == 'two_states':
        return ['forward', 'reversed'] + STATES[3:], direction_prior
    return list(STATES), direction_prior


def sample_cv(model, seed):
    start = {}
    if data_inits:
        start = {'nuts_sampler_kwargs': {'init_mean': init_mean(model)}}
    with model:
        return pm.sample(draws=cv_draws, tune=cv_tune, chains=cv_chains, nuts_sampler=sampler,
                         target_accept=target_accept, random_seed=seed, progressbar=False, **start)
"""),
md(r"""
# Predictive density of a new individual
For a left-out individual $i$ with dances $d$, the log predictive density is
$$
\log p(\boldsymbol{y}_i \mid \text{training data}) = \log \mathbb{E}_{\phi \mid \text{train}} \, \mathbb{E}_{b_i \mid \phi} \prod_{d} \int\!\!\int \sum_s \pi_{ds} \prod_j f_s(\theta_{dj} \mid \mu_d, \kappa_d) \; p(v_d \mid \phi)\, p(z_d \mid \phi) \, dv_d \, dz_d
$$
where $\phi$ are the population-level parameters, $b_i$ the individual's effects (direction offset $u_i$, concentration effect, and state-logit effects), $v_d$ the dance direction offset and $z_d$ the dance concentration effect.
- $\phi$: `S_draws` posterior draws, thinned evenly.
- $b_i$: `M_inner` Monte Carlo samples per draw from their population distributions.
- $v_d$: a grid of `G_grid` points (bounded priors: Normal, ±5 SD; von Mises priors: the whole circle), since the likelihood of a dance with many runs is sharply peaked in its direction, which Monte Carlo would sample poorly.
- $z_d$: `Q_nodes` Gauss–Hermite nodes.
- $s$: the dance's state, summed as in the model (the unrelated state is integrated over its direction in closed form).

The effective number of Monte Carlo samples (of $S \times M$) is reported for each individual; a small value means the estimate for that individual is noisy and probably too low.
"""),
code(r"""
from scipy.special import i0e, log_softmax, logsumexp


def np_lognorm(k):
    # log(2*pi*I0(kappa)), stable for large kappa
    return LOG_2PI + k + np.log(i0e(k))


def thin(post, name, idx):
    x = post[name].values # chain x draw x ...
    return x.reshape(-1, *x.shape[2:])[idx]


def heldout_lpd(idata, key, df_te, rng):
    # log predictive density of each left-out individual: Series indexed by individual label
    states, direction = model_spec(key)
    post = idata.posterior
    n_all = post.sizes['chain'] * post.sizes['draw']
    idx = np.linspace(0, n_all - 1, S_draws).astype(int)
    P = lambda name: thin(post, name, idx)
    has = lambda name: name in post
    mu_c = P('mu_condition') # S x C
    k0 = P('kappa_Intercept')
    keff = P('kappa_effect') if has('kappa_effect') else np.zeros((S_draws, 0))
    sk_i, sk_d = P('kappa_Individual_sigma'), P('kappa_Dance_sigma')
    kappa_u = P('mu_Individual_kappa')
    # dance direction offsets: grid and log weights (S x G)
    if direction == 'vonmises':
        v = np.broadcast_to(np.linspace(-np.pi, np.pi, G_grid, endpoint=False), (S_draws, G_grid))
        lw_v = P('mu_Dance_kappa')[:, None] * np.cos(v)
    else:
        t = np.linspace(-5, 5, G_grid)
        v = P('mu_Dance_sigma')[:, None] * t[None, :]
        lw_v = np.broadcast_to(-0.5 * t ** 2, (S_draws, G_grid))
    lw_v = lw_v - logsumexp(lw_v, axis=1, keepdims=True)
    # dance concentration effects: Gauss-Hermite nodes and log weights
    xq, wq = np.polynomial.hermite_e.hermegauss(Q_nodes)
    lw_q = np.log(wq / wq.sum())
    # state-probability parameters
    main = [s for s in states if s not in OUTLIER_STATES]
    out = [s for s in states if s in OUTLIER_STATES]
    w_ax = P('w_axial') if 'axial' in states else None
    res = {}
    for ind, d_ind in df_te.groupby('Individual_label'):
        # a new individual's effects: S x M
        u = rng.vonmises(0.0, np.broadcast_to(kappa_u[:, None], (S_draws, M_inner)))
        zk_i = rng.standard_normal((S_draws, M_inner))
        z_state = {s: rng.standard_normal((S_draws, M_inner)) for s in main[1:]}
        ll_ind = np.zeros((S_draws, M_inner))
        for dance, d_d in d_ind.groupby('Dance_label'):
            th = d_d['angle'].values
            c = int(d_d['cond_idx'].iloc[0])
            x_c = X_cond[c]
            n, C, Sn = len(th), np.cos(th).sum(), np.sin(th).sum()
            # direction: S x M x G
            off = u[:, :, None] + v[:, None, :]
            off = off if direction == 'vonmises' else bounded_np(off, MAX_OFFSET)
            mu = mu_c[:, c][:, None, None] + off
            # concentration: S x M x Q
            eta_k = (k0 + keff @ x_c)[:, None, None] + (sk_i[:, None] * zk_i)[:, :, None] + sk_d[:, None, None] * xq
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
                L['disoriented'] = np.full((S_draws, M_inner), -n * LOG_2PI)
            # state probabilities: S x M x states
            etas = [np.zeros((S_draws, M_inner))]
            for s in main[1:]:
                e = P(f'{s}_Intercept')[:, None] + np.zeros((1, M_inner))
                if has(f'{s}_effect'):
                    e = e + (P(f'{s}_effect') @ x_c)[:, None]
                e = e + P(f'{s}_Individual_sigma')[:, None] * z_state[s]
                etas.append(e)
            log_pi = log_softmax(np.stack(etas, axis=-1), axis=-1)
            if out:
                e_out = P(f'{out[0]}_Intercept') + (P(f'{out[0]}_effect') @ x_c if has(f'{out[0]}_effect') else 0)
                e_out = e_out[:, None, None]
                log_pi = np.concatenate([log_pi - np.logaddexp(0, e_out),
                                         np.broadcast_to(-np.logaddexp(0, -e_out), (S_draws, M_inner, 1))], axis=-1)
            L_s = np.stack([L[s] for s in main + out], axis=-1)
            ll_ind += logsumexp(log_pi + L_s, axis=-1)
        flat = ll_ind.ravel()
        w = np.exp(flat - flat.max())
        res[ind] = {'lpd': logsumexp(flat) - np.log(flat.size), 'mc_ess': w.sum() ** 2 / (w ** 2).sum(),
                    'n_dances': d_ind['Dance_label'].nunique(), 'n_runs': len(d_ind)}
    return pd.DataFrame(res).T


def bounded_np(x, bound):
    return bound * np.tanh(x / bound)
"""),
md(r"""
# Run the folds
Each fold refits every model in `cv_models` to the training individuals and calculates the predictive density of the left-out individuals.
Results are saved after each fold, so an interrupted run can be resumed (completed folds are skipped).
"""),
code(r"""
import os, time
import logging
logging.getLogger('pymc').setLevel(logging.WARNING)

os.makedirs('2Results/state_models', exist_ok=True)
out_file = f'2Results/state_models/HS_states_{elevation}_{fit_tag}_kfold{"_test" if test_run else ""}.csv'
done = pd.read_csv(out_file) if os.path.exists(out_file) else pd.DataFrame(columns=['fold', 'model'])
rng = np.random.default_rng(cv_seed)
rows = [done]
for k, test_inds in enumerate(folds):
    df_tr, dances_tr = recode(df_full[~df_full['Individual_label'].isin(test_inds)])
    df_te = df_full[df_full['Individual_label'].isin(test_inds)]
    for key in cv_models:
        if ((done['fold'] == k) & (done['model'] == key)).any():
            continue
        t0 = time.time()
        model = build_cv_model(key, df_tr, dances_tr)
        idata_k = sample_cv(model, seed=cv_seed + k)
        n_div = int(idata_k.sample_stats['diverging'].sum())
        lp = idata_k.sample_stats['logp' if 'logp' in idata_k.sample_stats else 'lp'].mean('draw').values
        res = heldout_lpd(idata_k, key, df_te, rng)
        res = res.assign(fold=k, model=key, divergences=n_div, chain_logp_range=float(lp.max() - lp.min()),
                         minutes=(time.time() - t0) / 60).rename_axis('individual').reset_index()
        rows.append(res)
        pd.concat(rows, ignore_index=True).to_csv(out_file, index=False)
        print(f'fold {k} ({", ".join(test_inds)}), {key}: lpd {res["lpd"].sum():.1f}, '
              f'min MC ESS {res["mc_ess"].min():.0f}, divergences {n_div}, '
              f'chain logp range {lp.max() - lp.min():.1f}, {(time.time() - t0) / 60:.1f} min', flush=True)
        del model, idata_k
cv = pd.concat(rows, ignore_index=True)
df, dances, n_dance = df_full, dances_full, len(dances_full) # restore the full data
"""),
md(r"""
# Comparison
The expected log predictive density (elpd) of each model is the sum over individuals; the standard error of the difference is $\sqrt{n \, \text{Var}(\Delta_i)}$, over the $n$ individuals.
Check `mc_ess` (individuals with a small effective sample size have noisy estimates) and `chain_logp_range` (a large range suggests that a refit's chains disagreed).
"""),
code(r"""
cv = pd.read_csv(out_file)
tab = cv.pivot_table(index='individual', columns='model', values='lpd')
summary = pd.DataFrame({'elpd': tab.sum(), 'n_individuals': tab.count()})
best = summary['elpd'].idxmax()
diff = tab.sub(tab[best], axis=0)
summary['elpd_diff'] = diff.sum()
summary['se_diff'] = np.sqrt(len(tab) * diff.var(ddof=1))
print(summary.sort_values('elpd', ascending=False).round(1))
print('\nPer individual (lpd), with the difference from the best model:')
print(tab.assign(n_runs=cv.groupby('individual')['n_runs'].first(),
                 min_mc_ess=cv.groupby('individual')['mc_ess'].min().round(0),
                 **{f'diff_{m}': diff[m] for m in tab.columns if m != best}).round(1).sort_values('n_runs', ascending=False))
print('\nRefit diagnostics:')
print(cv.groupby(['model'])[['divergences', 'chain_logp_range', 'minutes']].describe().round(1).T)
"""),
]

kf = {'cells': cells, 'metadata': kernel_meta, 'nbformat': 4, 'nbformat_minor': 5}
with open('HS_circGLMM_states_kfold.ipynb', 'w', encoding='utf8', newline='\r\n') as f:
    f.write(json.dumps(kf, indent=1, ensure_ascii=False) + '\n')
print('created HS_circGLMM_states_kfold.ipynb')
