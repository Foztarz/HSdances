"""Create HS_circGLMM_states_joint.ipynb: the dance state model fitted jointly across solar elevation bands."""
import json

kernel_meta = json.load(open('HS_circGLMM_states.ipynb', encoding='utf8'))['metadata']


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': s.strip('\n').splitlines(keepends=True)}


cells = [
md(r"""
# A joint state-switching model of honeybee waggle-dances across solar elevations
---

## Details

&ensp;AUTHOR:&emsp;   James Foster &emsp; 2026 10 21

MODIFIED:&emsp; James Foster &emsp; 2026 10 21

DESCRIPTION:&emsp; The state-switching model of [HS_circGLMM_states.ipynb](HS_circGLMM_states.ipynb), fitted to all three solar elevation bands at once (low, $< 15°$; medium, $30°$–$45°$; high, $> 75°$).
Each dance is in one of four states: **forward**, **reversed**, **axial**, or **unrelated** (concentrated around a direction unrelated to the feeder), marginalised per dance.
- The vertical direction is shared across elevations, since vertical dances should point towards the feeder at all elevations (a central part of the study design).
- Directions of the other conditions, concentrations and state probabilities can differ between elevation bands (condition × band effects).
- Individual effects are shared across bands (19 of 35 bees danced in more than one band).
- A null model with only unimodal dances (unbounded von Mises offsets, as in the `GUVdances` analysis), and a model with all directions fixed at the feeder, are fitted for comparison.

Design matrices are built per dance from its condition and band, so that a continuous effect of sun elevation can replace the bands later.

INPUTS:&emsp;    `0Data/hs_dance_reorg.csv`

OUTPUTS:&emsp;  `0Data/hs_dance_states_joint.feather`, `2Results/state_models/HS_states_joint_<fit_tag>_<model>.nc`
"""),
md(r"""
# Set up
Same environment as [HS_circGLMM_states.ipynb](HS_circGLMM_states.ipynb) (`HS-env`).
"""),
code(r"""
%set_env PYTENSOR_FLAGS=blas__ldflags='-llapack -lblas -lcblas'
"""),
md(r"""
## Settings
- `bands`: solar elevation ranges (degrees, exclusive), as in `HS_ExploratoryPlots.Rmd`, and `conditions_by_band`: the conditions included in each band. Low-elevation antisolar (4 dances, 1 bee) and zenith (5 dances) are excluded, as in the single-elevation models, and tilted dances are excluded throughout.
- `reference_band`: the band in which condition effects are defined (medium, the only band with all four conditions); band effects and condition × band interactions are relative to it.
- `direction_prior`, `condition_bound`, `outlier_state`, `data_inits`: as in `HS_circGLMM_states.ipynb`.
- `models_to_fit`: which models (below) to fit; with `skip_existing`, models already saved are loaded instead.
"""),
code(r"""
bands = {'low': (-90, 15), 'medium': (30, 45), 'high': (75, 90)} # degrees, exclusive
conditions_by_band = {
    'low': ['vertical', 'horizontal'],
    'medium': ['vertical', 'horizontal', 'antisolar', 'solar'],
    'high': ['vertical', 'horizontal', 'antisolar', 'solar'],
}
conditions_all = ['vertical', 'horizontal', 'antisolar', 'solar'] # successive-difference order
reference_band = 'medium'
direction_prior = 'bounded_vm' # bounded condition directions and dance offsets, von Mises individual offsets
condition_bound = 'feeder' # each condition's direction within 60° of the feeder
outlier_state = 'unrelated' # 'unrelated', 'disoriented' or None
data_inits = True
models_to_fit = ['states_cond_x_band']
skip_existing = True # load models already saved with this fit_tag instead of refitting

sampler = "nutpie"
n_draws = 1000
n_tune = 1000
n_chains = 4 # 8 chains ran out of memory (16 GB) with 312 dances
target_accept = 0.95
random_seed = 20261021
logp_threshold = 20 # chains this far below the best (mean log posterior) are flagged, not dropped
fit_tag = f'{direction_prior}_{outlier_state}' if outlier_state else direction_prior
"""),
md(r"""
# Data
"""),
code(r"""
import os
import numpy as np
import pandas as pd

df_all = pd.read_csv('0Data/hs_dance_reorg.csv', sep=',')
df_all['band'] = None
for b, (lo, hi) in bands.items():
    df_all.loc[(df_all['sun_el'] > lo) & (df_all['sun_el'] < hi), 'band'] = b
keep = pd.Series(False, index=df_all.index)
for b, conds in conditions_by_band.items():
    keep |= (df_all['band'] == b) & df_all['condition'].isin(conds)
df = df_all[keep].copy()

# angles relative to the feeder, between -pi and pi; drop missing angles
df['angle'] = np.angle(np.exp(1j * pd.to_numeric(df['feeder_angle'], errors='coerce')))
df = df.dropna(subset=['angle'])
df = df.rename(columns={'ID': 'Individual_label', 'ID_d': 'Dance_label'})
df['condition'] = pd.Categorical(df['condition'], categories=conditions_all)
df['band'] = pd.Categorical(df['band'], categories=list(bands))
df = df.sort_values(['Dance_label', 'waggle_run']).reset_index(drop=True)
df['Dance'] = df['Dance_label'].astype('category').cat.codes
df['Individual'] = df['Individual_label'].astype('category').cat.codes
assert (df.groupby('Dance')['band'].nunique() == 1).all(), 'a dance spans two elevation bands'

# condition x band cells, in band then condition order
cells_tab = (df[['band', 'condition']].drop_duplicates()
             .sort_values(['band', 'condition']).reset_index(drop=True))
cells_tab['cell'] = cells_tab['band'].astype(str) + '_' + cells_tab['condition'].astype(str)
cell_of = {(b, c): k for k, (b, c) in enumerate(zip(cells_tab['band'], cells_tab['condition']))}
df['cell_idx'] = [cell_of[(b, c)] for b, c in zip(df['band'], df['condition'])]


def dance_moments(a):
    z1, z2 = np.exp(1j * a).mean(), np.exp(2j * a).mean()
    return pd.Series({'R1': np.abs(z1), 'mu1': np.angle(z1), 'R2': np.abs(z2)})


dances = (df.groupby('Dance')
            .agg(Dance_label=('Dance_label', 'first'), Individual=('Individual', 'first'),
                 Individual_label=('Individual_label', 'first'), condition=('condition', 'first'),
                 band=('band', 'first'), cell_idx=('cell_idx', 'first'),
                 n_runs=('angle', 'size'), sun_el=('sun_el', 'mean'))
            .reset_index())
dances = dances.join(df.groupby('Dance')['angle'].apply(dance_moments).unstack(), on='Dance')
n_dance, n_ind = len(dances), df['Individual'].nunique()

df[['angle', 'Individual', 'Individual_label', 'Dance', 'Dance_label', 'condition', 'band', 'sun_el']].to_feather(
    '0Data/hs_dance_states_joint.feather')
print(pd.crosstab(dances['condition'], dances['band']))
print(f'{n_ind} individuals, {n_dance} dances, {len(df)} runs')
print('individuals per number of bands:', dances.groupby('Individual')['band'].nunique().value_counts().sort_index().to_dict())
"""),
md(r"""
## Design matrices
Effects are built per dance from named terms:
- condition effects, as successive differences: `H` (horizontal − vertical), `A` (antisolar − horizontal), `S` (solar − antisolar);
- band effects relative to the reference band (`low`, `high`);
- their interactions (e.g. `H:high`), only where that combination has data.

Formulas used below:
- `condition`: `H`, `A`, `S`;
- `condition + band`: also `low`, `high`;
- `condition * band`: also the interactions;
- `condition + condition:band`: condition effects and interactions but no band main effect, used for directions, so that the **vertical direction is the same in every band**.
"""),
code(r"""
COND_EFFECTS = {'vertical': [], 'horizontal': ['H'], 'antisolar': ['H', 'A'], 'solar': ['H', 'A', 'S']}
other_bands = [b for b in bands if b != reference_band]


def term_column(tab, term):
    col = np.ones(len(tab))
    for q in term.split(':'):
        if q in ('H', 'A', 'S'):
            col = col * tab['condition'].map(lambda c: float(q in COND_EFFECTS[c])).values
        elif q in bands:
            col = col * (tab['band'].astype(str) == q).values
        else:
            raise ValueError(f'unknown term {q}')
    return col


interactions = [f'{e}:{b}' for e in ['H', 'A', 'S'] for b in other_bands
                if term_column(cells_tab, f'{e}:{b}').any()]
FORMULAS = {
    '1': [],
    'condition': ['H', 'A', 'S'],
    'condition + band': ['H', 'A', 'S'] + other_bands,
    'condition * band': ['H', 'A', 'S'] + other_bands + interactions,
    'condition + condition:band': ['H', 'A', 'S'] + interactions,
}


def design(tab, formula):
    terms = FORMULAS[formula]
    X = np.column_stack([term_column(tab, t) for t in terms]) if terms else np.zeros((len(tab), 0))
    return X, terms


print('interaction terms with data:', interactions)
for f in FORMULAS:
    X, t = design(cells_tab, f)
    print(f"{f}: {len(t)} effects, design rank {np.linalg.matrix_rank(X) if t else 0}")
"""),
md(r"""
# The joint model
As in [HS_circGLMM_states.ipynb](HS_circGLMM_states.ipynb), with design matrices per dance rather than per condition:

- **Direction**: for each cell (condition × band), $\mu_\text{cell} = B \tanh\left((0.5 z + X^\mu_\text{cell} \beta^\mu) / B\right)$, within $B = 60°$ of the feeder; dance directions $\mu_d = \mu_{\text{cell}[d]} + B \tanh\left((u_{i[d]} + v_d)/B\right)$, with von Mises individual offsets and Normal dance offsets.
  With `fixed_directions`, every cell's forward direction is at the feeder. In the unimodal null model, directions are unbounded with von Mises priors.
- **Concentration**: $\text{softplus}^{-1}(\kappa_d) = \alpha^\kappa + X^\kappa_d \beta^\kappa + $ individual and dance effects.
- **States**: multinomial logit for forward, reversed and axial (individual random effects), and an outlier state (unrelated) with its own probability, $\text{logit}^{-1}(\alpha^\text{out} + X^\text{out}_d \beta^\text{out})$, the other states sharing the remainder.
"""),
code(r"""
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import xarray as xr

STATES = ['forward', 'reversed', 'axial'] + ([outlier_state] if outlier_state else [])
OUTLIER_STATES = ('unrelated', 'disoriented')
LOG_2PI = np.log(2 * np.pi)
MAX_POP = np.pi / 3
MAX_COND = np.pi / 3
MAX_OFFSET = np.pi / 3
MAX_SIGMA = 0.6


def inverse_softplus(x):
    return np.log(np.expm1(x))


def vonmises_lognorm(kappa):
    return LOG_2PI + kappa + pt.log(pt.ive(0, kappa))


def circ_sd(kappa):
    return pt.sqrt(-2 * pt.log(pt.ive(1, kappa) / pt.ive(0, kappa)))


def bounded(x, bound):
    return bound * pt.tanh(x / bound)


def build_joint_model(df, dances, spec):
    # spec: dict with 'states', 'direction', 'mu', 'kappa', 'state' (formulas) and optionally 'fixed_directions'
    states, direction = spec['states'], spec['direction']
    fixed = spec.get('fixed_directions', False)
    Xmu_c, mu_terms = design(cells_tab, spec['mu'])
    Xk_d, k_terms = design(dances, spec['kappa'])
    Xs_d, s_terms = design(dances, spec['state'])
    cell_d = dances['cell_idx'].values
    ind_idx = dances['Individual'].values
    dance_idx = df['Dance'].values
    n_d = np.bincount(dance_idx, minlength=n_dance).astype(float)
    C_d = np.bincount(dance_idx, weights=np.cos(df['angle'].values), minlength=n_dance)
    S_d = np.bincount(dance_idx, weights=np.sin(df['angle'].values), minlength=n_dance)
    M = np.zeros((n_dance, len(df)))
    M[dance_idx, np.arange(len(df))] = 1.0
    coords = {'dance': dances['Dance_label'].values,
              'individual': np.sort(dances['Individual_label'].unique()),
              'cell': cells_tab['cell'].values,
              'mu_effect_term': mu_terms, 'kappa_effect_term': k_terms, 'state_effect_term': s_terms,
              'state': list(states)}

    with pm.Model(coords=coords) as model:
        theta = pm.Data('theta', df['angle'].values)

        # --- Direction ---
        if direction == 'vonmises':
            mu0 = pm.VonMises('mu_Intercept', mu=0.0, kappa=5.0)
            mu_c = mu0
            if mu_terms:
                mu_c = mu0 + pt.dot(Xmu_c, pm.VonMises('mu_effect', mu=0.0, kappa=2.0, dims='mu_effect_term'))
            mu_c = pm.Deterministic('mu_cell', mu_c * pt.ones(len(cells_tab)), dims='cell')
            kappa_mu_ind = pm.LogNormal('mu_Individual_kappa', mu=np.log(30.0), sigma=0.6)
            kappa_mu_dance = pm.LogNormal('mu_Dance_kappa', mu=np.log(5.0), sigma=0.5)
            pm.Deterministic('mu_Individual_sigma', circ_sd(kappa_mu_ind))
            pm.Deterministic('mu_Dance_sigma', circ_sd(kappa_mu_dance))
            u_ind = pm.VonMises('mu_Individual', mu=0.0, kappa=kappa_mu_ind, dims='individual')
            v_dance = pm.VonMises('mu_Dance', mu=0.0, kappa=kappa_mu_dance, dims='dance')
            offset = pm.Deterministic('mu_offset', u_ind[ind_idx] + v_dance, dims='dance')
        else:
            if fixed:
                mu_c = pm.Deterministic('mu_cell', pt.zeros(len(cells_tab)), dims='cell')
            else:
                mu0_z = 0.5 * pm.Normal('mu_Intercept_z', 0.0, 1.0)
                lat = mu0_z * pt.ones(len(cells_tab))
                if mu_terms:
                    lat = lat + pt.dot(Xmu_c, pm.Normal('mu_effect', mu=0.0, sigma=1.0, dims='mu_effect_term'))
                mu_c = pm.Deterministic('mu_cell', bounded(lat, MAX_COND), dims='cell')
            kappa_mu_ind = pm.LogNormal('mu_Individual_kappa', mu=np.log(30.0), sigma=0.6)
            pm.Deterministic('mu_Individual_sigma', circ_sd(kappa_mu_ind))
            u_ind = pm.VonMises('mu_Individual', mu=0.0, kappa=kappa_mu_ind, dims='individual')
            sigma_mu_dance = pm.TruncatedNormal('mu_Dance_sigma', mu=0.0, sigma=0.5, lower=0.0, upper=MAX_SIGMA)
            v_dance = sigma_mu_dance * pm.Normal('mu_Dance_z', 0.0, 1.0, dims='dance')
            offset = pm.Deterministic('mu_offset', bounded(u_ind[ind_idx] + v_dance, MAX_OFFSET), dims='dance')
        mu_d = pm.Deterministic('mu', mu_c[cell_d] + offset, dims='dance')

        # --- Concentration ---
        k0 = pm.Normal('kappa_Intercept', mu=inverse_softplus(3.0), sigma=3.0)
        k_eff = pt.dot(Xk_d, pm.Normal('kappa_effect', mu=0.0, sigma=1.0, dims='kappa_effect_term')) if k_terms else 0.0
        sk_ind = pm.HalfNormal('kappa_Individual_sigma', sigma=1.0)
        sk_dance = pm.HalfNormal('kappa_Dance_sigma', sigma=1.0)
        zk_ind = pm.Normal('kappa_Individual_z', 0.0, 1.0, dims='individual')
        zk_dance = pm.Normal('kappa_Dance_z', 0.0, 1.0, dims='dance')
        kappa_d = pm.Deterministic('kappa', pt.softplus(k0 + k_eff + sk_ind * zk_ind[ind_idx] + sk_dance * zk_dance),
                                   dims='dance')

        # --- State probabilities ---
        main_states = [s for s in states if s not in OUTLIER_STATES]
        out_states = [s for s in states if s in OUTLIER_STATES]
        etas = [pt.zeros(n_dance)]
        for s in main_states[1:]:
            e = pm.Normal(f'{s}_Intercept', mu=-1.0, sigma=1.5) * pt.ones(n_dance)
            if s_terms:
                e = e + pt.dot(Xs_d, pm.Normal(f'{s}_effect', mu=0.0, sigma=1.5, dims='state_effect_term'))
            s_sig = pm.HalfNormal(f'{s}_Individual_sigma', sigma=1.0)
            e = e + s_sig * pm.Normal(f'{s}_Individual_z', 0.0, 1.0, dims='individual')[ind_idx]
            etas.append(e)
        log_pi = pt.special.log_softmax(pt.stack(etas, axis=1), axis=1)
        if out_states:
            s = out_states[0]
            eta_out = pm.Normal(f'{s}_Intercept', mu=-2.0, sigma=1.0) * pt.ones(n_dance)
            if s_terms:
                eta_out = eta_out + pt.dot(Xs_d, pm.Normal(f'{s}_effect', mu=0.0, sigma=1.0, dims='state_effect_term'))
            log_pi = pt.concatenate([log_pi - pt.softplus(eta_out)[:, None], -pt.softplus(-eta_out)[:, None]], axis=1)

        # --- Likelihood for each state, marginalised over the state of each dance ---
        norm_d = n_d * vonmises_lognorm(kappa_d)
        proj_d = kappa_d * (C_d * pt.cos(mu_d) + S_d * pt.sin(mu_d))
        L = {'forward': proj_d - norm_d, 'reversed': -proj_d - norm_d}
        if 'axial' in states:
            w_ax = pm.Deterministic('w_axial', 0.25 + 0.5 * pm.math.sigmoid(pm.Normal('w_axial_logit', 0.0, 1.0)))
            proj_r = kappa_d[dance_idx] * pt.cos(theta - mu_d[dance_idx])
            L['axial'] = pt.dot(M, pt.logaddexp(pt.log(w_ax) + proj_r, pt.log1p(-w_ax) - proj_r)) - norm_d
        if 'unrelated' in states:
            R_d = np.hypot(C_d, S_d)
            L['unrelated'] = kappa_d * R_d + pt.log(pt.ive(0, kappa_d * R_d)) - norm_d
        if 'disoriented' in states:
            L['disoriented'] = pt.as_tensor(-n_d * LOG_2PI)
        joint = log_pi + pt.stack([L[s] for s in main_states + out_states], axis=1)
        loglik = pm.Deterministic('loglik_dance', pt.logsumexp(joint, axis=1), dims='dance')
        pm.Deterministic('p_state', pt.special.softmax(joint, axis=1), dims=('dance', 'state'))
        pm.Potential('obs', loglik.sum())
    return model
"""),
md(r"""
## Starting values from the data
As in `HS_circGLMM_states.ipynb`: each cell's direction from the axes of its dances (doubled-angle means, the end within 90° of the feeder taken as forward), and the concentration from the dances' mean resultant lengths.
nutpie jitters each chain's start around these values.
"""),
code(r"""
def data_initvals(model, spec):
    z2 = df.groupby('Dance')['angle'].apply(lambda a: np.exp(2j * a).mean()).values
    axis_cell = np.array([np.angle(z2[dances['cell_idx'].values == k].sum()) / 2 for k in range(len(cells_tab))])
    if spec['direction'] == 'vonmises':
        lat = axis_cell
    else:
        lat = MAX_COND * np.arctanh(np.clip(axis_cell / MAX_COND, -0.8, 0.8))
    vert = (cells_tab['condition'] == 'vertical').values
    lat0 = lat[vert].mean()
    R = np.clip(np.maximum(dances['R1'].values, np.abs(z2)), 0.05, 0.95)
    kappa_hat = np.median(R * (2 - R ** 2) / (1 - R ** 2))
    iv = {'mu_Intercept_z': 2 * lat0, 'mu_Intercept': lat0,
          'mu_Dance_sigma': 0.3, 'mu_Individual_kappa': 30.0,
          'kappa_Intercept': inverse_softplus(kappa_hat),
          'kappa_Individual_sigma': 0.5, 'kappa_Dance_sigma': 0.5,
          'reversed_Individual_sigma': 0.5, 'axial_Individual_sigma': 0.5,
          'unrelated_Intercept': -2.0, 'disoriented_Intercept': -2.0}
    Xmu_c, mu_terms = design(cells_tab, spec['mu'])
    if mu_terms:
        iv['mu_effect'] = np.linalg.lstsq(Xmu_c, lat - lat0, rcond=None)[0]
    return {k: v for k, v in iv.items() if k in model.named_vars and model[k] in model.free_RVs}


def init_mean(model, spec):
    for name, val in data_initvals(model, spec).items():
        model.set_initval(model[name], np.asarray(val, dtype=float))
    ip = model.initial_point()
    return np.concatenate([np.ravel(ip[model.rvs_to_values[rv].name]) for rv in model.free_RVs])
"""),
md(r"""
# Models
All state models have directions ~ `condition + condition:band` (vertical shared across bands) and concentration ~ `condition * band`; they differ in the state probabilities (forward/reversed/axial logits and the unrelated probability):

| model | state probabilities | directions |
|---|---|---|
| `unimodal` | forward only (null model) | unbounded von Mises |
| `states_cond` | condition | estimated |
| `states_cond_band` | condition + band | estimated |
| `states_cond_x_band` | condition × band | estimated |
| `states_cond_x_band_fixed` | condition × band | fixed at the feeder |

Comparing `states_cond`, `states_cond_band` and `states_cond_x_band` tests whether the probability of each state differs between elevation bands, and whether that difference depends on condition; comparing `states_cond_x_band` with `states_cond_x_band_fixed` tests whether population-level shifts in direction explain the data better than individual and dance offsets; the null model tests whether states are needed at all.
"""),
code(r"""
base = {'states': STATES, 'direction': direction_prior, 'mu': 'condition + condition:band', 'kappa': 'condition * band'}
model_specs = {
    'unimodal': {'states': ['forward'], 'direction': 'vonmises', 'mu': 'condition + condition:band',
                 'kappa': 'condition * band', 'state': '1'},
    'states_cond': {**base, 'state': 'condition'},
    'states_cond_band': {**base, 'state': 'condition + band'},
    'states_cond_x_band': {**base, 'state': 'condition * band'},
    'states_cond_x_band_fixed': {**base, 'state': 'condition * band', 'fixed_directions': True},
}
models = {k: build_joint_model(df, dances, model_specs[k]) for k in models_to_fit}
for k, m in models.items():
    print(f'{k}: {len(m.free_RVs)} free variables, {sum(np.prod(v.shape.eval()) if v.ndim else 1 for v in m.free_RVs)} parameters')
"""),
md(r"""
# Fit the models
All chains are kept; chains whose mean log posterior is more than `logp_threshold` below the best are reported.
"""),
code(r"""
idatas = {}
os.makedirs('2Results/state_models', exist_ok=True)
for key, model in models.items():
    spec = model_specs[key]
    nc_file = f'2Results/state_models/HS_states_joint_{fit_tag}_{key}.nc'
    print(f'\n===== {key} =====', flush=True)
    if skip_existing and os.path.exists(nc_file):
        idatas[key] = az.from_netcdf(nc_file)
        print(f'loaded {nc_file}')
        continue
    start = {'nuts_sampler_kwargs': {'init_mean': init_mean(model, spec)}} if data_inits else {}
    with model:
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, nuts_sampler=sampler,
                          target_accept=target_accept, random_seed=random_seed, progressbar=False, **start)
    lp = idata.sample_stats['logp'].mean('draw')
    flagged = np.where((lp < lp.max() - logp_threshold).values)[0].tolist()
    print(f'Mean log posterior per chain: {np.round(lp.values, 1)}; flagged (not dropped): {flagged}')
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")
    idata.add_groups(log_likelihood=xr.Dataset({'loglik_dance': idata.posterior['loglik_dance']}))
    az.to_netcdf(idata, nc_file)
    idatas[key] = idata
"""),
md(r"""
# Convergence
"""),
code(r"""
main_vars = ['mu_cell', 'mu_effect', 'kappa_Intercept', 'kappa_effect',
             'reversed_Intercept', 'reversed_effect', 'axial_Intercept', 'axial_effect',
             'unrelated_Intercept', 'unrelated_effect', 'w_axial',
             'mu_Individual_sigma', 'mu_Dance_sigma', 'kappa_Individual_sigma', 'kappa_Dance_sigma',
             'reversed_Individual_sigma', 'axial_Individual_sigma']
summaries = {}
for key, idata in idatas.items():
    vs = [v for v in main_vars if v in idata.posterior]
    summ = az.summary(idata, var_names=vs, circ_var_names=['mu_cell'] if 'mu_cell' in vs else [])
    summaries[key] = summ
    print(f"{key}: max R-hat {summ['r_hat'].max():.3f} ({summ['r_hat'].idxmax()}), "
          f"min bulk ESS {summ['ess_bulk'].min():.0f} ({summ['ess_bulk'].idxmin()})")
"""),
code(r"""
# chain means of the cell directions, for every chain
for key, idata in idatas.items():
    z = np.exp(1j * idata.posterior['mu_cell']).mean('draw')
    tab = pd.DataFrame(np.rad2deg(np.angle(z.values)).round(0), columns=cells_tab['cell'],
                       index=[f'chain {c}' for c in range(z.sizes['chain'])])
    tab['mean logp'] = idata.sample_stats['logp'].mean('draw').values.round(1)
    print(f'\n===== {key} =====')
    print(tab.T)
"""),
md(r"""
# State probabilities by condition and band
For a typical individual (individual effects at zero).
"""),
code(r"""
from scipy.special import softmax


def cell_state_probs(idata, spec):
    post = idata.posterior
    n_s = post.sizes['chain'] * post.sizes['draw']
    D = lambda name: post[name].values.reshape(n_s, *post[name].shape[2:])
    Xs_c, s_terms = design(cells_tab, spec['state'])
    states = spec['states']
    main = [s for s in states if s not in OUTLIER_STATES]
    out = [s for s in states if s in OUTLIER_STATES]
    eta = np.zeros((n_s, len(cells_tab), len(main)))
    for j, s in enumerate(main[1:], start=1):
        eta[:, :, j] = D(f'{s}_Intercept')[:, None] + (D(f'{s}_effect') @ Xs_c.T if s_terms else 0)
    p = softmax(eta, axis=2)
    if out:
        e = D(f'{out[0]}_Intercept')[:, None] + (D(f'{out[0]}_effect') @ Xs_c.T if s_terms else 0)
        p_out = (1 / (1 + np.exp(-e)))[:, :, None]
        p = np.concatenate([p * (1 - p_out), p_out], axis=2)
    return p # samples x cells x states


for key, idata in idatas.items():
    spec = model_specs[key]
    if len(spec['states']) < 2:
        continue
    p = cell_state_probs(idata, spec)
    rows = []
    for k, cell in enumerate(cells_tab['cell']):
        for j, s in enumerate(spec['states']):
            q = np.percentile(p[:, k, j], [50, 2.5, 97.5])
            rows.append({'band': cells_tab['band'][k], 'condition': cells_tab['condition'][k], 'state': s,
                         'median': q[0], 'lower': q[1], 'upper': q[2]})
    st = pd.DataFrame(rows)
    print(f'\n===== {key} =====')
    print(st.pivot_table(index=['band', 'condition'], columns='state', values='median', observed=True)
            .reindex(columns=spec['states']).round(3))
"""),
md(r"""
# Directions and concentrations by condition and band
"""),
code(r"""
def circ_q(x):
    m = np.angle(np.exp(1j * x).mean())
    return np.round(np.rad2deg(np.percentile(np.angle(np.exp(1j * (x - m))), [50, 2.5, 97.5]) + m))


for key, idata in idatas.items():
    post = idata.posterior
    mu = post['mu_cell'].values.reshape(-1, len(cells_tab))
    print(f'\n===== {key} =====')
    print(pd.DataFrame([{'cell': c, **dict(zip(['mu_median', 'mu_lower', 'mu_upper'], circ_q(mu[:, k])))}
                        for k, c in enumerate(cells_tab['cell'])]).to_string(index=False))
"""),
md(r"""
# Model comparison (provisional)
PSIS-LOO over dances, if more than one model has been fitted; as for the single-elevation models, many dances are expected to have Pareto $\hat{k} > 0.7$, so this is provisional pending cross-validation by individual.
"""),
code(r"""
if len(idatas) > 1:
    loos = {k: az.loo(v) for k, v in idatas.items()}
    print(az.compare(loos, ic='loo').round(2))
    for k, l in loos.items():
        print(f'{k}: {(l.pareto_k > 0.7).sum().item()} of {l.n_data_points} dances with Pareto k > 0.7')
"""),
]

nb = {'cells': cells, 'metadata': kernel_meta, 'nbformat': 4, 'nbformat_minor': 5}
with open('HS_circGLMM_states_joint.ipynb', 'w', encoding='utf8', newline='\r\n') as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print('created HS_circGLMM_states_joint.ipynb')
