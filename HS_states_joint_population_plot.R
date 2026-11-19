#  Population-level predictions of the joint dance-state model, plotted as in
#  GUVdances (Foster et al. 2026, J Comp Physiol A, Fig. 4a), and saved as an editable PDF.
#
#  Based on "Extract predictions" in HS_circGLMM_summary_states.Rmd (knitted with
#  elevation = "joint" as HS_circGLMM_summary_states_joint.html). For each cell
#  (elevation band x condition):
#   - dots:          the primary bimodal mean of each dance (PrimaryMu: bim_mu1 for
#                    bimodal dances, otherwise the unimodal mean), from 0Data/hs_mean_vectors.csv
#   - thin lines:    each dance's predicted modes (mu_d and mu_d + 180°, length A1(kappa_d),
#                    width scaled by the expected weight of each mode, w1)
#   - shaded areas:  posterior density of the population mean vector for each mode
#                    (forward mode red-yellow, reverse mode blue, opacity scaled by weight)
#   - thick arrows:  the population median modes for a typical individual
#
#  Also saves, as separate editable PDFs, the Rmd's "State probabilities by condition"
#  and "Effects of condition on states, direction and concentration" (contrast) plots.
#
#  Run from the project directory (e.g. open HSdances.Rproj). The model is read with
#  ncdf4, so (unlike the Rmd) no conda environment is needed.

# Settings ----------------------------------------------------------------
model_dir = '2Results/state_models'
direction_prior = 'bounded_vm_unrelated'
model = 'states_cond_x_band'
plot_stem = file.path('Plots', paste0('HS_states_joint_', direction_prior, '_', model))
plot_file = paste0(plot_stem, '_population_predictions.pdf')
states_file = paste0(plot_stem, '_state_probabilities.pdf')
contrasts_file = paste0(plot_stem, '_condition_contrasts.pdf')
#show predicted modes for dances most probably in an outlier state (unrelated/disoriented)?
#their runs are not expected around mu_d, but mu_d is still the model's prediction for them
show_outlier_dances = TRUE
shrink = 2 #space around the circle for stacked dots
dance_lwd = 1 #maximum line width of dance modes (scaled by weight)
dance_lwd_min = 0.25 #minimum line width, so that minor modes remain visible
pdf_width = 8 #inches
pdf_height = 6
panel_size = 2.5 #inches, width and height of each circular panel saved separately

# Packages ----------------------------------------------------------------
source('HS_functions.R')
require(arrow) #read .feather files
require(ncdf4) #read the model's posterior directly from the arviz netCDF file (no Python needed)

#condition colours, as in HS_ExploratoryPlots.Rmd
cond_cols = list(
  vertical = 'darkred',
  horizontal = 'darkgreen',
  antisolar = 'cyan4',
  solar = 'blue4'
)
#successive model effects for each condition
cond_effs_all = list(vertical = character(0),
                     horizontal = c('H'),
                     antisolar = c('H', 'A'),
                     solar = c('H', 'A', 'S'))
bands_all = c('low', 'medium', 'high')
#state colours
state_cols = list(
  forward = '#1b7837',
  reversed = '#762a83',
  axial = '#e08214',
  unrelated = '#808080',
  disoriented = '#808080'
)

# Data --------------------------------------------------------------------
hd = as.data.frame(read_feather(file.path('0Data', 'hs_dance_states_joint.feather')))
hd = within(hd,
            {
              condition = droplevels(factor(as.character(condition), levels = names(cond_cols)))
              band = factor(as.character(band), levels = bands_all)
            })
conds = levels(hd$condition)
#groups: cells (band_condition) in band then condition order, as in the model
gt = unique(hd[, c('band', 'condition')])
gt = gt[order(gt$band, gt$condition), ]
group_tab = data.frame(group = paste(gt$band, gt$condition, sep = '_'),
                       band = as.character(gt$band), condition = as.character(gt$condition))
groups = group_tab$group
rownames(group_tab) = groups
hd$group = factor(paste(hd$band, hd$condition, sep = '_'), levels = groups)
GroupCol = function(g){cond_cols[[group_tab[g, 'condition']]]}
#one row per dance (0-based codes, as in the model)
dance_tab = unique(hd[, c('Dance', 'Dance_label', 'Individual', 'condition', 'group')])
dance_tab = dance_tab[order(dance_tab$Dance), ]

#primary bimodal mean of each dance (degrees)
mv = read.csv(file.path('0Data', 'hs_mean_vectors.csv'))
mv_d = mv[match(dance_tab$Dance_label, mv$ID_d), ]
stopifnot(!anyNA(mv_d$ID_d))
dance_tab$primary_mu = with(mv_d, PrimaryMu(mu, bim_mu1, bim_mu2))

# Model -------------------------------------------------------------------
model_file = file.path(model_dir,
                       paste0('HS_states_joint_', direction_prior, '_', model, '.nc'))
print(model_file)
nc = nc_open(model_file)
#variables in the posterior group (arviz stores each group as a netCDF4 group)
post_vars = sub('^posterior/', '', grep('^posterior/', names(nc$var), value = TRUE))
#ncdf4 returns dimensions in reverse order: (levels..., draw, chain)
n_draws = length(ncvar_get(nc, 'posterior/draw'))
n_chains = length(ncvar_get(nc, 'posterior/chain'))
n_samp = n_chains * n_draws

#Extract a parameter as samples (x levels (x levels)), draws within chains
Draws = function(name)
{
  vals = ncvar_get(nc, paste0('posterior/', name), collapse_degen = FALSE)
  d = dim(vals)
  if(length(d) == 2) # draw x chain
  {
    as.vector(vals)
  }else if(length(d) == 3) # level x draw x chain
  {
    matrix(aperm(vals, c(2, 3, 1)), ncol = d[1])
  }else # level2 x level1 x draw x chain
  {
    array(aperm(vals, c(3, 4, 2, 1)), dim = c(n_samp, d[2], d[1]))
  }
}
Coords = function(dim_name)
{
  as.character(ncvar_get(nc, paste0('posterior/', dim_name)))
}
model_states = Coords('state')
outlier_states = intersect(model_states, c('unrelated', 'disoriented'))
main_states = setdiff(model_states, outlier_states)
stopifnot(all(Coords('dance') == dance_tab$Dance_label),
          all(Coords('cell') == groups))
kappa_terms = if('kappa_effect' %in% post_vars){Coords('kappa_effect_term')}else{character(0)}
state_terms = if(any(paste0(model_states, '_effect') %in% post_vars)){Coords('state_effect_term')}else{character(0)}

#population-level parameters
mu_grp = Draws('mu_cell') # samples x groups
colnames(mu_grp) = groups
kappa0 = Draws('kappa_Intercept')
kappa_eff = if(length(kappa_terms)){Draws('kappa_effect')}else{NULL}
state_int = lapply(setNames(model_states[-1], model_states[-1]),
                   function(s){Draws(paste0(s, '_Intercept'))})
state_eff = lapply(setNames(main_states[-1], main_states[-1]),
                   function(s)
                   {
                     nm = paste0(s, '_effect')
                     if(nm %in% post_vars){m = Draws(nm); colnames(m) = state_terms; m}else{NULL}
                   })
out_eff = if(length(outlier_states) && paste0(outlier_states, '_effect') %in% post_vars)
{
  m = Draws(paste0(outlier_states, '_effect')); colnames(m) = state_terms; m
}else{NULL}
w_axial = if('w_axial' %in% post_vars){Draws('w_axial')}else{rep(0.5, n_samp)}
#dance-level parameters
mu_dance = Draws('mu') # samples x dances: forward direction of each dance
kappa_dance = Draws('kappa') # samples x dances
p_state = Draws('p_state') # samples x dances x states: posterior state probabilities
nc_close(nc)

# Prediction functions (as in HS_circGLMM_summary_states.Rmd) --------------
#circular median in (-pi, pi)
MedCDraws = function(x, templ = 'none', medn = 1)
{
  median.circular(circular(x = mod_circular(x), template = templ))[medn]
}
#Design row for a group: products of condition effects (H, A, S) and band indicators (low, high)
XGroup = function(g, terms)
{
  cnd = group_tab[g, 'condition']
  bnd = group_tab[g, 'band']
  sapply(terms,
         function(term)
         {
           prod(sapply(strsplit(term, ':')[[1]],
                       function(q)
                       {
                         if(q %in% c('H', 'A', 'S')){q %in% cond_effs_all[[cnd]]}else{bnd == q}
                       }))
         })
}
#Population concentration for a group
KappaGroup = function(g)
{
  eta = kappa0
  if(length(kappa_terms)){eta = eta + kappa_eff %*% XGroup(g, kappa_terms)}
  softplus(as.vector(eta))
}
#Population state probabilities for a group (typical individual): samples x states
StateProbs = function(g)
{
  eta = sapply(main_states,
               function(s)
               {
                 if(s == 'forward'){return(rep(0, n_samp))}
                 e = state_int[[s]]
                 if(!is.null(state_eff[[s]])){e = e + as.vector(state_eff[[s]] %*% XGroup(g, state_terms))}
                 e
               })
  eta = matrix(eta, nrow = n_samp)
  colnames(eta) = main_states
  ps = exp(eta - apply(eta, 1, max)) / rowSums(exp(eta - apply(eta, 1, max)))
  if(length(outlier_states))
  {
    e_out = state_int[[outlier_states]]
    if(!is.null(out_eff)){e_out = e_out + as.vector(out_eff %*% XGroup(g, state_terms))}
    p_out = plogis(e_out)
    ps = cbind(ps * (1 - p_out), p_out)
  }
  colnames(ps) = model_states
  ps
}
#Expected weight of the forward direction (mu1), among dances oriented to the feeder axis
WForward = function(ps, w = w_axial)
{
  w1 = ps[, 'forward'] + if('axial' %in% colnames(ps)){w * ps[, 'axial']}else{0}
  if(length(outlier_states)){w1 = w1 / (1 - ps[, outlier_states])}
  w1
}

# Predictions -------------------------------------------------------------
pop_draws = lapply(setNames(groups, groups),
                   function(g)
                   {
                     ps = StateProbs(g)
                     list(mu = mu_grp[, g], kappa = KappaGroup(g), p = ps, w1 = WForward(ps))
                   })
#dances: median forward direction and concentration, posterior expected weight towards mu1
p_state_mean = matrix(apply(p_state, c(2, 3), mean), nrow = nrow(dance_tab),
                      dimnames = list(NULL, model_states))
dance_tab = within(dance_tab,
                   {
                     mu_med = apply(mu_dance, 2, function(x){as.numeric(MedCDraws(x))})
                     kappa_med = apply(kappa_dance, 2, median)
                     w1 = WForward(p_state_mean, w = median(w_axial))
                     state_map = model_states[apply(p_state_mean, 1, which.max)]
                   })

# Plotting functions (as in HS_circGLMM_summary_states.Rmd) ----------------
#Mean vectors for each mode (mu1 and mu1 + pi), length A1(kappa), width and opacity scaled by weight
ModeArrows = function(mu, kappa, w1, col, lwd = 5, lwd_min = 0.5, length = 0, alpha = 200/255)
{
  mus = c(mu, mod_circular(mu + pi))
  ws = c(w1, 1 - w1)
  for(md in 1:2)
  {
    arrows.circular(
      x = circular(mus[md], units = 'radians', rotation = 'clock', zero = pi/2),
      y = A1(kappa),
      lwd = max(lwd_min, lwd * ws[md]),
      length = length,
      col = adjustcolor(col, alpha.f = alpha * max(0.25, ws[md]))
    )
  }
}
#Posterior density of the mean vector of each mode, opacity scaled by its weight
ModeContours = function(mu, kappa, w1)
{
  dl = list(mu = mu, kappa = kappa)
  w_med = median(w1)
  Draws2Cont(dl, alpha = 200/255 * w_med,
             x_string = 'sin(mu) * A1(kappa)', y_string = 'cos(mu) * A1(kappa)')
  Draws2Cont(dl, palette = 'Blues 3', alpha = 200/255 * (1 - w_med),
             x_string = '-sin(mu) * A1(kappa)', y_string = '-cos(mu) * A1(kappa)')
}
PopPanel = function(g)
{
  dg = subset(dance_tab, group == g)
  PCfun(angles = dg$primary_mu, col = GroupCol(g),
        shrink = shrink, plot_rho = FALSE, title = gsub('_', ', ', g))
  for(k in seq_len(nrow(dg)))
  {
    if(!show_outlier_dances && dg$state_map[k] %in% outlier_states){next}
    ModeArrows(dg$mu_med[k], dg$kappa_med[k], dg$w1[k],
               col = GroupCol(g), lwd = dance_lwd, lwd_min = dance_lwd_min)
  }
  pd = pop_draws[[g]]
  ModeContours(pd$mu, pd$kappa, pd$w1)
  ModeArrows(MedCDraws(pd$mu), median(pd$kappa), median(pd$w1),
             col = adjustcolor(GroupCol(g), offset = c(rep(-0.2, 3), 0)),
             lwd = 5, length = 0.1/1.25)
}
#one row per band, one column per condition (empty where a band has no dances in that condition)
panel_layout = matrix(sapply(bands_all,
                             function(b)
                             {
                               ifelse(paste(b, conds, sep = '_') %in% groups,
                                      paste(b, conds, sep = '_'), NA)
                             }),
                      nrow = length(bands_all), byrow = TRUE)
PlotAll = function()
{
  par(pty = 's', mfrow = dim(panel_layout), mar = c(0, 0, 0, 0))
  for(g in t(panel_layout)){if(is.na(g)){plot.new()}else{PopPanel(g)}}
}

# State probabilities by condition (as in the Rmd) --------------------------
#Bars: population-level probability of each state in each group (median and 95% interval,
#typical individual); points: mean posterior state probability across observed dances
PlotStates = function()
{
  par(mar = c(8, 4, 3, 1))
  plot(NULL, xlim = c(0.5, length(groups) + 0.5), ylim = c(0, 1), xaxt = 'n',
       xlab = '', ylab = 'Probability of state',
       main = 'Dance states, joint solar elevation')
  axis(1, at = seq_along(groups), labels = groups, las = 2)
  abline(v = cumsum(table(group_tab$band)[bands_all])[-length(bands_all)] + 0.5, col = 'gray')
  width = 0.75 / length(model_states)
  for(j in seq_along(model_states))
  {
    s = model_states[j]
    xs = seq_along(groups) + (j - (length(model_states) + 1) / 2) * width
    q = sapply(groups, function(g){quantile(pop_draws[[g]]$p[, s], c(0.5, 0.025, 0.975))})
    rect(xs - width / 2, 0, xs + width / 2, q[1, ], col = adjustcolor(state_cols[[s]], alpha.f = 0.8), border = NA)
    arrows(xs, q[2, ], xs, q[3, ], angle = 90, code = 3, length = 0.03)
    obs = sapply(groups, function(g){mean(p_state_mean[dance_tab$group == g, s])})
    points(xs, obs, pch = 21, bg = 'white')
  }
  legend('topright', legend = model_states, fill = unlist(state_cols[model_states]), border = NA, bty = 'n')
}

# Effects of condition on states (as in the Rmd) ----------------------------
#Contrasts between successive conditions within each band, and each condition vs vertical,
#then each band vs medium for each condition
CondContrasts = function(gs)
{
  if(length(gs) < 2){return(list())}
  c(lapply(gs[-1], function(g){c(g, gs[1])}),
    if(length(gs) > 2){lapply(3:length(gs), function(k){c(gs[k], gs[k - 1])})})
}
contrasts = c(do.call(c, lapply(bands_all, function(b){CondContrasts(groups[group_tab$band == b])})),
              do.call(c, lapply(conds,
                                function(cnd)
                                {
                                  lapply(intersect(paste(c('low', 'high'), cnd, sep = '_'), groups),
                                         function(g){c(g, paste('medium', cnd, sep = '_'))})
                                })))
names(contrasts) = sapply(contrasts, paste, collapse = ' - ')
#Posterior of the change in each state's probability, as violins coloured by the first group's condition
ContrastPanel = function(s)
{
  dl = lapply(contrasts, function(cn){pop_draws[[cn[1]]]$p[, s] - pop_draws[[cn[2]]]$p[, s]})
  plot(NULL, xlim = c(0.5, length(dl) + 0.5), ylim = c(-1, 1), xaxt = 'n',
       xlab = '', ylab = paste('Change in P(', s, ')'), main = s)
  axis(1, at = seq_along(dl), labels = names(dl), cex.axis = 0.7, las = 2)
  abline(h = 0, lty = 2)
  for(k in seq_along(dl))
  {
    hst = hist(dl[[k]], breaks = seq(-1, 1, by = 0.02), plot = FALSE)
    dens = 0.4 * hst$density / max(hst$density)
    rect(k - dens, head(hst$breaks, -1), k + dens, hst$breaks[-1],
         col = adjustcolor(GroupCol(contrasts[[k]][1]), alpha.f = 0.5), border = NA)
  }
}
PlotContrasts = function()
{
  par(mfrow = c(length(model_states), 1), mar = c(9, 4, 2, 1))
  for(s in model_states){ContrastPanel(s)}
}

# Plot and save -----------------------------------------------------------
#base pdf device: text stays as text and contours/lines as vector paths, editable in e.g. Illustrator/Inkscape
SavePDF = function(plot_fun, file, width, height)
{
  plot_fun() #preview
  pdf(file = file, width = width, height = height,
      useDingbats = FALSE, family = 'Helvetica', title = basename(file))
  plot_fun()
  dev.off()
  print(paste('Saved', file))
}
dir.create('Plots', showWarnings = FALSE)
SavePDF(PlotAll, plot_file, width = pdf_width, height = pdf_height)
SavePDF(PlotStates, states_file, width = 12, height = 5) #sizes as in the Rmd
SavePDF(PlotContrasts, contrasts_file, width = 12, height = 12)

#each panel also as its own PDF (in a subfolder), so that panels can be sized and arranged separately
panel_dir = paste0(plot_stem, '_panels')
dir.create(panel_dir, showWarnings = FALSE)
for(g in groups)
{
  SavePDF(function(){par(pty = 's', mar = c(0, 0, 0, 0)); PopPanel(g)},
          file.path(panel_dir, paste0('population_predictions_', g, '.pdf')),
          width = panel_size, height = panel_size)
}
for(s in model_states)
{
  SavePDF(function(){par(mar = c(9, 4, 2, 1)); ContrastPanel(s)},
          file.path(panel_dir, paste0('condition_contrasts_', s, '.pdf')),
          width = 12, height = 3)
}
for(g in groups)
{
  print(paste(g, '- dances:', sum(dance_tab$group == g),
              '; outlier dances:', sum(dance_tab$group == g & dance_tab$state_map %in% outlier_states)))
}
