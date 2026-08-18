# Details ---------------------------------------------------------------
#       AUTHOR:	James Foster              DATE: 2026 06 03
#     MODIFIED:	James Foster              DATE: 2026 07 21
#
#  DESCRIPTION: Functions for Hildebrandt F et al. (in prep).
#
#       INPUTS:
#
#      OUTPUTS:
#
#	   CHANGES: -
#             -
#
#   REFERENCES:
#
#    EXAMPLES:
#
#
#TODO   ---------------------------------------------
#TODO

# Set up workspace --------------------------------------------------------

## Load packages ----------------------------------------------------------
#needs installing before first use (in Rstudio, see automatic message)
suppressMessages(
  #these are disturbing users unnecessarily
  {
    require(circular) #package for handling circular data
    require(CircMLE) #package for circular mixture models
    require(brms) #package for preparing Stan models
    require(ARTool) #package for non-parametric linear models
    require(emmeans) #package for post-hoc comparisons
    require(geosphere) #for bearing calculations
  }
)

## System parameters -----------------------------------------------------

#Check the operating system and assign a logical flag (T or F)
sys_win <- Sys.info()[['sysname']] == 'Windows'
#User profile instead of home directory
if (sys_win) {
  #get rid of all the backslashes
  ltp <- gsub('\\\\', '/', Sys.getenv('USERPROFILE')) #Why does windows have to make this so difficult
} else {
  #Root directory should be the "HOME" directory on a Mac (or Linux?)
  ltp <- Sys.getenv('HOME') #Life was easier on Mac
}

## Select file ---------------------------------------------------------
SelectCSV <- function(path = file.path(ltp, 'Documents', "*.csv")) {
  # set path to file
  if (sys_win) {
    #choose.files is only available on Windows
    message('\n\nPlease select the ".csv" file\n\n')
    Sys.sleep(0.5) #goes too fast for the user to see the message on some computers
    path_file <- choose.files(
      default = path, #For some reason this is not possible in the "root" user
      caption = 'Please select the ".csv" file'
    )
  } else {
    message('\n\nPlease select the ".csv" file\n\n')
    Sys.sleep(0.5) #goes too fast for the user to see the message on some computers
    path_file <- file.choose(new = F)
  }
  #show the user the path they have selected
  if (is.null(path_file) | !length(path_file)) {
    stop('No file selected.')
  } else {
    print(path_file)
  }
}

# General functions -----------------------------------------------------

#Open file with default program on any OS
# https://stackoverflow.com/a/35044209/3745353
shell.exec.OS <- function(x) {
  # Cross-platform replacement for shell.exec(): uses base::shell.exec on Windows,
  # or the system 'open' command on macOS/Linux.
  if (exists("shell.exec", where = "package:base")) {
    return(base::shell.exec(x))
  } else {
    comm <- paste0('open "', x, '"')
    return(system(comm))
  }
}

#convert angles to signed angles in (-180, 180)
Mod360.180 <- function(x) {
  #use atan2 to convert any angle to the range (-180,180)
  deg(
    atan2(y = sin(rad(x)), x = cos(rad(x)))
  )
}

#Wrap a radian angle to (-pi, pi) using atan2; radian equivalent of Mod360.180.
mod_circular <- function(x) {
  atan2(y = sin(x), x = cos(x))
}

#Shift circular angles so they are centred on the mean, removing the wrap-around discontinuity.
#Returns angles in radians; useful before computing statistics that assume linearity.
unwrap_circular <- function(x) {
  mux <- mean.circular(x = circular(x = x, template = 'none'))
  centx <- atan2(y = sin(x - mux), x = cos(x - mux))
  unwrx <- centx + mux
}

#Degree version of unwrap_circular; returns unwrapped angles in degrees.
unwrap_circular_deg <- function(x) {
  mux <- mean.circular(x = circular(x = x, template = 'none'))
  centx <- atan2(y = sin(x - mux), x = cos(x - mux))
  unwrx <- centx + mux
  return(deg(unwrx))
}

#invert the softplus link
#https://en.wikipedia.org/wiki/Softplus
#we are using this as our _inverse_ link function for kappa,
#maps almost 1:1 but keeps values >0 for low estimates
softplus <- function(x) {
  log(exp(x) + 1)
}
#this would return our kappa estimates back to the original scale
inv_softplus <- function(x) {
  log(exp(x) - 1)
}

#convert inv_softplus scaled kappa to mean vector estimate
Softpl_to_meanvec <- function(x) {
  circular::A1(
    softplus(x)
  )
}

#convert circular to normalised
NormCirc <- function(x, plusmean = TRUE) {
  mn <- mean.circular(x) * as.numeric(plusmean)
  return(mod_circular(x - mn) + mn)
}


## Plot spacing function -------------------------------------------------
#generates the same spacing as R's default barplot function
BarSpacer <- function(n, spa = 0.2, wdt = 1.0) {
  seq(from = spa + 1 - wdt / 2, to = n * (1 + spa) - wdt / 2, length.out = n)
}


# Data summary functions --------------------------------------------------

#Fit a von Mises distribution by maximum likelihood to a numeric angle vector.
#Returns a named vector with mu (mean direction) and kappa (concentration).
MLE_est <- function(x) {
  with(
    mle.vonmises(circular(x, template = 'none'), bias = TRUE),
    c(mu = as.numeric(mu), kappa = kappa)
  )
}

#Perform the Rayleigh test of uniformity on a numeric angle vector.
#Returns a named scalar p-value; small values indicate non-uniform (oriented) data.
Rayleigh_p <- function(x) {
  with(rayleigh.test(circular(x, template = 'none')), c(p = p.value))
}

#find unique combinations of two variables
IndCond <- function(id, dta, vars = c('cel_body', 'Shade.NoShade')) {
  # Subset rows by ID
  sub_dta <- subset(dta, ID %in% id)

  # Select only the target columns
  target_cols <- sub_dta[, vars, drop = FALSE]

  # Paste row-wise across all selected columns
  row_pastes <- do.call(paste, target_cols)

  # Return the number of unique combinations
  return(length(unique(row_pastes)))
}

#Angular difference between two conditions for one individual
MuDiff <- function(
  id, #individual name
  dta, #dataframe to use, contains columns "mu" (degrees) & "ID"
  vars = c('cel_body', 'Shade.NoShade'), #variables of interest
  val, #name of target condition
  ref_val = 'sun NoShade' # name of reference condition
) {
  # Create the combined target column dynamically
  target_cols <- dta[, vars, drop = FALSE]
  dta$stim <- do.call(paste, target_cols)

  # Paste the comparison values together so they match the 'stim' format
  # e.g., val = c('sun', 'NoShade') becomes "sun NoShade"
  target_stim <- paste(val, collapse = " ")
  ref_stim <- paste(ref_val, collapse = " ")

  # Subset by ID and calculate the circular difference
  with(
    subset(dta, ID %in% id),
    {
      # Extract mu for the target and reference conditions
      mu_target <- mu[stim == target_stim]
      mu_ref <- mu[stim == ref_stim]

      # Ensure both conditions exist to avoid empty vector errors
      if (length(mu_target) == 0 || length(mu_ref) == 0) {
        # warning("Target or reference values not found in the specified 'vars' columns.")
        return(NA)
      }

      # Calculate circular difference
      return(
        deg(
          mod_circular(
            rad(mu_target - mu_ref)
          )
        )
      )
    }
  )
}

#Concentration difference between two conditions for one individual
KappaDiff <- function(
  id, #individual name
  dta, #dataframe to use, contains columns "mu" (degrees) & "ID"
  vars = c('cel_body', 'Shade.NoShade'), #variables of interest
  val, #name of target condition
  ref_val = 'sun NoShade' # name of reference condition
) {
  # Create the combined target column dynamically
  target_cols <- dta[, vars, drop = FALSE]
  dta$stim <- do.call(paste, target_cols)

  # Paste the comparison values together so they match the 'stim' format
  # e.g., val = c('sun', 'NoShade') becomes "sun NoShade"
  target_stim <- paste(val, collapse = " ")
  ref_stim <- paste(ref_val, collapse = " ")

  # Subset by ID and calculate the circular difference
  with(
    subset(dta, ID %in% id),
    {
      # Extract mu for the target and reference conditions
      kappa_target <- kappa[stim == target_stim]
      kappa_ref <- kappa[stim == ref_stim]

      # Ensure both conditions exist to avoid empty vector errors
      if (length(kappa_target) == 0 || length(kappa_ref) == 0) {
        # warning("Target or reference values not found in the specified 'vars' columns.")
        return(NA)
      }

      # Calculate circular difference
      return(
        kappa_target - kappa_ref
      )
    }
  )
}

#Mean vector length difference between two conditions for one individual
RhoDiff <- function(
  id, #individual name
  dta, #dataframe to use, contains columns "mu" (degrees) & "ID"
  vars = c('cel_body', 'Shade.NoShade'), #variables of interest
  val, #name of target condition
  ref_val = 'sun NoShade' # name of reference condition
) {
  # Create the combined target column dynamically
  target_cols <- dta[, vars, drop = FALSE]
  dta$stim <- do.call(paste, target_cols)

  # Paste the comparison values together so they match the 'stim' format
  # e.g., val = c('sun', 'NoShade') becomes "sun NoShade"
  target_stim <- paste(val, collapse = " ")
  ref_stim <- paste(ref_val, collapse = " ")

  # Subset by ID and calculate the circular difference
  with(
    subset(dta, ID %in% id),
    {
      # Extract mu for the target and reference conditions
      rho_target <- mean_vector[stim == target_stim]
      rho_ref <- mean_vector[stim == ref_stim]

      # Ensure both conditions exist to avoid empty vector errors
      if (length(rho_target) == 0 || length(rho_ref) == 0) {
        # warning("Target or reference values not found in the specified 'vars' columns.")
        return(NA)
      }

      # Calculate circular difference
      return(
        rho_target - rho_ref
      )
    }
  )
}

#Boxplot and stripchart together
BoxStripLine <- function(
  x,
  bxcol = 'gray75',
  pars = list(boxwex = 0.5),
  stpch = 21,
  stbg = gray(0, alpha = 0.5),
  stcol = "white",
  stmethod = "jitter",
  stcex = 1.5,
  ablineh = 0,
  ...
) {
  #passed to boxplot
  boxplot(
    x,
    ...,
    pars = pars,
    outline = FALSE,
    col = bxcol,
    # Fill color for the boxes
    border = "black" # Line color for the boxes
  )
  stripchart(
    x,
    add = TRUE,
    vertical = TRUE,
    method = stmethod,
    pch = stpch,
    bg = stbg,
    col = stcol,
    cex = stcex
  )
  abline(h = ablineh, lty = 3)
}

PropDiff <- function(x, ref = 0, alternative = "two.sided", na.rm = TRUE) {
  switch(
    EXPR = alternative,
    #for two sided, indicate direction with sign
    two.sided = sign(mean(x > ref, na.rm = na.rm) - 0.5) *
      max(c(
        mean(x > ref, na.rm = na.rm),
        mean(x < ref, na.rm = na.rm)
      )),
    less = mean(x < ref, na.rm = na.rm),
    greater = mean(x > ref, na.rm = na.rm),
    #for two sided, indicate direction with sign
    sign(mean(x > ref, na.rm = na.rm) - 0.5) *
      max(c(
        mean(x > ref, na.rm = na.rm),
        mean(x < ref, na.rm = na.rm)
      ))
  )
}

# Astronomical information -----------------------------------------------------

#handling function (sunAngle is bad with NAs and vectors)
GetSaz <- function(tm, lon, lat) {
  if (!is.na(tm)) {
    oce::sunAngle(
      t = tm,
      longitude = lon,
      latitude = lat,
      useRefraction = TRUE
    )$azimuth
  } else {
    NA # return precisely an NA if is.na
  }
}
#handling function (sunAngle is bad with NAs and vectors)
GetSel <- function(tm, lon, lat) {
  if (!is.na(tm)) {
    oce::sunAngle(
      t = tm,
      longitude = lon,
      latitude = lat,
      useRefraction = TRUE
    )$altitude
  } else {
    NA # return precisely an NA if is.na
  }
}

#handling function (sunAngle is bad with NAs and vectors)
GetMaz <- function(tm, lon, lat) {
  if (!is.na(tm)) {
    oce::moonAngle(
      t = tm,
      longitude = lon,
      latitude = lat,
      useRefraction = TRUE
    )$azimuth
  } else {
    tm
  }
}
#handling function (sunAngle is bad with NAs and vectors)
GetMel <- function(tm, lon, lat) {
  if (!is.na(tm)) {
    oce::moonAngle(
      t = tm,
      longitude = lon,
      latitude = lat,
      useRefraction = TRUE
    )$altitude
  } else {
    tm
  }
}

#time of day from date
GetToD <- function(tm, tz = attr(tm, "tzone")) {
  t_form <- format(tm, "%H:%M:%S")
  t_string <- paste("1970-01-01", t_form)
  tod <- as.POSIXct(x = t_string, tz = tz, format = "%Y-%m-%d %H:%M:%OS")
  return(tod)
}


#convert an alphanumeric longitude or latitude
#in degrees, minutes, seconds, cardinal direction
#to its decimal equivalent
Alphnum2decimal <- function(alphnum) {
  # cardinal direction (character)
  card <- alphnum[4]
  # number components (reformat to numeric)
  numb <- sapply(X = alphnum[-4], FUN = as.numeric)
  dec <- switch(
    EXPR = card,
    N = 1,
    E = 1,
    S = -1,
    W = -1,
    North = 1,
    East = 1,
    South = -1,
    West = -1,
    1
  ) *
    (
      numb[1] + #degrees
        numb[2] / 60 + #minutes
        numb[3] / (60 * 60) #seconds
    )
  return(dec)
}

#convert an alphanumeric string of latitude and longitude
#to a pair of signed decimals of longitude and latitude
Dms2decimal <- function(alphanum) {
  # Accept the single alphanumeric coordinate pair string and clean it up
  # This splits by any sequence of degrees, minutes, seconds symbols, spaces, or commas
  # e.g., "25° 34' 18" S, 31° 10' 48" E" -> c("25", "34", "18", "S", "31", "10", "48", "E")
  raw_tokens <- unlist(strsplit(
    x = alphanum,
    split = "[°'\",\\s]+",
    perl = TRUE
  ))

  # Remove any accidental trailing empty strings from the split
  raw_tokens <- raw_tokens[raw_tokens != ""]

  # Group the flat tokens into two distinct sub-lists: [[lat_components], [lon_components]]
  # Each sub-list will match your exact expected layout: c(Deg, Min, Sec, Direction)
  lst <- list(
    latitude = switch(
      raw_tokens[4], # check 4th position is a N–S cardinal direction
      N = raw_tokens[1:4],
      S = raw_tokens[1:4],
      North = raw_tokens[1:4],
      South = raw_tokens[1:4],
      E = raw_tokens[5:8], #otherwise use 2nd component
      W = raw_tokens[5:8],
      East = raw_tokens[5:8],
      West = raw_tokens[5:8],
      raw_tokens[1:4]
    ),

    longitude = switch(
      raw_tokens[8], # check 8th position is an E–W cardinal direction
      E = raw_tokens[5:8],
      W = raw_tokens[5:8],
      East = raw_tokens[5:8],
      West = raw_tokens[5:8],
      N = raw_tokens[1:4], #otherwise use 1st component
      S = raw_tokens[1:4],
      North = raw_tokens[1:4],
      South = raw_tokens[1:4],
      raw_tokens[5:8]
    )
  )

  #Convert each component to decimal, collapse to vector
  dec <- sapply(X = lst, FUN = Alphnum2decimal)
  #inherit names from list version
  names(dec) <- names(lst)

  # for some reason geosphere takes bearings in the order: longitude, latitude
  dec <- dec[c('longitude', 'latitude')] # should inherit names, in for troubleshooting
  return(dec)
}


# Circular plotting functions ---------------------------------------------

#Plot a circular dataset as a stacked dot plot (clockwise, north-up convention).
#Optionally overlays a mean-vector arrow scaled by rho (circular concentration).
PCfun <- function(
  angles,
  col = 'darkblue',
  shrink = 1.5,
  title = '',
  plot_rho = TRUE,
  titleline = -2,
  side = 1
) {
  ca <- circular(x = as.numeric(angles), units = 'degrees', rotation = 'clock')
  plot.circular(
    x = ca,
    col = col,
    stack = TRUE,
    sep = 0.1,
    bins = 355 / 5,
    units = 'degrees',
    rotation = 'clock',
    zero = pi / 2,
    shrink = shrink
  )
  mtext(text = title, side = side, line = titleline)
  lines(x = c(0, 0), y = c(-1, 1), col = 'gray')
  if (plot_rho) {
    arrows.circular(
      x = mean.circular(ca, na.rm = TRUE),
      y = rho.circular(ca, na.rm = TRUE),
      zero = pi / 2,
      rotation = 'clock',
      col = col,
      length = 0.1
    )
  }
}


#Plot the best-fitting circular mixture model (from CircMLE) alongside the raw data.
#Runs circ_mle() internally and passes the result to plot_circMLE().
Plt_cmle <- function(dt, col = 'black', title = '') {
  cdt <- circular(x = dt, units = 'degrees', rotation = 'clock', zero = pi / 2)
  plot_circMLE(
    data = cdt,
    table = circ_mle(data = cdt),
    col = c(col, col, 'gray20', 'gray20')
  )
  text(x = 0, y = -1.5, labels = title)
}


#Open an empty circular plot (no data) with a dashed significance-threshold circle.
#The threshold radius is derived from the Rayleigh test at alpha = 0.05 for n_sample observations.
OpenCplot <- function(
  x,
  angle_unit = 'degrees',
  angle_rot = 'clock',
  n_sample = 10
) {
  plot.circular(
    x = circular(
      x = NULL,
      type = 'angles',
      unit = angle_unit,
      # template = 'geographics',
      modulo = '2pi',
      zero = pi / 2,
      rotation = angle_rot
    ),
    col = NA
  )
  lines.circular(
    x = circular(
      x = seq(from = -180, to = 180, length.out = 1e3),
      units = angle_unit,
      rotation = angle_rot
    ),
    y = rep(x = sqrt(-log(0.05) / n_sample) - 1, times = 1e3),
    col = 'black',
    lty = 2,
    lwd = 0.25,
    lend = 'butt'
  )
}

#Draw an arc from angle a1 (at radius r1) to angle a2 (at radius r2) on an existing circular plot.
#Marked endpoints use col1 (open circle) and col2 (filled circle); handles wrap-around correctly.
#20260625 fixed to deal with points at -180° 180° boundary.
DiffArc <- function(
  a1,
  a2,
  r1,
  r2,
  col1 = 'cyan4',
  col2 = 'darkblue',
  angle_rot = 'clock',
  ...
) {
  ma1 <- Mod360.180(a1)
  ma2 <- Mod360.180(a2)

  # Calculate the shortest angular distance (-180 to 180)
  ang_diff <- Mod360.180(ma2 - ma1)

  # Define a continuous target angle based on the shortest path
  ma2_shortest <- ma1 + ang_diff

  # Plot the single continuous shortest sequence
  lines.circular(
    x = circular(
      x = seq(from = ma1, to = ma2_shortest, length.out = 1e2),
      type = 'angles',
      unit = 'degrees',
      modulo = '2pi',
      zero = pi / 2,
      rotation = angle_rot
    ),
    y = seq(from = r1, to = r2, length.out = 1e2) - 1,
    ...
  )

  # Draw the starting and ending points
  points(
    x = c(sin(rad(a1)) * r1, sin(rad(a2)) * r2),
    y = c(cos(rad(a1)) * r1, cos(rad(a2)) * r2),
    col = adjustcolor(col = c(col1, col2), alpha.f = 0.5),
    pch = c(21, 19),
    lwd = 2
  )
}

#plot vectors for each ID
Plt_IDvectors <- function(dta, ids, vtype = 'kappa', angle_name = 'mu', ...) {
  for (i in ids) {
    with(subset(dta, ID %in% i), {
      if (length(mu) > 0) {
        switch(
          EXPR = vtype,
          rho = lines(
            x = c(0, sin(rad(get(angle_name))) * mean_vector),
            y = c(0, cos(rad(get(angle_name))) * mean_vector),
            ...
          ),
          kappa = lines(
            x = c(0, sin(rad(get(angle_name))) * A1(kappa)),
            y = c(0, cos(rad(get(angle_name))) * A1(kappa)),
            ...
          ),
          lines(
            x = c(0, sin(rad(get(angle_name))) * A1(kappa)),
            y = c(0, cos(rad(get(angle_name))) * A1(kappa)),
            ...
          )
        )
      }
    })
  }
}

Plt_mvec <- function(
  id,
  dta,
  vars = c('cel_body', 'Shade.NoShade'),
  ord = c(4, 1, 3, 2), #alphabetical to factor order, was previously c(1,3,2,4)
  ...
) {
  #passed to lines
  # Select only the target columns
  target_cols <- dta[, vars, drop = FALSE]

  # Paste row-wise across all selected columns
  row_pastes <- do.call(paste, target_cols)

  dta$stim <- row_pastes

  with(subset(x = dta, subset = ID %in% id), {
    lines(
      x = 1:length(unique(row_pastes)),
      y = mean_vector[match(
        x = unique(row_pastes)[ord], #TODO check why this order
        table = stim,
        nomatch = NA
      )],
      col = gray(0, 0.2),
      ...
    )
  })
}

Plt_kappa <- function(
  id,
  dta,
  vars = c('cel_body', 'Shade.NoShade'),
  ord = c(4, 1, 3, 2), #alphabetical to factor order, was previously c(1,3,2,4)
  ...
) {
  #passed to lines
  # Select only the target columns
  target_cols <- dta[, vars, drop = FALSE]

  # Paste row-wise across all selected columns
  row_pastes <- do.call(paste, target_cols)

  dta$stim <- row_pastes

  with(subset(x = dta, subset = ID %in% id), {
    lines(
      x = 1:length(unique(row_pastes)),
      y = kappa[match(
        x = unique(row_pastes)[ord], #TODO check why this order
        table = stim,
        nomatch = NA
      )],
      col = gray(0, 0.2),
      ...
    )
  })
}

Plt_iskappa <- function(
  id,
  dta,
  vars = c('cel_body', 'Shade.NoShade'),
  ord = c(4, 1, 3, 2), #alphabetical to factor order, was previously c(1,3,2,4)
  ...
) {
  #passed to lines
  # Select only the target columns
  target_cols <- dta[, vars, drop = FALSE]

  # Paste row-wise across all selected columns
  row_pastes <- do.call(paste, target_cols)

  dta$stim <- row_pastes

  with(subset(x = dta, subset = ID %in% id), {
    lines(
      x = 1:length(unique(row_pastes)),
      y = iskappa[match(
        x = unique(row_pastes)[ord], #TODO check why this order
        table = stim,
        nomatch = NA
      )],
      col = gray(0, 0.2),
      ...
    )
  })
}

Plt_mu <- function(id, dta, vars = c('cel_body', 'Shade.NoShade'), ...) {
  #passed to lines
  # Select only the target columns
  target_cols <- dta[, vars, drop = FALSE]

  # Paste row-wise across all selected columns
  row_pastes <- do.call(paste, target_cols)

  dta$stim <- row_pastes

  with(subset(x = dta, subset = ID %in% id), {
    lines(
      x = 1:length(unique(row_pastes)),
      y = mu[match(
        x = unique(row_pastes)[c(4, 1, 3, 2)], #c(1,2,4,3),#expected order, was previously c(1,3,2,4)
        table = stim,
        nomatch = NA
      )],
      col = gray(0, 0.2),
      ...
    )
  })
}


# Set up circular formats -----------------------------------------------
pipi0 <- list(
  units = 'radians',
  type = 'angles',
  modulo = '2pi',
  zero = 0,
  rotation = 'clock',
  template = 'none'
)

deg360 <- list(
  units = 'degrees',
  type = 'angles',
  modulo = '2pi',
  zero = 0,
  rotation = 'clock',
  template = 'none'
)


# Generate circular parameters based on input data ------------------------
#Generate parameters for each individual
ParGenerator <- function(i, params, nn = 1) {
  list(
    mu = switch(
      EXPR = params$mu, #mu can be "uniform" or "vonmises" (i.e. distribution)
      uniform = runif(
        n = nn,
        min = 0,
        max = 360 - .Machine$double.xmin
      ),
      vonmises = circular::rvonmises(
        n = nn,
        mu = as.circular(x = 0, control.circular = deg360),
        # kappa = (params$sd*pi/180)^-2),#kappa ~= 1/(sd)^2
        kappa = A1inv(exp(((params$sd * pi / 180)^2) / (-2))),
        control.circular = deg360
      ), #sd = sqrt(-2 * log(r))
      runif(
        n = nn,
        min = 0,
        max = 360 - .Machine$double.xmin
      )
    ),
    kappa = exp(
      rep(x = log(params$kappa), times = nn) +
        rnorm(n = nn, mean = 0, sd = params$sd_logkappa)
    )
  )
}

#generate full distribution from those parameters
RVMgenerator <- function(
  param,
  nn = 20,
  cc = list(
    units = 'degrees',
    type = 'angles',
    modulo = '2pi',
    zero = 0,
    rotation = 'clock',
    template = 'none'
  )
) {
  xx <- circular::rvonmises(
    n = nn,
    mu = circular::as.circular(x = param$mu, control.circular = cc),
    kappa = param$kappa
  ) -
    pi
  return(xx)
}


# Maximum Likelihood Modelling --------------------------------------------

#Extract the parameters of the top model fit by circ_mle
MD_extract <- function(md) {
  #extract the order of models in the circm_mle results
  md_order <- with(md, {
    rownames(results)
  })
  #extract the results for just the "best model" lowest AIC
  md_best <- with(md, {
    results[md_order %in% bestmodel, ]
  })
  #extract just the relevant parameters
  with(md_best, {
    return(
      list(
        mu1 = deg(q1), #convert to degrees
        kappa1 = k1,
        mu2 = deg(q2), #convert to degrees
        kappa2 = k2,
        weight1 = lamda,
        loglikelihood = -Likelihood #convert to true log likelihood
      )
    )
  })
}

#function to calculate the likelihood of a sample, given the input ML parameters
LLcalc <- function(ml, angles, au = 'degrees', ar = 'clock', ...) {
  with(ml, {
    sum(
      # add together
      dvonmises(
        x = circular(x = angles, units = au, rotation = ar), # probability density for each observed angle
        mu = mu, # ML estimated mean
        kappa = kappa, # ML estimated concentration
        log = TRUE
      ) # on a log scale (i.e. add instead of multiplying)
    )
  })
}

#generic mean angle simulator
MeanRvm <- function(
  n, #representative sample size
  mu = circular(0), #mean (defaults to 0rad)
  kappa, #kappa required
  au = 'degrees', #units
  ar = 'clock'
) {
  #rotation direction
  mean.circular(rvonmises(
    n = n,
    mu = circular(mu, units = au, rotation = ar),
    kappa = kappa,
    control.circular = list(units = au, rotation = ar)
  ))
}

#Simulate confidence intervals for a unimodal or bimodal distribution
#fitted to a vector of "angles"
CI_vM <- function(
  angles, #vector of angles fitted (used for sample size)
  m1, #primary mean
  k1, #primary concentration
  m2 = NA, #secondary mean (ignored if NULL or NA)
  k2 = NA, #secondary kappa
  w1 = 1, #weighting of primary mean
  n = 1e4, #number of simulations
  au = 'degrees',
  ar = 'clock',
  calc_q = TRUE,
  alternative = 'one.sided', #two.sided less conservative
  interval = 0.95, #confidence interval to calculate
  speedup_parallel = TRUE
) {
  if (speedup_parallel) {
    #3x faster
    cl <- parallel::makePSOCKcluster(parallel::detectCores() - 1)
    parallel::clusterExport(
      cl = cl,
      varlist = c('mean.circular', 'circular', 'rvonmises'),
      envir = .GlobalEnv
    )
    parallel::clusterExport(
      cl = cl,
      varlist = c(
        'MeanRvm',
        'angles',
        'm1',
        'k1',
        'm2',
        'k2',
        'w1',
        'n',
        'au',
        'ar'
      ),
      envir = environment()
    )
    #simulate primary mean
    m1_est <-
      parallel::parSapply(
        cl = cl,
        X = 1:n,
        FUN = function(i) {
          eval.parent(
            {
              MeanRvm(
                n = round(length(angles) * w1), #estimate number of observations at primary mean
                mu = m1,
                kappa = k1,
                au = au,
                ar = ar
              )
            }
          )
        },
        simplify = 'array' #return an array of simulated angles
      )
    if (!is.na(m2)) {
      #if there is a valid secondary mean
      m2_est <-
        parallel::parSapply(
          cl = cl,
          X = 1:n,
          FUN = function(i) {
            eval.parent(
              {
                MeanRvm(
                  n = round(length(angles) * (1 - w1)), #estimate number of observations at secondary mean
                  mu = m2,
                  kappa = k2,
                  au = au,
                  ar = ar
                )
              }
            )
          },
          simplify = 'array' #return an array of simulated angles
        )
    }
    parallel::stopCluster(cl)
  } else {
    #if not using parallel, use the slower version via replicate()
    m1_est <- replicate(
      n = n,
      MeanRvm(
        n = round(length(angles) * w1),
        mu = m1,
        kappa = k1,
        au = au,
        ar = ar
      )
    )
    if (!is.na(m2)) {
      m2_est <- replicate(
        n = n,
        MeanRvm(
          n = round(length(angles) * (1 - w1)),
          mu = m2,
          kappa = k2,
          au = au,
          ar = ar
        )
      )
    }
  }
  return(
    if (calc_q) {
      #calculate quantiles only if requested
      #either two-sided, symmetrical around mean change
      #or one-sided, from zero change towards mean change
      probs1 <- switch(
        alternative,
        two.sided = sort(c(c(0, 1) + c(1, -1) * (1 - interval) / 2, 0.5)),
        one.sided = sort(c(
          c(0, 1) +
            (if (Mod360.180(m1) > 0) {
              #N.B. quantile.circular counts anticlockwise
              c(1, 0)
            } else {
              c(0, -1)
            }) *
              (1 - interval),
          0.5
        )),
        sort(c(
          c(0, 1) + #default to one-sided
            (if (Mod360.180(m1) > 0) {
              c(1, 0)
            } else {
              c(0, -1)
            }) *
              (1 - interval),
          0.5
        ))
      )
      if (is.na(m2)) {
        Mod360.180(
          quantile.circular(
            x = circular(x = m1_est, units = au, rotation = ar),
            probs = probs1
          )
        )
      } else {
        probs2 <- switch(
          alternative,
          two.sided = sort(c(c(0, 1) + c(1, -1) * (1 - interval) / 2, 0.5)),
          one.sided = sort(c(
            c(0, 1) +
              (if (Mod360.180(m2) > 0) {
                c(1, 0)
              } else {
                c(0, -1)
              }) *
                (1 - interval),
            0.5
          )),
          sort(c(
            c(0, 1) + #default to one-sided
              (if (Mod360.180(m2) < 0) {
                c(1, 0)
              } else {
                c(0, -1)
              }) *
                (1 - interval),
            0.5
          ))
        )
        list(
          m1 = Mod360.180(
            quantile.circular(
              x = circular(x = m1_est, units = au, rotation = ar),
              probs = probs1
            )
          ),
          m2 = Mod360.180(
            quantile.circular(
              x = circular(x = m2_est, units = au, rotation = ar),
              probs = probs2
            )
          )
        )
      }
    } else {
      #if quantiles not requested, return the simulations (mainly for troubleshooting)
      if (is.na(m2)) {
        m1_est <-
          sapply(X = m1_est, FUN = Mod360.180)
      } else {
        list(
          m1_est = sapply(X = m1_est, FUN = Mod360.180),
          m2_est = sapply(X = m2_est, FUN = Mod360.180),
        )
      }
    }
  )
}

#Format CI simulation results from CI_vM into a data frame.
#Each row is one dataset; columns give lower/median/upper for m1 (and m2 if bimodal).
Table_vCI <- function(ci) {
  rowCI <- function(i) {
    ci_lab <- c('lower', 'median', 'upper')
    if (length(i) == 3) {
      #only m1 simulated, CI extremes and median returned
      nm1 <- as.numeric(i)
      m2 <- c(NA, NA, NA) #set m2 to NA
      names(nm1) <- ci_lab #give same labels as m1 for consistency
      names(m2) <- ci_lab #give same labels as m1 for consistency
      return(c(
        m1 = nm1, #remove circular formatting to collapse vector
        m2 = m2
      ))
    } else {
      #both m1 and m2 simulated
      return(
        with(i, {
          nm1 <- as.numeric(m1)
          nm2 <- as.numeric(m2)
          # names(nm1) = names(m1)
          # names(nm2) = names(m2)
          names(nm1) <- ci_lab
          names(nm2) <- ci_lab
          c(
            m1 = nm1, #remove circular formatting to collapse vector
            m2 = nm2
          ) #remove circular formatting to collapse vector
        })
      )
    }
  }
  lst <- lapply(X = ci, FUN = rowCI)
  do.call(what = rbind, lst)
}

#Draw a simulated confidence interval as an arc on an existing circular plot.
#ci_vec is a 3-element (unimodal) or 6-element (bimodal) vector of lower/median/upper angles in degrees.
PlotCI_vM <- function(ci_vec, col = 'salmon', lwd = 2, radius = 0.95, ...) {
  #passed to lines()
  angle_seq1 <-
    c(
      seq(
        from = ci_vec[1], #lower
        to = ci_vec[1] +
          Mod360.180(ci_vec[2] - ci_vec[1]), #median
        length.out = 1e2 / 2
      ),
      seq(
        from = ci_vec[2], #median
        to = ci_vec[2] +
          Mod360.180(ci_vec[3] - ci_vec[2]), #upper
        length.out = 1e2 / 2
      )
    )
  lines(
    x = radius * sin(rad(angle_seq1)),
    y = radius * cos(rad(angle_seq1)),
    col = col,
    lwd = 2,
    lend = 'butt',
    ...
  )
  if (!is.na(ci_vec[4])) {
    angle_seq2 <-
      c(
        seq(
          from = ci_vec[1 + 3],
          to = ci_vec[1 + 3] +
            Mod360.180(ci_vec[2 + 3] - ci_vec[1 + 3]),
          length.out = 1e2 / 2
        ),
        seq(
          from = ci_vec[2 + 3],
          to = ci_vec[2 + 3] +
            Mod360.180(ci_vec[3 + 3] - ci_vec[2 + 3]),
          length.out = 1e2 / 2
        )
      )
    lines(
      x = radius * sin(rad(angle_seq2)),
      y = radius * cos(rad(angle_seq2)),
      col = col,
      lwd = 2,
      lend = 'butt',
      ...
    )
  }
}

#Overlay mean-vector arrow(s) from circMLE model parameters onto an existing circular plot.
#Arrow length is scaled by A1(kappa) (mean resultant length); width scales with mixture weight.
PlotMV_circMLE <- function(
  mod_par,
  au = 'degrees',
  ar = 'clock',
  col1 = 'salmon',
  col2 = col1,
  ...
) {
  #passed to arrows.circular()
  m1 <- with(mod_par, {
    circular(mu1, unit = au, rotation = ar, modulo = '2pi', zero = pi / 2)
  })
  with(mod_par, {
    arrows.circular(
      x = m1,
      y = A1(kappa1),
      col = col1,
      lwd = 5 * weight1,
      length = 0.1,
      ...
    )
  })
  if (!is.na(mod_par$mu2)) {
    m2 <- with(mod_par, {
      circular(mu2, unit = au, rotation = ar, modulo = '2pi', zero = pi / 2)
    })
    with(mod_par, {
      arrows.circular(
        x = m2,
        y = A1(kappa2),
        col = col2,
        lwd = 5 * (1 - weight1),
        length = 0.1,
        ...
      )
    })
  }
}

#Assemble a model-comparison table (log-likelihood, deviance, rank, df) for LR testing.
#Set bimod = TRUE to include a bimodal ('pairs multi') model alongside same/diff models.
CollectDetails <- function(
  ll,
  dts = 'all',
  bimod = FALSE, # whether a bimodal version is also considered
  ...
) {
  if (!bimod) {
    with(subset(ll, dataset == dts), {
      data.frame(
        modnm = c('pairs same', 'pairs diff'),
        ll = c(
          loglikelihood[model %in% 'same'],
          loglikelihood[model %in% 'diff']
        ),
        deviance = c(
          -2 * loglikelihood[model %in% 'same'],
          -2 * loglikelihood[model %in% 'diff']
        ),
        rnk = rank(
          c(
            -2 * loglikelihood[model %in% 'same'],
            -2 * loglikelihood[model %in% 'diff']
          )
        ), #paired differences use fewer observations, don't include in ranking
        df = c(1, 2)
      )
    })
  } else {
    with(subset(ll, dataset == dts), {
      data.frame(
        modnm = c('pairs same', 'pairs diff', 'pairs multi'),
        ll = c(
          loglikelihood[model %in% 'same'],
          loglikelihood[model %in% 'diff'],
          loglikelihood[model %in% 'multi']
        ),
        deviance = c(
          -2 * loglikelihood[model %in% 'same'],
          -2 * loglikelihood[model %in% 'diff'],
          -2 * loglikelihood[model %in% 'multi']
        ),
        rnk = rank(
          c(
            -2 * loglikelihood[model %in% 'same'],
            -2 * loglikelihood[model %in% 'diff'],
            -2 * loglikelihood[model %in% 'multi']
          )
        ), #paired differences use fewer observations, don't include in ranking
        df = c(1, 2, 5)
      )
    })
  }
}


#Perform a single likelihood ratio test specified by tst on a CollectDetails() table.
#Supported tests: 'uniformity', 'trials_same_mean', 'pairs_diff_zero',
#'pairs_multi_zero', 'pairs_multi_diff'. Returns chi-squared, df, p, and BH-adjusted p.
LR_calc <- function(tst, mdt) {
  #collect the deviances for h0 and h1 and calculate the difference in degrees of freedom
  lr_res <-
    with(mdt, {
      switch(
        EXPR = tst,
        #test for uniformity (i.e. more comprehensive Rayleigh test)
        uniformity = data.frame(
          dev0 = deviance[modnm == 'uniform'], #null hypothesis: uniform distribution
          dev1 = deviance[rnk == 1], #lowest rank is most likely model, any non-uniform distribution
          d.f. = df[rnk == 1] #uniform has 0 degrees of freedom, test degrees of freedom are best model - 0
        ),
        #test for unpaired grand mean
        trials_same_mean = data.frame(
          dev0 = deviance[rnk == 2], #null hypothesis: trials don't differ
          dev1 = deviance[modnm == 'trial mean'], #within trial obs. share a mean, expect lower deviance with more params
          d.f. = df[modnm == 'trial mean'] -
            df[rnk == 2] #2nd best fitting model
        ),
        #test for nonzero differences of pairs
        pairs_diff_zero = data.frame(
          dev0 = deviance[modnm == 'pairs same'], #null hypothesis: trials don't differ
          dev1 = deviance[modnm == 'pairs diff'], #within trial obs. share a mean, expect lower deviance with more params
          d.f. = df[modnm == 'pairs diff'] -
            df[modnm == 'pairs same'] #grand mean has half the number of params
        ),
        #test for nonzero multimodal differences of pairs
        pairs_multi_zero = data.frame(
          dev0 = deviance[modnm == 'pairs same'], #null hypothesis: trials don't differ
          dev1 = deviance[modnm == 'pairs multi'], #within trial obs. share a mean, expect lower deviance with more params
          d.f. = df[modnm == 'pairs multi'] -
            df[modnm == 'pairs same'] #grand mean has half the number of params
        ),
        #test for multimodal rather than unimodal differences of pairs
        pairs_multi_diff = data.frame(
          dev0 = deviance[modnm == 'pairs diff'], #null hypothesis: trials don't differ
          dev1 = deviance[modnm == 'pairs multi'], #within trial obs. share a mean, expect lower deviance with more params
          d.f. = df[modnm == 'pairs multi'] -
            df[modnm == 'pairs diff'] #grand mean has half the number of params
        ),
      )
    })
  #calculate the change in deviance (chi-squared distributed)
  lr_res <- within(lr_res, {
    chi_squared <- abs(unlist(dev0) - unlist(dev1))
  })
  #calculate the p value
  lr_res <- within(lr_res, {
    p <- pchisq(q = unlist(chi_squared), df = unlist(d.f.), lower.tail = FALSE)
  })
  #adjust for multiple comparisons (3 in this case)
  lr_res <- within(lr_res, {
    p_adjusted <- p.adjust(p = p, method = 'BH', n = length(p)) #could this be flexible?
  })
  return(lr_res)
}
#Perform multiple likelihood ratio tests in one call; tst is a character vector of test names.
#Returns a matrix with columns dev0/dev1/d.f./chi_squared/p/p_adjusted, one row per test.
LR_calc_lst <- function(tst, mdt, digits = 6) {
  #collect the deviances for h0 and h1 and calculate the difference in degrees of freedom
  lr_res <-
    with(mdt, {
      data.frame(
        t(
          sapply(
            X = tst,
            FUN = switch,
            #test for uniformity (i.e. more comprehensive Rayleigh test)
            uniformity = data.frame(
              dev0 = deviance[modnm == 'uniform'], #null hypothesis: uniform distribution
              dev1 = deviance[rnk == 1], #lowest rank is most likely model, any non-uniform distribution
              d.f. = df[rnk == 1] #uniform has 0 degrees of freedom, test degrees of freedom are best model - 0
            ),
            #test for unpaired grand mean
            trials_same_mean = data.frame(
              dev0 = deviance[rnk == 2], #null hypothesis: trials don't differ
              dev1 = deviance[modnm == 'trial mean'], #within trial obs. share a mean, expect lower deviance with more params
              d.f. = df[modnm == 'trial mean'] -
                df[rnk == 2] #2nd best fitting model
            ),
            #test for nonzero differences of pairs
            pairs_diff_zero = data.frame(
              dev0 = deviance[modnm == 'pairs same'], #null hypothesis: trials don't differ
              dev1 = deviance[modnm == 'pairs diff'], #within trial obs. share a mean, expect lower deviance with more params
              d.f. = df[modnm == 'pairs diff'] -
                df[modnm == 'pairs same'] #grand mean has half the number of params
            ),
            #test for nonzero multimodal differences of pairs
            pairs_multi_zero = data.frame(
              dev0 = deviance[modnm == 'pairs same'], #null hypothesis: trials don't differ
              dev1 = deviance[modnm == 'pairs multi'], #within trial obs. share a mean, expect lower deviance with more params
              d.f. = df[modnm == 'pairs multi'] -
                df[modnm == 'pairs same'] #grand mean has half the number of params
            ),
            #test for multimodal rather than unimodal differences of pairs
            pairs_multi_diff = data.frame(
              dev0 = deviance[modnm == 'pairs diff'], #null hypothesis: trials don't differ
              dev1 = deviance[modnm == 'pairs multi'], #within trial obs. share a mean, expect lower deviance with more params
              d.f. = df[modnm == 'pairs multi'] -
                df[modnm == 'pairs diff'] #grand mean has half the number of params
            ),
          )
        )
      )
    })
  #calculate the change in deviance (chi-squared distributed)
  lr_res <- within(lr_res, expr = {
    chi_squared <- abs(unlist(dev0) - unlist(dev1))
  })

  #calculate the p value
  lr_res <- within(lr_res, {
    p <- pchisq(q = unlist(chi_squared), df = unlist(d.f.), lower.tail = FALSE)
  })
  #adjust for multiple comparisons (3 in this case)
  lr_res <- within(lr_res, {
    p_adjusted <- p.adjust(p = p, method = 'BH', n = length(p)) #could this be flexible?
  })
  lr_res <- apply(X = lr_res, MARGIN = 2, FUN = unlist) #for whatever reason entries are still in list format
  return(lr_res)
}

#Build a BIC-based model-comparison table analogous to CollectDetails().
#BIC is computed as deviance + df * log(n_obs); n_obs must be supplied explicitly.
BFDetails <- function(
  ll,
  dts = 'all',
  bimod = FALSE,
  n_obs = NULL, # Pass the sample size length(pair_diffs_lst[[dts]])
  ...
) {
  if (is.null(n_obs)) {
    stop(
      "You must provide the sample size 'n_obs' to calculate BIC/Bayes Factors."
    )
  }

  # Base models
  mod_names <- c('pairs same', 'pairs diff')
  models_filter <- c('same', 'diff')
  degrees_of_freedom <- c(1, 2)

  if (bimod) {
    mod_names <- c(mod_names, 'pairs multi')
    models_filter <- c(models_filter, 'multi')
    degrees_of_freedom <- c(degrees_of_freedom, 5)
  }

  with(subset(ll, dataset == dts), {
    sub_ll <- sapply(models_filter, function(m) loglikelihood[model %in% m])
    devs <- -2 * sub_ll

    data.frame(
      modnm = mod_names,
      ll = sub_ll,
      deviance = devs,
      rnk = rank(devs),
      df = degrees_of_freedom,
      n = rep(n_obs, length(mod_names)), # Store sample size
      bic = devs + degrees_of_freedom * log(n_obs) # Calculate BIC
    )
  })
}

#Return a human-readable interpretation string for a hypothesis test result.
#Considers both the direction of the deviance change and the adjusted p-value.
H1label <- function(tst, d0, d1, pa) {
  switch(
    EXPR = tst,
    uniformity = if (d0 > d1 & pa < 0.05) {
      # data may be oriented, not significantly oriented, or significantly disoriented
      '*data are significantly oriented'
    } else {
      'data _are not_ significantly oriented'
    },
    trials_same_mean = if (d0 > d1 & pa < 0.05) {
      # trials significantly differ in mean, not significantly differ in mean, or share a significant mean
      '*trial means differ significantly'
    } else {
      'trial means _do not_ differ significantly'
    },
    pairs_diff_zero = if (d0 > d1 & pa < 0.05) {
      # trials significantly differ in mean, not significantly differ in mean, or share a significant mean
      '*pairs differ significantly'
    } else {
      'pairs _do not_ differ significantly'
    },
    pairs_multi_zero = if (d0 > d1 & pa < 0.05) {
      # trials significantly differ with multiple means, not significantly differ in mean, or share a significant mean
      '*pairs differ significantly with multiple means'
    } else {
      'pairs _do not_ differ significantly with multiple means'
    },
    pairs_multi_diff = if (d0 > d1 & pa < 0.05) {
      # trials significantly differ with multiple means, not significantly differ in mean, or share a significant mean
      '*pairs differ significantly with no single mean'
    } else {
      'pairs _do not_ differ significantly with multiple means'
    }
  )
}


#Calculate approximate Bayes factors from BIC differences (Schwarz approximation).
#Returns delta_bic, BF10 (evidence for H1 over H0), and BF01 (evidence for H0 over H1).
BF_calc <- function(tst, mdt) {
  # 1. Isolate H0 and H1 based on the requested test
  hypotheses <- with(mdt, {
    switch(
      EXPR = tst,
      uniformity = data.frame(
        bic0 = bic[modnm == 'uniform'],
        bic1 = bic[rnk == 1]
      ),
      trials_same_mean = data.frame(
        bic0 = bic[rnk == 2],
        bic1 = bic[modnm == 'trial mean']
      ),
      pairs_diff_zero = data.frame(
        bic0 = bic[modnm == 'pairs same'],
        bic1 = bic[modnm == 'pairs diff']
      ),
      pairs_multi_zero = data.frame(
        bic0 = bic[modnm == 'pairs same'],
        bic1 = bic[modnm == 'pairs multi']
      ),
      pairs_multi_diff = data.frame(
        bic0 = bic[modnm == 'pairs diff'],
        bic1 = bic[modnm == 'pairs multi']
      )
    )
  })

  # 2. Calculate Bayes Factor using the BIC difference
  hypotheses <- within(hypotheses, {
    delta_bic <- bic0 - bic1 # Positive value means H1 fits better relative to its complexity
    Bayes_factor_1.0 <- exp(delta_bic / 2) # Evidence for H1 vs H0
    BF01 <- 1 / Bayes_factor_1.0 # Evidence for H0 vs H1
  })

  return(hypotheses)
}

#Return a text label describing strength of evidence given a Bayes factor value (BF10).
#Thresholds follow the Jeffreys/Kass-Raftery scale (Decisive ≥100, Strong ≥10, Moderate ≥3).
BFlabel <- function(tst, Bayes_factor_1.0) {
  evidence <- if (Bayes_factor_1.0 >= 100) {
    "Decisive evidence for H1"
  } else if (Bayes_factor_1.0 >= 10) {
    "*Strong evidence for H1"
  } else if (Bayes_factor_1.0 >= 3) {
    "*Moderate evidence for H1"
  } else if (Bayes_factor_1.0 > 1) {
    "Anecdotal evidence for H1"
  } else if (Bayes_factor_1.0 == 1) {
    "No evidence preference"
  } else if (Bayes_factor_1.0 > 1 / 3) {
    "Anecdotal evidence for H0"
  } else if (Bayes_factor_1.0 > 1 / 10) {
    "Moderate evidence for H0"
  } else if (Bayes_factor_1.0 > 1 / 100) {
    "Strong evidence for H0"
  } else {
    "Decisive evidence for H0"
  }
  return(evidence)
}

#Bootstrap a confidence interval for a single parameter of a bimodal CircMLE model.
#Resamples angles with replacement n times, refits the model, and returns quantiles.
#param selects which model parameter to summarise (default 5 = mixture weight lambda).
BootBimod <- function(
  angles, #vector of angles fitted (used for sample size)
  model = 'M5A',
  m1 = 'right',
  param = 5, #weighting of primary mean
  n = 1e4, #number of simulations
  niter = 1000,
  calc_q = TRUE,
  interval = 0.95, #confidence interval to calculate
  speedup_parallel = TRUE
) {
  rad_angles <- rad(as.numeric(angles))
  Estfun <- function(ang = rad_angles, model = model, niter = niter) {
    pr <- suppressWarnings(
      Exec(str2expression(
        paste0(
          model,
          '(data = circular(
                        
                          sample(ang,
                                 replace = TRUE,
                                 size = length(ang)
                        ),
                        template = "none",
                        zero = 0),
                        niter = ',
          niter,
          ')$par'
        )
      ))
    )
    #mod the estimates to (-pi, pi)
    pr[c(1, 3)] <- mod_circular(pr[c(1, 3)])
    #find the mode of interest
    pr[5] <- switch(
      EXPR = m1,
      right = if (pr[1] < pr[3]) {
        # rightmost
        1 - pr[5]
      } else {
        pr[5]
      },
      left = if (pr[1] > pr[3]) {
        # leftmost
        1 - pr[5]
      } else {
        pr[5]
      },
      max = max(pr[5], 1 - pr[5]), # largest proportion
      pr[5] # original
    )
    #return the parameter of interest
    return(pr[param])
  }
  if (speedup_parallel) {
    #3x faster
    cl <- parallel::makePSOCKcluster(parallel::detectCores() - 1)
    parallel::clusterExport(
      cl = cl,
      varlist = c(
        model,

        'circular',
        'mod_circular'
      ),
      envir = .GlobalEnv
    )
    parallel::clusterExport(
      cl = cl,
      varlist = c('rad_angles', 'model', 'niter'),
      envir = environment()
    )
    #bootstrap the parameter
    pr_est <-
      parallel::parSapply(
        cl = cl,
        X = 1:n,
        FUN = function(i) {
          eval.parent(
            {
              Estfun(ang = rad_angles, model = model, niter = niter)
            }
          )
        },
        simplify = 'array' #return an array of simulated values
      )
    parallel::stopCluster(cl)
  } else {
    pr_est <- sapply(
      X = 1:n,
      FUN = function(i) {
        eval.parent(
          {
            Estfun(ang = rad_angles, model = model, niter = niter)
          }
        )
      },
      simplify = TRUE #return an array of simulated values
    )
  }
  return(
    if (calc_q) {
      quantile(
        pr_est,
        probs = c(0, 0.5, 1) + c(1, 0, -1) * ((1 - interval) / 2)
      )
    } else {
      pr_est
    }
  )
}


# Extract model summaries -------------------------------------------------
ART_extract <- function(mod, co, adjust = 'sidak', method = 'pairwise', ...) {
  #passed to emmeans::contrast
  co1 <- sub(x = co, pattern = ':', replacement = '')
  fo <- as.formula(paste('~', co1))
  em <- emmeans(artlm.con(mod, co), specs = fo)
  ct <- contrast(em, method = method, adjust = adjust, ...)
  return(ct)
}

#calculate rhat for circular variables with extreme ranges
Rhat_unwrap <- function(x) {
  rhat(unwrap_circular(x))
}

#Compute Rhat convergence diagnostics for circular parameters in a brms model.
#Unwraps angles before computing Rhat to avoid inflated values from circular discontinuities.
#variable is a regex pattern matched against draw variable names (default: '^b_zmu').
UnwrapRhats <- function(
  uwmod,
  variable = '^b_zmu',
  regex = TRUE,
  digits = 5,
  ...
) {
  rh <-
    apply(
      X = as_draws_df(uwmod, variable = variable, regex = regex, ...),
      MARGIN = 2,
      FUN = Rhat_unwrap
    )
  nrh <- names(rh)
  rh <- rh[!(nrh %in% c(".chain", ".iteration", ".draw"))]
  return(round(rh, digits = digits))
}

#summarise all circular rhats
SumUwRhats <- function(x, var = NULL) {
  summary(UnwrapRhats(x, variable = var))
}

#calculate log likelihood for a brms von Mises model
log_lik_unwrap_von_mises <- function(i, prep) {
  #remove circular formatting?
  prep$data$Y <- as.numeric(prep$data$Y)

  args <- list(
    mu = get_dpar(prep, "mu", i),
    kappa = get_dpar(prep, "kappa", i = i)
  )

  # ----------- log_lik helper-functions -----------
  # compute (possibly censored) log_lik values
  # @param dist name of a distribution for which the functions
  #   d<dist> (pdf) and p<dist> (cdf) are available
  # @param args additional arguments passed to pdf and cdf
  # @param prep a brmsprep object
  # @return vector of log_lik values
  log_lik_censor <- function(dist, args, i, prep) {
    pdf <- get(paste0("d", dist), mode = "function")
    cdf <- get(paste0("p", dist), mode = "function")
    y <- prep$data$Y[i]
    cens <- prep$data$cens[i]
    if (is.null(cens) || cens == 0) {
      x <- do_call(pdf, c(y, args, log = TRUE))
    } else if (cens == 1) {
      x <- do_call(cdf, c(y, args, lower.tail = FALSE, log.p = TRUE))
    } else if (cens == -1) {
      x <- do_call(cdf, c(y, args, log.p = TRUE))
    } else if (cens == 2) {
      rcens <- prep$data$rcens[i]
      x <- log(do_call(cdf, c(rcens, args)) - do_call(cdf, c(y, args)))
    }
    x
  }

  # adjust log_lik in truncated models
  # @param x vector of log_lik values
  # @param cdf a cumulative distribution function
  # @param args arguments passed to cdf
  # @param i observation number
  # @param prep a brmsprep object
  # @return vector of log_lik values
  log_lik_truncate <- function(x, cdf, args, i, prep) {
    lb <- prep$data[["lb"]][i]
    ub <- prep$data[["ub"]][i]
    if (is.null(lb) && is.null(ub)) {
      return(x)
    }
    if (!is.null(lb)) {
      log_cdf_lb <- do_call(cdf, c(lb, args, log.p = TRUE))
    } else {
      log_cdf_lb <- rep(-Inf, length(x))
    }
    if (!is.null(ub)) {
      log_cdf_ub <- do_call(cdf, c(ub, args, log.p = TRUE))
    } else {
      log_cdf_ub <- rep(0, length(x))
    }
    x - log_diff_exp(log_cdf_ub, log_cdf_lb)
  }

  # weight log_lik values according to defined weights
  # @param x vector of log_lik values
  # @param i observation number
  # @param prep a brmsprep object
  # @return vector of log_lik values
  log_lik_weight <- function(x, i, prep) {
    weight <- prep$data$weights[i]
    if (!is.null(weight)) {
      x <- x * weight
    }
    x
  }

  out <- log_lik_censor(
    dist = "von_mises",
    args = args,
    i = i,
    prep = prep
  )
  out <- log_lik_truncate(
    out,
    cdf = pvon_mises,
    args = args,
    i = i,
    prep = prep
  )
  log_lik_weight(out, i = i, prep = prep)
}

#Add filled 2D density contours to an existing plot from posterior draw samples.
#x_string and y_string are evaluated expressions (typically Cartesian projections of mu/kappa).
#Optionally crops density outside the unit circle (cropc = TRUE).
Draws2Cont <- function(
  draws,
  palette = 'Heat 2',
  nlevels = 20,
  x_string = 'sin(Intercept)*A1(softplus(Intercept_kappa))',
  y_string = 'cos(Intercept)*A1(softplus(Intercept_kappa))',
  alpha = 200 / 255,
  ngrid = 25, # defaults to a 25x25 grid
  cropc = FALSE, #crop region outside circle
  denstype = 'relative' # 'normalised' or 'relative' (normalised fails to plot low densities)
) {
  kdc <- with(draws, {
    MASS::kde2d(
      x = eval(str2lang(x_string)),
      y = eval(str2lang(y_string)),
      n = ngrid
    )
  })
  if (cropc) {
    xy <- with(kdc, expand.grid(x = x, y = y)) #find coordinates of z variable
    idc <- with(xy, x^2 + y^2 < 1.0)
    #crop edge of circle
    kdc <- within(kdc, {
      z[!idc] <- 0
    })
  }
  with(kdc, {
    .filled.contour(
      x = x,
      y = y,
      z = z,
      levels = (1:nlevels) *
        switch(
          EXPR = denstype,
          relative = max(z / nlevels),
          normalised = sum(z / nlevels), #warning, fails to plot low densities
          max(z / nlevels)
        ),
      col = hcl.colors(
        n = nlevels,
        palette = palette,
        rev = TRUE,
        alpha = alpha
      )
    )
  })
}


#Plot a histogram with the density axis horizontal and the data axis vertical.
#Useful for placing a distribution alongside a scatter plot that shares the y-axis.
VertHist <- function(
  data, # numerical data vector
  breaks = 1e2,
  ylab = 'data',
  xlab = 'density',
  ylim = NULL,
  main = '',
  col = 'gray',
  border = NA,
  axes = TRUE,
  ...
) {
  hst <- hist(
    x = data, # calculate the histogram but don't plot it
    breaks = breaks, # user defined breaks
    plot = FALSE
  )
  with(hst, {
    plot(
      x = NULL, #open an empty plot
      xlim = c(0, max(density)),
      ylim = if (is.null(ylim)) {
        range(mids)
      } else {
        ylim
      },
      xlab = xlab,
      ylab = ylab,
      main = main,
      axes = axes
    )
    #plot each bar
    for (i in 1:length(mids)) {
      rect(
        xleft = 0,
        xright = density[i],
        ybottom = breaks[i],
        ytop = breaks[i + 1],
        col = col,
        border = border,
        ...
      )
    }
  })
}

#Compute Mardia's circular standard deviation from a von Mises concentration parameter kappa.
#Uses the analytical formula sqrt(-2 * log(A1(kappa))) by default;
#a simulation-based alternative is also available via method = 'simulation'.
MardiaSD <- function(k, method = 'analytical', n = 1e4) {
  switch(
    method,
    simulation = circular::sd.circular(
      circular::rvonmises(
        n = n,
        mu = circular::circular(
          pi, # this is directly in the middle of (0,2pi)
          template = 'none'
        ),
        kappa = k
      )
    ),
    analytical = sqrt(-2 * log(circular::A1(k))),
    sqrt(-2 * log(circular::A1(k)))
  )
}


# Azimuth-Elevation Error Modelling ---------------------------------------
#mu,phi parametrisation of beta distribution
#probability density
dBeta <- function(x, mu, phi, ...) {
  alpha <- mu * phi
  beta <- (1 - mu) * phi
  return(dbeta(x, alpha, beta, ...))
}

#mu,phi parametrisation of beta distribution
#quantile
qBeta <- function(p, mu, phi, ...) {
  alpha <- mu * phi
  beta <- (1 - mu) * phi
  return(qbeta(p, alpha, beta, ...))
}

#estimate minimum circular SD and concentration (across population)
AzElEst <- function(
  sigma_phi, #circular SD and concentration
  alpha, #observed circular CD
  elevation, #known elevation
  priors = c(
    #sd of a normal distribution
    sigma_sd = pi / 2, #expect low error when sun near horizon
    #mean of lognormal (exp)
    phi_mu = 1000, # expect high concentration (individuals similar)
    #sd of lognormal (identity)
    phi_sd = 1.0 # wide prior
  )
) {
  sigma <- abs(mod_circular(sigma_phi[1])) #a number between 0 and pi
  phi <- exp(sigma_phi[2]) #a number between 0 and Inf
  #expected circular SD for each elevation
  est_alpha <- 2 * atan(tan(sigma / 2) / cos(rad(elevation)))
  #priors on low sigma and high concentration
  pr_ll <- dnorm(x = sigma, mean = 0, sd = priors[1], log = TRUE) +
    dnorm(x = log(phi), mean = log(priors[2]), sd = priors[3], log = TRUE)
  #can per-elevation alpha be approximated by a beta distribution
  #centred on the estimate?

  #calculate log likelihood for all SD angles
  #divide angles by 180° to get [0,1]
  ll <- sum(mapply(
    FUN = dBeta,
    x = alpha / pi, #circular SD [0,1]
    mu = est_alpha / pi, #estimated SD [0,1]
    phi = phi, #concentration
    log = TRUE
  ))
  # negative log likelihood for data and priors
  return(-(ll + pr_ll))
}

#wrapper function for optimising and processing
AzElOptim <- function(
  alpha,
  elevation,
  start_par = rnorm(2),
  ... # passed to optim
) {
  sig_opt <- optim(
    par = rnorm(2), #starting values from normal(0,1)
    f = AzElEst,
    alpha = alpha,
    elevation = elevation,
    ...
  )
  sig_opt <- within(sig_opt, {
    #optimised parameters
    sigma <- abs(par[1])
    phi <- exp(par[2])
    #quantiles
    lower <- pi * qBeta(p = 0.025, mu = sigma / pi, phi = phi)
    upper <- pi * qBeta(p = 0.975, mu = sigma / pi, phi = phi)
    #variance of a beta distribution
    var <- pi * ((sigma / pi) * (1 - sigma / pi)) / (phi + 1)
    #perhaps the standard error?
    se <- pi * (sqrt(var / pi))
  })
}

PredEleError <-
  function(
    elevation, # elevation in radians
    sigma = NULL, # minimum error in radians (elevation == 0)
    kappa = NULL
  ) {
    # kappa on identity scale
    if (all(is.null(c(sigma, kappa)))) {
      stop("Either sigma or kappa needed to calculate azimuth error")
    }

    if (is.null(sigma) & !is.null(kappa)) {
      sigma <- MardiaSD(kappa)
    }

    alpha <- 2 *
      atan(
        tan(sigma / 2) /
          cos(elevation)
      )
    return(alpha)
  }

LinesEleError <- function(
  azel_list,
  elevation = rad(seq(from = 0, to = 89, by = 1)),
  col = 'black',
  ...
) {
  #additional arguments passed to lines()
  with(azel_list, {
    lines(
      x = xx,
      y = deg(PredEleError(
        elevation = elevation,
        sigma = sigma
      )),
      lty = 2,
      col = col,
      ...
    )
    lines(
      x = xx,
      y = deg(PredEleError(
        elevation = elevation,
        sigma = lower
      )),
      lty = 3,
      col = col,
      ...
    )
    lines(
      x = xx,
      y = deg(PredEleError(
        elevation = elevation,
        sigma = upper
      )),
      lty = 3,
      col = col,
      ...
    )
  })
}


BetaEst <- function(x, mu_phi) {
  return(
    -sum(mapply(
      FUN = dBeta,
      x = x,
      #circular SD [0,1]
      mu = plogis(mu_phi[1]),
      #estimated SD [0,1]
      phi = exp(mu_phi[2]),
      #concentration
      log = TRUE
    ))
  )
}

#LRtest for effect of elevation on azimuth error
AzElTest <- function(
  alpha,
  elevation,
  azel_list = AzElOptim(alpha, elevation),
  ...
) {
  #passed to optim
  est_alpha <- with(azel_list, {
    2 * atan(tan(sigma / 2) / cos(rad(elevation)))
  })

  ll_AzEl <- with(azel_list, {
    sum(mapply(
      FUN = dBeta,
      x = alpha / pi,
      #circular SD [0,1]
      mu = est_alpha / pi,
      #estimated SD [0,1]
      phi = phi,
      #concentration
      log = TRUE
    ))
  })

  null_opt <- optim(
    par = rnorm(2),
    #starting values from normal(0,1)
    f = BetaEst,
    x = alpha / pi,
    ...
  )

  res_table <- data.frame(
    mu = c(
      alt = deg(mean(est_alpha)),
      null = deg(plogis(null_opt$par[1])) *
        pi
    ),
    phi = c(alt = azel_list$phi, null = exp(null_opt$par[2])),
    log_likelihood = c(alt = ll_AzEl, null = -(null_opt$value)),
    bic = -2 *
      c(alt = ll_AzEl, null = -(null_opt$value)) +
      c(2, 2) * #N.B. both models really have the same d.f.
        log(length(alpha)),
    chi_sq = c(
      NA,
      abs(
        -2 *
          diff(c(
            alt = ll_AzEl,
            null = -(null_opt$value)
          ))
      )
    )[rank(c(ll_AzEl, -(null_opt$value)))], #order test by likelihood
    p_value = c(
      NA,
      pchisq(
        q = abs(
          -2 *
            diff(c(
              alt = ll_AzEl,
              null = -(null_opt$value)
            ))
        ),
        df = 1, #complexity penalty of one d.f. difference (although currently both same d.f.)
        lower.tail = FALSE
      )
    )[rank(c(ll_AzEl, -(null_opt$value)))], #order test by likelihood
    Bayes_factor = c(
      NA,
      1 /
        (exp(
          diff(
            -2 *
              c(alt = ll_AzEl, null = -(null_opt$value)) +
              c(2, 2) * #N.B. both models really have the same d.f.
                log(length(alpha))
          ) /
            2
        ))
    )[rev(rank(
      -2 *
        c(
          alt = ll_AzEl,
          null = -(null_opt$value)
        ) +
        c(2, 2) * log(length(alpha))
    ))] #order by BIC
  )
  row.names(res_table) <- c('alternative', 'null')
  return(res_table)
}
