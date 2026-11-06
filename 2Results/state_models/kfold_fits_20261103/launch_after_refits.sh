#!/bin/zsh
# Wait for the refit run to finish, then run all k-fold fits (saving posteriors) under the watchdog (from the repo root).
cd /Users/jamesfoster/Documents/Github/HSdances
R=2Results/state_models/kfold_refit_20261103 F=2Results/state_models/kfold_fits_20261103
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> $F/launcher.log }
log "waiting for the refit run"
until [ -f $R/status.json ] && ! pgrep -f "kfold_watchdog.py $R" >/dev/null; do
    grep -q 'not started' $R/launcher.log 2>/dev/null && { log "refit run was not started; fits not started"; exit 1 }
    sleep 60
done
if ! grep -q '"state": "finished"' $R/status.json; then log "refit run did not finish successfully; fits not started"; exit 1; fi
log "refit run finished; starting fits"
exec caffeinate -i -s /opt/miniconda3/envs/HS-env/bin/python -u tools/watchdog/kfold_watchdog.py $F $F/frozen_HS_circGLMM_states_joint_kfold_fits.ipynb . --hours 72 "setup_notebook='$F/frozen_HS_circGLMM_states_joint.ipynb'" >> $F/watchdog_stdout.log 2>&1
