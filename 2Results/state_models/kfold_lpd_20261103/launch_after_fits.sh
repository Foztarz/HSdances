#!/bin/zsh
# Wait for the k-fold fits run to finish, then calculate the predictive densities under the watchdog (from the repo root).
cd /Users/jamesfoster/Documents/Github/HSdances
F=2Results/state_models/kfold_fits_20261103 L=2Results/state_models/kfold_lpd_20261103
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> $L/launcher.log }
log "waiting for the fits run"
until [ -f $F/status.json ] && ! pgrep -f "kfold_watchdog.py $F" >/dev/null; do
    grep -q 'not started' $F/launcher.log 2>/dev/null && { log "fits run was not started; lpd not started"; exit 1 }
    sleep 60
done
if ! grep -q '"state": "finished"' $F/status.json; then log "fits run did not finish successfully; lpd not started"; exit 1; fi
log "fits run finished; starting lpd"
exec caffeinate -i -s /opt/miniconda3/envs/HS-env/bin/python -u tools/watchdog/kfold_watchdog.py $L $L/frozen_HS_circGLMM_states_joint_kfold_lpd.ipynb . --hours 48 "setup_notebook='$L/frozen_HS_circGLMM_states_joint.ipynb'" >> $L/watchdog_stdout.log 2>&1
