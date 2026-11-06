#!/bin/zsh
# Wait for the main k-fold run to finish, then run the refits under the watchdog (from the repo root).
cd /Users/jamesfoster/Documents/Github/HSdances
M=2Results/state_models/kfold_run_20261027 R=2Results/state_models/kfold_refit_20261103
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> $R/launcher.log }
log "waiting for the main run"
while pgrep -f "kfold_watchdog.py $M" >/dev/null; do sleep 60; done
if ! grep -q '"state": "finished"' $M/status.json; then log "main run did not finish successfully; refits not started"; exit 1; fi
log "main run finished; starting refits"
exec caffeinate -i -s /opt/miniconda3/envs/HS-env/bin/python -u tools/watchdog/kfold_watchdog.py $R $R/HS_circGLMM_states_joint_kfold_refit.ipynb . --hours 48 >> $R/watchdog_stdout.log 2>&1
