"""Run a notebook unattended with tools/run_nb.py, pausing or stopping it if the Mac is under strain.

Usage (HS-env python): python kfold_watchdog.py RUN_DIR NOTEBOOK WORKDIR [--hours H] [name=value ...]

Every CHECK_S seconds the watchdog reads memory, CPU, temperature, disk and power:
- a thermostat freezes the run for a while (SIGSTOP) whenever the hottest CPU sensor is above THERMO_HIGH (set just
  below the 100 °C limit, at which the CPU throttles itself), and lets it continue below THERMO_LOW
- warning signs pause the run (SIGSTOP to run_nb.py and its kernel); it resumes after CLEAR_S seconds without warnings
- critical signs, or a pause longer than MAX_PAUSE_S, stop the run (the k-fold notebook saves after each model,
  so a restart resumes from the last saved model); the run is restarted when conditions are normal again,
  at most MAX_RESTARTS times
- the run is stopped for good at the deadline (--hours), after too many restarts, or when RUN_DIR/STOP exists
- RUN_DIR/PAUSE pauses the run by hand until it is removed
Output in RUN_DIR: watchdog.log (events), metrics.csv (every 5 min), status.json (latest state), run_nb.log,
executed.ipynb (the notebook with outputs).
"""
import csv, json, os, re, signal, subprocess, sys, time
from collections import deque
from datetime import datetime, timedelta

import psutil

CHECK_S = 30
THERMO_S = 5 # temperature checks of the thermostat
THERMO_HIGH, THERMO_LOW = 99.0, 92.0 # °C, mean of the last two readings / single reading (a backstop: the CPU throttles itself at 100 °C)
MAX_COOLING_S = 20 * 60 # still hot after this long frozen: stop the run
CLEAR_S = 10 * 60 # warning-free time before resuming
MAX_PAUSE_S = 2.5 * 3600 # run_nb.py gives up on a cell after 6 h, pauses included
RESTART_WAIT_S = 15 * 60 # normal conditions needed before a restart
MAX_RESTARTS = 4
METRICS_EVERY_S = 5 * 60
GB = 1024 ** 3

# thresholds: (pause, stop)
CPU_TEMP = (99.5, 103.0) # °C, 3-min mean of the hottest CPU sensor (i9-9980HK: Tjmax 100 °C)
CPU_TEMP_RESUME = 88.0 # °C, needed before resuming after a temperature pause
BATT_TEMP = (45.0, 50.0) # °C
SPEED_LIMIT = 50 # % (pmset): pause if the OS throttles the CPU below this on average over 5 min
# (macOS throttles briefly under any sustained load; that is its own protection, not a warning sign)
MEM_AVAIL = (3.0, 1.5) # GB available
SWAP_USED = (8.0, 14.0) # GB
OWN_RSS_FRAC = 0.6 # stop (rather than pause) on memory pressure if the run itself holds this share of RAM
LOAD_PER_CPU = 2.5 # 5-min load per logical CPU: a runaway, pause
DISK_FREE = (5.0, 2.5) # GB
BATTERY_STOP = 30 # % when running on battery (on battery at all: pause)

HERE = os.path.dirname(os.path.abspath(__file__))
SMC = os.path.join(HERE, 'smc_temp')
RUN_NB = os.path.join(os.path.dirname(HERE), 'run_nb.py')


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ''


def read_metrics(root_proc):
    m = {'time': datetime.now().isoformat(timespec='seconds')}
    temps = {}
    for line in sh([SMC]).splitlines():
        k, v = line.split()
        temps[k] = float(v)
    cpu = [v for k, v in temps.items() if k.startswith('TC')]
    m['cpu_temp'] = max(cpu) if cpu else None
    m['batt_temp'] = temps.get('TB0T')
    therm = sh(['pmset', '-g', 'therm'])
    s = re.search(r'CPU_Speed_Limit\s*=\s*(\d+)', therm)
    m['speed_limit'] = int(s.group(1)) if s else 100
    w = re.search(r'thermal warning level[^\n\d]*(\d+)', therm, re.I)
    m['thermal_warning'] = int(w.group(1)) if w and 'No thermal warning' not in therm else 0
    p = sh(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level']).strip()
    m['mem_pressure'] = int(p) if p.isdigit() else 1 # 1 normal, 2 warning, 4 critical
    vm = psutil.virtual_memory()
    m['mem_avail_gb'] = vm.available / GB
    m['swap_gb'] = psutil.swap_memory().used / GB
    m['load5'] = os.getloadavg()[1]
    m['disk_free_gb'] = psutil.disk_usage(os.getcwd()).free / GB
    b = psutil.sensors_battery()
    m['on_battery'] = bool(b and not b.power_plugged)
    m['battery_pct'] = b.percent if b else None
    rss = 0
    if root_proc is not None:
        for pr in [root_proc] + safe_children(root_proc):
            try:
                rss += pr.memory_info().rss
            except psutil.Error:
                pass
    m['run_rss_gb'] = rss / GB
    m['total_gb'] = vm.total / GB
    return m


def safe_children(p):
    try:
        return p.children(recursive=True)
    except psutil.Error:
        return []


def assess(m, temp_hist, speed_hist):
    """Return (critical reasons, warning reasons)."""
    crit, warn = [], []
    t = temp_hist and sum(temp_hist) / len(temp_hist)
    if m['cpu_temp'] is not None and m['cpu_temp'] >= CPU_TEMP[1]:
        crit.append(f"CPU {m['cpu_temp']:.0f} °C")
    elif t and t >= CPU_TEMP[0]:
        warn.append(f'CPU mean {t:.0f} °C')
    if m['batt_temp'] is not None:
        if m['batt_temp'] >= BATT_TEMP[1]:
            crit.append(f"battery {m['batt_temp']:.0f} °C")
        elif m['batt_temp'] >= BATT_TEMP[0]:
            warn.append(f"battery {m['batt_temp']:.0f} °C")
    sl = sum(speed_hist) / len(speed_hist) if speed_hist else 100
    if len(speed_hist) == speed_hist.maxlen and sl < SPEED_LIMIT:
        warn.append(f"CPU speed limit {sl:.0f}% (5-min mean)")
    if m['thermal_warning']:
        warn.append(f"thermal warning level {m['thermal_warning']}")
    mem_bad = None
    if m['mem_pressure'] >= 4 or m['mem_avail_gb'] < MEM_AVAIL[1] or m['swap_gb'] >= SWAP_USED[1]:
        mem_bad = 'crit'
    elif m['mem_pressure'] >= 2 or m['mem_avail_gb'] < MEM_AVAIL[0] or m['swap_gb'] >= SWAP_USED[0]:
        mem_bad = 'warn'
    if mem_bad:
        msg = (f"memory pressure {m['mem_pressure']}, {m['mem_avail_gb']:.1f} GB available, "
               f"swap {m['swap_gb']:.1f} GB, run {m['run_rss_gb']:.1f} GB")
        # pausing does not free the run's own memory
        if mem_bad == 'crit' or m['run_rss_gb'] >= OWN_RSS_FRAC * m['total_gb']:
            crit.append(msg)
        else:
            warn.append(msg)
    if m['load5'] > LOAD_PER_CPU * psutil.cpu_count():
        warn.append(f"5-min load {m['load5']:.1f}")
    if m['disk_free_gb'] < DISK_FREE[1]:
        crit.append(f"disk free {m['disk_free_gb']:.1f} GB")
    elif m['disk_free_gb'] < DISK_FREE[0]:
        warn.append(f"disk free {m['disk_free_gb']:.1f} GB")
    if m['on_battery']:
        if m['battery_pct'] is not None and m['battery_pct'] < BATTERY_STOP:
            crit.append(f"on battery, {m['battery_pct']}%")
        else:
            warn.append('on battery')
    return crit, warn


class Watchdog:
    def __init__(self, run_dir, notebook, workdir, hours, overrides):
        self.run_dir, self.notebook, self.workdir, self.overrides = run_dir, notebook, workdir, overrides
        self.deadline = datetime.now() + timedelta(hours=hours)
        self.proc = self.ps = None
        self.paused_since = None
        self.clear_since = None
        self.pause_for_temp = False
        self.restarts = 0
        self.state = 'starting'
        self.temp_hist = deque(maxlen=6) # 3 min
        self.speed_hist = deque(maxlen=10) # 5 min
        self.cooling_since = None
        self.thermo_hist = deque(maxlen=2)
        self.duty = deque(maxlen=int(3600 / THERMO_S)) # 1 = running, 0 = frozen by the thermostat, last hour
        self.last_metrics_row = 0
        self.metrics_file = os.path.join(run_dir, 'metrics.csv')

    def log(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(os.path.join(self.run_dir, 'watchdog.log'), 'a') as f:
            f.write(line + '\n')

    def tree(self):
        return [self.ps] + safe_children(self.ps) if self.ps else []

    def signal_tree(self, sig):
        for pr in self.tree():
            try:
                pr.send_signal(sig)
            except psutil.Error:
                pass

    def start(self):
        cmd = ['nice', '-n', '5', sys.executable, '-u', RUN_NB, self.notebook,
               os.path.join(self.run_dir, 'executed.ipynb'), self.workdir] + self.overrides
        out = open(os.path.join(self.run_dir, 'run_nb.log'), 'a')
        out.write(f'\n===== start {datetime.now():%Y-%m-%d %H:%M:%S} =====\n')
        out.flush()
        self.proc = subprocess.Popen(cmd, cwd=self.workdir, stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True,
                                     env=dict(os.environ, RUN_NB_TIMEOUT_H='24')) # thermostat pauses slow the cells
        self.ps = psutil.Process(self.proc.pid)
        self.state, self.paused_since = 'running', None
        self.log(f'started run (pid {self.proc.pid}), restarts so far {self.restarts}')

    def stop_run(self, why):
        if not self.proc:
            return
        self.log(f'STOPPING run: {why}')
        procs = self.tree()
        self.signal_tree(signal.SIGCONT)
        for pr in procs:
            try:
                pr.terminate()
            except psutil.Error:
                pass
        gone, alive = psutil.wait_procs(procs, timeout=30)
        for pr in alive:
            try:
                pr.kill()
            except psutil.Error:
                pass
        self.proc = self.ps = None
        self.paused_since = self.cooling_since = None
        self.state = 'stopped'

    def pause(self, why):
        if self.paused_since is None:
            self.cooling_since = None
            self.signal_tree(signal.SIGSTOP)
            self.paused_since = time.time()
            self.state = 'paused'
            self.log(f'PAUSED: {why}')

    def resume(self):
        self.signal_tree(signal.SIGCONT)
        self.log(f'resumed after {(time.time() - self.paused_since) / 60:.0f} min pause')
        self.paused_since, self.clear_since, self.state, self.pause_for_temp = None, None, 'running', False
        self.cooling_since = None

    def write_status(self, m, crit, warn):
        st = {'state': self.state, 'restarts': self.restarts, 'deadline': self.deadline.isoformat(timespec='seconds'),
              'pid': self.proc.pid if self.proc else None, 'critical': crit, 'warnings': warn,
              'thermostat_cooling': self.cooling_since is not None,
              'running_share_last_hour': round(sum(self.duty) / len(self.duty), 2) if self.duty else None, 'metrics': m}
        with open(os.path.join(self.run_dir, 'status.json'), 'w') as f:
            json.dump(st, f, indent=1)
        if time.time() - self.last_metrics_row >= METRICS_EVERY_S:
            self.last_metrics_row = time.time()
            new = not os.path.exists(self.metrics_file)
            with open(self.metrics_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['state'] + list(m))
                if new:
                    w.writeheader()
                w.writerow({'state': self.state, **m})

    def cpu_temp(self):
        cpu = [float(v) for k, v in (l.split() for l in sh([SMC]).splitlines()) if k.startswith('TC')]
        return max(cpu) if cpu else None

    def thermostat(self, seconds):
        """Sleep for `seconds`, freezing the run while the CPU is hot. Returns True if the run was stopped."""
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(THERMO_S)
            if self.proc is None or self.paused_since is not None:
                self.cooling_since = None
                continue
            try:
                t = self.cpu_temp()
            except Exception:
                t = None
            if t is None:
                continue
            self.thermo_hist.append(t)
            if self.cooling_since is None:
                if sum(self.thermo_hist) / len(self.thermo_hist) >= THERMO_HIGH:
                    self.signal_tree(signal.SIGSTOP)
                    self.cooling_since = time.time()
                    self.state = 'cooling'
            elif t <= THERMO_LOW:
                self.signal_tree(signal.SIGCONT)
                self.cooling_since = None
                self.thermo_hist.clear()
                self.state = 'running'
            elif time.time() - self.cooling_since > MAX_COOLING_S:
                self.stop_run(f'CPU still {t:.0f} °C after {MAX_COOLING_S / 60:.0f} min frozen')
                return True
            self.duty.append(0 if self.cooling_since else 1)
        return False

    def run(self):
        try:
            self.loop()
        except BaseException as e:
            self.log(f'watchdog error: {e!r}')
            raise
        finally:
            if self.proc is not None:
                self.stop_run('watchdog exiting')

    def loop(self):
        self.log(f'watchdog started; deadline {self.deadline:%Y-%m-%d %H:%M}; notebook {self.notebook}')
        ok_since = time.time() - RESTART_WAIT_S # allow an immediate first start
        first = True
        errors = 0
        while True:
            try:
                m = read_metrics(self.ps)
            except Exception as e: # keep watching through a failed reading
                errors += 1
                self.log(f'metrics error {errors}: {e!r}')
                if errors > 20:
                    raise
                time.sleep(CHECK_S)
                continue
            errors = 0
            if m['cpu_temp'] is not None:
                self.temp_hist.append(m['cpu_temp'])
            if self.paused_since is None:
                self.speed_hist.append(m['speed_limit'])
            else:
                self.speed_hist.clear()
            crit, warn = assess(m, self.temp_hist, self.speed_hist)
            if os.path.exists(os.path.join(self.run_dir, 'PAUSE')):
                warn.append('PAUSE file')
            self.write_status(m, crit, warn)

            if datetime.now() >= self.deadline:
                self.stop_run('deadline reached')
                break
            if os.path.exists(os.path.join(self.run_dir, 'STOP')):
                self.stop_run('STOP file')
                break

            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    self.proc = self.ps = None
                    if rc == 0:
                        self.log('run FINISHED successfully')
                        self.state = 'finished'
                        self.write_status(m, crit, warn)
                        break
                    self.log(f'run exited with code {rc} (see run_nb.log)')
                    self.state = 'stopped'
                    if self.restarts >= MAX_RESTARTS:
                        self.log('too many restarts; giving up')
                        break
                    ok_since = time.time() # wait before restarting

            if self.proc is not None:
                if crit:
                    self.stop_run('; '.join(crit))
                    ok_since = time.time()
                elif warn:
                    self.clear_since = None
                    if any('°C' in w or 'speed' in w or 'thermal' in w for w in warn):
                        self.pause_for_temp = True
                    self.pause('; '.join(warn))
                    if time.time() - self.paused_since > MAX_PAUSE_S:
                        self.stop_run(f'paused longer than {MAX_PAUSE_S / 3600:.1f} h')
                        ok_since = time.time()
                elif self.paused_since is not None:
                    cool = not self.pause_for_temp or (m['cpu_temp'] or 0) < CPU_TEMP_RESUME
                    if not cool:
                        self.clear_since = None
                    elif self.clear_since is None:
                        self.clear_since = time.time()
                    elif time.time() - self.clear_since >= CLEAR_S:
                        self.resume()
                    if self.paused_since is not None and time.time() - self.paused_since > MAX_PAUSE_S:
                        self.stop_run(f'paused longer than {MAX_PAUSE_S / 3600:.1f} h')
                        ok_since = time.time()
            else:
                if crit or warn:
                    ok_since = time.time()
                elif time.time() - ok_since >= RESTART_WAIT_S:
                    if not first:
                        if self.restarts >= MAX_RESTARTS:
                            self.log('too many restarts; giving up')
                            break
                        self.restarts += 1
                    first = False
                    self.start()
            if self.thermostat(CHECK_S):
                ok_since = time.time()
        self.state = self.state if self.state == 'finished' else 'ended'
        self.write_status(read_metrics(None), [], [])
        self.log(f'watchdog exiting ({self.state})')


if __name__ == '__main__':
    args = sys.argv[1:]
    hours = 120.0
    if '--hours' in args:
        i = args.index('--hours')
        hours = float(args[i + 1])
        del args[i:i + 2]
    run_dir, notebook, workdir, *overrides = args
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(workdir)
    Watchdog(os.path.abspath(run_dir), notebook, os.path.abspath(workdir), hours, overrides).run()
