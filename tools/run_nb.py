"""Execute a notebook cell by cell with jupyter_client (no nbconvert in HS-env).
Usage: python run_nb.py in.ipynb out.ipynb workdir [name=value ...]
Each name=value replaces the line starting 'name =' in the first code cell defining it."""
import json, sys, re, time, queue
from jupyter_client.manager import start_new_kernel

src, dst, workdir, *overrides = sys.argv[1:]
nb = json.load(open(src, encoding='utf8'))
for ov in overrides:
    name, val = ov.split('=', 1)
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        new = re.sub(rf'(?m)^{name}\s*=.*?(?=\s*#|$)', f'{name} = {val}', s, count=1)
        if new != s:
            c['source'] = new.splitlines(keepends=True)
            print(f'set {name} = {val}', flush=True)
            break

km, kc = start_new_kernel(kernel_name='python3', cwd=workdir)
def save():
    json.dump(nb, open(dst, 'w', encoding='utf8'), indent=1)
n = 0
try:
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        n += 1
        t0 = time.time()
        print(f'[cell {i} / code {n}] start', flush=True)
        msg_id = kc.execute(''.join(c['source']))
        outs, err = [], False
        while True:
            try:
                m = kc.get_iopub_msg(timeout=3600 * 6)
            except queue.Empty:
                raise RuntimeError('timeout')
            if m['parent_header'].get('msg_id') != msg_id:
                continue
            t, ct = m['msg_type'], m['content']
            if t == 'status' and ct['execution_state'] == 'idle':
                break
            if t == 'stream':
                if outs and outs[-1].get('output_type') == 'stream' and outs[-1]['name'] == ct['name']:
                    outs[-1]['text'] += ct['text']
                else:
                    outs.append({'output_type': 'stream', 'name': ct['name'], 'text': ct['text']})
                if ct['name'] == 'stderr' and len(ct['text']) < 400:
                    print('  ', ct['text'].strip()[:300], flush=True)
            elif t in ('execute_result', 'display_data'):
                o = {'output_type': t, 'data': ct['data'], 'metadata': ct.get('metadata', {})}
                if t == 'execute_result':
                    o['execution_count'] = ct['execution_count']
                outs.append(o)
            elif t == 'error':
                outs.append({'output_type': 'error', 'ename': ct['ename'], 'evalue': ct['evalue'], 'traceback': ct['traceback']})
                err = True
                print('ERROR', ct['ename'], ct['evalue'], flush=True)
        for o in outs:
            if o['output_type'] == 'stream':
                o['text'] = o['text'].splitlines(keepends=True)
        c['outputs'], c['execution_count'] = outs, n
        save()
        print(f'[cell {i}] done in {time.time() - t0:.0f}s', flush=True)
        if err:
            sys.exit(1)
finally:
    save()
    kc.stop_channels()
    km.shutdown_kernel(now=True)
print('finished', flush=True)
