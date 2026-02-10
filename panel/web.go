package main

// dashboardHTML is the embedded single-page web UI for the panel.
const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hyperfanity Panel</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.4em; }
  h2 { color: #8b949e; margin: 24px 0 12px; font-size: 1.1em; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .stat { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 14px; text-align: center; }
  .stat .val { font-size: 1.6em; color: #58a6ff; font-weight: bold; }
  .stat .lbl { font-size: 0.75em; color: #8b949e; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #21262d; font-size: 0.85em; }
  th { color: #8b949e; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }
  .badge-pending { background: #1f2937; color: #9ca3af; }
  .badge-active { background: #064e3b; color: #34d399; }
  .badge-complete { background: #1e3a5f; color: #58a6ff; }
  .badge-online { background: #064e3b; color: #34d399; }
  .badge-offline { background: #3b1c1c; color: #f87171; }
  .form-row { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  input, select, button { font-family: inherit; font-size: 0.85em; padding: 8px 12px; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #c9d1d9; }
  button { background: #238636; border-color: #238636; color: #fff; cursor: pointer; font-weight: 600; }
  button:hover { background: #2ea043; }
  .key { font-family: inherit; font-size: 0.8em; word-break: break-all; color: #f0883e; }
  .muted { color: #484f58; }
  .addr { color: #d2a8ff; }
</style>
</head>
<body>
<div class="container">
  <h1>Hyperfanity Panel</h1>
  <p class="muted" style="margin-bottom:16px">CUDA Vanity Address Mining Coordinator</p>

  <div class="stats" id="stats"></div>

  <h2>Create Job</h2>
  <div class="form-row">
    <select id="chain"><option value="eth">ETH</option><option value="trx">TRX</option><option value="sol">SOL</option><option value="btc">BTC</option></select>
    <select id="matchType"><option value="prefix">Prefix</option><option value="suffix">Suffix</option><option value="contains">Contains</option></select>
    <input id="pattern" placeholder="Pattern (e.g. dead)" style="flex:1;min-width:120px">
    <button onclick="createJob()">Submit Job</button>
  </div>

  <h2>Jobs</h2>
  <table>
    <thead><tr><th>ID</th><th>Chain</th><th>Mode</th><th>Pattern</th><th>Status</th><th>Result</th></tr></thead>
    <tbody id="jobs"></tbody>
  </table>

  <h2>Workers</h2>
  <table>
    <thead><tr><th>ID</th><th>Host</th><th>GPUs</th><th>Hashrate</th><th>Checked</th><th>Best</th><th>Status</th></tr></thead>
    <tbody id="workers"></tbody>
  </table>
</div>
<script>
function fmt(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}
function badge(cls, txt) { return '<span class="badge badge-'+cls+'">'+txt+'</span>'; }
function short(id) { return id ? id.substring(0, 12) : '-'; }

async function refresh() {
  try {
    const [stats, jobs, workers] = await Promise.all([
      fetch('/api/stats').then(r => r.json()),
      fetch('/api/jobs').then(r => r.json()),
      fetch('/api/workers').then(r => r.json()),
    ]);
    document.getElementById('stats').innerHTML =
      '<div class="stat"><div class="val">'+stats.OnlineWorkers+'/'+stats.TotalWorkers+'</div><div class="lbl">Workers Online</div></div>'+
      '<div class="stat"><div class="val">'+stats.TotalHashrate.toFixed(1)+'</div><div class="lbl">Total MH/s</div></div>'+
      '<div class="stat"><div class="val">'+stats.PendingJobs+'</div><div class="lbl">Pending Jobs</div></div>'+
      '<div class="stat"><div class="val">'+stats.ActiveJobs+'</div><div class="lbl">Active Jobs</div></div>'+
      '<div class="stat"><div class="val">'+stats.CompletedJobs+'</div><div class="lbl">Completed</div></div>';

    let jhtml = '';
    (jobs || []).forEach(j => {
      let res = '-';
      if (j.status === 'complete') res = '<span class="addr">'+j.result_address+'</span><br><span class="key">'+j.result_key+'</span>';
      jhtml += '<tr><td>'+short(j.id)+'</td><td>'+j.chain.toUpperCase()+'</td><td>'+j.match_type+'</td><td>'+j.pattern+'</td><td>'+badge(j.status,j.status)+'</td><td>'+res+'</td></tr>';
    });
    document.getElementById('jobs').innerHTML = jhtml || '<tr><td colspan="6" class="muted">No jobs</td></tr>';

    let whtml = '';
    (workers || []).forEach(w => {
      let gpus = w.gpu_names ? w.gpu_names.join(', ') : w.gpu_count+' GPU(s)';
      let st = w.online ? badge('online','online') : badge('offline','offline');
      whtml += '<tr><td>'+short(w.id)+'</td><td>'+w.hostname+'</td><td>'+gpus+'</td><td>'+w.hashrate_mhs.toFixed(1)+' MH/s</td><td>'+fmt(w.total_checked)+'</td><td>'+w.best_score+'</td><td>'+st+'</td></tr>';
    });
    document.getElementById('workers').innerHTML = whtml || '<tr><td colspan="7" class="muted">No workers</td></tr>';
  } catch(e) { console.error('refresh error:', e); }
}

async function createJob() {
  const chain = document.getElementById('chain').value;
  const matchType = document.getElementById('matchType').value;
  const pattern = document.getElementById('pattern').value.trim();
  if (!pattern) { alert('Enter a pattern'); return; }
  await fetch('/api/jobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({chain, match_type: matchType, pattern}),
  });
  document.getElementById('pattern').value = '';
  refresh();
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
`
