/* =========================================================================
   QuantMind dashboard — rendering & interactivity
   Reads window.QM_DATA (exported from the real trained prototype).
   ========================================================================= */
(function () {
  const D = window.QM_DATA;
  if (!D) { document.body.innerHTML = "<p style='padding:40px;color:#fff'>data.js not loaded — run export_frontend_data.py</p>"; return; }

  const COLORS = { AAPL:"#60a5fa", MSFT:"#f472b6", JPM:"#34d399", XOM:"#a78bfa", JNJ:"#fbbf24", CASH:"#64748b" };
  const BRAND1 = "#818cf8", BRAND2 = "#2dd4bf", POS = "#34d399", NEG = "#f87171", MUTED = "#aeb7d0";
  const assetColor = (k) => COLORS[k] || "#94a3b8";

  // ---- formatting ---------------------------------------------------------
  const fmtMoney = (v) => "£" + Math.round(v).toLocaleString("en-GB");
  const fmtPct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const fmtSignedPct = (v, d = 1) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
  const fmtNum = (v, d = 2) => v.toFixed(d);

  // ---- Chart.js global theme ---------------------------------------------
  Chart.defaults.color = MUTED;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(11,16,32,.96)";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,.12)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.cornerRadius = 10;
  Chart.defaults.plugins.tooltip.titleColor = "#eef2fb";
  Chart.defaults.plugins.tooltip.bodyColor = "#aeb7d0";
  const GRID = "rgba(255,255,255,.05)";

  // ---- header / footer ----------------------------------------------------
  document.getElementById("top-value").textContent = fmtMoney(D.kpis.portfolio_value);
  document.getElementById("foot-source").textContent = D.meta.data_source;
  document.getElementById("equity-range").textContent = `${D.meta.test_start} → ${D.meta.test_end} · ${D.meta.test_days} trading days`;
  document.getElementById("perf-range").textContent = `${D.meta.test_start} → ${D.meta.test_end} · out-of-sample (${D.meta.test_days} days)`;

  // =========================================================================
  //  NAV / VIEW SWITCHING
  // =========================================================================
  const TITLES = {
    dashboard: ["Portfolio Dashboard", "Active, explainable allocation across 5 S&P 500 assets"],
    recommendations: ["Recommendations", "What QuantMind suggests today — and why"],
    explainability: ["Explainability", "Open the black box: what drives every decision"],
    assistant: ["AI Assistant", "Ask about your portfolio in plain English"],
    performance: ["Performance", "Backtested results vs passive baselines"],
  };
  let chatBooted = false;
  function activateView(v) {
    if (!TITLES[v]) v = "dashboard";
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === v));
    document.querySelectorAll(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
    document.getElementById("pg-title").textContent = TITLES[v][0];
    document.getElementById("pg-sub").innerHTML = TITLES[v][1];
    if (v === "assistant" && !chatBooted) { bootChat(); chatBooted = true; }
  }
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      const v = item.dataset.view;
      if (history.replaceState) history.replaceState(null, "", "#" + v);
      activateView(v);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
  // (deep-link activation happens at the end, once all helpers are defined)

  // =========================================================================
  //  KPI CARDS
  // =========================================================================
  const ICONS = {
    wallet: '<path d="M19 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0 0 4h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7"/><circle cx="16" cy="13" r="1.4"/>',
    trend: '<path d="M3 17l5-5 4 4 8-9"/><path d="M16 7h5v5"/>',
    shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    drop: '<path d="M12 22a7 7 0 0 0 7-7c0-5-7-13-7-13S5 10 5 15a7 7 0 0 0 7 7z"/>',
  };
  const ico = (k) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${ICONS[k]}</svg>`;
  const k = D.kpis;
  const kpis = [
    { label: "Portfolio Value", icon: "wallet", val: fmtMoney(k.portfolio_value), delta: fmtSignedPct(k.total_return), up: k.total_return >= 0, note: "total return" },
    { label: "Sharpe Ratio", icon: "trend", val: fmtNum(k.sharpe), delta: "+" + fmtNum(k.sharpe - D.metrics["Buy & Hold (1/N)"].Sharpe), up: true, note: "vs buy & hold" },
    { label: "CAGR", icon: "shield", val: fmtPct(k.cagr), delta: fmtSignedPct(k.cagr - D.metrics["Buy & Hold (1/N)"].CAGR), up: true, note: "annualised" },
    { label: "Max Drawdown", icon: "drop", val: fmtPct(k.max_drawdown), delta: "vol " + fmtPct(k.volatility), up: false, note: "peak-to-trough", neutral: true },
  ];
  document.getElementById("kpi-grid").innerHTML = kpis.map((c) => `
    <div class="card kpi">
      <div class="top"><span class="label">${c.label}</span><span class="ico">${ico(c.icon)}</span></div>
      <div class="val num">${c.val}</div>
      <div class="delta ${c.neutral ? "" : c.up ? "up" : "down"}">
        ${c.neutral ? "" : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">${c.up ? '<path d="M6 15l6-6 6 6"/>' : '<path d="M6 9l6 6 6-6"/>'}</svg>`}
        <span class="num">${c.delta}</span><span class="muted">${c.note}</span>
      </div>
    </div>`).join("");

  // =========================================================================
  //  EQUITY CHART
  // =========================================================================
  const eq = D.equity;
  const mkGradient = (ctx, hex) => {
    const g = ctx.createLinearGradient(0, 0, 0, 300);
    g.addColorStop(0, hex + "55"); g.addColorStop(1, hex + "02"); return g;
  };
  let equityChart;
  function renderEquity(cmp) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    const cmpData = cmp === "best" ? eq.best : eq.buyhold;
    const cmpLabel = cmp === "best" ? `Best asset (${D.meta.best_asset_name.replace("Best asset ", "").replace(/[()]/g, "")})` : "Buy & Hold (1/N)";
    const ds = [
      { label: "QuantMind PPO", data: eq.agent, borderColor: BRAND2, backgroundColor: mkGradient(ctx, "#2dd4bf"), fill: true, borderWidth: 2.6, tension: .25, pointRadius: 0, pointHoverRadius: 4 },
      { label: cmpLabel, data: cmpData, borderColor: "#8b93a7", backgroundColor: "transparent", borderWidth: 1.6, borderDash: [5, 4], tension: .25, pointRadius: 0, pointHoverRadius: 4 },
    ];
    if (equityChart) { equityChart.data.labels = eq.dates; equityChart.data.datasets = ds; equityChart.update(); }
    else equityChart = new Chart(ctx, {
      type: "line",
      data: { labels: eq.dates, datasets: ds },
      options: {
        responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtMoney(c.parsed.y)}` } } },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: "#6b7493" } },
          y: { grid: { color: GRID }, ticks: { callback: (v) => "£" + (v / 1000).toFixed(0) + "k", color: "#6b7493" } },
        },
      },
    });
    document.getElementById("equity-legend").innerHTML =
      `<div class="item"><span class="swatch" style="background:${BRAND2}"></span>QuantMind PPO</div>
       <div class="item"><span class="swatch" style="background:#8b93a7"></span>${cmpLabel}</div>`;
  }
  renderEquity("buyhold");
  document.querySelectorAll("#equity-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#equity-seg button").forEach((x) => x.classList.toggle("active", x === b));
      renderEquity(b.dataset.cmp);
    }));

  // =========================================================================
  //  CURRENT ALLOCATION — donut + list
  // =========================================================================
  const ca = D.current_allocation;
  const caKeys = Object.keys(ca).sort((a, b) => ca[b] - ca[a]);
  new Chart(document.getElementById("allocDonut").getContext("2d"), {
    type: "doughnut",
    data: { labels: caKeys, datasets: [{ data: caKeys.map((x) => ca[x]), backgroundColor: caKeys.map(assetColor), borderColor: "#0b1020", borderWidth: 3, hoverOffset: 6 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "72%", plugins: { tooltip: { callbacks: { label: (c) => `${c.label}: ${fmtPct(c.parsed)}` } } } },
  });
  document.getElementById("donut-invested").textContent = fmtPct(1 - (ca.CASH || 0), 0);
  document.getElementById("alloc-list").innerHTML = caKeys.map((key) => `
    <div class="alloc-row">
      <div class="tkr"><span class="d" style="background:${assetColor(key)}"></span>${key}</div>
      <div class="alloc-bar"><span style="width:${(ca[key] * 100).toFixed(1)}%;background:${assetColor(key)}"></span></div>
      <div class="pct num">${fmtPct(ca[key])}</div>
    </div>`).join("");

  // =========================================================================
  //  ALLOCATION OVER TIME — stacked area
  // =========================================================================
  const at = D.allocation_ts;
  new Chart(document.getElementById("allocArea").getContext("2d"), {
    type: "line",
    data: {
      labels: at.dates,
      datasets: at.columns.map((col) => ({
        label: col, data: at.series[col], borderColor: assetColor(col),
        backgroundColor: assetColor(col) + "cc", fill: true, borderWidth: 0,
        tension: .3, pointRadius: 0,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtPct(c.parsed.y)}` } } },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { maxTicksLimit: 7, color: "#6b7493" } },
        y: { stacked: true, min: 0, max: 1, grid: { color: GRID }, ticks: { callback: (v) => (v * 100) + "%", color: "#6b7493" } },
      },
    },
  });
  document.getElementById("area-legend").innerHTML = at.columns.map((c) =>
    `<div class="item"><span class="swatch" style="background:${assetColor(c)}"></span>${c}</div>`).join("");

  // =========================================================================
  //  RECOMMENDATIONS
  // =========================================================================
  document.getElementById("rationale-text").textContent = D.rationale.replace(/\n/g, " ");
  const badgeClass = (a) => a === "Overweight" ? "over" : a === "Hold" ? "hold" : "under";
  const maxDriver = Math.max(...D.recommendations.flatMap((r) => r.drivers.map((d) => d.magnitude)), 1e-9);
  document.getElementById("reco-list").innerHTML = D.recommendations.map((r) => `
    <div class="reco-card">
      <div class="reco-main">
        <div class="reco-logo" style="background:${assetColor(r.ticker)}">${r.ticker.slice(0, 2)}</div>
        <div class="reco-id"><div class="t">${r.ticker}</div><div class="s">${r.name} · ${r.sector}</div></div>
        <div class="reco-weight">
          <div class="wl"><span>Target weight</span><span class="num">${fmtPct(r.weight)}</span></div>
          <div class="wbar"><span style="width:${Math.max(r.weight * 100, 1.5)}%"></span></div>
        </div>
        <span class="badge ${badgeClass(r.action)}">${r.action}</span>
        <svg class="reco-chevron" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
      </div>
      <div class="reco-why"><div class="reco-why-inner">
        <div class="h">Why — top Shapley drivers</div>
        ${r.drivers.map((d) => `
          <div class="driver">
            <div class="dir ${d.direction}">${d.direction === "up"
              ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>'
              : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>'}</div>
            <div>${d.phrase} <span style="color:var(--text-3)">${d.direction === "up" ? "increased" : "reduced"} the weight</span></div>
            <div class="mbar"><span style="width:${(d.magnitude / maxDriver * 100).toFixed(0)}%;background:${d.direction === "up" ? POS : NEG}"></span></div>
          </div>`).join("")}
      </div></div>
    </div>`).join("");
  document.querySelectorAll(".reco-card").forEach((card) =>
    card.querySelector(".reco-main").addEventListener("click", () => card.classList.toggle("open")));

  // =========================================================================
  //  EXPLAINABILITY — SHAP bar chart
  // =========================================================================
  const prettyFeat = (f) => {
    if (f.startsWith("w_")) return f.slice(2) + " weight (inertia)";
    const [tk, ft] = f.split(":");
    const map = { ret_1d: "1-day return", ret_5d: "5-day momentum", rsi_14: "RSI", macd_hist: "MACD trend", vol_20: "volatility", px_sma20: "price vs SMA20", sma5_sma20: "trend (SMA5/20)" };
    return `${tk} · ${map[ft] || ft}`;
  };
  const fi = D.feature_importance.slice().reverse();
  new Chart(document.getElementById("shapChart").getContext("2d"), {
    type: "bar",
    data: { labels: fi.map((x) => prettyFeat(x.feature)), datasets: [{ data: fi.map((x) => x.value), backgroundColor: fi.map((x) => x.feature.startsWith("w_") ? BRAND1 : BRAND2), borderRadius: 5, barThickness: 14 }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { tooltip: { callbacks: { label: (c) => "impact: " + c.parsed.x.toFixed(4) } } },
      scales: { x: { grid: { color: GRID }, ticks: { color: "#6b7493" } }, y: { grid: { display: false }, ticks: { color: "#aeb7d0" } } },
    },
  });

  // =========================================================================
  //  PERFORMANCE — table + risk/return + ratios
  // =========================================================================
  const order = ["QuantMind PPO", "Buy & Hold (1/N)"];
  const bestKey = Object.keys(D.metrics).find((x) => x.startsWith("Best asset"));
  if (bestKey) order.push(bestKey);
  const stratColor = { "QuantMind PPO": BRAND2, "Buy & Hold (1/N)": "#8b93a7" };
  if (bestKey) stratColor[bestKey] = "#a78bfa";
  const bestSharpe = Math.max(...order.map((s) => D.metrics[s].Sharpe));

  const cols = [["Total Return", "Total Return"], ["CAGR", "CAGR"], ["Ann. Volatility", "Volatility"], ["Sharpe", "Sharpe"], ["Sortino", "Sortino"], ["Max Drawdown", "Max DD"]];
  document.getElementById("perf-table").innerHTML = `
    <thead><tr><th>Strategy</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead>
    <tbody>${order.map((s) => {
      const m = D.metrics[s];
      const cell = (key) => {
        const v = m[key];
        if (key === "Sharpe" || key === "Sortino") return `<td class="${v === bestSharpe && key === "Sharpe" ? "best" : ""} num">${v.toFixed(2)}</td>`;
        if (key === "Max Drawdown") return `<td class="cell-neg num">${fmtPct(v)}</td>`;
        return `<td class="${v >= 0 ? "cell-pos" : "cell-neg"} num">${fmtSignedPct(v)}</td>`;
      };
      return `<tr class="${s === "QuantMind PPO" ? "hero" : ""}">
        <td><div class="strat"><span class="d" style="background:${stratColor[s]}"></span>${s}</div></td>
        ${cols.map((c) => cell(c[0])).join("")}</tr>`;
    }).join("")}</tbody>`;
  document.getElementById("perf-note").innerHTML =
    `Average daily turnover ${fmtPct(D.kpis.avg_turnover)} · ${(D.meta.transaction_cost * 100).toFixed(1)}% transaction cost · trained ${D.meta.timesteps.toLocaleString()} steps. "Best asset" is chosen with hindsight and is not investable.`;

  new Chart(document.getElementById("riskReturn").getContext("2d"), {
    type: "scatter",
    data: { datasets: order.map((s) => ({ label: s, data: [{ x: D.metrics[s]["Ann. Volatility"], y: D.metrics[s].CAGR }], backgroundColor: stratColor[s], pointRadius: 9, pointHoverRadius: 11 })) },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { usePointStyle: true, boxWidth: 8 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: vol ${fmtPct(c.parsed.x)}, CAGR ${fmtPct(c.parsed.y)}` } } },
      scales: { x: { grid: { color: GRID }, title: { display: true, text: "Volatility", color: "#6b7493" }, ticks: { callback: (v) => (v * 100).toFixed(0) + "%", color: "#6b7493" } }, y: { grid: { color: GRID }, title: { display: true, text: "CAGR", color: "#6b7493" }, ticks: { callback: (v) => (v * 100).toFixed(0) + "%", color: "#6b7493" } } },
    },
  });

  new Chart(document.getElementById("ratioChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: order.map((s) => s.replace(" (1/N)", "").replace(" PPO", "")),
      datasets: [
        { label: "Sharpe", data: order.map((s) => D.metrics[s].Sharpe), backgroundColor: BRAND2, borderRadius: 6, barPercentage: .6, categoryPercentage: .6 },
        { label: "Sortino", data: order.map((s) => D.metrics[s].Sortino), backgroundColor: BRAND1, borderRadius: 6, barPercentage: .6, categoryPercentage: .6 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { usePointStyle: true, boxWidth: 8 } } }, scales: { x: { grid: { display: false }, ticks: { color: "#aeb7d0" } }, y: { grid: { color: GRID }, ticks: { color: "#6b7493" } } } },
  });

  // =========================================================================
  //  AI ASSISTANT (rule-based answers grounded in the real data)
  // =========================================================================
  const top = D.recommendations[0];
  const bh = D.metrics["Buy & Hold (1/N)"];
  const answerFor = (qRaw) => {
    const q = qRaw.toLowerCase();
    const tk = D.meta.tickers.find((t) => q.includes(t.toLowerCase()));
    if (tk) {
      const r = D.recommendations.find((x) => x.ticker === tk);
      const dl = r.drivers.map((d) => `<b>${d.phrase}</b> (${d.direction === "up" ? "↑" : "↓"})`).join(", ");
      return `I currently set <b>${tk}</b> (${r.name}) to <b>${fmtPct(r.weight)}</b> — a <b>${r.action.toLowerCase()}</b> stance. The biggest drivers were ${dl}. You can see the full Shapley breakdown on the Recommendations tab.`;
    }
    if (/(risk|drawdown|volatil|safe|lose)/.test(q))
      return `The portfolio's annualised volatility is <b>${fmtPct(k.volatility)}</b> with a worst peak-to-trough drawdown of <b>${fmtPct(k.max_drawdown)}</b> over the test period. A 40% per-asset cap enforces diversification, and a turnover penalty keeps trading low (avg <b>${fmtPct(k.avg_turnover)}</b>/day), which controls cost and risk.`;
    if (/(beat|market|benchmark|buy.?and.?hold|outperform|better)/.test(q))
      return `Over the out-of-sample period I returned <b>${fmtSignedPct(k.total_return)}</b> vs <b>${fmtSignedPct(bh["Total Return"])}</b> for an equal-weight buy-and-hold, with a higher Sharpe ratio (<b>${k.sharpe.toFixed(2)}</b> vs <b>${bh.Sharpe.toFixed(2)}</b>) at similar volatility. So yes — better risk-adjusted returns, though not a guarantee of future performance.`;
    if (/(how|explain|work|decide|why|driver|shap|feature)/.test(q))
      return `Each decision is explained with <b>Shapley-value attribution</b>: I measure how every market signal moves the allocation. Right now my decisions lean most on <b>position inertia</b> (avoiding needless trades), then <b>MACD trend</b> and <b>RSI</b> signals. Head to the Explainability tab for the full ranking.`;
    if (/(sharpe|sortino|ratio)/.test(q))
      return `My Sharpe ratio is <b>${k.sharpe.toFixed(2)}</b> and Sortino is <b>${k.sortino.toFixed(2)}</b>, both above the buy-and-hold baseline (${bh.Sharpe.toFixed(2)} / ${bh.Sortino.toFixed(2)}). Sortino is higher because it only penalises downside volatility.`;
    return `I'm QuantMind — an explainable RL portfolio agent. Your portfolio is worth <b>${fmtMoney(k.portfolio_value)}</b> (<b>${fmtSignedPct(k.total_return)}</b>), my biggest position is <b>${top.ticker}</b> at <b>${fmtPct(top.weight)}</b>, and every recommendation comes with a plain-English reason. Try asking why I chose a specific stock, or how risky the portfolio is.`;
  };

  const stream = document.getElementById("chat-stream");
  const addMsg = (html, who) => {
    const el = document.createElement("div");
    el.className = "msg " + who;
    el.innerHTML = who === "bot"
      ? `<div class="av"><svg viewBox="0 0 24 24" fill="none" stroke="#04121a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0 0 8v1a4 4 0 0 0 8 0v-1a4 4 0 0 0 0-8V6a4 4 0 0 0-4-4z"/></svg></div><div class="bubble">${html}</div>`
      : `<div class="bubble">${html}</div>`;
    stream.appendChild(el);
    stream.scrollTop = stream.scrollHeight;
  };
  const send = (q) => { addMsg(q, "user"); setTimeout(() => addMsg(answerFor(q), "bot"), 320); };

  const SUGGEST = [`Why did you overweight ${top.ticker}?`, "Did you beat the market?", "How risky is this portfolio?", "How do you make decisions?"];
  function bootChat() {
    addMsg(answerFor(""), "bot");
    document.getElementById("suggest").innerHTML = SUGGEST.map((s) => `<button>${s}</button>`).join("");
    document.querySelectorAll("#suggest button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
  }
  const input = document.getElementById("chat-input");
  const doSend = () => { const v = input.value.trim(); if (!v) return; send(v); input.value = ""; };
  document.getElementById("chat-send").addEventListener("click", doSend);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") doSend(); });

  // Deep-link support: open directly to a view via #recommendations etc.
  // Placed last so all view helpers (incl. bootChat dependencies) are defined.
  if (location.hash) activateView(location.hash.slice(1));
})();
