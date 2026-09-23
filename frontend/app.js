/* =========================================================================
   QuantMind dashboard — rendering & interactivity
   Reads window.QM_DATA (exported from the real trained prototype).
   ========================================================================= */
(function () {
  const D = window.QM_DATA;

  // ---- study build --------------------------------------------------------
  // Build A (default) shows explanations. Build B (?build=B) is the
  // explanations-hidden variant used by the user study's comparison arm: the
  // same data, layout and controls, with every explanation surface removed --
  // the "why" drivers, the rationale banner, the Explainability view and any
  // explanatory chat answer -- so a difference in trust can be attributed to
  // the explanations rather than to the tool.
  const BUILD = new URLSearchParams(location.search).get("build") === "B" ? "B" : "A";
  const SHOW_XAI = BUILD === "A";
  document.body.dataset.build = BUILD;
  if (!SHOW_XAI) document.title = "QuantMind — AI Portfolio Advisor";
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
  // ?still turns off chart animation so a screenshot shows the finished charts.
  if (new URLSearchParams(location.search).has("still")) Chart.defaults.animation = false;

  // ---- initial-amount rescale -------------------------------------
  // Every £ figure in the backtest scales linearly with the starting cash
  // (INITIAL_CASH = £100,000 in src/config.py); percentages, Sharpe, weights
  // and allocations are scale-invariant. So the user's own amount is applied
  // as a pure client-side multiplier over the one locked payload -- no
  // retraining, no recomputation, honest about being a rescale of the same
  // backtest rather than a new one.
  const BASE_AMOUNT = D.meta.initial_cash;
  let scaleFactor = 1;


  // ---- larger text -------------------------------------------------------
  const sizeBtn = document.getElementById("text-size");
  const applySize = (on) => { document.body.dataset.large = on ? "1" : "0"; sizeBtn.setAttribute("aria-pressed", on ? "true" : "false"); };
  let largeOn = false;
  try { largeOn = localStorage.getItem("qm-large") === "1"; } catch (e) { /* storage unavailable */ }
  applySize(largeOn);
  sizeBtn.addEventListener("click", () => {
    largeOn = !largeOn; applySize(largeOn);
    try { localStorage.setItem("qm-large", largeOn ? "1" : "0"); } catch (e) { /* ignore */ }
  });

  // ---- header / footer ----------------------------------------------------
  function renderTopValue() {
    document.getElementById("top-value").textContent = fmtMoney(D.kpis.portfolio_value * scaleFactor);
  }
  renderTopValue();
  document.getElementById("foot-source").textContent = D.meta.data_source;
  document.getElementById("equity-range").textContent = `${D.meta.test_start} → ${D.meta.test_end} · ${D.meta.test_days} trading days`;
  document.getElementById("perf-range").textContent = `${D.meta.test_start} → ${D.meta.test_end} · out-of-sample (${D.meta.test_days} days)`;

  // =========================================================================
  //  NAV / VIEW SWITCHING
  // =========================================================================
  const TITLES = SHOW_XAI ? {
    dashboard: ["Portfolio Dashboard", "Active, explainable allocation across 5 S&P 500 assets"],
    build: ["Build a portfolio", "Get advice for your own amount and date"],
    recommendations: ["Recommendations", "What QuantMind suggests today — and why"],
    explainability: ["Explainability", "Open the black box: what drives every decision"],
    assistant: ["AI Assistant", "Ask about your portfolio in plain English"],
    performance: ["Performance", "Backtested results vs passive baselines"],
  } : {
    dashboard: ["Portfolio Dashboard", "Active allocation across 5 S&P 500 assets"],
    build: ["Build a portfolio", "Get advice for your own amount and date"],
    recommendations: ["Recommendations", "What QuantMind suggests today"],
    assistant: ["AI Assistant", "Ask about your portfolio in plain English"],
    performance: ["Performance", "Backtested results vs passive baselines"],
  };
  if (!SHOW_XAI) {
    // Remove every explanation surface from the page (build B).
    document.querySelectorAll(".xai-only").forEach((el) => el.remove());
    document.getElementById("pg-sub").textContent = TITLES.dashboard[1];
    document.getElementById("reco-sub").textContent = "The suggested split across the five companies";
    document.getElementById("reco-tag").textContent = "Backtested";
  }
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
  function renderKpis() {
    const kpis = [
      { label: "Portfolio Value", icon: "wallet", val: fmtMoney(k.portfolio_value * scaleFactor), delta: fmtSignedPct(k.total_return), up: k.total_return >= 0, note: "total return" },
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
    document.getElementById("kpi-note").textContent =
      `Best of ${k.n_seeds} training seeds; across all ${k.n_seeds} the average Sharpe is ${fmtNum(k.mean_sharpe_5seed)}.`;
  }
  renderKpis();

  // =========================================================================
  //  EQUITY CHART
  // =========================================================================
  const eq = D.equity;
  const mkGradient = (ctx, hex) => {
    const g = ctx.createLinearGradient(0, 0, 0, 300);
    g.addColorStop(0, hex + "55"); g.addColorStop(1, hex + "02"); return g;
  };
  let equityChart, lastCmp = "buyhold";
  const scaled = (arr) => arr.map((v) => v * scaleFactor);
  function renderEquity(cmp) {
    lastCmp = cmp;
    const ctx = document.getElementById("equityChart").getContext("2d");
    const cmpData = cmp === "best" ? eq.best : eq.buyhold;
    const cmpLabel = cmp === "best" ? `Best asset (${D.meta.best_asset_name.replace("Best asset ", "").replace(/[()]/g, "")})` : "Buy & Hold (1/N)";
    const ds = [
      { label: "QuantMind PPO", data: scaled(eq.agent), borderColor: BRAND2, backgroundColor: mkGradient(ctx, "#2dd4bf"), fill: true, borderWidth: 2.6, tension: .25, pointRadius: 0, pointHoverRadius: 4 },
      { label: cmpLabel, data: scaled(cmpData), borderColor: "#8b93a7", backgroundColor: "transparent", borderWidth: 1.6, borderDash: [5, 4], tension: .25, pointRadius: 0, pointHoverRadius: 4 },
    ];
    if (equityChart) { equityChart.data.labels = eq.dates; equityChart.data.datasets = ds; equityChart.update(); }
    else equityChart = new Chart(ctx, {
      type: "line",
      data: { labels: eq.dates, datasets: ds },
      options: {
        responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtMoney(c.parsed.y)}` } } },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: "#7d86a6" } },
          y: { grid: { color: GRID }, ticks: { callback: (v) => "£" + (v / 1000).toFixed(0) + "k", color: "#7d86a6" } },
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

  // ---- initial-amount input: rescale every £ display, nothing retrained ---
  const amountInput = document.getElementById("amount-input");
  if (amountInput) {
    amountInput.value = Math.round(BASE_AMOUNT);
    amountInput.addEventListener("input", () => {
      const v = parseFloat(amountInput.value);
      scaleFactor = v > 0 ? v / BASE_AMOUNT : 1;
      renderTopValue();
      renderKpis();
      renderEquity(lastCmp);
    });
  }

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
        x: { stacked: true, grid: { display: false }, ticks: { maxTicksLimit: 7, color: "#7d86a6" } },
        y: { stacked: true, min: 0, max: 1, grid: { color: GRID }, ticks: { callback: (v) => (v * 100) + "%", color: "#7d86a6" } },
      },
    },
  });
  document.getElementById("area-legend").innerHTML = at.columns.map((c) =>
    `<div class="item"><span class="swatch" style="background:${assetColor(c)}"></span>${c}</div>`).join("");

  // =========================================================================
  //  RECOMMENDATIONS
  // =========================================================================
  if (SHOW_XAI) document.getElementById("rationale-text").textContent = D.rationale.replace(/\n/g, " ");
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
        ${SHOW_XAI ? '<svg class="reco-chevron" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>' : ""}
      </div>
      ${SHOW_XAI ? `<div class="reco-why"><div class="reco-why-inner">
        <div class="h">Why — top Shapley drivers</div>
        ${r.drivers.map((d) => `
          <div class="driver">
            <div class="dir ${d.direction}">${d.direction === "up"
              ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>'
              : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>'}</div>
            <div>${d.phrase} <span style="color:var(--text-3)">${d.direction === "up" ? "increased" : "reduced"} the weight</span></div>
            <div class="mbar"><span style="width:${(d.magnitude / maxDriver * 100).toFixed(0)}%;background:${d.direction === "up" ? POS : NEG}"></span></div>
          </div>`).join("")}
      </div></div>` : ""}
    </div>`).join("");
  if (SHOW_XAI) {
    document.querySelectorAll(".reco-card").forEach((card) =>
      card.querySelector(".reco-main").addEventListener("click", () => card.classList.toggle("open")));
    // Open the top recommendation by default so its reasoning is visible at a glance.
    document.querySelector(".reco-card")?.classList.add("open");
  }

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
  if (SHOW_XAI) new Chart(document.getElementById("shapChart").getContext("2d"), {
    type: "bar",
    data: { labels: fi.map((x) => prettyFeat(x.feature)), datasets: [{ data: fi.map((x) => x.value), backgroundColor: fi.map((x) => x.feature.startsWith("w_") ? BRAND1 : BRAND2), borderRadius: 5, barThickness: 14 }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { tooltip: { callbacks: { label: (c) => "impact: " + c.parsed.x.toFixed(4) } } },
      scales: { x: { grid: { color: GRID }, ticks: { color: "#7d86a6" } }, y: { grid: { display: false }, ticks: { color: "#aeb7d0" } } },
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
    `Best of ${k.n_seeds} training seeds; across all ${k.n_seeds} the average Sharpe is ${fmtNum(k.mean_sharpe_5seed)}. Average daily turnover ${fmtPct(D.kpis.avg_turnover)} · ${(D.meta.transaction_cost * 100).toFixed(1)}% transaction cost · trained ${D.meta.timesteps.toLocaleString()} steps. "Best asset" is chosen with hindsight and is not investable.`;

  new Chart(document.getElementById("riskReturn").getContext("2d"), {
    type: "scatter",
    data: { datasets: order.map((s) => ({ label: s, data: [{ x: D.metrics[s]["Ann. Volatility"], y: D.metrics[s].CAGR }], backgroundColor: stratColor[s], pointRadius: 9, pointHoverRadius: 11 })) },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { usePointStyle: true, boxWidth: 8 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: vol ${fmtPct(c.parsed.x)}, CAGR ${fmtPct(c.parsed.y)}` } } },
      scales: { x: { grid: { color: GRID }, title: { display: true, text: "Volatility", color: "#7d86a6" }, ticks: { callback: (v) => (v * 100).toFixed(0) + "%", color: "#7d86a6" } }, y: { grid: { color: GRID }, title: { display: true, text: "CAGR", color: "#7d86a6" }, ticks: { callback: (v) => (v * 100).toFixed(0) + "%", color: "#7d86a6" } } },
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
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { usePointStyle: true, boxWidth: 8 } } }, scales: { x: { grid: { display: false }, ticks: { color: "#aeb7d0" } }, y: { grid: { color: GRID }, ticks: { color: "#7d86a6" } } } },
  });

  // =========================================================================
  //  AI ASSISTANT (rule-based answers grounded in the real data)
  // =========================================================================
  const top = D.recommendations[0];
  const bh = D.metrics["Buy & Hold (1/N)"];
  // Build B: the assistant still answers questions about weights, performance
  // and risk, but gives no reasons, drivers or method description.
  const answerForB = (qRaw) => {
    const q = qRaw.toLowerCase();
    const tk = D.meta.tickers.find((t) => q.includes(t.toLowerCase()));
    if (/(why|how.*(decide|choose|work)|explain|reason|driver|shap|feature)/.test(q))
      return `This version doesn't give reasons for its choices. I can tell you the suggested weights, the performance against buy-and-hold, and how risky the portfolio is.`;
    if (tk) {
      const r = D.recommendations.find((x) => x.ticker === tk);
      return `I currently set <b>${tk}</b> (${r.name}) to <b>${fmtPct(r.weight)}</b> of the portfolio — a <b>${r.action.toLowerCase()}</b> stance.`;
    }
    if (/(risk|drawdown|volatil|safe|lose)/.test(q))
      return `The portfolio's annualised volatility is <b>${fmtPct(k.volatility)}</b> with a worst peak-to-trough drawdown of <b>${fmtPct(k.max_drawdown)}</b> over the test period.`;
    if (/(beat|market|benchmark|buy.?and.?hold|outperform|better|sharpe|sortino|ratio)/.test(q))
      return `This screen shows the best of ${k.n_seeds} training seeds: it returned <b>${fmtSignedPct(k.total_return)}</b> vs <b>${fmtSignedPct(bh["Total Return"])}</b> for an equal-weight buy-and-hold, with a Sharpe ratio of <b>${k.sharpe.toFixed(2)}</b> vs <b>${bh.Sharpe.toFixed(2)}</b>. Across all ${k.n_seeds} seeds the average Sharpe is <b>${k.mean_sharpe_5seed.toFixed(2)}</b>, below buy-and-hold, so this is not a reliable edge.`;
    return `I'm QuantMind. Your portfolio is worth <b>${fmtMoney(k.portfolio_value)}</b> (<b>${fmtSignedPct(k.total_return)}</b>) and my biggest position is <b>${top.ticker}</b> at <b>${fmtPct(top.weight)}</b>. You can ask me about a company's weight, the performance, or the risk.`;
  };
  const answerFor = (qRaw) => {
    if (!SHOW_XAI) return answerForB(qRaw);
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
      return `This is the best of my ${k.n_seeds} training seeds: over the out-of-sample period it returned <b>${fmtSignedPct(k.total_return)}</b> vs <b>${fmtSignedPct(bh["Total Return"])}</b> for an equal-weight buy-and-hold, with a Sharpe ratio of <b>${k.sharpe.toFixed(2)}</b> vs <b>${bh.Sharpe.toFixed(2)}</b>. But across all ${k.n_seeds} seeds the average Sharpe is <b>${k.mean_sharpe_5seed.toFixed(2)}</b>, below buy-and-hold — so I don't have a reliable edge, only a good seed.`;
    if (/(how|explain|work|decide|why|driver|shap|feature)/.test(q))
      return `Each decision is explained with <b>Shapley-value attribution</b>: I measure how every market signal moves the allocation. Right now my decisions lean most on <b>position inertia</b> (avoiding needless trades), then <b>MACD trend</b> and <b>RSI</b> signals. Head to the Explainability tab for the full ranking.`;
    if (/(sharpe|sortino|ratio)/.test(q))
      return `This best-seed run has a Sharpe ratio of <b>${k.sharpe.toFixed(2)}</b> and Sortino of <b>${k.sortino.toFixed(2)}</b>, both above the buy-and-hold baseline (${bh.Sharpe.toFixed(2)} / ${bh.Sortino.toFixed(2)}). That is one seed, not the typical result: across all ${k.n_seeds} training seeds the average Sharpe is <b>${k.mean_sharpe_5seed.toFixed(2)}</b>, below buy-and-hold.`;
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
  // Prefer the live, audited backend (/api/chat) when a server is running;
  // fall back to the local rule-based answerFor() for the no-server static
  // build, so the two modes described in the report both work from this
  // one file.
  const send = (q) => {
    addMsg(q, "user");
    // Build B never calls the audited-rationale backend: it has no explanation to give.
    if (!SHOW_XAI) { setTimeout(() => addMsg(answerFor(q), "bot"), 320); return; }
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    })
      .then((r) => { if (!r.ok) throw new Error("no server"); return r.json(); })
      .then((data) => addMsg(data.answer, "bot"))
      .catch(() => setTimeout(() => addMsg(answerFor(q), "bot"), 320));
  };

  const SUGGEST = SHOW_XAI
    ? [`Why did you overweight ${top.ticker}?`, "Did you beat the market?", "How risky is this portfolio?", "How do you make decisions?"]
    : [`How much is in ${top.ticker}?`, "Did you beat the market?", "How risky is this portfolio?"];
  function bootChat() {
    addMsg(answerFor(""), "bot");
    document.getElementById("suggest").innerHTML = SUGGEST.map((s) => `<button>${s}</button>`).join("");
    document.querySelectorAll("#suggest button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
  }
  const input = document.getElementById("chat-input");
  const doSend = () => { const v = input.value.trim(); if (!v) return; send(v); input.value = ""; };
  document.getElementById("chat-send").addEventListener("click", doSend);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") doSend(); });


  // =========================================================================
  //  BUILD A PORTFOLIO (live path: POST /api/recommend, nothing retrained)
  // =========================================================================
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const bForm = document.getElementById("build-form");
  const bStatus = document.getElementById("build-status");
  const bResult = document.getElementById("build-result");
  document.getElementById("b-exclude").innerHTML = D.meta.tickers.map((t) =>
    `<label><input type="checkbox" value="${t}"> ${t}</label>`).join("");
  fetch("/api/recommend/range").then((r) => { if (!r.ok) throw new Error(); return r.json(); }).then((rg) => {
    const d = document.getElementById("b-date");
    d.min = rg.first; d.max = rg.last; d.value = rg.last;
    // ?autorun submits the form once with its defaults (used for screenshots and demos).
    if (new URLSearchParams(location.search).has("autorun")) bForm.requestSubmit();
  }).catch(() => {
    bStatus.textContent = "Live advice needs the QuantMind server. Start it with: uvicorn serve:app --port 8000";
    document.getElementById("b-go").disabled = true;
  });
  bForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn = document.getElementById("b-go");
    const body = {
      amount: parseFloat(document.getElementById("b-amount").value),
      as_of: document.getElementById("b-date").value || null,
      max_weight: parseFloat(document.getElementById("b-cap").value),
      exclude: [...document.querySelectorAll("#b-exclude input:checked")].map((c) => c.value),
      explanations: SHOW_XAI,
    };
    btn.disabled = true; bResult.innerHTML = ""; bStatus.textContent = "Working out your split…";
    fetch("/api/recommend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => r.json().then((j) => ({ ok: r.ok, j })))
      .then(({ ok, j }) => {
        if (!ok) throw new Error(typeof j.detail === "string" ? j.detail : "That request could not be used. Check the amount and date.");
        renderBuild(j); bStatus.textContent = "Done. Your suggested split is below.";
      })
      .catch((err) => { bStatus.textContent = err.message === "Failed to fetch" ? "Could not reach the server." : err.message; })
      .finally(() => { btn.disabled = false; });
  });
  function renderBuild(r) {
    const notes = [];
    if (r.date_note) notes.push(r.date_note);
    if (r.in_sample) notes.push("This date is inside the period the model was trained on, so the advice is not a fair test of it.");
    notes.push("This assumes you are starting from cash. " + r.notice);
    if (r.excluded.length) notes.push("Left out by you: " + r.excluded.join(", ") + ". Their share is held as cash.");
    const rows = r.holdings.map((h) => `
      <tr><td><b>${esc(h.ticker)}</b> <span class="muted">${esc(h.name)}</span></td>
          <td class="r num">${fmtPct(h.weight)}</td><td class="r num">${fmtMoney(h.amount)}</td>
          <td><span class="badge ${badgeClass(h.action)}">${esc(h.action)}</span></td></tr>
      ${SHOW_XAI && h.drivers && h.weight > 0 ? `<tr><td colspan="4" class="b-why">Why: ${h.drivers.map((d) =>
        `${esc(d.phrase)} ${d.direction === "up" ? "increased" : "reduced"} the weight`).join("; ")}.</td></tr>` : ""}`).join("");
    bResult.innerHTML = `
      ${notes.map((n) => `<p class="notice">${esc(n)}</p>`).join("")}
      <table class="b-table"><caption class="muted">Suggested split of ${fmtMoney(r.amount)} as of ${esc(r.as_of)}</caption>
        <thead><tr><th>Company</th><th class="r">Share</th><th class="r">Amount</th><th>Stance</th></tr></thead>
        <tbody>${rows}
          <tr><td><b>Cash</b></td><td class="r num">${fmtPct(r.cash.weight)}</td><td class="r num">${fmtMoney(r.cash.amount)}</td><td></td></tr>
        </tbody></table>
      ${SHOW_XAI && r.rationale ? `<p class="explain-note" style="margin-top:14px"><b>In one sentence:</b> ${esc(r.rationale.sentence)}</p>` : ""}`;
  }

    // Deep-link support: open directly to a view via #recommendations etc.
  // Placed last so all view helpers (incl. bootChat dependencies) are defined.
  if (location.hash) activateView(location.hash.slice(1));
})();
