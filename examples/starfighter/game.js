/* cu-Jev Starfighter — a lane shooter flown by typed decisions.
 *
 * Every tick the game writes what it sees as plain text, POSTs it to
 * /v1/systemone with four questions and applies the answers through visible
 * game rules (one lane per tick, gun cooldown, shield charge). The model never
 * generates text; it picks options. Everything the pilot "knows" is in the
 * state string shown on the right. */
(() => {
  const qs = new URLSearchParams(location.search);
  const API = (qs.get("api") || "").replace(/\/$/, "");
  const MODEL = qs.get("model") || "jev-latest";
  const cv = document.getElementById("game"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, LANES = 5, LANE_W = W / LANES;
  const laneX = i => LANE_W * i + LANE_W / 2;
  const UNIT = 8; // px per "unit" in the state text
  const $ = id => document.getElementById(id);
  const rnd = (a, b) => a + Math.random() * (b - a);
  const NAME = { asteroid: "asteroid", belt: "asteroid wall", enemy: "enemy fighter", ebullet: "enemy bullet", powerup: "shield pickup", credit: "credits" };

  // ---------------------------------------------------------------- state
  let g;
  function reset() {
    g = {
      t: 0, score: 0, running: false, over: false, overAt: 0, wave: 1, shake: 0,
      ship: { lane: 2, target: 2, x: laneX(2), y: H - 70, hp: 3, shield: 1, shieldUntil: 0, gunAt: 0, tilt: 0, moveAt: 0 },
      ents: [], parts: [], pops: [], stars: [], nextSpawn: 700, nextBelt: 5000, nextCredits: 1500,
      decisions: 0, decT0: 0, lastLat: 0, lastTok: 0, kills: 0, credits: 0,
    };
    for (let i = 0; i < 140; i++) g.stars.push({ x: Math.random() * W, y: Math.random() * H, z: 0.3 + Math.random() * 0.7 });
  }
  reset();

  function pop(x, y, text, color) { g.pops.push({ x, y, text, color, life: 900 }); }
  function boom(x, y, color, n = 18) { for (let i = 0; i < n; i++) g.parts.push({ x, y, vx: rnd(-170, 170), vy: rnd(-170, 170), life: rnd(300, 700), color }); }

  function spawn() {
    const r = Math.random(), lane = Math.floor(Math.random() * LANES), pace = 1 + g.t / 90000;
    if (r < 0.35) g.ents.push({ type: "asteroid", lane, y: -30, hp: 2, vy: rnd(95, 135) * pace, r: 22, rot: 0, vr: rnd(-2, 2) });
    else if (r < 0.85) g.ents.push({ type: "enemy", lane, y: -30, hp: 1, vy: rnd(70, 110) * pace, r: 18, fireAt: g.t + rnd(700, 1500), driftAt: g.t + rnd(900, 2200) });
    else g.ents.push({ type: "powerup", lane, y: -20, hp: 1, vy: 85, r: 14 });
  }
  function spawnBelt() {
    const gap = Math.floor(Math.random() * LANES), pace = 1 + g.t / 90000;
    for (let l = 0; l < LANES; l++) if (l !== gap) g.ents.push({ type: "belt", lane: l, y: -40, hp: 3, vy: 95 * pace, r: 26, rot: rnd(0, 6), vr: rnd(-1, 1) });
    g.wave++;
  }
  function spawnCredits() {
    const lane = Math.floor(Math.random() * LANES), n = 3 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) g.ents.push({ type: "credit", lane, y: -20 - i * 34, hp: 1, vy: 110, r: 9 });
  }

  // ---------------------------------------------------------------- rules
  function act(targetLane, fire, shield) {
    const s = g.ship;
    if (targetLane !== null && targetLane >= 0 && targetLane < LANES) s.target = targetLane;
    if (fire && g.t >= s.gunAt) { s.gunAt = g.t + 240; g.ents.push({ type: "bullet", lane: s.lane, x: s.x, y: s.y - 24, vy: -560, r: 4 }); }
    if (shield && s.shield >= 1 && g.t >= s.shieldUntil) { s.shieldUntil = g.t + 1500; s.shield = 0; pop(s.x, s.y - 40, "SHIELD", "#54c7b6"); }
  }

  function update(dt) {
    const s = g.ship;
    g.t += dt;
    if (g.over) { if (g.t - g.overAt > 2500) { const keep = g.stars; reset(); g.stars = keep; g.running = true; startDecisions(); } return; }
    g.score += dt / 1000;
    // one lane per 140 ms toward the target
    if (s.target !== s.lane && g.t >= s.moveAt) { s.lane += Math.sign(s.target - s.lane); s.moveAt = g.t + 140; }
    s.x += (laneX(s.lane) - s.x) * Math.min(1, dt / 80);
    s.tilt = (laneX(s.lane) - s.x) / LANE_W;
    if (g.t >= s.shieldUntil) s.shield = Math.min(1, s.shield + dt / 8000);
    g.shake = Math.max(0, g.shake - dt);
    g.nextSpawn -= dt; g.nextBelt -= dt; g.nextCredits -= dt;
    if (g.nextSpawn <= 0) { spawn(); g.nextSpawn = Math.max(420, 950 - g.t / 70); }
    if (g.nextBelt <= 0) { spawnBelt(); g.nextBelt = Math.max(5000, 8500 - g.t / 40); }
    if (g.nextCredits <= 0) { spawnCredits(); g.nextCredits = rnd(2200, 3800); }
    for (const e of g.ents) {
      e.y += e.vy * dt / 1000;
      if (e.type !== "bullet") e.x += (laneX(e.lane) - e.x) * Math.min(1, dt / 120) || 0;
      if (e.x === undefined || Number.isNaN(e.x)) e.x = laneX(e.lane);
      if (e.rot !== undefined) e.rot += e.vr * dt / 1000;
      if (e.type === "enemy") {
        if (g.t >= e.fireAt && e.y < s.y - 60) { e.fireAt = g.t + rnd(900, 1700); g.ents.push({ type: "ebullet", lane: e.lane, x: e.x, y: e.y + 16, vy: 260, r: 4 }); }
        if (g.t >= e.driftAt && e.y < s.y - 55 * UNIT) { e.driftAt = g.t + rnd(1200, 2500); e.lane = Math.max(0, Math.min(LANES - 1, e.lane + (Math.random() < 0.5 ? -1 : 1))); }
      }
    }
    for (const b of g.ents) if (b.type === "bullet") for (const e of g.ents) {
      if ((e.type === "asteroid" || e.type === "enemy" || e.type === "belt") && e.lane === b.lane && !e.dead && Math.abs(e.y - b.y) < e.r + 6) {
        b.dead = true; e.hp--; boom(e.x, e.y, e.type === "enemy" ? "#ff5470" : "#b8b8c8", 6);
        if (e.hp <= 0) { e.dead = true; const v = e.type === "enemy" ? 100 : 50; g.score += v; g.kills++; pop(e.x, e.y, `+${v}`, "#f4aa42"); boom(e.x, e.y, e.type === "enemy" ? "#ff5470" : "#b8b8c8", 24); }
        break;
      }
    }
    for (const e of g.ents) {
      if (e.dead || e.type === "bullet") continue;
      if (!(e.lane === s.lane && Math.abs(e.y - s.y) < e.r + 16 && Math.abs(e.x - s.x) < LANE_W * 0.6)) continue;
      e.dead = true;
      if (e.type === "credit") { g.score += 20; g.credits++; pop(e.x, e.y, "+20", "#ffe27a"); continue; }
      if (e.type === "powerup") { g.score += 25; s.shield = 1; pop(e.x, e.y, "shield", "#54c7b6"); boom(e.x, e.y, "#54c7b6", 16); continue; }
      boom(s.x, s.y, "#f4aa42", 30); g.shake = 350;
      if (g.t < s.shieldUntil) { pop(s.x, s.y - 40, "absorbed", "#54c7b6"); continue; }
      s.hp--; pop(s.x, s.y - 40, "HULL HIT", "#ff5470");
      (window.__hits = window.__hits || []).push({ t: Math.round(g.t), by: e.type, lane: s.lane + 1, target: s.target + 1, transit: s.target !== s.lane, state: $("state").textContent });
      if (s.hp <= 0) { g.over = true; g.overAt = g.t; boom(s.x, s.y, "#ffffff", 70); }
    }
    g.ents = g.ents.filter(e => !e.dead && e.y < H + 40 && e.y > -240);
    for (const p of g.parts) { p.x += p.vx * dt / 1000; p.y += p.vy * dt / 1000; p.life -= dt; }
    g.parts = g.parts.filter(p => p.life > 0);
    for (const p of g.pops) { p.y -= 30 * dt / 1000; p.life -= dt; }
    g.pops = g.pops.filter(p => p.life > 0);
    for (const st of g.stars) { st.y += 45 * st.z * dt / 1000; if (st.y > H) { st.y = 0; st.x = Math.random() * W; } }
  }

  // ---------------------------------------------------------------- the state the pilot sees
  function laneItems(l) {
    const s = g.ship;
    return g.ents.filter(e => e.lane === l && e.type !== "bullet" && e.y < s.y + 24)
      .map(e => ({ e, d: Math.max(0, Math.round((s.y - e.y) / UNIT)) })).sort((a, b) => a.d - b.d);
  }
  function laneDesc(l) {
    const items = laneItems(l);
    if (!items.length) return "clear (safe)";
    const danger = items.find(o => o.e.type !== "powerup" && o.e.type !== "credit");
    const bonus = items.filter(o => o.e.type === "credit").length;
    const parts = [];
    const seen = new Set();
    for (const o of items.slice(0, 4)) {
      if (o.e.type === "credit") { if (!seen.has("credit")) { parts.push(`${bonus} credits ${o.d}u (bonus)`); seen.add("credit"); } continue; }
      parts.push(o.d === 0 ? `${NAME[o.e.type]} beside you` : `${NAME[o.e.type]} ${o.d}u`);
    }
    let tag;
    if (!danger) tag = " (safe)";
    else if (danger.d < 28) tag = " (DANGER: hit within a second)";
    else if (danger.d < 60) tag = " (risky)";
    else tag = danger.e.type === "enemy" ? " (far, shootable)" : " (far)";
    return parts.join(", ") + tag;
  }
  function buildRequest() {
    const s = g.ship, L = s.lane + 1;
    const shield = g.t < s.shieldUntil ? "ACTIVE" : s.shield >= 1 ? "ready" : `charging (${Math.round(s.shield * 100)}%)`;
    const gun = g.t >= s.gunAt ? "ready" : "cooling";
    const lines = [`Ship: lane ${L} of ${LANES}, hull ${s.hp}/3, shield ${shield}, gun ${gun}. The ship moves one lane per tick toward the chosen lane; objects only hit the ship in their own lane.`];
    for (let l = 0; l < LANES; l++) lines.push(`Lane ${l + 1}${l === s.lane ? " (yours)" : ""}: ${laneDesc(l)}.`);
    lines.push(`Pilot orders: ${$("policy").value.trim()}`);
    const state = lines.join("\n");
    const crit = {};
    for (let l = Math.max(0, s.lane - 1); l <= Math.min(LANES - 1, s.lane + 1); l++)
      crit[`lane ${l + 1}`] = laneDesc(l) + (l === s.lane ? " [stay]" : l < s.lane ? " [one step left]" : " [one step right]");
    const own = laneDesc(s.lane);
    const questions = {
      move: { type: "choice", instructions: "Which lane for the next moment: stay, or step one lane left or right? Never a DANGER lane; leave a risky lane if a neighbour is better; then prefer credits, then a far shootable fighter, then clear.", criteria: crit },
      fire: { type: "noul", instructions: `Is an enemy fighter, asteroid or asteroid wall ahead in the ship's own lane ${L}?`,
              criteria: { true: "something shootable is ahead in the ship's lane", false: "nothing to shoot in the ship's lane" } },
      shield: { type: "noul", instructions: `Does the state tag the ship's own lane ${L} with the word DANGER? Only DANGER counts; "risky", "far" and "safe" do not.`,
                criteria: { true: "the ship's lane is tagged DANGER: a hit is unavoidable", false: "the ship's lane is safe, risky or far: keep the shield charged" } },
      threat: { type: "score", instructions: "How dangerous is the ship's situation?",
                criteria: ["calm: everything clear or far", "caution: risky objects in other lanes", "danger: own lane risky", "critical: own lane DANGER or hull 1"] },
    };
    return { state, questions, own, body: { model: MODEL, state, questions } };
  }

  // ---------------------------------------------------------------- decision loop
  let deciding = false;
  async function startDecisions() {
    if (deciding) return; deciding = true;
    while (g.running && !$("human").checked) {
      if (g.over) { await new Promise(r => setTimeout(r, 100)); continue; }
      const req = buildRequest();
      $("state").textContent = req.state;
      const t0 = performance.now();
      let res;
      try {
        const r = await fetch(`${API}/v1/systemone`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req.body) });
        if (!r.ok) { $("log").textContent = `HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`; await new Promise(x => setTimeout(x, 500)); continue; }
        res = await r.json();
      } catch (e) { $("log").textContent = `cannot reach cu-Jev at ${API || location.origin}: ${e}`; await new Promise(x => setTimeout(x, 800)); continue; }
      $("log").textContent = "";
      const lat = performance.now() - t0;
      const a = res.answers;
      const lane = parseInt(a.move.choice.replace("lane ", ""), 10) - 1;
      act(lane, a.fire.noul > 0.5, a.shield.noul > 0.5);
      g.decisions++; g.lastLat = lat; g.lastTok = res.usage.input_tokens;
      if (!g.decT0) g.decT0 = performance.now();
      renderPanel(a, res.model, lat);
    }
    deciding = false;
  }

  function bars(id, rows) {
    const el = $(id).querySelector(".bars");
    const best = Math.max(...rows.map(r => r.p));
    el.innerHTML = rows.map(r => `<div class="bar${r.p === best ? " win" : ""}"><span class="lab">${r.k}</span><span class="tr"><span class="fl" style="width:${(r.p * 100).toFixed(1)}%"></span></span><span class="v">${(r.p * 100).toFixed(0)}%</span></div>`).join("");
  }
  function renderPanel(a, model, lat) {
    bars("q-move", Object.entries(a.move.probabilities).map(([k, p]) => ({ k, p })));
    bars("q-fire", [{ k: "yes", p: a.fire.noul }, { k: "no", p: 1 - a.fire.noul }]);
    bars("q-shield", [{ k: "yes", p: a.shield.noul }, { k: "no", p: 1 - a.shield.noul }]);
    bars("q-threat", Object.entries(a.threat.probabilities).map(([k, p]) => ({ k: a.threat.legend[k].split(":")[0], p })));
    $("threat-meter").querySelector("i").style.width = `${(a.threat.score / 3 * 100).toFixed(0)}%`;
    $("lat").textContent = lat.toFixed(0);
    $("dps").textContent = g.decT0 ? (g.decisions / ((performance.now() - g.decT0) / 1000)).toFixed(1) : "–";
    $("tok").textContent = g.lastTok;
    $("model").textContent = model.replace("cujev/", "");
  }

  // ---------------------------------------------------------------- render
  function drawShip(s) {
    ctx.save(); ctx.translate(s.x, s.y); ctx.rotate(s.tilt * 0.35);
    if (g.t < s.shieldUntil) { ctx.beginPath(); ctx.arc(0, -4, 30, 0, Math.PI * 2); ctx.strokeStyle = "rgba(84,199,182,.9)"; ctx.lineWidth = 3; ctx.shadowColor = "#54c7b6"; ctx.shadowBlur = 18; ctx.stroke(); ctx.shadowBlur = 0; }
    const grad = ctx.createLinearGradient(0, -22, 0, 18); grad.addColorStop(0, "#f4f6ff"); grad.addColorStop(1, "#7f8cff");
    ctx.fillStyle = grad; ctx.shadowColor = "#7f8cff"; ctx.shadowBlur = 16;
    ctx.beginPath(); ctx.moveTo(0, -24); ctx.lineTo(16, 16); ctx.lineTo(6, 10); ctx.lineTo(0, 16); ctx.lineTo(-6, 10); ctx.lineTo(-16, 16); ctx.closePath(); ctx.fill();
    ctx.shadowBlur = 0;
    const fl = 10 + Math.random() * 10; ctx.fillStyle = "rgba(244,170,66,.9)"; ctx.beginPath(); ctx.moveTo(-5, 14); ctx.lineTo(0, 14 + fl); ctx.lineTo(5, 14); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  function rock(e, base) {
    ctx.rotate(e.rot); ctx.fillStyle = base; ctx.shadowColor = "#000"; ctx.shadowBlur = 8;
    ctx.beginPath(); for (let i = 0; i < 8; i++) { const a = i / 8 * Math.PI * 2, r = e.r * (0.8 + ((i * 7) % 3) * 0.12); ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r); } ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#6f7285"; ctx.beginPath(); ctx.arc(-5, -3, 5, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(7, 6, 3, 0, 7); ctx.fill();
  }
  function drawEnt(e) {
    ctx.save(); ctx.translate(e.x, e.y);
    if (e.type === "asteroid") rock(e, e.hp === 2 ? "#a7a9b8" : "#d7d9e8");
    else if (e.type === "belt") rock(e, e.hp === 3 ? "#8c6f5a" : e.hp === 2 ? "#a8886f" : "#c9a98c");
    else if (e.type === "enemy") {
      ctx.fillStyle = "#ff5470"; ctx.shadowColor = "#ff5470"; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.moveTo(0, 18); ctx.lineTo(18, -12); ctx.lineTo(6, -6); ctx.lineTo(0, -14); ctx.lineTo(-6, -6); ctx.lineTo(-18, -12); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(0, 2, 3, 0, 7); ctx.fill();
    } else if (e.type === "powerup") {
      const pul = 1 + 0.15 * Math.sin(g.t / 120); ctx.scale(pul, pul);
      ctx.strokeStyle = "#54c7b6"; ctx.lineWidth = 3; ctx.shadowColor = "#54c7b6"; ctx.shadowBlur = 16;
      ctx.beginPath(); for (let i = 0; i < 6; i++) { const a = i / 6 * Math.PI * 2; ctx.lineTo(Math.cos(a) * 13, Math.sin(a) * 13); } ctx.closePath(); ctx.stroke();
      ctx.fillStyle = "#54c7b6"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center"; ctx.fillText("S", 0, 4);
    } else if (e.type === "credit") {
      const w = Math.abs(Math.cos(g.t / 200 + e.y / 40)); ctx.scale(Math.max(0.2, w), 1);
      ctx.fillStyle = "#ffe27a"; ctx.shadowColor = "#ffd23f"; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(0, 0, 9, 0, 7); ctx.fill();
      ctx.fillStyle = "#b8860b"; ctx.beginPath(); ctx.arc(0, 0, 5, 0, 7); ctx.fill();
    } else if (e.type === "bullet") {
      ctx.fillStyle = "#f4aa42"; ctx.shadowColor = "#f4aa42"; ctx.shadowBlur = 10; ctx.fillRect(-2, -10, 4, 16);
    } else if (e.type === "ebullet") {
      ctx.fillStyle = "#ff5470"; ctx.shadowColor = "#ff5470"; ctx.shadowBlur = 10; ctx.beginPath(); ctx.arc(0, 0, 4, 0, 7); ctx.fill();
    }
    ctx.restore();
  }
  function render() {
    ctx.save();
    if (g.shake > 0) ctx.translate(rnd(-4, 4) * g.shake / 350, rnd(-4, 4) * g.shake / 350);
    const bg = ctx.createLinearGradient(0, 0, W, H); bg.addColorStop(0, "#050716"); bg.addColorStop(0.5, "#0a0620"); bg.addColorStop(1, "#04060f");
    ctx.fillStyle = bg; ctx.fillRect(-10, -10, W + 20, H + 20);
    const neb = ctx.createRadialGradient(W * 0.7, H * 0.3 + Math.sin(g.t / 4000) * 40, 20, W * 0.7, H * 0.3, 300); neb.addColorStop(0, "rgba(120,80,255,.12)"); neb.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = neb; ctx.fillRect(0, 0, W, H);
    for (const st of g.stars) { ctx.fillStyle = `rgba(255,255,255,${0.25 + st.z * 0.6})`; ctx.fillRect(st.x, st.y, st.z * 2, st.z * 2); }
    for (let i = 1; i < LANES; i++) { ctx.strokeStyle = "rgba(120,140,255,.08)"; ctx.beginPath(); ctx.moveTo(LANE_W * i, 0); ctx.lineTo(LANE_W * i, H); ctx.stroke(); }
    ctx.fillStyle = "rgba(84,199,182,.06)"; ctx.fillRect(LANE_W * g.ship.lane, 0, LANE_W, H);
    if (g.ship.target !== g.ship.lane) { ctx.strokeStyle = "rgba(244,170,66,.5)"; ctx.setLineDash([6, 6]); ctx.strokeRect(LANE_W * g.ship.target + 4, 4, LANE_W - 8, H - 8); ctx.setLineDash([]); }
    ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = "rgba(138,147,184,.6)"; ctx.textAlign = "center";
    for (let i = 0; i < LANES; i++) ctx.fillText(`lane ${i + 1}`, laneX(i), 16);
    for (const e of g.ents) drawEnt(e);
    if (!g.over) drawShip(g.ship);
    for (const p of g.parts) { ctx.globalAlpha = Math.max(0, p.life / 700); ctx.fillStyle = p.color; ctx.fillRect(p.x, p.y, 3, 3); }
    ctx.globalAlpha = 1;
    for (const p of g.pops) { ctx.globalAlpha = Math.min(1, p.life / 400); ctx.fillStyle = p.color; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "center"; ctx.fillText(p.text, p.x, p.y); }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#161b33"; ctx.fillRect(12, H - 18, 100, 6); ctx.fillStyle = g.t < g.ship.shieldUntil ? "#f4aa42" : "#54c7b6"; ctx.fillRect(12, H - 18, 100 * g.ship.shield, 6);
    ctx.fillStyle = "#8a93b8"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText("shield", 12, H - 24);
    ctx.textAlign = "right"; ctx.fillText(`wave ${g.wave} · ${g.kills} kills · ${g.credits} credits`, W - 12, H - 24);
    if (g.over) { ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.font = "bold 34px sans-serif"; ctx.textAlign = "center"; ctx.fillText("HULL LOST", W / 2, H / 2 - 10); ctx.font = "16px sans-serif"; ctx.fillStyle = "#8a93b8"; ctx.fillText(`score ${Math.floor(g.score)} · ${g.kills} kills · restarting`, W / 2, H / 2 + 24); }
    else if (!g.running) { ctx.fillStyle = "rgba(0,0,0,.5)"; ctx.fillRect(0, 0, W, H); ctx.fillStyle = "#fff"; ctx.font = "bold 28px sans-serif"; ctx.textAlign = "center"; ctx.fillText("cu-Jev Starfighter", W / 2, H / 2 - 10); ctx.font = "14px sans-serif"; ctx.fillStyle = "#8a93b8"; ctx.fillText("press Start — the model flies", W / 2, H / 2 + 20); }
    ctx.restore();
    $("score").textContent = `SCORE ${Math.floor(g.score)}`;
    $("hp").textContent = "♥".repeat(Math.max(0, g.ship.hp)) + "♡".repeat(3 - Math.max(0, g.ship.hp));
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(50, now - last); last = now;
    if (g.running) update(dt);
    render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // ---------------------------------------------------------------- controls
  $("start").onclick = () => { g.running = true; $("start").disabled = true; $("pause").disabled = false; startDecisions(); };
  $("pause").onclick = () => { g.running = false; $("start").disabled = false; $("pause").disabled = true; };
  $("human").onchange = () => { if (!$("human").checked) startDecisions(); };
  window.addEventListener("keydown", e => {
    if (!$("human").checked || !g.running) return;
    if (e.key === "ArrowLeft") act(Math.max(0, g.ship.lane - 1), false, false);
    if (e.key === "ArrowRight") act(Math.min(LANES - 1, g.ship.lane + 1), false, false);
    if (e.key === " ") { act(null, true, false); e.preventDefault(); }
    if (e.key === "Shift") act(null, false, true);
  });
  if (qs.get("policy")) $("policy").value = qs.get("policy");
  if (qs.get("autostart")) setTimeout(() => $("start").click(), 300);
})();
