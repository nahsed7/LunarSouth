// Runs the browser game's physics engine (static/mission_sim.html, <ENGINE> block) under Node for tests/test_mission_sim.py.
// usage: node tests/sim_engine_check.js <payload.json> <points.json>   -> prints JSON results
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "static", "mission_sim.html"), "utf8");
const m = html.match(/\/\/<ENGINE>([\s\S]*?)\/\/<\/ENGINE>/);
if (!m) throw new Error("engine block not found");
const module_ = { exports: {} };
new Function("module", "Buffer", m[1] + "\nmodule.exports = ENG;")(module_, Buffer);
const E = module_.exports;

const P = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const pts = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const T = E.makeTerrain(P.dem, P.far_hz);
const out = { profiles: [], soc: [], served: [], crit: 0 };
for (const [x, y] of pts) out.profiles.push(Array.from(E.horizonProfile(T, x, y, 1.5)));

// Outpost replay with no player overrides at the base step -> must equal energy.simulate (no activities).
const st = { e: P.outpost.battery.capacity_wh * P.outpost.battery.initial_soc };
const h = P.step_min / 60;
for (let i = 0; i < P.n; i++) {
  const loads = E.outpostLoads(P, i * h, P.plan.sunlit[i] === 1, P.s.site_earth[i] === 1, null, null);
  const r = E.batteryStep(P.outpost.battery, st, P.s.out_gen[i] || 0, loads, h);
  out.soc.push(r.soc * 100);
  out.served.push(r.served.reduce((a, b) => a + b, 0));
  if (r.crit) out.crit++;
}
// drive power must match activities.Traverse.drive_w
out.drive = [0, 5, 15].map((s) => E.driveW(P.rover, 0.1, s * E.D2R));
// LOS sanity: outpost to itself is clear
out.los_self = E.losClear(T, 0, 0, 3, 10, 0, 1.5);
process.stdout.write(JSON.stringify(out));
