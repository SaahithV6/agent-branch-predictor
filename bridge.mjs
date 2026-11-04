import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createInterface } from "node:readline";
import { homedir } from "node:os";
import { callSim, simReadOnly, simTools, world } from "./sim.mjs";
import { callReal, entities as realEntities, realReadOnly, realTools } from "./real.mjs";
import { existsSync } from "node:fs";
const BIZ = process.env.BIZ === "1";
const { bizTools, bizReadOnly, callBiz } = BIZ ? await import("./biz.mjs") : { bizTools: [], bizReadOnly: new Set(), callBiz: async () => null };
const RESEARCH = process.env.RESEARCH === "1";
const { researchTools, researchReadOnly, callResearch } = RESEARCH ? await import("./research.mjs") : { researchTools: [], researchReadOnly: new Set(), callResearch: async () => null };

const BIN = process.env.NATIVE_BIN ?? `${homedir()}/native/target/release/mcp-stdio`;
const DB = process.env.NATIVE_DB ?? new URL("./native.db", import.meta.url).pathname;
const PORT = Number(process.env.BRIDGE_PORT ?? 8787);

const child = spawn(BIN, [DB], {
  env: { ...process.env, NATIVE_CE_MCP_SURFACE: "legacy", NATIVE_CE_MCP_TOOL_PROFILE: "focused" },
  stdio: ["pipe", "pipe", "inherit"],
});
child.on("exit", (code) => {
  console.error(`native exited (${code})`);
  process.exit(1);
});
const lines = createInterface({ input: child.stdout });
const waiting = new Map();
lines.on("line", (line) => {
  const msg = JSON.parse(line);
  waiting.get(msg.id)?.(msg);
  waiting.delete(msg.id);
});

let nextId = 0;
let queue = Promise.resolve();
function rpc(method, params) {
  const run = () =>
    new Promise((resolve) => {
      const id = ++nextId;
      waiting.set(id, resolve);
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  const p = queue.then(run);
  queue = p.catch(() => {});
  return p;
}

async function call(name, args) {
  const msg = await rpc("tools/call", { name, arguments: { run_key: runKey, ...args } });
  if (msg.error) throw new Error(msg.error.message);
  return msg.result;
}

async function json(name, args) {
  const r = await call(name, { ...args, format: "json" });
  if (r.isError) throw new Error(r.content?.[0]?.text ?? `${name} failed`);
  return r.structuredContent ?? JSON.parse(r.content[0].text);
}

await rpc("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "qm-bridge", version: "0" } });
child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
let runKey;
const boot = await call("bootstrap", { run_key: "new:qm", format: "json" });
runKey = (boot.structuredContent ?? JSON.parse(boot.content[0].text)).session.run_key;
const ZOD_UNSUPPORTED = new Set(["not", "dependentSchemas", "dependentRequired"]);
const zodSafe = (v) =>
  Array.isArray(v)
    ? v.map(zodSafe)
    : v && typeof v === "object"
      ? Object.fromEntries(Object.entries(v).filter(([k]) => !ZOD_UNSUPPORTED.has(k)).map(([k, x]) => [k, zodSafe(x)]))
      : v;
const nativeTools = (await rpc("tools/list", {})).result.tools.map((t) => {
  const { oneOf, anyOf, allOf, ...inputSchema } = zodSafe(t.inputSchema);
  return { ...t, inputSchema };
});

const folders = new Map();
async function folderFor(scope) {
  if (folders.has(scope)) return folders.get(scope);
  const found = await json("query_record", {
    steps: [{ step: "filter", kinds: ["folder"], facets: [{ key: "qm_scope_id", eq: scope }] }],
  });
  const hit = (found.items ?? found.records ?? found.results ?? [])[0];
  const id =
    hit?.id ??
    (
      await json("create_record", {
        type: "Collection",
        kind: "folder",
        name: `QM ${scope}`,
        facets: { qm_scope_id: scope },
        reason: `Memory folder for QM scope ${scope}`,
        sources: [],
      })
    ).id;
  folders.set(scope, id);
  return id;
}

const STOP = new Set("the and for with that this from have what your about into when were will would there their them they then than which while".split(" "));

async function recall({ query = "", scope = "org", limit = 20 }) {
  const home = await folderFor(scope);
  const terms = [...new Set(query.toLowerCase().match(/[a-z0-9][a-z0-9._-]{2,}/g) ?? [])].filter((t) => !STOP.has(t)).slice(0, 8);
  const score = new Map();
  if (terms.length) {
    for (const term of terms) {
      const r = await json("search", { query: term, scope: home, limit: 50 });
      for (const h of r.hits ?? []) {
        if (h.superseded_by?.total_count) continue;
        const cur = score.get(h.id) ?? { n: 0, text: h.name };
        cur.n++;
        score.set(h.id, cur);
      }
    }
  } else {
    const r = await json("query_record", { steps: [{ step: "filter", kinds: ["fact"], facets: [{ key: "qm_scope_id", eq: scope }] }] });
    for (const h of r.items ?? r.records ?? r.results ?? []) score.set(h.id, { n: 1, text: h.name });
  }
  return [...score.values()].sort((a, b) => b.n - a.n).slice(0, limit).map((h) => `- ${h.text}`).join("\n");
}

async function remember({ content = "", scope = "org", acting_user = "unknown", source = "explicit" }) {
  const home = await folderFor(scope);
  const facts = content.split("\n").map((l) => l.replace(/^\s*[-*]\s*/, "").trim()).filter(Boolean);
  for (const fact of facts)
    await json("create_record", {
      type: "Document",
      kind: "fact",
      name: fact.slice(0, 200),
      body: fact,
      home_id: home,
      facets: { qm_scope_id: scope, qm_actor: acting_user },
      reason: `QM ${source} memory from ${acting_user}`,
      sources: [],
    });
  return `stored ${facts.length} fact(s)`;
}

let PREDICTOR = process.env.PREDICTOR ?? "speculate";
const THETA = Number(process.env.PREDICTOR_THETA ?? 0.7);
const LIVE_WAIT_MS = Number(process.env.LIVE_WAIT_MS ?? 1500);
const TAU = Number(process.env.PREDICTOR_TAU ?? 0.5);
const SIM = process.env.SIM === "1";
const REAL = process.env.REAL === "1";
const READ_ONLY = new Set([...(SIM ? simReadOnly : []), ...(REAL ? realReadOnly : []), ...bizReadOnly, ...researchReadOnly, "get_structure", "read_guide", "get_record", "get_history", "whats_changed", "resolve_facets", "query_record", "resolve_rollup", "search", "scan", "read_attachment"]);
const IGNORED_ARGS = new Set(["run_key", "format", "acting_user"]);
const clean = (args) => Object.fromEntries(Object.entries(args).filter(([k]) => !IGNORED_ARGS.has(k)));
const lowerStrings = (v) => (typeof v === "string" ? v.toLowerCase() : Array.isArray(v) ? v.map(lowerStrings) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, lowerStrings(x)])) : v);
const callKey = (name, args) => {
  const a = lowerStrings(clean(args));
  return name + JSON.stringify(a, Object.keys(a).sort());
};
const words = (q) => new Set((q.toLowerCase().match(/[a-z0-9][a-z0-9._-]{2,}/g) ?? []).filter((t) => !STOP.has(t)));
const jaccard = (a, b) => {
  const inter = [...a].filter((x) => b.has(x)).length;
  return inter / (a.size + b.size - inter || 1);
};
const stats = { turns: 0, predicted: 0, speculated: 0, hits: 0, readOnlyCalls: 0, calls: 0, savedMs: 0, squashed: 0, injected: 0, chainSpeculated: 0, chainHits: 0, chainSavedMs: 0 };
const history = [];
const enumsByArg = new Map();
let turn = null;
const timelines = [];
let idle;

function indexEnums(tools) {
  for (const t of tools)
    for (const [k, v] of Object.entries(t.inputSchema?.properties ?? {}))
      if (Array.isArray(v.enum)) enumsByArg.set(k, new Set([...(enumsByArg.get(k) ?? []), ...v.enum.filter((x) => typeof x === "string")]));
}

function entityOf(query) {
  const used = new Set(history.flatMap((h) => h.calls.flatMap((c) => Object.keys(c.args))));
  const seenValues = history.flatMap((h) => h.calls.flatMap((c) => Object.values(c.args).filter((v) => typeof v === "string" && v.length < 60)));
  const canonical = new Map();
  for (const v of [...(REAL ? realEntities : []), ...seenValues, ...skillTools.flatMap((t) => t.inputSchema.properties[entityParam(t)]?.enum ?? []), ...[...enumsByArg].filter(([k]) => used.has(k)).flatMap(([, v]) => [...v])])
    canonical.set(v.toLowerCase(), v);
  const q = ` ${query.toLowerCase().replace(/[^a-z0-9_-]+/g, " ")} `;
  const hit = [...canonical.keys()].filter((v) => v.length > 2 && q.includes(` ${v.replace(/[^a-z0-9_-]+/g, " ").trim()} `)).sort((a, b) => b.length - a.length)[0];
  return hit ? canonical.get(hit) : null;
}

const mapStrings = (v, f) => (typeof v === "string" ? f(v) : Array.isArray(v) ? v.map((x) => mapStrings(x, f)) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, mapStrings(x, f)])) : v);
const slotArgs = (args, e) => (e ? mapStrings(args, (v) => (v.toLowerCase() === e.toLowerCase() ? "$E" : v)) : args);
const fillArgs = (args, e) => (e ? mapStrings(args, (v) => (v === "$E" ? e : v)) : args);

function learn(query, calls) {
  const e = entityOf(query);
  const w = words(query);
  if (e) for (const part of e.toLowerCase().split(/[^a-z0-9_-]+/)) w.delete(part);
  const reads = calls.filter((c) => READ_ONLY.has(c.name));
  history.push({ query, calls, w, templates: new Map(reads.map((c) => [callKey(c.name, slotArgs(c.args, e)), { name: c.name, args: slotArgs(c.args, e) }])) });
}

const CLASSIFIER = process.env.CLASSIFIER ?? `${homedir()}/qm-native/call-classifier.pkl`;
const CLASSIFIER_THETA = Number(process.env.CLASSIFIER_THETA ?? 0.5);
const skillTools = [...(SIM ? simTools : []), ...(REAL ? realTools : []), ...bizTools, ...researchTools];
const entityParam = (t) => t.inputSchema?.required?.[0];
const skillEnums = new Set(skillTools.flatMap((t) => t.inputSchema.properties[entityParam(t)]?.enum ?? []).map((v) => v.toLowerCase()));
let classifier = null;
if (skillTools.length && existsSync(CLASSIFIER)) {
  const proc = spawn(`${homedir()}/qm-native/.venv/bin/python`, [`${homedir()}/qm-native/classify.py`, CLASSIFIER], { stdio: ["pipe", "pipe", "inherit"] });
  const out = createInterface({ input: proc.stdout });
  const queue = [];
  out.on("line", (line) => queue.shift()?.(JSON.parse(line).probs));
  let chain = Promise.resolve();
  classifier = (text) => {
    const p = chain.then(() => new Promise((resolve) => (queue.push(resolve), proc.stdin.write(JSON.stringify({ text }) + "\n"))));
    chain = p.catch(() => {});
    return p;
  };
}

function applicable(tool, e) {
  const values = tool.inputSchema.properties[entityParam(tool)]?.enum;
  return values ? values.map((v) => v.toLowerCase()).includes(e.toLowerCase()) : !skillEnums.has(e.toLowerCase());
}

function toolArgTemplates() {
  const counts = new Map();
  for (const h of history)
    for (const raw of h.templates.values()) {
      const tool = skillTools.find((x) => x.name === raw.name);
      const param = tool && entityParam(tool);
      const t = param && param in raw.args ? { ...raw, args: { ...raw.args, [param]: "$E" } } : raw;
      const byTool = counts.get(t.name) ?? new Map();
      const k = JSON.stringify(t.args, Object.keys(t.args).sort());
      byTool.set(k, (byTool.get(k) ?? 0) + 1);
      counts.set(t.name, byTool);
    }
  return new Map([...counts].map(([name, m]) => [name, JSON.parse([...m].sort((a, b) => b[1] - a[1])[0][0])]));
}

async function classify(query, e) {
  if (!classifier || !e) return [];
  const probs = await classifier(query.replace(new RegExp(e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig"), " ENTITY "));
  const templates = toolArgTemplates();
  return skillTools
    .filter((t) => READ_ONLY.has(t.name) && (probs[t.name] ?? 0) >= CLASSIFIER_THETA && applicable(t, e))
    .map((t) => {
      const args = fillArgs(templates.get(t.name) ?? { [entityParam(t)]: "$E" }, e);
      return { name: t.name, args, key: callKey(t.name, args), confidence: probs[t.name], source: "classifier" };
    });
}

let PREDICT_SCOPE = process.env.PREDICT_SCOPE ?? "calls";
let FAST_PATH = process.env.FAST_PATH === "1";
let DEEP = process.env.DEEP === "1";
let FAST_LOOSE = process.env.FAST_LOOSE === "1";
let NO_THINK = process.env.NO_THINK === "1";
let COMPACT = process.env.COMPACT === "1";
let PRUNE = process.env.PRUNE === "1";
let MEMO = process.env.MEMO === "1";
const FAST_MEMO_MARKER = "[[qm-fast-memo]]";
const CONNECTOR_ID = process.env.CONNECTOR_ID ?? "native";
const NO_THINK_MARKER = "[[qm-no-think]]";

function compactText(text) {
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    return text;
  }
  const cell = (v) => (v && typeof v === "object" ? JSON.stringify(v) : String(v ?? ""));
  if (Array.isArray(value) && value.length && value.every((r) => r && typeof r === "object" && !Array.isArray(r))) {
    const cols = [...new Set(value.flatMap((r) => Object.keys(r)))];
    return [cols.join("|"), ...value.map((r) => cols.map((c) => cell(r[c])).join("|"))].join("\n");
  }
  if (value && typeof value === "object" && !Array.isArray(value)) return Object.entries(value).map(([k, v]) => `${k}=${cell(v)}`).join("; ");
  return text;
}

const resultText = (result, limit) => {
  const raw = result.content?.[0]?.text ?? "";
  return (COMPACT ? compactText(raw) : raw).slice(0, limit);
};
const FAST_LOOSE_THETA = Number(process.env.FAST_LOOSE_THETA ?? 0.5);
const FAST_THETA = Number(process.env.FAST_THETA ?? 0.8);
const FAST_PATH_MARKER = "[[qm-fast-path]]";

function sameEntity(p, e) {
  if (!e) return true;
  const tool = skillTools.find((t) => t.name === p.name);
  if (!tool || !applicable(tool, e)) return true;
  const value = p.args?.[entityParam(tool)];
  return typeof value !== "string" || value.toLowerCase() === e.toLowerCase();
}

async function predict(query) {
  return (await predictRaw(query)).filter((p) => sameEntity(p, entityOf(query)));
}

async function predictRaw(query) {
  const e = entityOf(query);
  const fromHistory = predictFromHistory(query, e);
  const base = fromHistory.length ? fromHistory : await classify(query, e).catch((err) => (console.error(`classifier failed: ${err.message}`), []));
  if (PREDICT_SCOPE !== "entity" || !e || !base.length) return base;
  const have = new Set(base.map((p) => p.name));
  const templates = toolArgTemplates();
  const rest = skillTools
    .filter((t) => READ_ONLY.has(t.name) && !have.has(t.name) && applicable(t, e))
    .map((t) => {
      const args = fillArgs(templates.get(t.name) ?? { [entityParam(t)]: "$E" }, e);
      return { name: t.name, args, key: callKey(t.name, args), confidence: 0, source: "entity-scope" };
    });
  return [...base, ...rest];
}

function predictFromHistory(query, e) {
  const w = words(query);
  if (e) for (const part of e.toLowerCase().split(/[^a-z0-9_-]+/)) w.delete(part);
  const nbrs = history.filter((h) => jaccard(w, h.w) >= TAU);
  if (nbrs.length < 2) return [];
  const support = new Map();
  for (const h of nbrs)
    for (const [k, t] of h.templates) {
      const s = support.get(k) ?? { ...t, n: 0 };
      s.n++;
      support.set(k, s);
    }
  return [...support.values()]
    .filter((s) => s.n >= 2 && s.n / nbrs.length >= THETA)
    .map((s) => {
      const args = fillArgs(s.args, e);
      return { name: s.name, args, key: callKey(s.name, args), confidence: s.n / nbrs.length };
    });
}

async function loadHistory() {
  for (let offset = 0; ; ) {
    const page = await json("query_record", { steps: [{ step: "filter", kinds: ["tool-trace"] }], offset, limit: 200 });
    for (const r of page.records ?? []) {
      const d = JSON.parse(r.body ?? "{}");
      if (d.query && d.calls) learn(d.query, d.calls.map((c) => ({ name: c.name, args: c.args ?? {} })));
    }
    if (!page.has_more) return;
    offset += page.returned;
  }
}

async function exec(name, args) {
  if (name.startsWith("web_")) return (await callResearch(name, args)) ?? { content: [{ type: "text", text: "unknown skill" }], isError: true };
  if (name.startsWith("biz_")) return (await callBiz(name, args)) ?? { content: [{ type: "text", text: "unknown skill" }], isError: true };
  if (name.startsWith("real_")) return (await callReal(name, args)) ?? { content: [{ type: "text", text: "unknown tool" }], isError: true };
  if (name.startsWith("sim_")) return (await callSim(name, args)) ?? { content: [{ type: "text", text: "unknown skill" }], isError: true };
  const own = qmTools.find((t) => t.name === name);
  if (!own) return call(name, args);
  try {
    return { content: [{ type: "text", text: await own.run(args) }] };
  } catch (e) {
    return { content: [{ type: "text", text: String(e.message ?? e) }], isError: true };
  }
}

async function closeTurn() {
  clearTimeout(idle);
  const t = turn;
  turn = null;
  if (!t || PREDICTOR === "off") return;
  stats.augmentReused = (stats.augmentReused ?? 0) + t.calls.filter((c) => t.augmentedKeys?.has(c.key)).length;
  const used = new Set(t.calls.map((c) => c.key));
  stats.hits += t.predicted.filter((p) => used.has(p.key)).length;
  if (!t.calls.length || PREDICTOR === "augment") return;
  learn(t.query, t.calls);
  await json("create_record", {
    type: "Document",
    kind: "tool-trace",
    name: t.query.slice(0, 200),
    body: JSON.stringify({ query: t.query, calls: t.calls, predicted: t.predicted.map((p) => p.key), mode: PREDICTOR }),
    home_id: await folderFor("qm-tool-traces"),
    facets: { qm_trace: "1" },
    reason: "QM turn tool trace for the branch predictor",
    sources: [],
  }).catch((e) => console.error(`trace write failed: ${e.message}`));
}

async function startTurn(query) {
  if (turn && turn.query === query && Date.now() - turn.at < 10_000) return "";
  await closeTurn();
  turn = { query, at: Date.now(), calls: [], predicted: [], spec: new Map(), timeline: [] };
  timelines.push({ query, at: turn.at, timeline: turn.timeline });
  if (timelines.length > 200) timelines.shift();
  stats.turns++;
  if (PREDICTOR === "off") return "";
  turn.predicted = await predict(query);
  stats.predicted += turn.predicted.length;
  if (PREDICTOR !== "speculate" && PREDICTOR !== "inject") return "";
  const live = [];
  const raw = [];
  for (const p of turn.predicted) {
    const started = Date.now();
    const pending = exec(p.name, p.args).then((result) => ({ result, ms: Date.now() - started }));
    pending.catch(() => {});
    turn.spec.set(p.key, pending);
    stats.speculated++;
    raw.push(pending);
    live.push(pending.then(({ result }) => (result && !result.isError ? `- ${p.name} ${JSON.stringify(p.args)} → ${resultText(result, 2500)}` : null)));
  }
  if (PREDICTOR !== "inject" || !live.length) return "";
  const deadlineAt = Date.now() + LIVE_WAIT_MS;
  const deadline = new Promise((r) => setTimeout(() => r(null), LIVE_WAIT_MS));
  const done = await Promise.all(live.map((l) => Promise.race([l, deadline])));
  const primary = done.filter((line, i) => line && turn.predicted[i].source !== "entity-scope");
  const entryNow = timelines.at(-1);
  if (entryNow && entryNow.query === turn.query)
    entryNow.prefetch = await Promise.all(
      raw.map((pending, i) =>
        Promise.race([pending.then(({ result, ms }) => ({ name: turn.predicted[i].name, args: turn.predicted[i].args, ms, ok: !result?.isError })), new Promise((r) => setTimeout(() => r({ name: turn.predicted[i].name, args: turn.predicted[i].args, ms: null, ok: false }), 0))]),
      ),
    );
  const deepStarted = Date.now();
  const deepLines = DEEP ? await deepFollowUps(raw, deadlineAt) : [];
  if (entryNow && entryNow.query === turn.query) entryNow.deep = { count: deepLines.length, ms: Date.now() - deepStarted };
  primary.push(...deepLines);
  const core = turn.predicted.filter((p) => p.source !== "entity-scope");
  const wide = turn.predicted.filter((p) => p.source === "entity-scope");
  const allFetched = done.every(Boolean);
  const theta = FAST_LOOSE && (wide.length || deepLines.length) ? FAST_LOOSE_THETA : FAST_THETA;
  const reason = !core.length ? "no-prediction" : !allFetched ? "missing-result" : core.some((p) => p.confidence < theta) ? "low-confidence" : "fast";
  const fast = FAST_PATH && reason === "fast";
  if (FAST_PATH) stats.fastReasons = { ...(stats.fastReasons ?? {}), [reason]: ((stats.fastReasons ?? {})[reason] ?? 0) + 1 };
  if (fast) stats.fastPath = (stats.fastPath ?? 0) + 1;
  const entry = timelines.at(-1);
  if (entry && entry.query === turn.query) {
    entry.fastReason = FAST_PATH ? reason : "off";
    entry.predicted = turn.predicted.map((p) => ({ key: p.key, confidence: Math.round((p.confidence ?? 0) * 100) / 100, source: p.source ?? "history" }));
  }
  const background = done.filter((line, i) => line && turn.predicted[i].source === "entity-scope");
  stats.injected += primary.length + background.length;
  const at = MEMO ? "the start of this turn" : new Date().toISOString();
  return (
    (primary.length
      ? `\n\n### Live tool results for this request\nExecuted for you at ${at}, moments ago. These are current results, not stored memory. Use them directly; call these tools again only if you need something they do not contain.\n${primary.join("\n")}`
      : "") +
    (background.length
      ? `\n\n### Background data (same account, fetched at ${at})\nCurrent results kept on hand so you do not need extra lookups. Do not summarize them; use one only if the request asks for it or it changes your answer.\n${background.join("\n")}`
      : "") +
    (fast ? `\n${FAST_PATH_MARKER}` : "") +
    (fast && MEMO ? `\n${FAST_MEMO_MARKER}` : "") +
    (NO_THINK && !fast && primary.length ? `\n${NO_THINK_MARKER}` : "") +
    (PRUNE && !fast && turn.predicted.length ? `\n${toolScopeMarker()}` : "")
  );
}

let PRUNE_FAMILY = process.env.PRUNE_FAMILY === "1";
const family = (name) => name.split("_")[0];

function toolScopeMarker() {
  const names = new Set();
  for (const p of turn.predicted) {
    names.add(p.name);
    for (const t of followers(p.name)) names.add(t.name);
  }
  if (PRUNE_FAMILY) {
    const fams = new Set([...names].map(family));
    for (const t of skillTools) if (fams.has(family(t.name))) names.add(t.name);
  }
  return `[[qm-tools:${[...names].map((n) => `${CONNECTOR_ID}_${n}`).join(",")}]]`;
}

const DEEP_K = Number(process.env.DEEP_K ?? 12);
const DEEP_CHARS = Number(process.env.DEEP_CHARS ?? 700);

function followers(prev) {
  const prevParam = entityParam(skillTools.find((t) => t.name === prev) ?? {});
  const counts = new Map();
  let total = 0;
  for (const h of history) {
    const at = h.calls.findIndex((c) => c.name === prev);
    if (at < 0) continue;
    total++;
    for (const name of new Set(h.calls.slice(at + 1).map((c) => c.name))) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return [...counts]
    .filter(([name, n]) => total && n / total >= CHAIN_P && READ_ONLY.has(name) && name !== prev)
    .map(([name]) => skillTools.find((t) => t.name === name))
    .filter((t) => t && entityParam(t) !== prevParam);
}

async function deepFollowUps(raw, deadlineAt) {
  const jobs = [];
  for (const [i, pending] of raw.entries()) {
    const p = turn.predicted[i];
    if (p.source === "entity-scope") continue;
    const settled = await Promise.race([pending, new Promise((r) => setTimeout(() => r(null), Math.max(0, deadlineAt - Date.now())))]);
    const text = settled?.result?.content?.[0]?.text;
    if (!text || settled.result.isError) continue;
    for (const tool of followers(p.name)) {
      const param = entityParam(tool);
      for (const value of candidates(text, param).slice(0, DEEP_K)) {
        const args = { [param]: value };
        const key = callKey(tool.name, args);
        if (turn.spec.has(key)) continue;
        const started = Date.now();
        const pending2 = exec(tool.name, args).then((result) => ({ result, ms: Date.now() - started }));
        pending2.catch(() => {});
        turn.spec.set(key, pending2);
        stats.speculated++;
        jobs.push(pending2.then(({ result }) => (result && !result.isError ? `- ${tool.name} ${JSON.stringify(args)} → ${resultText(result, DEEP_CHARS)}` : null)));
      }
    }
  }
  const late = new Promise((r) => setTimeout(() => r(null), Math.max(0, deadlineAt - Date.now())));
  const lines = (await Promise.all(jobs.map((j) => Promise.race([j, late])))).filter(Boolean);
  stats.deep = (stats.deep ?? 0) + lines.length;
  return lines;
}

async function run(name, args) {
  const key = callKey(name, args);
  if (turn) {
    turn.calls.push({ name, args: clean(args), key });
    stats.calls++;
    clearTimeout(idle);
    idle = setTimeout(() => void closeTurn(), 120_000);
    if (READ_ONLY.has(name)) {
      stats.readOnlyCalls++;
      const pending = turn.spec.get(key);
      if (pending) {
        turn.spec.delete(key);
        const asked = Date.now();
        const { result, ms } = await pending.catch(() => ({}));
        if (result && !result.isError) {
          const saved = Math.max(0, ms - (Date.now() - asked));
          stats.savedMs += saved;
          if (turn.chained?.has(key)) (stats.chainHits++, (stats.chainSavedMs += saved));
          chain(name, result);
          return result;
        }
      }
    } else if (turn.spec.size) {
      stats.squashed += turn.spec.size;
      turn.spec.clear();
    }
  }
  const started = Date.now();
  const result = await exec(name, args);
  if (turn && PREDICTOR === "augment" && READ_ONLY.has(name) && !result.isError) {
    const augmented = await augment(name, result);
    turn?.timeline.push({ name, start: started - turn.at, end: Date.now() - turn.at });
    return augmented;
  }
  turn?.timeline.push({ name, start: started - turn.at, end: Date.now() - turn.at });
  if (turn && READ_ONLY.has(name) && !result.isError) chain(name, result);
  return result;
}

const AUGMENT_K = Number(process.env.AUGMENT_K ?? 3);
const AUGMENT_WAIT_MS = Number(process.env.AUGMENT_WAIT_MS ?? 3000);
const AUGMENT_CHARS = Number(process.env.AUGMENT_CHARS ?? 1500);

async function augment(prev, result) {
  const text = result.content?.[0]?.text ?? "";
  const follow = [];
  for (const nextName of transitions(prev)) {
    const tool = skillTools.find((t) => t.name === nextName);
    if (!tool || !READ_ONLY.has(nextName) || nextName === prev) continue;
    const param = entityParam(tool);
    for (const value of candidates(text, param).slice(0, AUGMENT_K)) follow.push({ name: nextName, args: { [param]: value } });
  }
  if (!follow.length) return result;
  const deadline = new Promise((r) => setTimeout(() => r(null), AUGMENT_WAIT_MS));
  const done = await Promise.all(follow.map((f) => Promise.race([exec(f.name, f.args).catch(() => null), deadline])));
  const lines = follow.map((f, i) => (done[i] && !done[i].isError ? `- ${f.name} ${JSON.stringify(f.args)} → ${done[i].content?.[0]?.text?.slice(0, AUGMENT_CHARS)}` : null)).filter(Boolean);
  stats.augmented = (stats.augmented ?? 0) + lines.length;
  if (!lines.length) return result;
  for (const [i, f] of follow.entries()) if (done[i] && !done[i].isError) (turn.augmentedKeys ??= new Set()).add(callKey(f.name, f.args));
  return { ...result, content: [{ type: "text", text: `${text}\n\nFollow-up results already fetched for you (the likely next lookups, run just now; use these instead of calling the tools again):\n${lines.join("\n")}` }] };
}

const CHAIN_K = Number(process.env.CHAIN_K ?? 8);
const CHAIN_P = Number(process.env.CHAIN_P ?? 0.3);

function transitions(prev) {
  const next = new Map();
  let total = 0;
  for (const h of [...history, ...(turn ? [{ calls: turn.calls }] : [])])
    for (let i = 0; i + 1 < h.calls.length; i++)
      if (h.calls[i].name === prev) {
        total++;
        next.set(h.calls[i + 1].name, (next.get(h.calls[i + 1].name) ?? 0) + 1);
      }
  return [...next].filter(([, n]) => total && n / total >= CHAIN_P).map(([n]) => n);
}

function candidates(text, param) {
  const found = [];
  const walk = (v) => {
    if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") for (const [k, x] of Object.entries(v)) k === param && typeof x === "string" ? found.push(x) : walk(x);
  };
  try {
    walk(JSON.parse(text));
  } catch {
    if (/url/i.test(param)) found.push(...(text.match(/https?:\/\/[^\s"'<>)\]]+/g) ?? []));
  }
  return [...new Set(found)];
}

function chain(prev, result) {
  if (!turn || (PREDICTOR !== "speculate" && PREDICTOR !== "inject")) return;
  const text = result.content?.[0]?.text ?? "";
  for (const nextName of transitions(prev)) {
    const tool = skillTools.find((t) => t.name === nextName);
    if (!tool || !READ_ONLY.has(nextName)) continue;
    const param = entityParam(tool);
    for (const value of candidates(text, param).slice(0, CHAIN_K)) {
      const args = { [param]: value };
      const k = callKey(nextName, args);
      if (turn.spec.has(k) || turn.calls.some((c) => c.key === k)) continue;
      const started = Date.now();
      const pending = exec(nextName, args).then((r) => ({ result: r, ms: Date.now() - started }));
      pending.catch(() => {});
      turn.spec.set(k, pending);
      (turn.chained ??= new Set()).add(k);
      stats.chainSpeculated++;
    }
  }
}

const str = { type: "string" };
const qmTools = [
  { name: "qm_recall", description: "Recall QM memory facts from Native.", inputSchema: { type: "object", properties: { query: str, scope: str, acting_user: str } }, run: recall },
  { name: "qm_remember", description: "Store QM memory facts in Native, one per line.", inputSchema: { type: "object", required: ["content"], properties: { content: str, scope: str, acting_user: str, source: str } }, run: remember },
];

async function handle(req) {
  const { id, method, params = {} } = req;
  if (method === "initialize") return { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "native-qm-bridge", version: "0" } };
  if (method === "tools/list") return { tools: [...qmTools.map(({ run, ...t }) => t), ...nativeTools.filter((t) => t.name !== "bootstrap"), ...(SIM ? simTools : []), ...(REAL ? realTools : []), ...bizTools, ...researchTools] };
  if (method === "tools/call") {
    const args = params.arguments ?? {};
    if (params.name === "qm_recall" && args.max_chars) {
      const facts = await recall({ ...args, query: "" });
      const live = (await startTurn(args.query ?? "")) ?? "";
      const entry = timelines.at(-1);
      if (entry && entry.query === (args.query ?? "")) entry.recallDone = Date.now() - entry.at;
      const scopeMark = live.match(/\n(\[\[qm-tools:[^\]]*\]\])/)?.[1];
      const noThinkMark = (live.includes(NO_THINK_MARKER) ? `${NO_THINK_MARKER}\n` : "") + (scopeMark ? `${scopeMark}\n` : "");
      const plain = facts + live.replace(`\n${FAST_PATH_MARKER}`, "").replace(`\n${NO_THINK_MARKER}`, "").replace(/\n\[\[qm-tools:[^\]]*\]\]/, "");
      const fits = !args.max_chars || plain.length + FAST_PATH_MARKER.length + 1 <= args.max_chars;
      if (live.includes(FAST_PATH_MARKER) && !fits) {
        stats.fastPath = Math.max(0, (stats.fastPath ?? 1) - 1);
        const entry = timelines.at(-1);
        if (entry) entry.fastReason = "over-budget";
      }
      const memoMark = live.includes(FAST_MEMO_MARKER) ? `${FAST_MEMO_MARKER}\n` : "";
      const body = live.includes(FAST_PATH_MARKER) && fits ? `${FAST_PATH_MARKER}\n${memoMark}${plain.replace(`\n${FAST_MEMO_MARKER}`, "")}` : `${noThinkMark}${plain.replace(`\n${FAST_MEMO_MARKER}`, "")}`;
      return { content: [{ type: "text", text: body }] };
    }
    const plumbing = (params.name === "qm_remember" && args.source) || (params.name === "qm_recall" && !args.query);
    return plumbing ? exec(params.name, args) : run(params.name, args);
  }
  if (id === undefined) return null;
  throw Object.assign(new Error(`unsupported method ${method}`), { code: -32601 });
}

indexEnums([...nativeTools, ...(SIM ? simTools : []), ...(REAL ? realTools : []), ...bizTools, ...researchTools]);
await loadHistory();

createServer(async (req, res) => {
  let body = "";
  for await (const chunk of req) body += chunk;
  if (req.url === "/token") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ access_token: "local", token_type: "bearer", expires_in: 3600 }));
  }
  if (req.url === "/sim/world") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify(world()));
  }
  if (req.url.startsWith("/mode/") && req.method === "POST") {
    const [m, scope = process.env.PREDICT_SCOPE ?? "calls", ...flags] = req.url.slice(6).split("+");
    if (!["off", "shadow", "speculate", "inject", "augment"].includes(m) || !["calls", "entity"].includes(scope)) return res.writeHead(400), res.end();
    await closeTurn();
    PREDICTOR = m;
    PREDICT_SCOPE = scope;
    FAST_PATH = flags.includes("fast");
    DEEP = flags.includes("deep");
    FAST_LOOSE = flags.includes("loose");
    NO_THINK = flags.includes("nothink");
    COMPACT = flags.includes("compact");
    PRUNE = flags.includes("prune") || flags.includes("prunefam");
    PRUNE_FAMILY = flags.includes("prunefam");
    MEMO = flags.includes("memo");
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ mode: PREDICTOR, scope: PREDICT_SCOPE, fast: FAST_PATH, deep: DEEP, loose: FAST_LOOSE, nothink: NO_THINK, compact: COMPACT, prune: PRUNE }));
  }
  if (req.url === "/timelines") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify(timelines));
  }
  if (req.url === "/predictor") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ mode: PREDICTOR, theta: THETA, tau: TAU, history: history.length, ...stats, precision: stats.predicted ? stats.hits / stats.predicted : null, coverage: stats.readOnlyCalls ? stats.hits / stats.readOnlyCalls : null }));
  }
  if (req.url !== "/mcp" || req.method !== "POST") {
    res.writeHead(404);
    return res.end();
  }
  const msg = JSON.parse(body || "{}");
  let out;
  try {
    const result = await handle(msg);
    if (result === null) {
      res.writeHead(202);
      return res.end();
    }
    out = { jsonrpc: "2.0", id: msg.id, result };
  } catch (e) {
    out = { jsonrpc: "2.0", id: msg.id, error: { code: e.code ?? -32000, message: String(e.message ?? e) } };
  }
  res.writeHead(200, { "content-type": "application/json" });
  res.end(JSON.stringify(out));
}).listen(PORT, "127.0.0.1", () => console.error(`native bridge on http://127.0.0.1:${PORT} (db ${DB}, run ${runKey})`));
