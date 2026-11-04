import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.DEMO_PORT ?? 8090);
const ARMS = [
  { id: "plain", name: "Plain agent", qm: 8081, bridge: 8788, log: join(here, "plain.log") },
  { id: "ours", name: "With speculation", qm: 8080, bridge: 8787, log: join(here, "ours.log") },
];

const runs = new Map();

function emit(run, event) {
  run.events.push(event);
  for (const res of run.listeners) res.write(`data: ${JSON.stringify(event)}\n\n`);
}

async function json(url, init) {
  const res = await fetch(url, init);
  return res.json();
}

function usageFor(arm, runId) {
  let found = null;
  for (const line of readFileSync(arm.log, "utf8").split("\n"))
    if (line.startsWith("[claude] usage ") && line.includes(`"runId":"${runId}"`)) found = JSON.parse(line.slice("[claude] usage ".length));
  return found;
}

async function traceFor(arm, question, runId) {
  for (let i = 0; i < 12; i++) {
    const usage = usageFor(arm, runId);
    if (usage) {
      const timelines = await json(`http://127.0.0.1:${arm.bridge}/timelines`);
      const entry = [...timelines].reverse().find((t) => t.query === question) ?? {};
      return { usage, fastReason: entry.fastReason ?? null, predicted: entry.predicted ?? [], prefetch: entry.prefetch ?? [], deep: entry.deep ?? null, recallDone: entry.recallDone ?? null, tools: entry.timeline ?? [] };
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return { usage: null };
}

async function runArm(run, arm, question) {
  const t0 = Date.now();
  const t = () => Date.now() - t0;
  emit(run, { arm: arm.id, type: "start", t: 0 });
  try {
    const queued = await json(`http://127.0.0.1:${arm.qm}/v1/turns?async=1`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ surface: "api", actor: { externalId: "saahith" }, conversation: { kind: "dm", threadRef: `demo-${arm.id}-${Date.now()}-${Math.random()}` }, text: question }),
    });
    const res = await fetch(`http://127.0.0.1:${arm.qm}/v1/runs/${queued.runId}/events`);
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    for await (const chunk of res.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.name === "delta") emit(run, { arm: arm.id, type: "delta", t: t(), text: ev.value.delta });
        else if (ev.type === "TOOL_CALL_START") emit(run, { arm: arm.id, type: "tool", t: t(), name: ev.toolCallName });
        else if (ev.type === "RUN_FINISHED" || ev.type === "RUN_ERROR") {
          emit(run, { arm: arm.id, type: "done", t: t(), status: ev.type });
          finished = true;
        }
      }
      if (finished) break;
    }
    emit(run, { arm: arm.id, type: "trace", trace: await traceFor(arm, question, queued.runId) });
  } catch (error) {
    emit(run, { arm: arm.id, type: "done", t: t(), status: `error: ${error.message}` });
  }
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/" || url.pathname === "/index.html") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    return res.end(readFileSync(join(here, "index.html")));
  }
  if (url.pathname === "/api/ask" && req.method === "POST") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const question = String(JSON.parse(body || "{}").q ?? "").trim();
    if (!question) return res.writeHead(400), res.end();
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const run = { events: [], listeners: new Set() };
    runs.set(id, run);
    if (runs.size > 50) runs.delete(runs.keys().next().value);
    for (const arm of ARMS) void runArm(run, arm, question);
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ id, arms: ARMS.map(({ id: armId, name }) => ({ id: armId, name })) }));
  }
  if (url.pathname === "/api/events") {
    const run = runs.get(url.searchParams.get("id"));
    if (!run) return res.writeHead(404), res.end();
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    for (const event of run.events) res.write(`data: ${JSON.stringify(event)}\n\n`);
    run.listeners.add(res);
    req.on("close", () => run.listeners.delete(res));
    return;
  }
  res.writeHead(404);
  res.end();
}).listen(PORT, "127.0.0.1", () => console.error(`demo on http://127.0.0.1:${PORT}`));
