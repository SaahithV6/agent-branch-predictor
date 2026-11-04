import { createHash } from "node:crypto";

const SERVICES = ["auth", "billing", "search", "checkout", "notify", "ingest", "gateway", "reports"];
const CUSTOMERS = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka"];
const PEOPLE = ["Priya", "Marcus", "Lena", "Tomas", "Aiko", "Dev", "Ruth", "Omar"];

const h = (s) => createHash("sha256").update(s).digest().readUInt32BE(0);
const pick = (arr, s) => arr[h(s) % arr.length];

export function world() {
  const services = Object.fromEntries(
    SERVICES.map((s) => [
      s,
      {
        healthy: h(`${s}:health`) % 3 !== 0,
        error: `E-${1000 + (h(`${s}:err`) % 9000)}`,
        oncall: pick(PEOPLE, `${s}:oncall`),
        ci: h(`${s}:ci`) % 4 === 0 ? "failing" : "passing",
        blockers: h(`${s}:blk`) % 3,
        version: `v${1 + (h(`${s}:v`) % 9)}.${h(`${s}:m`) % 20}`,
      },
    ]),
  );
  const customers = Object.fromEntries(
    CUSTOMERS.map((c) => [
      c,
      {
        tier: pick(["enterprise", "growth", "starter"], `${c}:tier`),
        openTickets: h(`${c}:t`) % 6,
        overdue: h(`${c}:inv`) % 3 === 0 ? `$${(h(`${c}:amt`) % 90) * 1000 + 1000}` : "none",
        usageTrend: pick(["up 40%", "flat", "down 25%"], `${c}:use`),
      },
    ]),
  );
  return { services, customers };
}

const W = world();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jitter = (median) => Math.round(median * Math.exp((Math.random() - 0.5) * 0.6));
const svc = (a) => W.services[a.service] ?? { unknown: `no service named ${a.service}` };
const cust = (a) => W.customers[a.customer] ?? { unknown: `no customer named ${a.customer}` };

const S = { type: "object", required: ["service"], properties: { service: { type: "string", enum: SERVICES } } };
const C = { type: "object", required: ["customer"], properties: { customer: { type: "string", enum: CUSTOMERS } } };

const SKILLS = [
  ["service_status", "Current health of a service.", S, 800, true, (a) => ({ service: a.service, healthy: svc(a).healthy })],
  ["recent_deploys", "Last deploys of a service.", S, 1500, true, (a) => ({ service: a.service, latest: svc(a).version, deployedHoursAgo: h(`${a.service}:d`) % 48 })],
  ["error_logs", "Top error in a service's logs over the last hour.", S, 3000, true, (a) => ({ service: a.service, topError: svc(a).healthy ? "none" : svc(a).error, count: svc(a).healthy ? 0 : 50 + (h(`${a.service}:c`) % 500) })],
  ["oncall_lookup", "Who is on call for a service.", S, 500, true, (a) => ({ service: a.service, oncall: svc(a).oncall })],
  ["runbook_fetch", "Fetch the runbook for a service.", S, 2000, true, (a) => ({ service: a.service, runbook: `Restart ${a.service} workers, then check ${a.service}-db replication lag.` })],
  ["ci_status", "CI pipeline status for a service's main branch.", S, 4000, true, (a) => ({ service: a.service, ci: svc(a).ci })],
  ["open_blockers", "Open release blockers for a service.", S, 1200, true, (a) => ({ service: a.service, blockers: svc(a).blockers })],
  ["changelog", "Unreleased changelog for a service.", S, 1000, true, (a) => ({ service: a.service, entries: 2 + (h(`${a.service}:cl`) % 6) })],
  ["customer_account", "Account details for a customer.", C, 700, true, (a) => ({ customer: a.customer, tier: cust(a).tier })],
  ["customer_tickets", "Open support tickets for a customer.", C, 1500, true, (a) => ({ customer: a.customer, openTickets: cust(a).openTickets })],
  ["invoices", "Invoice status for a customer.", C, 2500, true, (a) => ({ customer: a.customer, overdue: cust(a).overdue })],
  ["usage_metrics", "30-day usage trend for a customer.", C, 3500, true, (a) => ({ customer: a.customer, trend: cust(a).usageTrend })],
  ["web_search", "Search the public web.", { type: "object", required: ["query"], properties: { query: { type: "string" } } }, 6000, true, (a) => ({ query: a.query, results: [`No public incidents found for "${a.query}".`] })],
  ["page_oncall", "Page the on-call engineer for a service.", { type: "object", required: ["service", "message"], properties: { service: { type: "string", enum: SERVICES }, message: { type: "string" } } }, 1000, false, (a) => ({ paged: svc(a).oncall })],
  ["create_ticket", "Open a support ticket for a customer.", { type: "object", required: ["customer", "summary"], properties: { customer: { type: "string", enum: CUSTOMERS }, summary: { type: "string" } } }, 1000, false, (a) => ({ ticket: `T-${h(a.customer + a.summary) % 100000}` })],
  ["post_status_update", "Post a status update for a service.", { type: "object", required: ["service", "text"], properties: { service: { type: "string", enum: SERVICES }, text: { type: "string" } } }, 800, false, (a) => ({ posted: true, service: a.service })],
];

export const simTools = SKILLS.map(([name, description, inputSchema]) => ({ name: `sim_${name}`, description, inputSchema }));
export const simReadOnly = new Set(SKILLS.filter((s) => s[4]).map(([name]) => `sim_${name}`));
const byName = new Map(SKILLS.map((s) => [`sim_${s[0]}`, s]));

export async function callSim(name, args) {
  const skill = byName.get(name);
  if (!skill) return null;
  await sleep(jitter(skill[3]));
  return { content: [{ type: "text", text: JSON.stringify(skill[5](args)) }] };
}
