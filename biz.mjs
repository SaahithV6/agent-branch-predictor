import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { DatabaseSync } from "node:sqlite";

export const CUSTOMERS = ["Adobe", "Airbnb", "AMD", "Atlassian", "Boeing", "Caterpillar", "Cisco", "Coca-Cola", "Costco", "Delta Air Lines", "Dropbox", "eBay", "FedEx", "General Electric",
  "Honeywell", "IBM", "Intel", "Intuit", "Lockheed Martin", "Lyft", "Mastercard", "McDonald's", "Moderna", "Netflix", "Nike", "Oracle", "PayPal", "PepsiCo", "Pfizer", "Qualcomm",
  "Salesforce", "Shopify", "Snowflake", "Spotify", "Starbucks", "Target", "Tesla", "Uber", "UPS", "Visa", "Walmart", "Zoom"];
const PLANS = [["Starter", 1200], ["Growth", 4800], ["Enterprise", 24000]];
const REPS = ["Priya Shah", "Marcus Lee", "Lena Ortiz", "Tomas Berg", "Aiko Tanaka"];
const DB = process.env.BIZ_DB ?? new URL("./biz.db", import.meta.url).pathname;

const h = (s) => createHash("sha256").update(s).digest().readUInt32BE(0);

function build(db) {
  db.exec(`
    create table accounts (name text primary key, plan text, arr integer, csm text, renewal date, health text, region text);
    create table contacts (account text, name text, title text, email text);
    create table invoices (account text, month text, amount integer, status text);
    create table tickets (id integer primary key, account text, opened date, severity text, status text, subject text);
    create table events (account text, day date, feature text, users integer);
    create table tasks (id integer primary key, account text, note text, created text);
    create table outbox (id integer primary key, account text, recipient text, body text, created text);
    create index ev on events(account, day);`);
  const subjects = ["SSO login loop", "Export to CSV times out", "Invoice shows wrong seat count", "API rate limit errors", "Dashboard slow to load", "Webhook retries failing", "Need admin role change", "Data residency question"];
  const features = ["dashboards", "exports", "api", "alerts", "sso", "reports"];
  const ins = (sql) => db.prepare(sql);
  const acc = ins("insert into accounts values (?,?,?,?,?,?,?)"), con = ins("insert into contacts values (?,?,?,?)"), inv = ins("insert into invoices values (?,?,?,?)");
  const tik = ins("insert into tickets (account, opened, severity, status, subject) values (?,?,?,?,?)"), ev = ins("insert into events values (?,?,?,?)");
  db.exec("begin");
  for (const c of CUSTOMERS) {
    const [plan, base] = PLANS[h(c + "plan") % 3];
    const seats = 5 + (h(c + "seats") % 60);
    const health = ["green", "yellow", "red"][h(c + "health") % 3];
    acc.run(c, plan, base * Math.ceil(seats / 10), REPS[h(c + "csm") % REPS.length], `2026-${String(1 + (h(c + "ren") % 12)).padStart(2, "0")}-15`, health, ["NA", "EMEA", "APAC"][h(c + "reg") % 3]);
    for (const [i, title] of ["VP Operations", "IT Director", "Finance Manager"].entries()) con.run(c, `${["Alex", "Sam", "Jordan", "Taylor", "Casey"][h(c + i) % 5]} ${["Kim", "Patel", "Nguyen", "Garcia", "Smith"][h(c + "l" + i) % 5]}`, title, `${title.split(" ")[0].toLowerCase()}@${c.toLowerCase().replace(/[^a-z]/g, "")}.com`);
    for (let m = 0; m < 24; m++) {
      const month = `${2024 + Math.floor(m / 12)}-${String(1 + (m % 12)).padStart(2, "0")}`;
      inv.run(c, month, Math.round((base * Math.ceil(seats / 10)) / 12), m >= 22 && health === "red" ? "overdue" : m === 23 && health === "yellow" ? "disputed" : "paid");
    }
    for (let t = 0; t < 30 + (h(c + "t") % 120); t++) tik.run(c, `2025-${String(1 + (h(c + t) % 12)).padStart(2, "0")}-${String(1 + (h(c + "d" + t) % 28)).padStart(2, "0")}`, ["low", "medium", "high", "urgent"][h(c + "s" + t) % 4], h(c + "o" + t) % 5 === 0 ? "open" : "closed", subjects[h(c + "j" + t) % subjects.length]);
    const trend = { green: 1.002, yellow: 1.0, red: 0.993 }[health];
    for (let d = 0; d < 365; d++) {
      const day = new Date(Date.UTC(2025, 0, 1) + d * 86_400_000).toISOString().slice(0, 10);
      for (const f of features) for (let k = 0; k < 15; k++) ev.run(c, day, f, Math.max(0, Math.round(seats * Math.pow(trend, d) * ((h(c + day + f + k) % 100) / 100))));
    }
  }
  db.exec("commit");
}

const db = new DatabaseSync(DB);
if (!db.prepare("select name from sqlite_master where name='accounts'").get()) build(db);

const all = (sql, ...a) => db.prepare(sql).all(...a);
const C = { type: "object", required: ["account"], properties: { account: { type: "string", enum: CUSTOMERS } } };

const SKILLS = [
  ["list_accounts", "List every account a CSM owns, with plan, ARR, health and renewal date.", { type: "object", required: ["csm"], properties: { csm: { type: "string", enum: REPS } } }, true,
    ({ csm }) => all("select name as account, plan, arr, health, renewal from accounts where csm = ? order by name", csm)],
  ["crm_account", "CRM record for a customer account: plan, ARR, CSM owner, renewal date, health, region.", C, true, ({ account }) => all("select * from accounts where name = ?", account)[0] ?? `no account ${account}`],
  ["crm_contacts", "Key contacts at a customer account.", C, true, ({ account }) => all("select name, title, email from contacts where account = ?", account)],
  ["billing_invoices", "Last 6 invoices for an account with payment status.", C, true, ({ account }) => all("select month, amount, status from invoices where account = ? order by month desc limit 6", account)],
  ["support_tickets", "Open support tickets for an account, newest first.", C, true, ({ account }) => all("select id, opened, severity, subject from tickets where account = ? and status = 'open' order by opened desc limit 15", account)],
  ["usage_trend", "Product usage trend for an account: monthly active usage for the last 6 months by feature.", C, true, ({ account }) =>
    all("select substr(day,1,7) month, feature, sum(users) usage from events where account = ? and day >= '2025-07-01' group by month, feature order by month", account)],
  ["company_news", "Recent public information about a customer company (Wikipedia summary).", C, true, async ({ account }) => {
    const r = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(account.replace(/ /g, "_"))}`, { headers: { "user-agent": "qm-native-bench/0.1 (local research)" }, signal: AbortSignal.timeout(15_000) });
    const j = await r.json();
    return j.extract ?? `no summary (${r.status})`;
  }],
  ["create_task", "Create a follow-up task for an account in the CS task tracker.", { type: "object", required: ["account", "note"], properties: { account: C.properties.account, note: { type: "string" } } }, false,
    ({ account, note }) => ({ task: Number(db.prepare("insert into tasks (account, note, created) values (?,?,datetime('now'))").run(account, note).lastInsertRowid) })],
  ["draft_email", "Save an email draft to a customer contact in the outbox (not sent).", { type: "object", required: ["account", "recipient", "body"], properties: { account: C.properties.account, recipient: { type: "string" }, body: { type: "string" } } }, false,
    ({ account, recipient, body }) => ({ draft: Number(db.prepare("insert into outbox (account, recipient, body, created) values (?,?,?,datetime('now'))").run(account, recipient, body).lastInsertRowid) })],
];

export const bizTools = SKILLS.map(([name, description, inputSchema]) => ({ name: `biz_${name}`, description, inputSchema }));
export const bizReadOnly = new Set(SKILLS.filter((s) => s[3]).map(([name]) => `biz_${name}`));
const byName = new Map(SKILLS.map((s) => [`biz_${s[0]}`, s[4]]));

export async function callBiz(name, args) {
  const fn = byName.get(name);
  if (!fn) return null;
  try {
    const out = await fn(args);
    return { content: [{ type: "text", text: typeof out === "string" ? out : JSON.stringify(out) }] };
  } catch (e) {
    return { content: [{ type: "text", text: String(e.message ?? e) }], isError: true };
  }
}
