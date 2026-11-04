import itertools, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

TOKEN = open(os.path.join(os.path.dirname(__file__), "..", "claude.token")).read().strip()
GROUPS = {
    "a job-application company (e.g. NVIDIA, Zipline)": ["real_jobs_log", "real_jobs_queue", "real_jobs_cap_window"],
    "a git repo name (e.g. qm, scrubline)": ["real_repo_status", "real_repo_log"],
    "a backend service (e.g. auth, billing)": ["sim_service_status", "sim_recent_deploys", "sim_error_logs", "sim_oncall_lookup", "sim_runbook_fetch", "sim_ci_status", "sim_open_blockers", "sim_changelog"],
    "a customer account (e.g. acme, globex)": ["sim_customer_account", "sim_customer_tickets", "sim_invoices", "sim_usage_metrics"],
}
GROUPS["a B2B SaaS customer account, a real company (e.g. Netflix, Boeing)"] = ["biz_crm_account", "biz_crm_contacts", "biz_billing_invoices", "biz_support_tickets", "biz_usage_trend", "biz_company_news"]
GROUPS["a customer-success manager's name (e.g. Marcus Lee, Priya Shah), asking about their whole book of accounts"] = ["biz_list_accounts"]
ONLY = os.environ.get("DISTILL_ONLY")
if ONLY:
    GROUPS = {k: v for k, v in GROUPS.items() if v[0].startswith(ONLY)}
DESC = {t["name"]: t["description"] for f in ("tools-real.json", "tools-sim.json", "tools-biz.json") for t in json.load(open(f))}


def haiku(prompt):
    for _ in range(3):
        r = subprocess.run(["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--model", "claude-haiku-4-5", "--tools", "", "--no-session-persistence",
                            "--system-prompt", "Output only a JSON array of strings.", prompt], capture_output=True, text=True, timeout=180, env={**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": TOKEN}, cwd="/tmp")
        m = re.search(r"\[.*\]", r.stdout, re.S)
        if m:
            try:
                return [s for s in json.loads(m.group(0)) if isinstance(s, str) and "{E}" in s]
            except Exception:
                pass
    return []


def job(kind, combo):
    tools = "\n".join(f"- {t}: {DESC[t]}" for t in combo)
    prompt = (f"An assistant has many lookup tools. Write {30 if len(combo) == 1 and combo[0] == 'biz_list_accounts' else 15} diverse, realistic messages a user might send it that would need exactly these lookups and no others:\n{tools}\n"
              f"Each message is about one entity, {kind}; write the literal placeholder {{E}} where the entity name goes. Vary wording, length and tone; "
              "some terse, some chatty; never name the tools.")
    return [{"text": s, "tools": list(combo)} for s in haiku(prompt)]


combos = [(k, c) for k, tools in GROUPS.items() for n in (1, 2, 3) for c in itertools.combinations(tools, n)]
print(f"{len(combos)} tool combinations", flush=True)
with ThreadPoolExecutor(3) as ex:
    rows = [r for batch in ex.map(lambda kc: job(*kc), combos) for r in batch]
out = f"distill-data{'-' + ONLY.rstrip('_') if ONLY else ''}.json"
json.dump(rows, open(out, "w"))
print(f"{len(rows)} labeled requests -> {out}")
