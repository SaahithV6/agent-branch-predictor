const UA = { "user-agent": "qm-native-bench/0.1 (local research)" };
const text = (html) =>
  html.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<nav[\s\S]*?<\/nav>|<[^>]+>/g, " ").replace(/&[a-z#0-9]+;/g, " ").replace(/\s+/g, " ").trim();

const SKILLS = [
  ["hn_search", "Search Hacker News stories (live). Returns title, url, points, comment count, date.", { type: "object", required: ["query"], properties: { query: { type: "string" } } }, true,
    async ({ query }) => {
      const r = await fetch(`https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=10&query=${encodeURIComponent(query)}`, { headers: UA, signal: AbortSignal.timeout(15_000) });
      const j = await r.json();
      return j.hits.filter((x) => x.url).map((x) => ({ title: x.title, url: x.url, points: x.points, comments: x.num_comments, date: x.created_at?.slice(0, 10), hn: `https://news.ycombinator.com/item?id=${x.objectID}` }));
    }],
  ["fetch_url", "Fetch a web page (live) and return its readable text, truncated.", { type: "object", required: ["url"], properties: { url: { type: "string" } } }, true,
    async ({ url }) => {
      const r = await fetch(url, { headers: UA, redirect: "follow", signal: AbortSignal.timeout(20_000) });
      return { status: r.status, text: text(await r.text()).slice(0, 4000) };
    }],
];

export const researchTools = SKILLS.map(([name, description, inputSchema]) => ({ name: `web_${name}`, description, inputSchema }));
export const researchReadOnly = new Set(SKILLS.filter((s) => s[3]).map(([name]) => `web_${name}`));
const byName = new Map(SKILLS.map((s) => [`web_${s[0]}`, s[4]]));

export async function callResearch(name, args) {
  const fn = byName.get(name);
  if (!fn) return null;
  try {
    const out = await fn(args);
    return { content: [{ type: "text", text: typeof out === "string" ? out : JSON.stringify(out) }] };
  } catch (e) {
    return { content: [{ type: "text", text: `error: ${e.message ?? e}` }], isError: true };
  }
}
