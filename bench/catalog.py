"""Normalize public API catalogs into one tool catalog: one JSON line per distinct tool.

Usage: [VERSIONS=1] python3 -I bench/catalog.py <catalogs dir> <out.jsonl> [source]   (one source per process parallelizes it)
       python3 -I bench/catalog.py assemble <out.jsonl> <N> <tools.jsonl ...>   (real tools, then tenant copies up to N)
VERSIONS=1 keeps every API version of an operation as its own tool (id gets the spec file), the way a
gateway serving several versions side by side exposes them; they are near-identical lookalikes.

Sources (fetched by data/catalogs/fetch.sh and the ToolBench mirror; parsed as data, never run):
  apisguru   APIs.guru openapi-directory  (OpenAPI 2/3, JSON or YAML)
  azure      Azure/azure-rest-api-specs   (Swagger 2, JSON)
  msgraph    microsoftgraph/msgraph-metadata v1.0 (OpenAPI 3, YAML)
  botocore   AWS service models           (service-2.json)
  google     Google API discovery documents
  toolbench  ToolBench / StableToolBench RapidAPI tools

Record: {"id", "source", "server", "name", "read", "args": [{"name", "type", "required", "format"?, "enum"?}], "desc"}.
"id" is source/server/name; the same operation in several API versions is kept once (latest file wins).
"read" follows HTTP semantics (GET/HEAD) or the operation verb (Get/List/Describe...) where there is no method.
"""

import glob
import json
import os
import re
import sys

import yaml

VERSIONS = os.environ.get("VERSIONS") == "1"
READ_VERBS = ("get", "list", "describe", "search", "lookup", "query", "read", "fetch", "find", "check", "head", "show", "count")
HTTP = ("get", "put", "post", "delete", "patch", "head", "options")


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:120]


def arg(name, schema, required, spec=None):
    schema = resolve(schema or {}, spec) if spec else (schema or {})
    out = {"name": str(name), "type": schema.get("type", "string") if isinstance(schema.get("type", "string"), str) else "string",
           "required": bool(required)}
    for k in ("format", "enum", "pattern"):
        if schema.get(k) is not None:
            out[k] = schema[k][:20] if k == "enum" and isinstance(schema[k], list) else schema[k]
    return out


def resolve(node, spec, depth=0):
    """Follow a local $ref (#/components/..., #/definitions/..., #/parameters/...)."""
    while isinstance(node, dict) and "$ref" in node and depth < 8:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return {}
        cur = spec
        for part in ref[2:].split("/"):
            cur = cur.get(part.replace("~1", "/").replace("~0", "~"), {}) if isinstance(cur, dict) else {}
        node, depth = cur, depth + 1
    return node if isinstance(node, dict) else {}


def openapi(spec, source, server):
    """Operations of one OpenAPI 2/3 document: path/query parameters plus top-level body fields."""
    for path, item in (spec.get("paths") or {}).items():
        item = resolve(item, spec)
        shared = item.get("parameters") or []
        for method in HTTP:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            args = {}
            for p in shared + (op.get("parameters") or []):
                p = resolve(p, spec)
                if p.get("in") in ("path", "query") and p.get("name"):
                    args[p["name"]] = arg(p["name"], p.get("schema") or p, p.get("required"), spec)
                elif p.get("in") == "body":
                    body = resolve(p.get("schema") or {}, spec)
                    for k, v in (body.get("properties") or {}).items():
                        args.setdefault(k, arg(k, v, k in (body.get("required") or []), spec))
            content = (resolve(op.get("requestBody") or {}, spec).get("content") or {})
            body = resolve(next(iter(content.values()), {}).get("schema") or {}, spec) if content else {}
            for k, v in (body.get("properties") or {}).items():
                args.setdefault(k, arg(k, v, k in (body.get("required") or []), spec))
            name = op.get("operationId") or f"{method}_{path}"
            yield {"source": source, "server": server, "name": slug(name), "read": method in ("get", "head"),
                   "args": list(args.values()), "desc": (op.get("summary") or op.get("description") or "")[:200]}


def load_doc(path):
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".json"):
        return json.loads(raw)
    return yaml.load(raw, Loader=yaml.CSafeLoader)


def from_openapi_tree(root, source, server_of):
    for path in sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True)):
        if not re.search(r"\.(json|ya?ml)$", path) or "/examples/" in path or "/example/" in path:
            continue
        try:
            spec = load_doc(path)
        except Exception:
            continue
        if isinstance(spec, dict) and ("swagger" in spec or "openapi" in spec):
            for t in openapi(spec, source, server_of(path, spec)):
                if VERSIONS:
                    t["version"] = slug(os.path.relpath(path, root))
                yield t


def botocore(root):
    for path in sorted(glob.glob(os.path.join(root, "botocore/data/*/*/service-2.json"))):
        spec = json.load(open(path))
        server = spec.get("metadata", {}).get("serviceId") or path.split("/")[-3]
        shapes = spec.get("shapes", {})
        for name, op in spec.get("operations", {}).items():
            shape = shapes.get((op.get("input") or {}).get("shape"), {})
            req = set(shape.get("required", []))
            args = [arg(k, {"type": shapes.get(v.get("shape"), {}).get("type", "string"),
                            "enum": shapes.get(v.get("shape"), {}).get("enum"),
                            "pattern": shapes.get(v.get("shape"), {}).get("pattern")}, k in req)
                    for k, v in shape.get("members", {}).items()]
            verb = re.match(r"[A-Z][a-z]+", name)
            yield {"source": "botocore", "server": slug(server), "name": name, "read": bool(verb) and verb.group().lower() in READ_VERBS,
                   "args": args, "desc": ""}


def google(root):
    def walk(res, prefix):
        for m in (res.get("methods") or {}).values():
            yield prefix, m
        for k, r in (res.get("resources") or {}).items():
            yield from walk(r, f"{prefix}.{k}")
    for path in sorted(glob.glob(os.path.join(root, "doc_*.json"))):
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        server = f"{doc.get('name')}.{doc.get('version')}"
        for _, m in walk(doc, doc.get("name", "")):
            args = [arg(k, v, v.get("required")) for k, v in (m.get("parameters") or {}).items()]
            yield {"source": "google", "server": slug(server), "name": slug(m.get("id", "")),
                   "read": m.get("httpMethod", "").upper() in ("GET", "HEAD"), "args": args, "desc": (m.get("description") or "")[:200]}


def toolbench(root):
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            t = json.load(open(path))
        except Exception:
            continue
        server = slug(t.get("tool_name") or os.path.basename(path)[:-5])
        for a in t.get("api_list") or []:
            args = [arg(p.get("name"), {"type": str(p.get("type", "string")).lower()}, req)
                    for req, key in ((True, "required_parameters"), (False, "optional_parameters")) for p in a.get(key) or [] if p.get("name")]
            yield {"source": "toolbench", "server": server, "name": slug(a.get("name")), "read": str(a.get("method", "GET")).upper() in ("GET", "HEAD"),
                   "args": args, "desc": (a.get("description") or "")[:200]}


def sources(base, only=None):
    pick = lambda name: only in (None, name)  # noqa: E731
    yield from () if not pick("apisguru") else from_openapi_tree(os.path.join(base, "apisguru/APIs"), "apisguru",
                                 lambda p, s: slug(os.path.relpath(p, os.path.join(base, "apisguru/APIs")).split("/")[0] + "/" +
                                                   os.path.relpath(p, os.path.join(base, "apisguru/APIs")).split("/")[1]))
    yield from () if not pick("azure") else from_openapi_tree(os.path.join(base, "azure/specification"), "azure",
                                 lambda p, s: slug("/".join(os.path.relpath(p, os.path.join(base, "azure/specification")).split("/")[:3])))
    yield from () if not pick("msgraph") else from_openapi_tree(os.path.join(base, "msgraph/openapi/v1.0"), "msgraph", lambda p, s: "graph.v1.0")
    yield from () if not pick("botocore") else botocore(os.path.join(base, "botocore"))
    yield from () if not pick("google") else google(os.path.join(base, "google"))
    yield from () if not pick("toolbench") else toolbench(os.path.join(base, "toolbench/x"))


def assemble(out, target, paths):
    """Real records from paths, then tenant copies of them (same API under another tenant's server name,
    as a multi-tenant gateway registers it) until target tools; copies carry "tenant": k."""
    real = [line for p in paths for line in open(p)]
    with open(out, "w") as f:
        f.writelines(real)
        n, k = len(real), 1
        while n < target:
            for line in real:
                if n >= target:
                    break
                t = json.loads(line)
                t.update(server=f"{t['server']}~t{k}", id=f"t{k}/{t['id']}", tenant=k)
                f.write(json.dumps(t) + "\n")
                n += 1
            k += 1
    print(json.dumps({"real": len(real), "tenant_copies": max(0, target - len(real)), "total": max(target, len(real))}))


if __name__ == "__main__" and sys.argv[1] == "assemble":
    assemble(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
elif __name__ == "__main__":
    base, out = sys.argv[1], sys.argv[2]
    seen, n, counts = {}, 0, {}
    for t in sources(base, sys.argv[3] if len(sys.argv) > 3 else None):
        t["id"] = f"{t['source']}/{t['server']}/{t['name']}" + (f"@{t['version']}" if t.get("version") else "")
        seen[t["id"]] = t  # later API versions overwrite earlier ones (files are read in sorted order)
        n += 1
    with open(out, "w") as f:
        for t in seen.values():
            f.write(json.dumps(t, default=str) + "\n")  # YAML turns some enum values into dates
            counts[t["source"]] = counts.get(t["source"], 0) + 1
    print(json.dumps({"operations_read": n, "distinct_tools": len(seen), "by_source": counts,
                      "reads": sum(t["read"] for t in seen.values())}))
