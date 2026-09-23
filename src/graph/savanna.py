"""Find the workspace host through the Savanna control-plane API instead of hunting in the UI.

    python -m src.graph.savanna                 # discover and print
    python -m src.graph.savanna --write-env     # also write TG_HOST into .env

WHY THIS EXISTS. `TG_HOST` is the one credential that cannot be typed from memory: it is a
generated hostname buried in a console panel, and a wrong guess looks exactly like a stopped
workspace. Savanna has a control-plane API that knows the answer, so asking it is better than
asking a person to find a button.

WHAT WAS VERIFIED AGAINST THE LIVE API (2026-09-23), because none of it is in one doc page:

  base            https://api.tgcloud.io/controller/v4/v2
  auth header     X-Api-Key: <key>          <- NOT `Authorization: Bearer`, which returns 401
                                               "Could not authenticate the token from the header"
  list workgroups GET /workgroups
  get workspace   GET /workgroups/{workgroupID}/workspaces/{workspaceID}

Two failure modes worth naming, since both look like a broken key and neither is:

  * `Missing Authentication Token` with a 403 is AWS API Gateway saying the PATH does not exist.
    The control plane lives under `/controller/v4/v2`, not `/v2` - every shorter path 403s.
  * A 200 with `"Result": []` means the key authenticated but can see no workgroups. That is a
    SCOPE problem, not an auth problem: an API key created in a different organization than the
    workspace returns exactly this. The fix is to create the key from the same organization that
    owns the workgroup, under Admin -> Settings -> API Keys.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

from ..config import ROOT, get, load_env

BASE = "https://api.tgcloud.io/controller/v4/v2"
TIMEOUT = 25


def _call(path: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        headers={"X-Api-Key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"Error": True, "Message": f"HTTP {exc.code}: {body[:300]}"}
    except urllib.error.URLError as exc:
        return {"Error": True, "Message": f"could not reach {BASE}: {exc.reason}"}


def _host_from(workspace: dict) -> str:
    """Pull whatever the workspace record calls its endpoint.

    The field name is not documented and has changed across preview versions, so every plausible
    key is checked rather than assuming one. If none is present the whole record is printed -
    better to show the caller the payload than to report "not found" about a field we guessed.
    """
    # `nginx_host` is the real field, confirmed against the live API on 2026-09-23. It is not
    # called endpoint, host or url, and it is not in the endpoint documentation - which is why
    # this checks every plausible name and prints the raw record when none of them hit.
    for key in ("nginx_host", "endpoint", "host", "url", "dnsName", "publicEndpoint",
                "connectionUrl", "apiEndpoint", "restppEndpoint", "domain"):
        value = workspace.get(key)
        if value:
            value = str(value)
            return value if value.startswith("http") else f"https://{value}"
    return ""


def discover(api_key: str) -> dict:
    """Walk workgroups -> workspaces and return everything found, with the host if present."""
    out: dict = {"workgroups": [], "workspaces": [], "hosts": [], "diagnosis": ""}

    wg = _call("/workgroups", api_key)
    if wg.get("Error"):
        out["diagnosis"] = f"workgroup listing failed: {wg.get('Message')}"
        return out

    groups = wg.get("Result") or []
    out["workgroups"] = groups
    if not groups:
        out["diagnosis"] = (
            "The key authenticated but can see no workgroups. That is a scope problem, not an "
            "auth problem - an API key created in a different organization than the workspace "
            "returns exactly this. Create the key from the organization that owns "
            "MyWorkgroup (Admin -> Settings -> API Keys), or read TG_HOST off the workspace's "
            "Connect -> Connect from API panel and set it by hand.")
        return out

    for group in groups:
        gid = group.get("workgroup_id") or group.get("workgroupID") or group.get("id")
        for ws in group.get("workspaces") or []:
            wid = ws.get("workspace_id") or ws.get("workspaceID") or ws.get("id")
            detail = ws
            if gid and wid:
                got = _call(f"/workgroups/{gid}/workspaces/{wid}", api_key)
                if not got.get("Error") and got.get("Result"):
                    detail = got["Result"]
            host = _host_from(detail)
            out["workspaces"].append({"workgroup": group.get("name"), "name": ws.get("name"),
                                      "workgroup_id": gid, "workspace_id": wid,
                                      "status": detail.get("status"), "host": host,
                                      "record": detail})
            if host:
                out["hosts"].append(host)

    if not out["hosts"]:
        out["diagnosis"] = ("Workspaces were found but none exposed an endpoint field. The raw "
                            "records are printed below - the host is in there under some name.")
    return out


def write_env(host: str) -> None:
    """Set TG_HOST in .env without disturbing anything else in it."""
    p = pathlib.Path(ROOT) / ".env"
    s = p.read_text(encoding="utf-8")
    if re.search(r"^TG_HOST=", s, flags=re.M):
        s = re.sub(r"^TG_HOST=.*$", f"TG_HOST={host}", s, flags=re.M)
    else:
        s += f"\nTG_HOST={host}\n"
    p.write_text(s, encoding="utf-8")


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key", default="", help="defaults to TG_API_KEY in .env")
    ap.add_argument("--write-env", action="store_true", help="write the host into .env")
    a = ap.parse_args()

    api_key = a.api_key or get("TG_API_KEY")
    if not api_key:
        print("No API key. Set TG_API_KEY in .env, or pass --api-key.\n"
              "Create one in Savanna under Admin -> Settings -> API Keys, in the SAME "
              "organization that owns the workgroup.")
        return 2

    print(f"querying {BASE} with X-Api-Key ...{api_key[-6:]}\n")
    found = discover(api_key)

    print(f"workgroups visible: {len(found['workgroups'])}")
    for ws in found["workspaces"]:
        print(f"  {ws['workgroup']} / {ws['name']}  status={ws['status']}  "
              f"host={ws['host'] or '(no endpoint field)'}")

    if found["diagnosis"]:
        print("\n" + found["diagnosis"])
    if found["workspaces"] and not found["hosts"]:
        print("\nraw workspace records:")
        for ws in found["workspaces"]:
            print(json.dumps(ws["record"], indent=2)[:2000])

    hosts = found["hosts"]
    if hosts:
        host = hosts[0]
        print(f"\nhost: {host}")
        if a.write_env:
            write_env(host)
            print("written to .env as TG_HOST")
        else:
            print("re-run with --write-env to store it, then: python -m src.graph.deploy --check")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
