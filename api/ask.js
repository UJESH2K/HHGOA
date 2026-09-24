// The copilot, as a Vercel function: the public deploy's only server-side code.
//
// The deployed console is a static page with no Python server behind it, so the key has to
// live somewhere that is not the browser. This function holds it (ANTHROPIC_API_KEY in the
// Vercel project settings) and streams Claude's answer back as server-sent events, in the same
// event shape as `src/ui/app.py`'s /api/ask, so the page cannot tell the two apart.
//
// The system prompt, the Fraud Policy and every case's context are NOT written here. They are
// generated from `src/agent/copilot.py` by `python -m src.ui.build_static` into
// `api/_copilot.json`, so the Python copilot and this one cannot drift apart - and a caller can
// only ask about the 20 cases in that file, never send an arbitrary prompt or payload.
import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";

const CFG = JSON.parse(readFileSync(new URL("./_copilot.json", import.meta.url), "utf8"));

const MAX_QUESTION = 800;
const MAX_HISTORY = 6;
const PER_HOUR = Number(process.env.COPILOT_PER_HOUR || 20);

// Best-effort per-visitor limit. Serverless instances do not share memory, so this slows a
// single abuser rather than guaranteeing a cap - the hard ceiling is the spend limit on the
// Anthropic workspace the key belongs to.
const hits = new Map();
function limited(ip) {
  const now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < 3_600_000);
  if (recent.length >= PER_HOUR) return true;
  recent.push(now);
  hits.set(ip, recent);
  return false;
}

function cost(u) {
  const [inRate, outRate] = CFG.pricing[CFG.model] || [0, 0];
  return ((u.input_tokens || 0) * inRate + (u.cache_read_input_tokens || 0) * inRate * 0.1
    + (u.cache_creation_input_tokens || 0) * inRate * 1.25
    + (u.output_tokens || 0) * outRate) / 1e6;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ detail: "POST only" });
  if (!process.env.ANTHROPIC_API_KEY) {
    return res.status(503).json({ detail: "ANTHROPIC_API_KEY is not set in the Vercel project" });
  }
  // Only the console on this same deployment may call it.
  const origin = req.headers.origin;
  if (origin && new URL(origin).host !== req.headers.host) {
    return res.status(403).json({ detail: "cross-origin requests are not accepted" });
  }
  const ip = String(req.headers["x-forwarded-for"] || "anon").split(",")[0].trim();
  if (limited(ip)) {
    return res.status(429).json({ detail: `limit of ${PER_HOUR} questions an hour reached` });
  }

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  const context = CFG.cases[String(body.case_id || "")];
  const question = String(body.question || "").trim().slice(0, MAX_QUESTION);
  if (!context) return res.status(404).json({ detail: "unknown case" });
  if (!question) return res.status(400).json({ detail: "empty question" });

  const messages = [
    { role: "user", content: [
      // policy + case is one cached prefix, exactly as in src/agent/copilot.py
      { type: "text", text: context, cache_control: { type: "ephemeral" } },
      { type: "text", text: "I will ask questions about this case." },
    ] },
    { role: "assistant", content: "Understood. Ask away." },
  ];
  for (const turn of (Array.isArray(body.history) ? body.history : []).slice(-MAX_HISTORY)) {
    if ((turn.role === "user" || turn.role === "assistant") && turn.content) {
      messages.push({ role: turn.role, content: String(turn.content).slice(0, 4000) });
    }
  }
  messages.push({ role: "user", content: question });

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
  });
  const send = (event) => res.write(`data: ${JSON.stringify(event)}\n\n`);

  const client = new Anthropic();
  const t0 = Date.now();
  let firstToken = null;
  try {
    const stream = client.beta.messages.stream({
      model: CFG.model,
      max_tokens: CFG.max_tokens,
      system: [{ type: "text", text: CFG.system }],
      messages,
      output_config: { effort: CFG.effort },
      // If a safety classifier declines, re-run on Anthropic's recommended fallback model
      // rather than returning a refusal to an analyst mid-investigation.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    });
    for await (const event of stream) {
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        if (firstToken === null) firstToken = (Date.now() - t0) / 1000;
        send({ type: "text", text: event.delta.text });
      }
    }
    const final = await stream.finalMessage();
    const u = final.usage || {};
    send({
      type: "done",
      model: final.model || CFG.model,
      stop_reason: final.stop_reason,
      latency_s: Math.round((Date.now() - t0) / 10) / 100,
      first_token_s: firstToken === null ? null : Math.round(firstToken * 100) / 100,
      usage: {
        input_tokens: u.input_tokens || 0,
        output_tokens: u.output_tokens || 0,
        cache_read_tokens: u.cache_read_input_tokens || 0,
        cache_write_tokens: u.cache_creation_input_tokens || 0,
      },
      cost_usd: Math.round(cost(u) * 1e5) / 1e5,
    });
  } catch (err) {
    send({ type: "error", message: `${err?.name || "Error"}: ${err?.message || err}` });
  }
  res.end();
}
