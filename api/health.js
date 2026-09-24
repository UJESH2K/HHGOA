// Tells the static console whether the live copilot is available on this deployment.
import { readFileSync } from "node:fs";

const CFG = JSON.parse(readFileSync(new URL("./_copilot.json", import.meta.url), "utf8"));

export default function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  res.status(200).json({
    live: false,
    copilot: Boolean(process.env.ANTHROPIC_API_KEY),
    copilot_model: CFG.model,
    cases: Object.keys(CFG.cases).length,
  });
}
