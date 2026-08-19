// Scores customers with REAL pre-trained scikit-learn Random Forest models
// (see ml/train_models.py). No LLM involved in this endpoint — pure model
// inference, so results are deterministic and reproducible.

import fs from "fs";
import path from "path";
import { scoreChurn, scoreUpsell } from "./lib/forest.js";

// Netlify's function bundler auto-injects a working `__dirname` at build time —
// we deliberately do NOT redeclare it ourselves (that caused a SyntaxError),
// and we don't derive it from import.meta.url either (that resolved to an
// invalid path once bundled). We just use the injected global directly.

let churnModel = null;
let upsellModel = null;

function resolveModelsDir() {
  // Primary: Netlify's bundler-injected __dirname (works in most configs)
  if (typeof __dirname !== "undefined") {
    return path.join(__dirname, "models");
  }
  // Fallback: derive from the current working directory Netlify sets for the function
  return path.join(process.cwd(), "netlify", "functions", "models");
}

function loadModels() {
  const modelsDir = resolveModelsDir();
  if (!churnModel) {
    churnModel = JSON.parse(fs.readFileSync(path.join(modelsDir, "model_churn.json"), "utf8"));
  }
  if (!upsellModel) {
    upsellModel = JSON.parse(fs.readFileSync(path.join(modelsDir, "model_upsell.json"), "utf8"));
  }
}

export async function handler(event) {
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: { message: "Method not allowed" } }),
    };
  }

  try {
    loadModels();
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: { message: `Failed to load models: ${err.message}` } }),
    };
  }

  try {
    const { records } = JSON.parse(event.body);

    if (!Array.isArray(records) || records.length === 0) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: { message: "Provide a non-empty 'records' array" } }),
      };
    }

    const scored = records.map((record) => {
      const churn = scoreChurn(record, churnModel);
      const upsell = scoreUpsell(record, upsellModel);
      return {
        customer_id: record.customer_id,
        company_name: record.company_name,
        churn,
        upsell,
      };
    });

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scored, model_info: "RandomForestClassifier (scikit-learn), trained offline on uploaded CRM export" }),
    };
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: { message: err.message } }),
    };
  }
}
