// Upsell propensity engine — runs server-side on Netlify.
// Same API-key protection as analyze.js, but this one also turns on
// Claude's web_search tool so it can pull real, current signals about
// a named customer (funding news, hiring, incidents, cloud migration, etc.)
// instead of just simulating everything.

export async function handler(event) {
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: { message: "Method not allowed" } }),
    };
  }

  try {
    const { system, messages, max_tokens, enableWebSearch } = JSON.parse(event.body);

    const requestBody = {
      model: "claude-sonnet-4-6",
      max_tokens: max_tokens || 1500,
      system,
      messages,
    };

    // Only attach the web_search tool when the caller asks for it —
    // e.g. skip it for quick re-scoring calls that don't need fresh lookups.
    if (enableWebSearch) {
      requestBody.tools = [
        {
          type: "web_search_20250305",
          name: "web_search",
        },
      ];
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await response.json();

    return {
      statusCode: response.status,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    };
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: { message: err.message } }),
    };
  }
}
