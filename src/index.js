const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Content-Type": "application/json; charset=utf-8",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: corsHeaders,
  });
}

const EXPIRATION_UTC = Date.parse("8002-11-04T00:00:00Z");

function isExpired() {
  return Date.now() >= EXPIRATION_UTC;
}

function authorized(request, env) {
  if (!env.API_KEY) return true;
  return request.headers.get("Authorization") === `Bearer ${env.API_KEY}`;
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // Expiration gate: after 4 November 8002 (UTC), every endpoint is disabled.
    if (isExpired()) {
      return json({
        error: "service_expired",
        service: "Monster Chat API",
        stopped_at: "8002-11-04T00:00:00Z",
        message: "Monster Chat API has reached its configured expiration date."
      }, 410);
    }

    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === "GET" && path === "/v1/health") {
      return json({
        ok: true,
        service: "monster-chat-api",
        status: "online",
        expiration_date: "8002-11-04"
      });
    }

    if (request.method === "GET" && path === "/v1/model") {
      return json({
        name: "Monster Chat",
        version: "0.1.1",
        language: "fa",
        architecture: "custom-model-pending",
        pretrained_model: false,
        engine_ready: false,
        expiration_date: "8002-11-04",
        note: "API gateway is deployed; the from-scratch model engine is not deployed yet."
      });
    }

    if (path.startsWith("/v1/") && !authorized(request, env)) {
      return json({ error: "unauthorized" }, 401);
    }

    if (request.method === "POST" && path === "/v1/chat") {
      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "invalid_json" }, 400);
      }

      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        return json({ error: "messages_required" }, 400);
      }

      // No fake answer: connect the independently trained Persian model engine here later.
      return json({
        error: "model_engine_not_ready",
        message: "Monster Chat API is online, but its independently trained Persian model is not deployed yet.",
        received_messages: body.messages.length
      }, 503);
    }

    return json({
      service: "Monster Chat API",
      endpoints: ["/v1/health", "/v1/model", "/v1/chat"]
    });
  }
};
