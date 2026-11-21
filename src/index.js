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

function isAuthorized(request, env) {
  if (!env.API_KEY) return true;
  return request.headers.get("Authorization") === `Bearer ${env.API_KEY}`;
}

function backendUrl(path, requestUrl, env) {
  const base = env.MODEL_API_URL.replace(/\/+$/, "") + "/";
  const url = new URL(path.replace(/^\//, ""), base);
  url.search = new URL(requestUrl).search;
  return url;
}

async function forward(request, path, env) {
  if (!env.MODEL_API_URL) {
    return json({
      error: "model_backend_not_configured",
      service: "Monster-1 gateway",
      model_ready: false,
      message: "Configure MODEL_API_URL to point to the separately hosted Monster-1 FastAPI service.",
    }, 503);
  }
  try {
    const target = backendUrl(path, request.url, env);
    const response = await fetch(new Request(target, request));
    const headers = new Headers(response.headers);
    for (const [key, value] of Object.entries(corsHeaders)) headers.set(key, value);
    return new Response(response.body, { status: response.status, headers });
  } catch {
    return json({ error: "model_backend_unavailable", model_ready: false }, 502);
  }
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }
    const url = new URL(request.url);
    const path = url.pathname;

    if (path !== "/v1/health" && path !== "/v1/model" && path !== "/v1/chat") {
      return json({ service: "Monster-1 gateway", endpoints: ["/v1/health", "/v1/model", "/v1/chat"] });
    }
    if (!isAuthorized(request, env)) {
      return json({ error: "unauthorized" }, 401);
    }

    if (path === "/v1/health" && request.method === "GET" && !env.MODEL_API_URL) {
      return json({ ok: true, service: "monster-1-gateway", model_ready: false, status: "backend_not_configured" });
    }
    if (path === "/v1/model" && request.method === "GET" && !env.MODEL_API_URL) {
      return json({ name: "Monster-1", language: "fa", pretrained_model: false, engine_ready: false, status: "not_trained_or_unverified" });
    }
    if (path === "/v1/chat" && request.method === "POST") {
      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "invalid_json" }, 400);
      }
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        return json({ error: "messages_required" }, 400);
      }
      return forward(request, path, env);
    }
    if ((path === "/v1/health" || path === "/v1/model") && request.method === "GET") {
      return forward(request, path, env);
    }
    return json({ error: "method_not_allowed" }, 405);
  },
};
