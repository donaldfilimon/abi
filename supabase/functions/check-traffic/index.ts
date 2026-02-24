// deno-lint-ignore-file no-explicit-any
// Check Traffic Edge Function: classify bots vs humans and log to Postgres
// Routes:
// - POST /check-traffic  (JSON body optional)
// - GET  /check-traffic  (no body)
// Returns: { request_id, is_bot, bot_score, bot_reason }

import { createClient } from "npm:@supabase/supabase-js@2.46.1";

// Known bot UA fragments and hints
const KNOWN_BOT_UA = [
  "Googlebot", "Bingbot", "Slurp", "DuckDuckBot", "Baiduspider", "YandexBot",
  "Sogou", "Exabot", "facebot", "ia_archiver", "AhrefsBot", "SemrushBot",
  "MJ12bot", "curl/", "Wget/", "python-requests", "Go-http-client", "Apache-HttpClient",
  "HeadlessChrome", "node-fetch", "facebookexternalhit", "Twitterbot", "Discordbot"
];

function scoreBotLikeSignals(init: {
  userAgent?: string | null;
  method?: string | null;
  referrer?: string | null;
  acceptLanguage?: string | null;
}) {
  const ua = init.userAgent || "";
  const method = (init.method || "").toUpperCase();
  const ref = init.referrer || "";
  const al = init.acceptLanguage || "";

  let score = 0;
  const reasons: string[] = [];

  if (!ua || ua.length < 10) { score += 2; reasons.push("missing_or_short_user_agent"); }
  for (const frag of KNOWN_BOT_UA) {
    if (ua.includes(frag)) { score += 5; reasons.push(`ua_matches:${frag}`); break; }
  }
  if (!ref) { score += 1; reasons.push("no_referrer"); }
  if (!al) { score += 1; reasons.push("no_accept_language"); }
  if (method === "HEAD") { score += 2; reasons.push("head_method"); }

  const is_bot = score >= 4;
  return { score, is_bot, reasons: reasons.join(",") };
}

function getClientIP(req: Request) {
  // Common proxy headers (Supabase deploys behind a proxy)
  const hdr = req.headers;
  const xff = hdr.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  const cf = hdr.get("cf-connecting-ip");
  if (cf) return cf;
  // Deno.serve currently does not expose remote addr directly
  return null;
}

Deno.serve(async (req: Request) => {
  try {
    const url = new URL(req.url);

    if (url.pathname !== "/check-traffic") {
      return new Response("Not Found", { status: 404 });
    }

    const supabase = createClient<any>(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
      { auth: { persistSession: false } }
    );

    const headers = req.headers;
    const ua = headers.get("user-agent");
    const referrer = headers.get("referer") || headers.get("referrer");
    const acceptLanguage = headers.get("accept-language");
    const ip = getClientIP(req);

    const classification = scoreBotLikeSignals({
      userAgent: ua,
      method: req.method,
      referrer,
      acceptLanguage,
    });

    // Optional body metadata
    let bodyMeta: Record<string, unknown> = {};
    if (req.method === "POST") {
      try {
        bodyMeta = await req.json();
      } catch (_) {/* ignore */}
    }

    const row = {
      ip,
      method: req.method,
      path: url.searchParams.get("path") || url.pathname,
      referrer: referrer || null,
      user_agent: ua || null,
      accept_language: acceptLanguage || null,
      is_bot: classification.is_bot,
      bot_reason: classification.reasons,
      bot_score: classification.score,
      country: headers.get("cf-ipcountry") || headers.get("x-vercel-ip-country") || null,
      region: headers.get("x-vercel-ip-country-region") || null,
      city: headers.get("x-vercel-ip-city") || null,
      metadata: bodyMeta,
    };

    const { data, error } = await supabase
      .from("app.request_logs")
      .insert(row)
      .select("request_id, is_bot, bot_score, bot_reason")
      .single();

    if (error) {
      console.error("DB insert error", error);
      return new Response(JSON.stringify({ error: "db_insert_failed", details: error.message }), {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      headers: { "content-type": "application/json" },
    });
  } catch (e) {
    console.error(e);
    return new Response(JSON.stringify({ error: "internal_error" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
});