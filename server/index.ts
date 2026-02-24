import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "Method not allowed" }),
      {
        status: 405,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  let body
  try {
    body = await req.json()
  } catch {
    return new Response(
      JSON.stringify({ error: "Invalid JSON body" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  const { trajectory, winner, trapped, playerToken, buildVersion } = body

  if (!Array.isArray(trajectory)) {
    return new Response(
      JSON.stringify({ error: "Invalid trajectory" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  if (![0, 1, 2].includes(winner)) {
    return new Response(
      JSON.stringify({ error: "Invalid winner" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  if (trajectory.every(pair => pair.x === 1)) {
    return new Response(
      JSON.stringify({ error: "Invalid trajectory: all first actions are 1" }),
      { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders } }
    )
  }

  // ---- Validate trapped ----
  if (typeof trapped !== "boolean") {
    return new Response(
      JSON.stringify({ error: "Invalid trapped flag" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  // Validate buildVersion
  if (
    !buildVersion ||
    typeof buildVersion.x !== "number" ||
    typeof buildVersion.y !== "number" ||
    typeof buildVersion.z !== "number"
  ) {
    return new Response(
      JSON.stringify({ error: "Invalid buildVersion" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SERVICE_ROLE_KEY")!
  )

  const trajectory_length = trajectory.length
  if (trajectory_length < 5) {
    return new Response(
      JSON.stringify({ error: "Trajectory too short" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  if (trajectory_length > 100) {
    return new Response(
      JSON.stringify({ error: "Trajectory too long" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  const { error } = await supabase
    .from("episodes")
    .insert({
      trajectory: trajectory.map(pair => [pair.x, pair.y]),
      trajectory_length,
      winner,
      trapped,
      player_token: playerToken,
      version_major: buildVersion.x,
      version_minor: buildVersion.y,
      version_patch: buildVersion.z,
    })

  if (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  return new Response(
    JSON.stringify({ ok: true }),
    {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    }
  )
})