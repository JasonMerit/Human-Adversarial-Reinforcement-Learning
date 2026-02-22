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

  const { trajectory, winner } = body

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

  const allFirstAreOne = trajectory.every(
    (pair: any) => Array.isArray(pair) && pair[0] === 1
  )

  if (allFirstAreOne) {
    return new Response(
      JSON.stringify({
        error: "Invalid trajectory: all first actions are 1",
      }),
      {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    )
  }

  const trajectory_length = trajectory.length

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SERVICE_ROLE_KEY")!
  )

  const { error } = await supabase
    .from("episodes")
    .insert({
      trajectory,
      trajectory_length,
      winner,
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