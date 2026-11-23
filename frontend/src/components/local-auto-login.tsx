"use client";

import { useEffect } from "react";
import { useSession } from "@/lib/auth-client";

const ENABLED = ["1", "true", "yes"].includes(
  (process.env.NEXT_PUBLIC_LOCAL_AUTO_LOGIN ?? "").toLowerCase()
);
const GUARD_KEY = "supoclip-local-login-at";

/** Sends a signed-out browser to /api/auth/local-login when local auto login is on. */
export function LocalAutoLogin() {
  const { data, isPending } = useSession();

  useEffect(() => {
    if (!ENABLED || isPending || data) return;
    // Don't loop if the endpoint fails: at most one attempt every 15 seconds.
    try {
      const last = Number(sessionStorage.getItem(GUARD_KEY) ?? 0);
      if (Date.now() - last < 15_000) return;
      sessionStorage.setItem(GUARD_KEY, String(Date.now()));
    } catch {}
    const { pathname, search } = window.location;
    const params = new URLSearchParams(search);
    const next =
      pathname === "/sign-in" || pathname === "/sign-up"
        ? params.get("next") || "/"
        : pathname + search;
    window.location.replace(`/api/auth/local-login?${new URLSearchParams({ next })}`);
  }, [data, isPending]);

  return null;
}
