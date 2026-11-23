import type { BetterAuthPlugin } from "better-auth";
import { createAuthEndpoint } from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";

/**
 * Local-only convenience: when LOCAL_AUTO_LOGIN is set, GET /api/auth/local-login
 * signs the browser in as the local user without a password. Never enable it on a
 * server other people can reach.
 */
export function localAutoLoginEnabled(): boolean {
  return ["1", "true", "yes"].includes((process.env.LOCAL_AUTO_LOGIN ?? "").toLowerCase());
}

function safeNext(value: unknown): string {
  if (typeof value !== "string" || !value.startsWith("/") || value.startsWith("//")) return "/";
  if (value.startsWith("/sign-in") || value.startsWith("/sign-up")) return "/";
  return value;
}

export const localAutoLogin = () =>
  ({
    id: "local-auto-login",
    endpoints: {
      localLogin: createAuthEndpoint("/local-login", { method: "GET" }, async (ctx) => {
        if (!localAutoLoginEnabled()) {
          return ctx.json({ error: "Local auto login is disabled" }, { status: 404 });
        }

        const adapter = ctx.context.internalAdapter;
        const email = process.env.LOCAL_AUTO_LOGIN_EMAIL?.trim();
        let user = email
          ? (await adapter.findUserByEmail(email))?.user ?? null
          : (
              await ctx.context.adapter.findMany<{ id: string }>({
                model: "user",
                sortBy: { field: "createdAt", direction: "asc" },
                limit: 1,
              })
            )[0] ?? null;

        if (!user) {
          user = await adapter.createUser({
            email: email || "local@localhost",
            name: "Local",
            emailVerified: true,
          }, { method: "local-auto-login" });
        }

        const fullUser = await adapter.findUserById(user.id);
        if (!fullUser) return ctx.json({ error: "User not found" }, { status: 500 });

        const session = await adapter.createSession(fullUser.id, false);
        await setSessionCookie(ctx, { session, user: fullUser });

        const next = safeNext(new URL(ctx.request?.url ?? "http://localhost/").searchParams.get("next"));
        throw ctx.redirect(next);
      }),
    },
  }) satisfies BetterAuthPlugin;
