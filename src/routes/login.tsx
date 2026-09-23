import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { LeafIcon } from "../components/icons";
import { isAuthenticated, loginPatient } from "../lib/auth";

export const Route = createFileRoute("/login")({
  beforeLoad: () => {
    if (isAuthenticated()) {
      throw redirect({ to: "/" });
    }
  },
  head: () => ({
    meta: [{ title: "Log in — MindAura AI" }],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await loginPatient(email.trim(), password);
      await navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-sm flex-col items-center pt-10">
      <span className="text-primary">
        <LeafIcon size={36} />
      </span>
      <h1 className="mt-4 text-2xl text-primary-deep">MindAura AI</h1>
      <p className="mt-1 text-sm text-muted-foreground">Log in to continue your check-ins.</p>

      <form onSubmit={handleSubmit} className="card-soft mt-8 w-full p-7">
        <label className="block text-sm text-foreground">
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1.5 w-full rounded-xl border border-border bg-card px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-primary/40"
          />
        </label>

        <label className="mt-5 block text-sm text-foreground">
          Password
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1.5 w-full rounded-xl border border-border bg-card px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-primary/40"
          />
        </label>

        {error ? <p className="mt-4 text-sm text-destructive">{error}</p> : null}

        <button type="submit" disabled={busy} className="btn-primary mt-6 w-full disabled:opacity-60">
          {busy ? "Logging in..." : "Login"}
        </button>
      </form>

      <p className="mt-6 text-sm text-muted-foreground">
        Don't have an account?{" "}
        <Link to="/register" className="text-primary-deep underline">
          Create Account
        </Link>
      </p>
    </div>
  );
}
