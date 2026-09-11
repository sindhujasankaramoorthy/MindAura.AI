import { Link, useRouterState } from "@tanstack/react-router";
import { BrandMark } from "./AuraLogo";

const links = [
  { to: "/", label: "Home" },
  { to: "/check-ins", label: "Check-ins" },
  { to: "/practices", label: "Practices" },
  { to: "/profile", label: "Profile" },
] as const;

export function Header() {
  const path = useRouterState({ select: (s) => s.location.pathname });

  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
        <Link to="/" className="shrink-0">
          <BrandMark />
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => {
            const active = l.to === "/" ? path === "/" : path.startsWith(l.to);
            return (
              <Link
                key={l.to}
                to={l.to}
                className={`rounded-full px-4 py-2 text-sm transition-all duration-300 ${
                  active
                    ? "bg-primary-soft text-primary-deep"
                    : "text-muted-foreground hover:bg-secondary hover:text-primary-deep"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <button
            type="button"
            aria-label="Notifications"
            className="relative rounded-full border border-border bg-card p-2.5 text-muted-foreground transition-colors duration-300 hover:text-primary-deep"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 8a6 6 0 1 0-12 0c0 6-2 7-2 7h16s-2-1-2-7" />
              <path d="M13.7 20a2 2 0 0 1-3.4 0" />
            </svg>
            <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-peach" />
          </button>
          <Link
            to="/profile"
            aria-label="Profile"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-lavender-soft font-display text-sm text-primary-deep transition-transform duration-300 hover:scale-105"
          >
            S
          </Link>
        </div>
      </div>

      <nav className="flex items-center gap-1 overflow-x-auto px-5 pb-3 md:hidden">
        {links.map((l) => {
          const active = l.to === "/" ? path === "/" : path.startsWith(l.to);
          return (
            <Link
              key={l.to}
              to={l.to}
              className={`rounded-full px-3.5 py-1.5 text-sm whitespace-nowrap transition-colors duration-300 ${
                active ? "bg-primary-soft text-primary-deep" : "text-muted-foreground"
              }`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
