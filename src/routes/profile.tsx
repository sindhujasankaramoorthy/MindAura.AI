import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { LeafIcon } from "../components/icons";
import { useStore } from "../lib/store";

export const Route = createFileRoute("/profile")({
  head: () => ({
    meta: [
      { title: "Profile — MindAura AI" },
      {
        name: "description",
        content: "Manage your MindAura AI profile, notification preferences and privacy settings.",
      },
      { property: "og:title", content: "Profile — MindAura AI" },
      { property: "og:description", content: "Your preferences, notifications and privacy." },
    ],
  }),
  component: Profile,
});

function Profile() {
  const { checkins, practicesDone } = useStore();
  const [reminders, setReminders] = useState(true);
  const [practiceNudge, setPracticeNudge] = useState(false);

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <section className="card-soft flex flex-wrap items-center gap-6 p-8 animate-rise">
        <span className="relative flex h-20 w-20 items-center justify-center rounded-full bg-lavender-soft font-display text-2xl text-primary-deep">
          <span className="absolute inset-[-6px] rounded-full border border-primary/20 animate-ripple" />
          S
        </span>
        <div>
          <h1 className="text-2xl text-primary-deep">Sindhuja</h1>
          <p className="text-sm text-muted-foreground">sindhuja@example.com</p>
          <p className="mt-2 text-xs text-muted-foreground">
            {checkins.length} check-ins · {practicesDone.length} practices completed
          </p>
        </div>
      </section>

      <section className="card-soft p-8 animate-fade">
        <h2 className="text-lg text-primary-deep">Preferences</h2>
        <div className="mt-5 space-y-4">
          <Toggle
            label="Daily check-in reminder"
            hint="A gentle nudge at a time that suits you."
            on={reminders}
            onChange={setReminders}
          />
          <Toggle
            label="Practice reminders"
            hint="Occasional reminders for your assigned practices."
            on={practiceNudge}
            onChange={setPracticeNudge}
          />
        </div>
      </section>

      <section className="card-soft bg-primary-soft/50 p-8 animate-fade">
        <div className="flex items-center gap-3">
          <LeafIcon className="text-primary-deep" />
          <h2 className="text-lg text-primary-deep">Your privacy matters.</h2>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-primary-deep/80">
          Your check-ins, recordings and journal entries are stored securely and handled according
          to the application's privacy policy. They belong to you, and are only shared with the
          care team you choose to work with.
        </p>
      </section>

      <section className="card-soft divide-y divide-border p-2 animate-fade">
        {["Privacy policy", "Notification settings", "Help & support", "Sign out"].map((item) => (
          <button
            key={item}
            type="button"
            className="flex w-full items-center justify-between rounded-2xl px-6 py-4 text-left text-sm text-foreground transition-colors duration-300 hover:bg-secondary"
          >
            {item}
            <span className="text-muted-foreground">›</span>
          </button>
        ))}
      </section>
    </div>
  );
}

function Toggle({
  label,
  hint,
  on,
  onChange,
}: {
  label: string;
  hint: string;
  on: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-6 rounded-3xl bg-secondary/60 px-5 py-4">
      <div>
        <p className="text-sm text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={label}
        onClick={() => onChange(!on)}
        className={`h-7 w-12 shrink-0 rounded-full p-1 transition-colors duration-300 ${
          on ? "bg-primary" : "bg-muted-foreground/30"
        }`}
      >
        <span
          className={`block h-5 w-5 rounded-full bg-card transition-transform duration-300 ${
            on ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
    </div>
  );
}
