import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { CheckIcon, ClockIcon, LeafIcon, SparkIcon } from "../components/icons";
import { formatDuration, togglePracticeDone, useStore } from "../lib/store";

export const Route = createFileRoute("/practices")({
  head: () => ({
    meta: [
      { title: "Your Practices — MindAura AI" },
      {
        name: "description",
        content:
          "Gentle practices like breathing, reflection and gratitude journaling, to take at your own pace.",
      },
      { property: "og:title", content: "Your Practices — MindAura AI" },
      { property: "og:description", content: "Small steps you can take at your own pace." },
    ],
  }),
  component: Practices,
});

type Practice = {
  id: string;
  title: string;
  copy: string;
  minutes: number;
  kind: "breathing" | "writing" | "quiet";
};

const practices: Practice[] = [
  {
    id: "breathing",
    title: "5-Minute Breathing",
    copy: "Take a few quiet minutes to focus on your breathing.",
    minutes: 5,
    kind: "breathing",
  },
  {
    id: "reflection",
    title: "Daily Reflection",
    copy: "Spend a few minutes reflecting on your day.",
    minutes: 10,
    kind: "writing",
  },
  {
    id: "gratitude",
    title: "Gratitude Journal",
    copy: "Write down three things you're grateful for.",
    minutes: 5,
    kind: "writing",
  },
  {
    id: "mindfulness",
    title: "Mindfulness",
    copy: "Take a quiet moment to slow down and reconnect.",
    minutes: 5,
    kind: "quiet",
  },
  {
    id: "evening",
    title: "Evening Wind-down",
    copy: "Let the day settle with a slow, unhurried pause.",
    minutes: 8,
    kind: "quiet",
  },
];

function Practices() {
  const { practicesDone } = useStore();
  const [active, setActive] = useState<Practice | null>(null);
  const completed = practices.filter((p) => practicesDone.includes(p.id)).length;
  const pct = Math.round((completed / practices.length) * 100);

  if (active) {
    return (
      <PracticeSession
        practice={active}
        onExit={() => setActive(null)}
        onComplete={() => {
          if (!practicesDone.includes(active.id)) togglePracticeDone(active.id);
        }}
      />
    );
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-3xl text-primary-deep">Your Practices</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Small steps you can take at your own pace.
      </p>

      <div className="card-soft mt-7 p-7 animate-rise">
        <div className="flex items-center justify-between text-sm">
          <span className="text-primary-deep">
            {completed} of {practices.length} completed
          </span>
          <span className="text-muted-foreground">{pct}%</span>
        </div>
        <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-700 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="mt-6 grid gap-5 sm:grid-cols-2">
        {practices.map((p, i) => {
          const done = practicesDone.includes(p.id);
          return (
            <article
              key={p.id}
              className="card-soft lift flex flex-col p-7 animate-rise"
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <div className="flex items-start justify-between">
                <span
                  className={`flex h-11 w-11 items-center justify-center rounded-2xl ${
                    done ? "bg-primary text-primary-foreground" : "bg-primary-soft text-primary-deep"
                  }`}
                >
                  {done ? <CheckIcon size={20} /> : <LeafIcon size={20} />}
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-3 py-1 text-xs text-muted-foreground">
                  <ClockIcon size={13} /> {p.minutes} min
                </span>
              </div>
              <h2 className="mt-5 text-lg text-primary-deep">{p.title}</h2>
              <p className="mt-1.5 flex-1 text-sm text-muted-foreground">{p.copy}</p>
              <button type="button" className="btn-primary mt-6" onClick={() => setActive(p)}>
                {done ? "Do it again" : "Start"}
              </button>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function PracticeSession({
  practice,
  onExit,
  onComplete,
}: {
  practice: Practice;
  onExit: () => void;
  onComplete: () => void;
}) {
  const total = practice.minutes * 60;
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(true);
  const [note, setNote] = useState("");
  const finished = elapsed >= total;

  useEffect(() => {
    if (!running || finished) return;
    const id = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => window.clearInterval(id);
  }, [running, finished]);

  useEffect(() => {
    if (finished) onComplete();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);

  const phase = useMemo(() => {
    const t = elapsed % 12;
    if (t < 4) return "Inhale";
    if (t < 7) return "Hold";
    return "Exhale";
  }, [elapsed]);

  const scale = phase === "Inhale" ? 1.12 : phase === "Hold" ? 1.12 : 0.82;

  if (finished) {
    return (
      <div className="mx-auto flex max-w-xl flex-col items-center pt-14 text-center">
        <div className="relative flex h-40 w-40 items-center justify-center">
          <span className="absolute h-32 w-32 rounded-full bg-primary/20 blur-2xl animate-aura" />
          <span className="text-primary animate-leaf">
            <SparkIcon size={52} />
          </span>
        </div>
        <h1 className="mt-6 text-3xl text-primary-deep">Practice completed 🌱</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Nice work making space for {practice.title.toLowerCase()}.
        </p>
        <button type="button" className="btn-primary mt-9" onClick={onExit}>
          Back to practices
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-xl">
      <button
        type="button"
        onClick={onExit}
        className="text-sm text-muted-foreground transition-colors duration-300 hover:text-primary-deep"
      >
        ← Practices
      </button>

      <div className="mt-6 text-center animate-fade">
        <h1 className="text-3xl text-primary-deep">{practice.title}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{practice.copy}</p>
      </div>

      {practice.kind === "writing" ? (
        <div className="card-soft mt-8 p-7 animate-rise">
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={9}
            placeholder="Start writing here..."
            className="w-full resize-none bg-transparent text-base leading-relaxed outline-none placeholder:text-muted-foreground/70"
          />
        </div>
      ) : (
        <div className="card-soft mt-8 flex flex-col items-center gap-6 p-12 animate-rise">
          <div className="relative flex h-60 w-60 items-center justify-center">
            <span className="absolute h-48 w-48 rounded-full bg-lavender/20 blur-3xl animate-aura" />
            <span
              className="flex h-44 w-44 items-center justify-center rounded-full bg-primary/25"
              style={{
                transform: `scale(${scale})`,
                transition: "transform 3.4s cubic-bezier(0.37, 0, 0.63, 1)",
              }}
            >
              <span className="font-display text-xl tracking-wide text-primary-deep">{phase}</span>
            </span>
          </div>
        </div>
      )}

      <div className="card-soft mt-6 p-6 animate-fade">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>
            {formatDuration(elapsed)} / {formatDuration(total)}
          </span>
          <button
            type="button"
            onClick={() => setRunning((r) => !r)}
            className="text-primary-deep transition-opacity duration-300 hover:opacity-70"
          >
            {running ? "Pause" : "Resume"}
          </button>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-1000 ease-linear"
            style={{ width: `${Math.min(100, (elapsed / total) * 100)}%` }}
          />
        </div>
        <button
          type="button"
          className="btn-ghost mt-6 w-full"
          onClick={() => setElapsed(total)}
        >
          Finish now
        </button>
      </div>
    </div>
  );
}
