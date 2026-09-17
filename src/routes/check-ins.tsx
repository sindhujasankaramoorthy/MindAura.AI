import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { CameraIcon, LeafIcon, MicIcon, NotebookIcon } from "../components/icons";
import { dayLabel, formatDuration, useStore, type CheckIn } from "../lib/store";

export const Route = createFileRoute("/check-ins")({
  head: () => ({
    meta: [
      { title: "My Check-ins — MindAura AI" },
      {
        name: "description",
        content: "Look back on your journal entries, voice notes and video check-ins in one calm timeline.",
      },
      { property: "og:title", content: "My Check-ins — MindAura AI" },
      { property: "og:description", content: "Your own check-ins, gathered in one calm timeline." },
    ],
  }),
  component: CheckIns,
});

const meta = {
  write: { label: "Journal check-in", Icon: NotebookIcon, tint: "bg-primary-soft" },
  speak: { label: "Voice check-in", Icon: MicIcon, tint: "bg-lavender-soft" },
  video: { label: "Video check-in", Icon: CameraIcon, tint: "bg-peach-soft" },
} as const;

function CheckIns() {
  const { checkins } = useStore();
  const [open, setOpen] = useState<CheckIn | null>(null);

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-3xl text-primary-deep">My Check-ins</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Everything you've shared, kept just for you.
      </p>

      {checkins.length === 0 ? (
        <div className="card-soft mt-8 flex flex-col items-center gap-4 p-12 text-center animate-fade">
          <span className="text-primary/60 animate-float">
            <LeafIcon size={32} />
          </span>
          <p className="text-muted-foreground">Your check-ins will appear here.</p>
          <Link to="/check-in" className="btn-primary">
            Start Today's Check-in
          </Link>
        </div>
      ) : (
        <ol className="mt-8 space-y-5">
          {checkins.map((c, i) => (
            <li
              key={c.id}
              className="animate-rise"
              style={{ animationDelay: `${Math.min(i, 6) * 60}ms` }}
            >
              <button
                type="button"
                onClick={() => setOpen(c)}
                className="card-soft lift w-full p-6 text-left"
              >
                <div className="flex items-start gap-4">
                  <span className="mt-0.5 flex flex-col items-center gap-2">
                    {c.types.map((t) => {
                      const M = meta[t];
                      return (
                        <span
                          key={t}
                          className={`flex h-10 w-10 items-center justify-center rounded-2xl text-primary-deep ${M.tint}`}
                        >
                          <M.Icon size={18} />
                        </span>
                      );
                    })}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                      {dayLabel(c.createdAt)}
                    </p>
                    <p className="mt-1 text-primary-deep">
                      {c.types.map((t) => meta[t].label).join(" · ")}
                    </p>
                    {c.text ? (
                      <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{c.text}</p>
                    ) : null}
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
                      {c.voiceSeconds ? (
                        <span className="rounded-full bg-secondary px-3 py-1">
                          Voice {formatDuration(c.voiceSeconds)}
                        </span>
                      ) : null}
                      {c.videoSeconds ? (
                        <span className="rounded-full bg-secondary px-3 py-1">
                          Video {formatDuration(c.videoSeconds)}
                        </span>
                      ) : null}
                      <span className="rounded-full bg-secondary px-3 py-1">
                        {new Date(c.createdAt).toLocaleTimeString(undefined, {
                          hour: "numeric",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            </li>
          ))}
        </ol>
      )}

      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-end justify-center bg-foreground/25 p-4 backdrop-blur-sm animate-fade sm:items-center"
          onClick={() => setOpen(null)}
        >
          <div
            className="card-soft max-h-[80vh] w-full max-w-lg overflow-y-auto p-8 animate-rise"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
              {dayLabel(open.createdAt)}
            </p>
            <h2 className="mt-1 text-2xl text-primary-deep">
              {open.types.map((t) => meta[t].label).join(" · ")}
            </h2>
            {open.text ? (
              <p className="mt-5 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                {open.text}
              </p>
            ) : null}
            {open.voiceSeconds ? (
              <p className="mt-5 rounded-3xl bg-secondary px-5 py-4 text-sm text-muted-foreground">
                Voice recording saved · {formatDuration(open.voiceSeconds)}
              </p>
            ) : null}
            {open.videoSeconds ? (
              <p className="mt-3 rounded-3xl bg-secondary px-5 py-4 text-sm text-muted-foreground">
                Video recording saved · {formatDuration(open.videoSeconds)}
              </p>
            ) : null}
            <div className="mt-7 flex justify-end">
              <button type="button" className="btn-ghost" onClick={() => setOpen(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
