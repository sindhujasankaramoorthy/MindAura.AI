import { createFileRoute, Link } from "@tanstack/react-router";
import { CameraIcon, CheckIcon, LeafIcon, MicIcon, NotebookIcon, SparkIcon } from "../components/icons";
import { useStore } from "../lib/store";
import { requireAuth } from "../lib/auth";

export const Route = createFileRoute("/")({
  beforeLoad: requireAuth,
  head: () => ({
    meta: [
      { title: "MindAura AI — Take a moment for yourself" },
      {
        name: "description",
        content:
          "Start a daily check-in by writing, speaking or recording a short video, and follow gentle practices at your own pace.",
      },
      { property: "og:title", content: "MindAura AI — Take a moment for yourself" },
      {
        property: "og:description",
        content: "A calm space for daily check-ins and gentle wellness practices.",
      },
    ],
  }),
  component: Home,
});

const modes = [
  {
    key: "write",
    title: "Write",
    copy: "Put your thoughts into words.",
    Icon: NotebookIcon,
    tint: "bg-primary-soft text-primary-deep",
  },
  {
    key: "speak",
    title: "Speak",
    copy: "Sometimes talking is easier.",
    Icon: MicIcon,
    tint: "bg-lavender-soft text-primary-deep",
  },
  {
    key: "video",
    title: "Video",
    copy: "Share a little more about your day.",
    Icon: CameraIcon,
    tint: "bg-peach-soft text-primary-deep",
  },
] as const;

const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function Home() {
  const { checkins, practicesDone } = useStore();

  const activeDays = new Set(checkins.map((c) => new Date(c.createdAt).toDateString()));
  const now = new Date();
  const monday = new Date(now);
  monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));

  let streak = 0;
  for (let i = 0; i < 60; i++) {
    const d = new Date();
    d.setDate(now.getDate() - i);
    if (activeDays.has(d.toDateString())) streak++;
    else if (i > 0) break;
  }

  return (
    <div className="space-y-16">
      <section className="grid items-center gap-10 md:grid-cols-[1.1fr_0.9fr]">
        <div className="animate-rise">
          <p className="mb-4 inline-flex items-center gap-2 rounded-full bg-card px-4 py-1.5 text-xs text-muted-foreground shadow-soft">
            <LeafIcon size={14} /> Your private space
          </p>
          <h1 className="text-4xl leading-tight text-primary-deep sm:text-5xl">
            Take a moment for yourself.
          </h1>
          <p className="mt-4 max-w-md text-base text-muted-foreground">
            Express what's on your mind in a way that feels comfortable.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/check-in" className="btn-primary">
              Start Today's Check-in
            </Link>
            <Link to="/practices" className="btn-ghost">
              Your Practices
            </Link>
          </div>
        </div>

        <div className="relative mx-auto flex h-72 w-72 items-center justify-center">
          <span className="absolute h-64 w-64 rounded-full bg-lavender/20 blur-3xl animate-aura" />
          <span className="absolute h-56 w-56 rounded-full border border-primary/20 animate-ripple" />
          <span
            className="absolute h-44 w-44 rounded-full border border-peach/40 animate-ripple"
            style={{ animationDelay: "1.4s" }}
          />
          <span className="h-40 w-40 rounded-full bg-primary/30 blur-sm animate-breathe" />
          <span className="absolute font-display text-sm tracking-wide text-primary-deep">
            breathe
          </span>
          <span className="absolute -right-2 top-6 text-primary/50 animate-float">
            <LeafIcon size={26} />
          </span>
        </div>
      </section>

      <section className="animate-fade">
        <h2 className="text-2xl text-primary-deep">How would you like to check in?</h2>
        <div className="mt-6 grid gap-5 sm:grid-cols-3">
          {modes.map(({ key, title, copy, Icon, tint }) => (
            <Link
              key={key}
              to="/check-in"
              search={{ mode: key }}
              className="card-soft lift group p-7"
            >
              <span
                className={`inline-flex h-12 w-12 items-center justify-center rounded-2xl transition-transform duration-500 group-hover:-translate-y-1 group-hover:rotate-3 ${tint}`}
              >
                <Icon size={22} />
              </span>
              <h3 className="mt-5 text-lg text-primary-deep">{title}</h3>
              <p className="mt-1.5 text-sm text-muted-foreground">{copy}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="animate-fade">
        <h2 className="text-2xl text-primary-deep">Your week</h2>
        <p className="mt-1.5 text-sm text-muted-foreground">
          A gentle look at the moments you've taken for yourself.
        </p>

        <div className="card-soft mt-6 p-7">
          <div className="grid grid-cols-7 gap-2 sm:gap-4">
            {weekDays.map((label, i) => {
              const d = new Date(monday);
              d.setDate(monday.getDate() + i);
              const done = activeDays.has(d.toDateString());
              return (
                <div key={label} className="flex flex-col items-center gap-2">
                  <span className="text-xs text-muted-foreground">{label}</span>
                  <span
                    className={`flex h-11 w-11 items-center justify-center rounded-full transition-all duration-500 ${
                      done ? "bg-primary-soft text-primary-deep" : "bg-muted text-muted-foreground/60"
                    }`}
                  >
                    {done ? <CheckIcon size={18} /> : "—"}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-4">
            <Stat label="Check-ins" value={checkins.length} />
            <Stat label="Practices" value={practicesDone.length} />
            <Stat label="Current streak" value={`${streak} ${streak === 1 ? "day" : "days"}`} />
            <Stat label="Days active" value={activeDays.size} />
          </div>
        </div>
      </section>

      <section className="card-soft flex flex-wrap items-center justify-between gap-5 bg-primary-soft/60 p-7">
        <div className="flex items-center gap-3">
          <SparkIcon className="text-primary-deep" />
          <p className="text-sm text-primary-deep">
            Whatever you share stays yours. Take it at your own pace.
          </p>
        </div>
        <Link to="/check-ins" className="btn-ghost">
          View my check-ins
        </Link>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-3xl bg-secondary/70 px-5 py-4">
      <p className="font-display text-2xl text-primary-deep">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{label}</p>
    </div>
  );
}
