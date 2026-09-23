import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { CameraIcon, CheckIcon, LeafIcon, MicIcon, NotebookIcon } from "../components/icons";
import { addCheckIn, formatDuration, type CheckInType } from "../lib/store";
import { submitTextCheckIn, submitVoiceCheckIn, submitVideoCheckIn, submitCheckInSession } from "../lib/api";
import { requireAuth } from "../lib/auth";

export const Route = createFileRoute("/check-in")({
  beforeLoad: requireAuth,
  validateSearch: (search: Record<string, unknown>) => {
    const mode = search["mode"];
    return mode === "write" || mode === "speak" || mode === "video"
      ? { mode: mode as CheckInType }
      : {};
  },
  head: () => ({
    meta: [
      { title: "Today's Check-in — MindAura AI" },
      {
        name: "description",
        content:
          "Check in by writing, speaking or recording a short video — whichever feels most comfortable today.",
      },
      { property: "og:title", content: "Today's Check-in — MindAura AI" },
      { property: "og:description", content: "Express what's on your mind, your way." },
    ],
  }),
  component: CheckInFlow,
});

const options = [
  { key: "write", title: "Write", copy: "Put your thoughts into words.", Icon: NotebookIcon },
  { key: "speak", title: "Speak", copy: "Sometimes talking is easier.", Icon: MicIcon },
  { key: "video", title: "Video", copy: "Share a little more about your day.", Icon: CameraIcon },
] as const;

function CheckInFlow() {
  const { mode } = Route.useSearch();
  const navigate = useNavigate();

  const [step, setStep] = useState(mode ? 2 : 1);
  const [selected, setSelected] = useState<CheckInType[]>(mode ? [mode] : []);
  const [activityIndex, setActivityIndex] = useState(0);
  const [text, setText] = useState("");
  const [voiceSeconds, setVoiceSeconds] = useState(0);
  const [videoSeconds, setVideoSeconds] = useState(0);

  const ordered = (["write", "speak", "video"] as CheckInType[]).filter((t) =>
    selected.includes(t),
  );
  const current = ordered[activityIndex];

  function toggle(key: CheckInType) {
    setSelected((s) => (s.includes(key) ? s.filter((k) => k !== key) : [...s, key]));
  }

  function nextActivity(payload?: { text?: string; voice?: number; video?: number }) {
    if (payload?.text !== undefined) setText(payload.text);
    if (payload?.voice !== undefined) setVoiceSeconds(payload.voice);
    if (payload?.video !== undefined) setVideoSeconds(payload.video);

    if (activityIndex < ordered.length - 1) {
      setActivityIndex((i) => i + 1);
    } else {
      const finalText = payload?.text ?? text ?? undefined;
      const finalVoice = payload?.voice ?? voiceSeconds ?? undefined;
      const finalVideo = payload?.video ?? videoSeconds ?? undefined;
      addCheckIn({ types: ordered, text: finalText, voiceSeconds: finalVoice, videoSeconds: finalVideo });
      void submitCheckInSession(ordered, finalText, finalVoice, finalVideo);
      setStep(3);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <Steps step={step} />

      {step === 1 ? (
        <section className="mt-10 animate-rise">
          <h1 className="text-3xl text-primary-deep">How would you like to check in today?</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose one, or as many as you like. There's no wrong way.
          </p>

          <div className="mt-8 space-y-4">
            {options.map(({ key, title, copy, Icon }) => {
              const active = selected.includes(key);
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => toggle(key)}
                  className={`card-soft lift flex w-full items-center gap-5 p-6 text-left transition-colors duration-300 ${
                    active ? "ring-2 ring-primary/50" : ""
                  }`}
                >
                  <span
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl transition-colors duration-300 ${
                      active ? "bg-primary text-primary-foreground" : "bg-secondary text-primary-deep"
                    }`}
                  >
                    <Icon size={22} />
                  </span>
                  <span className="flex-1">
                    <span className="block text-lg text-primary-deep">{title}</span>
                    <span className="block text-sm text-muted-foreground">{copy}</span>
                  </span>
                  <span
                    className={`flex h-6 w-6 items-center justify-center rounded-full border transition-all duration-300 ${
                      active
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-border text-transparent"
                    }`}
                  >
                    <CheckIcon size={14} />
                  </span>
                </button>
              );
            })}
          </div>

          <button
            type="button"
            disabled={selected.length === 0}
            onClick={() => {
              setActivityIndex(0);
              setStep(2);
            }}
            className="btn-primary mt-8 w-full"
          >
            Continue
          </button>
        </section>
      ) : null}

      {step === 2 && current === "write" ? (
        <WriteStep initial={text} onDone={(t) => nextActivity({ text: t })} />
      ) : null}
      {step === 2 && current === "speak" ? (
        <VoiceStep onDone={(s) => nextActivity({ voice: s })} />
      ) : null}
      {step === 2 && current === "video" ? (
        <VideoStep onDone={(s) => nextActivity({ video: s })} />
      ) : null}

      {step === 3 ? (
        <section className="mt-14 flex flex-col items-center text-center">
          <div className="relative flex h-40 w-40 items-center justify-center">
            <span className="absolute h-32 w-32 rounded-full bg-primary/20 blur-2xl animate-aura" />
            <span className="absolute h-32 w-32 rounded-full border border-primary/30 animate-ripple" />
            <span className="text-primary animate-leaf">
              <LeafIcon size={54} />
            </span>
          </div>
          <h1 className="mt-6 text-3xl text-primary-deep">You're all checked in 🌿</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Thank you for taking a moment for yourself.
          </p>
          <p className="mt-6 inline-flex items-center gap-2 rounded-full bg-primary-soft px-5 py-2 text-sm text-primary-deep">
            <CheckIcon size={16} /> Check-in saved securely
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Link to="/check-ins" className="btn-primary">
              View My Check-ins
            </Link>
            <button type="button" className="btn-ghost" onClick={() => navigate({ to: "/" })}>
              Done
            </button>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function Steps({ step }: { step: number }) {
  return (
    <div className="flex items-center justify-center gap-3">
      <span className="mr-2 text-xs uppercase tracking-[0.2em] text-muted-foreground">
        Check-in
      </span>
      {[1, 2, 3].map((n) => (
        <span key={n} className="flex items-center gap-3">
          <span
            className={`flex h-8 w-8 items-center justify-center rounded-full text-sm transition-all duration-500 ${
              step >= n ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"
            }`}
          >
            {step > n ? <CheckIcon size={14} /> : n}
          </span>
          {n < 3 ? (
            <span
              className={`h-px w-8 transition-colors duration-500 ${
                step > n ? "bg-primary" : "bg-border"
              }`}
            />
          ) : null}
        </span>
      ))}
    </div>
  );
}

const prompts = [
  "What was the best part of your day?",
  "Was anything bothering you today?",
  "What would you like to talk about?",
];

type SubmitPhase = "idle" | "submitting" | "processing" | "error";

function WriteStep({ initial, onDone }: { initial: string; onDone: (text: string) => void }) {
  const [value, setValue] = useState(initial);
  const [phase, setPhase] = useState<SubmitPhase>("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!value.trim()) {
      setError("Please write something before continuing.");
      return;
    }
    setError(null);
    setPhase("submitting");
    const toProcessing = window.setTimeout(() => setPhase("processing"), 350);

    const result = await submitTextCheckIn(value);
    window.clearTimeout(toProcessing);

    if (!result.ok) {
      setPhase("error");
      setError(result.error);
      return;
    }
    onDone(value);
  }

  const busy = phase === "submitting" || phase === "processing";
  const buttonLabel =
    phase === "submitting" ? "Submitting..." : phase === "processing" ? "Processing..." : "Save & Continue";

  return (
    <section className="mt-10 animate-rise">
      <h1 className="text-3xl text-primary-deep">What's on your mind?</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Take your time. There is no right or wrong way to write.
      </p>

      <div className="relative mt-7">
        <span className="pointer-events-none absolute -right-3 -top-6 text-primary/40 animate-float">
          <LeafIcon size={26} />
        </span>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Start writing here..."
          rows={11}
          className="card-soft w-full resize-none p-7 text-base leading-relaxed text-foreground outline-none placeholder:text-muted-foreground/70 focus:ring-2 focus:ring-primary/40"
        />
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {prompts.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setValue((v) => (v ? `${v}\n\n${p}\n` : `${p}\n`))}
            className="rounded-full border border-border bg-card px-4 py-2 text-xs text-muted-foreground transition-all duration-300 hover:-translate-y-0.5 hover:text-primary-deep"
          >
            {p}
          </button>
        ))}
      </div>

      {error ? <p className="mt-4 text-sm text-destructive">{error}</p> : null}

      <button
        type="button"
        disabled={busy}
        onClick={handleSubmit}
        className="btn-primary mt-4 w-full disabled:opacity-60"
      >
        {buttonLabel}
      </button>
    </section>
  );
}

function useTimer(running: boolean) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [running]);
  return [seconds, setSeconds] as const;
}

function VoiceStep({ onDone }: { onDone: (seconds: number) => void }) {
  const [phase, setPhase] = useState<"idle" | "recording" | "paused" | "saved">("idle");
  const [seconds, setSeconds] = useTimer(phase === "recording");
  const [levels, setLevels] = useState<number[]>(Array.from({ length: 32 }, () => 0.15));
  const [uploadPhase, setUploadPhase] = useState<SubmitPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const blobRef = useRef<Blob | null>(null);

  function cleanup() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close().catch(() => {});
    streamRef.current = null;
    ctxRef.current = null;
  }

  useEffect(() => cleanup, []);

  async function start() {
    setSeconds(0);
    setPhase("recording");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        blobRef.current = new Blob(chunksRef.current, { type: recorder.mimeType });
      };
      recorder.start();
      recorderRef.current = recorder;

      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteFrequencyData(data);
        setLevels(Array.from(data.slice(0, 32), (v) => Math.max(0.12, v / 255)));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      const tick = () => {
        setLevels((prev) => prev.map((_, i) => 0.2 + Math.abs(Math.sin(Date.now() / 320 + i)) * 0.7));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    }
  }

  function finish() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    cleanup();
    setPhase("saved");
  }

  async function handleContinue() {
    if (!blobRef.current) {
      setError("No recording found. Please record again.");
      return;
    }
    setError(null);
    setUploadPhase("submitting");
    const toProcessing = window.setTimeout(() => setUploadPhase("processing"), 350);

    const result = await submitVoiceCheckIn(blobRef.current);
    window.clearTimeout(toProcessing);

    if (!result.ok) {
      setUploadPhase("error");
      setError(result.error);
      return;
    }
    onDone(seconds);
  }

  return (
    <section className="mt-10 animate-rise text-center">
      <h1 className="text-3xl text-primary-deep">Want to talk instead?</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Sometimes it's easier to say what's on your mind.
      </p>

      <div className="card-soft mt-8 flex flex-col items-center gap-6 p-10">
        <button
          type="button"
          onClick={phase === "idle" ? start : undefined}
          aria-label={phase === "idle" ? "Tap to start recording" : "Recording"}
          className={`relative flex h-36 w-36 items-center justify-center rounded-full transition-all duration-500 ${
            phase === "recording"
              ? "bg-primary text-primary-foreground animate-pulse-soft"
              : "bg-primary-soft text-primary-deep hover:scale-105"
          }`}
        >
          {phase === "recording" ? (
            <span className="absolute inset-0 rounded-full border border-primary/40 animate-ripple" />
          ) : null}
          <MicIcon size={44} />
        </button>

        {phase === "idle" ? (
          <p className="text-sm text-muted-foreground">Tap to start</p>
        ) : (
          <>
            <div className="flex h-16 items-end justify-center gap-1">
              {levels.map((l, i) => (
                <span
                  key={i}
                  className="w-1.5 rounded-full bg-primary/70 transition-[height] duration-150"
                  style={{ height: `${Math.max(6, l * 60)}px` }}
                />
              ))}
            </div>
            <p className="font-display text-2xl text-primary-deep">{formatDuration(seconds)}</p>
          </>
        )}

        {phase === "recording" || phase === "paused" ? (
          <div className="flex flex-wrap justify-center gap-3">
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setPhase(phase === "recording" ? "paused" : "recording")}
            >
              {phase === "recording" ? "Pause" : "Resume"}
            </button>
            <button type="button" className="btn-primary" onClick={finish}>
              Finish Recording
            </button>
          </div>
        ) : null}

        {phase === "saved" ? (
          <div className="flex flex-col items-center gap-4 animate-fade">
            <p className="inline-flex items-center gap-2 rounded-full bg-primary-soft px-5 py-2 text-sm text-primary-deep">
              <CheckIcon size={16} /> Your recording is saved.
            </p>
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <div className="flex flex-wrap justify-center gap-3">
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setSeconds(0);
                  setError(null);
                  setUploadPhase("idle");
                  setPhase("idle");
                }}
              >
                Record Again
              </button>
              <button
                type="button"
                disabled={uploadPhase === "submitting" || uploadPhase === "processing"}
                className="btn-primary disabled:opacity-60"
                onClick={handleContinue}
              >
                {uploadPhase === "submitting"
                  ? "Uploading..."
                  : uploadPhase === "processing"
                    ? "Processing voice..."
                    : "Continue"}
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function VideoStep({ onDone }: { onDone: (seconds: number) => void }) {
  const [phase, setPhase] = useState<"idle" | "recording" | "saved">("idle");
  const [seconds, setSeconds] = useTimer(phase === "recording");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const blobRef = useRef<Blob | null>(null);

  function stopStream() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  useEffect(() => stopStream, []);

  async function start() {
    setSeconds(0);
    setPhase("recording");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        blobRef.current = new Blob(chunksRef.current, { type: recorder.mimeType });
      };
      recorder.start();
      recorderRef.current = recorder;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
    } catch {
      /* camera unavailable — keep the calm placeholder */
    }
  }

  return (
    <section className="mt-10 animate-rise text-center">
      <h1 className="text-3xl text-primary-deep">A little more about your day</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        You can record a short video check-in if you'd like.
      </p>

      <div className="card-soft mt-8 p-6">
        <div className="relative aspect-video w-full overflow-hidden rounded-3xl bg-secondary">
          <span
            className={`pointer-events-none absolute inset-0 z-10 rounded-3xl border-2 transition-colors duration-700 ${
              phase === "recording" ? "border-primary/60 animate-pulse-soft" : "border-border"
            }`}
          />
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full object-cover"
            aria-label="Camera preview"
          />
          {phase === "idle" ? (
            <span className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
              <CameraIcon size={38} />
              <span className="text-sm">Your camera preview appears here</span>
            </span>
          ) : null}
          {phase === "recording" ? (
            <span className="absolute left-4 top-4 z-20 rounded-full bg-card/90 px-3 py-1 text-xs text-primary-deep">
              {formatDuration(seconds)}
            </span>
          ) : null}
        </div>

        <div className="mt-6 flex flex-wrap justify-center gap-3">
          {phase === "idle" ? (
            <button type="button" className="btn-primary" onClick={start}>
              Start Recording
            </button>
          ) : null}
          {phase === "recording" ? (
            <button
              type="button"
              className="btn-primary"
              onClick={() => {
                stopStream();
                setPhase("saved");
              }}
            >
              Stop
            </button>
          ) : null}
          {phase === "saved" ? (
            <>
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setSeconds(0);
                  setPhase("idle");
                }}
              >
                Retake
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => {
                  if (blobRef.current) submitVideoCheckIn(blobRef.current);
                  onDone(seconds);
                }}
              >
                Continue
              </button>
            </>
          ) : null}
        </div>

        {phase === "saved" ? (
          <p className="mt-5 inline-flex items-center gap-2 rounded-full bg-primary-soft px-5 py-2 text-sm text-primary-deep animate-fade">
            <CheckIcon size={16} /> Your recording is saved.
          </p>
        ) : null}
      </div>
    </section>
  );
}
