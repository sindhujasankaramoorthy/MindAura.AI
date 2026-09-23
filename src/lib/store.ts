import { useEffect, useState } from "react";
import { getStoredPatient } from "./auth";

export type CheckInType = "write" | "speak" | "video";

export type CheckIn = {
  id: string;
  createdAt: string;
  types: CheckInType[];
  text?: string;
  voiceSeconds?: number;
  videoSeconds?: number;
};

export type StoreState = {
  checkins: CheckIn[];
  practicesDone: string[];
};

// Namespaced per logged-in patient -- a shared, unscoped key here meant
// any patient logging into the same browser saw whatever local check-in/
// practice history the previous patient had left behind.
function storageKey(): string {
  const patient = getStoredPatient();
  return `mindaura.patient.v1.${patient?.id ?? "anonymous"}`;
}

const seed: StoreState = {
  checkins: [],
  practicesDone: [],
};

let state: StoreState = seed;
let hydratedKey: string | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

function hydrate() {
  if (typeof window === "undefined") return;
  const key = storageKey();
  if (hydratedKey === key) return;
  hydratedKey = key;
  state = seed;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw) state = { ...seed, ...(JSON.parse(raw) as StoreState) };
  } catch {
    /* ignore */
  }
  emit();
}

function persist() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(storageKey(), JSON.stringify(state));
  } catch {
    /* ignore */
  }
}

export function useStore(): StoreState {
  const [snapshot, setSnapshot] = useState<StoreState>(state);
  useEffect(() => {
    const listener = () => setSnapshot(state);
    listeners.add(listener);
    hydrate();
    listener();
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return snapshot;
}

export function addCheckIn(entry: Omit<CheckIn, "id" | "createdAt">) {
  hydrate();
  state = {
    ...state,
    checkins: [
      { ...entry, id: crypto.randomUUID(), createdAt: new Date().toISOString() },
      ...state.checkins,
    ],
  };
  persist();
  emit();
}

export function togglePracticeDone(id: string) {
  hydrate();
  const done = state.practicesDone.includes(id)
    ? state.practicesDone.filter((p) => p !== id)
    : [...state.practicesDone, id];
  state = { ...state, practicesDone: done };
  persist();
  emit();
}

export function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function dayLabel(iso: string) {
  const d = new Date(iso);
  const today = new Date();
  const diff = Math.floor(
    (new Date(today.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 86400000,
  );
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return d.toLocaleDateString(undefined, { weekday: "long" });
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
