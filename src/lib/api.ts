// Thin client for the MedTrust AI backend's check-in ingestion endpoints
// (backend/app/api/checkins.py). Sends the recorded/written check-in from
// the existing Write/Speak/Video boxes to the backend, which runs the
// appropriate AI pipeline and saves the result under the logged-in
// patient (see src/lib/auth.ts). Callers use the returned SubmitResult to
// show submitting/processing/error states without exposing the actual AI
// analysis to the patient.

import { getToken, clearAuth } from "./auth";

export const API_BASE_URL =
  (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

export type SubmitResult = { ok: true } | { ok: false; error: string };

async function postForm(path: string, form: FormData): Promise<SubmitResult> {
  const token = getToken();
  if (!token) {
    return { ok: false, error: "Please log in to submit a check-in." };
  }

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });

    if (res.status === 401) {
      clearAuth();
      window.location.href = "/login";
      return { ok: false, error: "Your session has expired. Please log in again." };
    }

    if (!res.ok) {
      let detail = "";
      try {
        detail = JSON.stringify(await res.json());
      } catch {
        detail = await res.text();
      }
      console.error(`[MindAura] ${path} failed:`, res.status, detail);
      return { ok: false, error: "Something went wrong. Please try again." };
    }

    const body = (await res.json()) as { status?: string; error?: string };
    if (body.status === "failed") {
      return { ok: false, error: body.error ?? "Something went wrong. Please try again." };
    }
    return { ok: true };
  } catch (err) {
    console.error(`[MindAura] Could not reach backend at ${API_BASE_URL}${path}:`, err);
    return { ok: false, error: "Could not reach the server. Check your connection and try again." };
  }
}

export function submitTextCheckIn(text: string): Promise<SubmitResult> {
  const form = new FormData();
  form.append("text", text);
  return postForm("/checkins/text", form);
}

export function submitVoiceCheckIn(blob: Blob): Promise<SubmitResult> {
  const form = new FormData();
  form.append("file", blob, `voice-checkin.${_extensionFor(blob.type)}`);
  return postForm("/checkins/voice", form);
}

export function submitVideoCheckIn(blob: Blob): Promise<SubmitResult> {
  const form = new FormData();
  form.append("file", blob, `video-checkin.${_extensionFor(blob.type)}`);
  return postForm("/checkins/video", form);
}

function _extensionFor(mimeType: string): string {
  if (mimeType.includes("mp4")) return "mp4";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

async function authedFetch(path: string, init: RequestInit = {}): Promise<Response | null> {
  const token = getToken();
  if (!token) return null;

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${token}` },
    });
    if (res.status === 401) {
      clearAuth();
      window.location.href = "/login";
      return null;
    }
    return res;
  } catch (err) {
    console.error(`[MindAura] Could not reach backend at ${API_BASE_URL}${path}:`, err);
    return null;
  }
}

async function sendJson(method: string, path: string, body: unknown): Promise<SubmitResult> {
  const res = await authedFetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res) return { ok: false, error: "Could not reach the server. Check your connection and try again." };
  if (!res.ok) {
    console.error(`[MindAura] ${path} failed:`, res.status, await res.text());
    return { ok: false, error: "Something went wrong. Please try again." };
  }
  return { ok: true };
}

function postJson(path: string, body: unknown): Promise<SubmitResult> {
  return sendJson("POST", path, body);
}

export function submitCheckInSession(
  types: string[],
  text?: string,
  voiceSeconds?: number,
  videoSeconds?: number,
): Promise<SubmitResult> {
  return postJson("/checkins/session", {
    types,
    text: text ?? null,
    voice_seconds: voiceSeconds ?? null,
    video_seconds: videoSeconds ?? null,
  });
}

export function submitPracticeCompletion(practiceId: string, note?: string): Promise<SubmitResult> {
  return postJson("/api/practices/complete", { practice_id: practiceId, note: note ?? null });
}

export interface PatientPreferences {
  daily_checkin_reminder: boolean;
  practice_reminders: boolean;
}

export async function fetchPreferences(): Promise<PatientPreferences | null> {
  const res = await authedFetch("/api/preferences/");
  if (!res || !res.ok) return null;
  return res.json();
}

export function updatePreferences(update: Partial<PatientPreferences>): Promise<SubmitResult> {
  return sendJson("PUT", "/api/preferences/", update);
}
