// Thin client for the MedTrust AI backend's check-in ingestion endpoints
// (backend/app/api/checkins.py). Sends the recorded/written check-in to
// the backend, which runs the appropriate AI pipeline, saves the
// structured result to the patient's record, and prints it to its own
// console. The AI output itself is never rendered back to the patient
// (see backend/app/api/patients.py's /history endpoint, used only by the
// doctor dashboard).

const API_BASE_URL =
  (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

// No patient login exists yet -- the whole app (like the backend's
// CURRENT_ACTIVE_USER_ID doctor stand-in) operates as the single seeded
// demo patient until real auth is built.
const DEMO_PATIENT_ID = "pat-1";

async function postForm(path: string, form: FormData): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, { method: "POST", body: form });
    if (!res.ok) {
      console.error(`[MindAura] ${path} failed:`, res.status, await res.text());
      return;
    }
    console.log(`[MindAura] ${path} accepted — check the backend console for the result.`);
  } catch (err) {
    // Non-fatal: the local check-in flow (src/lib/store.ts) still works
    // even if the backend is unreachable -- this is best-effort telemetry
    // to the AI pipeline, not a required part of saving a check-in.
    console.error(`[MindAura] Could not reach backend at ${API_BASE_URL}${path}:`, err);
  }
}

export function submitTextCheckIn(text: string): void {
  const form = new FormData();
  form.append("text", text);
  form.append("patient_id", DEMO_PATIENT_ID);
  void postForm("/checkins/text", form);
}

export function submitVoiceCheckIn(blob: Blob): void {
  const form = new FormData();
  form.append("file", blob, `voice-checkin.${_extensionFor(blob.type)}`);
  form.append("patient_id", DEMO_PATIENT_ID);
  void postForm("/checkins/voice", form);
}

export function submitVideoCheckIn(blob: Blob): void {
  const form = new FormData();
  form.append("file", blob, `video-checkin.${_extensionFor(blob.type)}`);
  form.append("patient_id", DEMO_PATIENT_ID);
  void postForm("/checkins/video", form);
}

function _extensionFor(mimeType: string): string {
  if (mimeType.includes("mp4")) return "mp4";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

// --- Doctor-side reads (backend/app/api/patients.py) ---

export interface PatientSummary {
  id: string;
  mrn: string;
  first_name: string;
  last_name: string;
  age: number;
  gender: string;
  blood_group: string;
  known_allergies: string[];
  chronic_conditions: string[];
  created_at: string;
}

export interface TextJournalRecord {
  id: string;
  patient_id: string;
  created_at: string;
  raw_input: string;
  analysis: Record<string, unknown>;
}

export interface VoiceRecord {
  id: string;
  patient_id: string;
  created_at: string;
  analysis: Record<string, unknown>;
}

export interface VideoAnalysisRecord {
  id: string;
  patient_id: string;
  video_id: string;
  analysis_id: string;
  created_at: string;
  face_model: string | null;
  voice_model: string | null;
  observation: Record<string, unknown>;
}

export interface PatientHistory {
  patient_id: string;
  text_journals: TextJournalRecord[];
  voice_records: VoiceRecord[];
  video_analyses: VideoAnalysisRecord[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function fetchPatients(search?: string): Promise<PatientSummary[]> {
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  return getJson(`/api/patients/${qs}`);
}

export function fetchPatient(patientId: string): Promise<PatientSummary> {
  return getJson(`/api/patients/${patientId}`);
}

export function fetchPatientHistory(patientId: string): Promise<PatientHistory> {
  return getJson(`/api/patients/${patientId}/history`);
}
