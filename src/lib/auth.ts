// Patient authentication client for backend/app/api/auth.py's
// register/login/logout endpoints. The session token is the only thing
// stored client-side; the backend always resolves patient_id from it
// (backend/app/security.get_current_patient_id), so nothing here can be
// used to impersonate another patient.

import { redirect } from "@tanstack/react-router";

const API_BASE_URL =
  (import.meta as { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

const TOKEN_KEY = "mindaura_token";
const PATIENT_KEY = "mindaura_patient";

export interface AuthPatient {
  id: string;
  name: string;
  email: string;
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getStoredPatient(): AuthPatient | null {
  try {
    const raw = localStorage.getItem(PATIENT_KEY);
    return raw ? (JSON.parse(raw) as AuthPatient) : null;
  } catch {
    return null;
  }
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

/** Route beforeLoad guard for protected pages -- redirects to /login when
 * there's no session token. Backend endpoints still independently verify
 * the token itself; this only keeps an unauthenticated visitor out of the
 * UI. */
export function requireAuth(): void {
  if (!isAuthenticated()) {
    throw redirect({ to: "/login" });
  }
}

export function clearAuth(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(PATIENT_KEY);
  } catch {
    // best-effort
  }
}

function persistAuth(token: string, patient: AuthPatient): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(PATIENT_KEY, JSON.stringify(patient));
  } catch {
    // best-effort -- the session still works for this tab via the token
    // held in memory by the caller, even if storage is blocked
  }
}

async function postJson(path: string, body: unknown): Promise<{ token: string; patient: AuthPatient }> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }

  const data = (await res.json().catch(() => ({}))) as {
    detail?: string;
    token?: string;
    patient?: AuthPatient;
  };

  if (!res.ok) {
    throw new Error(data.detail ?? "Something went wrong. Please try again.");
  }
  return { token: data.token as string, patient: data.patient as AuthPatient };
}

export async function registerPatient(
  name: string,
  email: string,
  password: string,
  confirmPassword: string,
): Promise<AuthPatient> {
  const { token, patient } = await postJson("/api/auth/register", {
    name,
    email,
    password,
    confirm_password: confirmPassword,
  });
  persistAuth(token, patient);
  return patient;
}

export async function loginPatient(email: string, password: string): Promise<AuthPatient> {
  const { token, patient } = await postJson("/api/auth/login", { email, password });
  persistAuth(token, patient);
  return patient;
}

export async function logoutPatient(): Promise<void> {
  const token = getToken();
  clearAuth();
  if (!token) return;
  try {
    await fetch(`${API_BASE_URL}/api/auth/logout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    // logout is already effective locally; the token will simply expire
    // unused server-side if this request doesn't land
  }
}
