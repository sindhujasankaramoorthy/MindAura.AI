import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  fetchPatient,
  fetchPatientHistory,
  type TextJournalRecord,
  type VoiceRecord,
  type VideoAnalysisRecord,
} from "../../../lib/api";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../../../components/ui/tabs";

export const Route = createFileRoute("/doctor/patients/$patientId")({
  head: () => ({
    meta: [{ title: "Patient — MindAura AI" }],
  }),
  component: PatientDetail,
});

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString();
}

function PatientDetail() {
  const { patientId } = Route.useParams();

  const patientQuery = useQuery({
    queryKey: ["patient", patientId],
    queryFn: () => fetchPatient(patientId),
  });
  const historyQuery = useQuery({
    queryKey: ["patient-history", patientId],
    queryFn: () => fetchPatientHistory(patientId),
  });

  if (patientQuery.isLoading || historyQuery.isLoading) {
    return <p className="mx-auto max-w-4xl text-sm text-muted-foreground">Loading patient…</p>;
  }
  if (patientQuery.isError || !patientQuery.data) {
    return (
      <div className="mx-auto max-w-4xl">
        <p className="text-sm text-destructive">Could not load this patient.</p>
        <Link to="/doctor/patients" className="btn-ghost mt-4 inline-block">
          Back to patients
        </Link>
      </div>
    );
  }

  const patient = patientQuery.data;
  const history = historyQuery.data;

  return (
    <div className="mx-auto max-w-4xl">
      <Link to="/doctor/patients" className="text-sm text-muted-foreground hover:text-primary-deep">
        ← Back to patients
      </Link>

      <div className="card-soft mt-4 p-6">
        <h1 className="text-2xl text-primary-deep">
          {patient.first_name} {patient.last_name}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {patient.mrn} · {patient.age} yrs · {patient.gender} · {patient.blood_group || "Unknown"}
        </p>

        {(patient.chronic_conditions?.length > 0 || patient.known_allergies?.length > 0) && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {patient.chronic_conditions?.length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  Clinical baseline
                </p>
                <p className="mt-1 text-sm text-foreground">
                  {patient.chronic_conditions.join(", ")}
                </p>
              </div>
            )}
            {patient.known_allergies?.length > 0 && (
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  Known allergies
                </p>
                <p className="mt-1 text-sm text-foreground">{patient.known_allergies.join(", ")}</p>
              </div>
            )}
          </div>
        )}
      </div>

      <Tabs defaultValue="text" className="mt-8">
        <TabsList>
          <TabsTrigger value="text">
            Text ({history?.text_journals.length ?? 0})
          </TabsTrigger>
          <TabsTrigger value="voice">
            Voice ({history?.voice_records.length ?? 0})
          </TabsTrigger>
          <TabsTrigger value="video">
            Video ({history?.video_analyses.length ?? 0})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="text">
          <TextHistory records={history?.text_journals ?? []} />
        </TabsContent>
        <TabsContent value="voice">
          <VoiceHistory records={history?.voice_records ?? []} />
        </TabsContent>
        <TabsContent value="video">
          <VideoHistory records={history?.video_analyses ?? []} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function EmptyState({ label }: { label: string }) {
  return (
    <p className="mt-6 text-sm text-muted-foreground">No {label} check-ins yet.</p>
  );
}

function TextHistory({ records }: { records: TextJournalRecord[] }) {
  if (records.length === 0) return <EmptyState label="text" />;
  return (
    <ol className="mt-6 space-y-4">
      {records.map((r) => {
        const analysis = r.analysis as Record<string, unknown>;
        const language = analysis["language"] as Record<string, unknown> | undefined;
        return (
          <li key={r.id} className="card-soft p-5">
            <p className="text-xs text-muted-foreground">{formatDate(r.created_at)}</p>
            <p className="mt-2 text-foreground">{r.raw_input}</p>
            {language && (
              <p className="mt-3 text-xs text-muted-foreground">
                Language: {String(language["primary_language"] ?? "unknown")}
                {language["mixed_language"] ? " (mixed)" : ""}
                {language["tanglish_fallback_used"] ? " · Tanglish fallback" : ""}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function VoiceHistory({ records }: { records: VoiceRecord[] }) {
  if (records.length === 0) return <EmptyState label="voice" />;
  return (
    <ol className="mt-6 space-y-4">
      {records.map((r) => {
        const a = r.analysis as Record<string, unknown>;
        const features = a["acoustic_features"] as Record<string, unknown> | undefined;
        return (
          <li key={r.id} className="card-soft p-5">
            <p className="text-xs text-muted-foreground">{formatDate(r.created_at)}</p>
            <p className="mt-2 text-foreground">
              {String(a["raw_transcript"] ?? "(no transcript)")}
            </p>
            {features && (
              <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground sm:grid-cols-3">
                <div>Pitch: {formatNum(features["pitch_mean_hz"])} Hz</div>
                <div>Rate: {formatNum(features["speaking_rate_wps"])} wps</div>
                <div>Pauses: {formatNum(features["pause_ratio"])}</div>
                <div>Variability: {formatNum(features["pitch_variability"])}</div>
                <div>Energy: {formatNum(features["energy_rms"])}</div>
              </dl>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function VideoHistory({ records }: { records: VideoAnalysisRecord[] }) {
  if (records.length === 0) return <EmptyState label="video" />;
  return (
    <ol className="mt-6 space-y-4">
      {records.map((r) => (
        <li key={r.id} className="card-soft p-5">
          <p className="text-xs text-muted-foreground">{formatDate(r.created_at)}</p>
          <p className="mt-2 text-sm text-foreground">
            Face model: {r.face_model ?? "not_available"} · Voice model:{" "}
            {r.voice_model ?? "not_available"}
          </p>
          <details className="mt-3 text-xs text-muted-foreground">
            <summary className="cursor-pointer">Full observation JSON</summary>
            <pre className="mt-2 overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(r.observation, null, 2)}
            </pre>
          </details>
        </li>
      ))}
    </ol>
  );
}

function formatNum(v: unknown): string {
  return typeof v === "number" ? v.toFixed(2) : String(v ?? "n/a");
}
