import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { fetchPatients } from "../../../lib/api";

export const Route = createFileRoute("/doctor/patients/")({
  head: () => ({
    meta: [{ title: "Patients — MindAura AI" }],
  }),
  component: PatientList,
});

function PatientList() {
  const [search, setSearch] = useState("");
  const { data: patients, isLoading, isError } = useQuery({
    queryKey: ["patients", search],
    queryFn: () => fetchPatients(search || undefined),
  });

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl text-primary-deep">Patients</h1>

      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search by name, MRN or phone"
        className="card-soft mt-6 w-full border border-border px-4 py-2.5 text-sm outline-none"
      />

      {isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading patients…</p>
      ) : isError ? (
        <p className="mt-6 text-sm text-destructive">
          Could not reach the backend. Is the API server running?
        </p>
      ) : !patients || patients.length === 0 ? (
        <p className="mt-6 text-sm text-muted-foreground">No patients found.</p>
      ) : (
        <ol className="mt-6 space-y-4">
          {patients.map((p) => (
            <li key={p.id}>
              <Link
                to="/doctor/patients/$patientId"
                params={{ patientId: p.id }}
                className="card-soft lift flex items-center justify-between p-5"
              >
                <div>
                  <p className="text-primary-deep">
                    {p.first_name} {p.last_name}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {p.mrn} · {p.age} yrs · {p.gender} · {p.blood_group || "Unknown"}
                  </p>
                </div>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
