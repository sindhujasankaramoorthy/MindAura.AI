import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { fetchPatients } from "../../lib/api";
import { LeafIcon } from "../../components/icons";

export const Route = createFileRoute("/doctor/")({
  head: () => ({
    meta: [{ title: "Doctor Dashboard — MindAura AI" }],
  }),
  component: DoctorHome,
});

function DoctorHome() {
  const { data: patients, isLoading, isError } = useQuery({
    queryKey: ["patients"],
    queryFn: () => fetchPatients(),
  });

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl text-primary-deep">Doctor Dashboard</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Patient check-ins and AI observations, gathered for clinical review.
      </p>

      <div className="mt-8 flex items-center justify-between">
        <h2 className="text-lg text-primary-deep">Patients</h2>
        <Link to="/doctor/patients" className="btn-ghost">
          View all
        </Link>
      </div>

      {isLoading ? (
        <p className="mt-6 text-sm text-muted-foreground">Loading patients…</p>
      ) : isError ? (
        <p className="mt-6 text-sm text-destructive">
          Could not reach the backend. Is the API server running?
        </p>
      ) : !patients || patients.length === 0 ? (
        <div className="card-soft mt-6 flex flex-col items-center gap-4 p-12 text-center">
          <span className="text-primary/60">
            <LeafIcon size={32} />
          </span>
          <p className="text-muted-foreground">No patients registered yet.</p>
        </div>
      ) : (
        <ol className="mt-6 space-y-4">
          {patients.slice(0, 5).map((p) => (
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
                    {p.mrn} · {p.age} yrs · {p.gender}
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
