"""
MedTrust AI - 17-Section Case Sheet & Doctor Sign-Off API
Handles AI structured case sheet synthesis, inline section editing,
doctor verification and electronic sign-off, immutable locking,
multilingual summaries (6 languages), and print/PDF formatting.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body, Response
from fastapi.responses import HTMLResponse
from backend.app.database import (
    get_db_connection,
    get_patient_by_id
)
from backend.app.services.ai_casesheet_service import (
    generate_case_sheet_from_transcript,
    compute_approval_hash,
    generate_multilingual_summaries
)

router = APIRouter(prefix="/api/casesheets", tags=["casesheets"])


@router.get("/{consultation_id}")
def get_case_sheet_by_consultation(consultation_id: str):
    """Retrieves the 17-section case sheet associated with a consultation."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT cs.*,
           p.first_name || ' ' || p.last_name as patient_name,
           p.mrn as patient_mrn,
           p.date_of_birth,
           p.age as patient_age,
           p.gender as patient_gender,
           p.blood_group as patient_blood_group,
           p.phone as patient_phone,
           u.name as doctor_name,
           u.registration_number as doctor_registration,
           u.specialization as doctor_specialization,
           s.name as student_name
    FROM casesheets cs
    JOIN patients p ON cs.patient_id = p.id
    JOIN users u ON cs.doctor_id = u.id
    LEFT JOIN users s ON cs.student_id = s.id
    WHERE cs.consultation_id = ?
    """, (consultation_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No case sheet found for this consultation.")

    d = dict(row)
    d["sections"] = json.loads(d.get("sections_json") or "{}")
    d["multilingual_summary"] = json.loads(d.get("multilingual_summary_json") or "{}")
    d["approval"] = json.loads(d.get("approval_json") or "{}")
    return d


@router.post("/generate/{consultation_id}")
def generate_ai_case_sheet(consultation_id: str):
    """
    Synthesizes the complete 17-section clinical case sheet from
    consultation transcripts using Gemini AI (or offline Clinical NLP fallback).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Fetch consultation metadata
    cursor.execute("SELECT * FROM consultations WHERE id = ?", (consultation_id,))
    consult_row = cursor.fetchone()
    if not consult_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Consultation not found.")

    consultation = dict(consult_row)

    # 2. Fetch patient and doctor details
    patient = get_patient_by_id(consultation["patient_id"])
    cursor.execute("SELECT * FROM users WHERE id = ?", (consultation["doctor_id"],))
    doctor = dict(cursor.fetchone())

    student = None
    if consultation.get("student_id"):
        cursor.execute("SELECT * FROM users WHERE id = ?", (consultation["student_id"],))
        st_row = cursor.fetchone()
        if st_row:
            student = dict(st_row)

    # 3. Fetch transcript turns
    cursor.execute("""
    SELECT * FROM transcripts
    WHERE consultation_id = ?
    ORDER BY turn_order ASC
    """, (consultation_id,))
    turns = [dict(t) for t in cursor.fetchall()]

    if not turns:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Cannot generate case sheet: No transcript dialogue found for this consultation. Speak in meeting or load a demo scenario first."
        )

    # 4. Generate 17-section case sheet & multilingual summaries
    ai_result = generate_case_sheet_from_transcript(turns, patient, doctor, student)
    sections = ai_result["sections"]
    multilingual = ai_result["multilingual_summary"]
    source = ai_result["extraction_source"]

    now = datetime.now().isoformat()
    cs_id = f"cs-{consultation_id}"

    # 5. Check if case sheet already exists for this consultation
    cursor.execute("SELECT id, status FROM casesheets WHERE consultation_id = ?", (consultation_id,))
    existing = cursor.fetchone()

    if existing:
        if existing["status"] == "approved_locked":
            conn.close()
            raise HTTPException(status_code=403, detail="Cannot regenerate: Case sheet is approved and locked.")
        
        cursor.execute("""
        UPDATE casesheets
        SET updated_at = ?,
            extraction_source = ?,
            sections_json = ?,
            multilingual_summary_json = ?
        WHERE id = ?
        """, (now, source, json.dumps(sections), json.dumps(multilingual), existing["id"]))
        cs_id = existing["id"]
    else:
        cursor.execute("""
        INSERT INTO casesheets (
            id, consultation_id, patient_id, doctor_id, student_id, status,
            created_at, updated_at, extraction_source, sections_json,
            multilingual_summary_json, approval_json
        ) VALUES (?, ?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?)
        """, (
            cs_id,
            consultation_id,
            consultation["patient_id"],
            consultation["doctor_id"],
            consultation.get("student_id"),
            now,
            now,
            source,
            json.dumps(sections),
            json.dumps(multilingual),
            json.dumps({"is_approved": False})
        ))
        # Update consultation pointer
        cursor.execute("UPDATE consultations SET case_sheet_id = ? WHERE id = ?", (cs_id, consultation_id))

    conn.commit()
    conn.close()

    return get_case_sheet_by_consultation(consultation_id)


@router.put("/{case_sheet_id}")
def update_case_sheet(case_sheet_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Allows inline editing of any of the 17 sections, including adding/removing
    medication rows, updating assessments, symptoms, and doctor observations.
    Blocked if the record is approved_locked.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM casesheets WHERE id = ?", (case_sheet_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Case sheet not found")

    current = dict(row)
    if current["status"] == "approved_locked":
        conn.close()
        raise HTTPException(status_code=403, detail="Record is locked and cannot be edited after Doctor Approval.")

    sections = payload.get("sections")
    if not sections:
        conn.close()
        raise HTTPException(status_code=400, detail="Missing 'sections' payload.")

    now = datetime.now().isoformat()
    cursor.execute("""
    UPDATE casesheets
    SET sections_json = ?,
        updated_at = ?
    WHERE id = ?
    """, (json.dumps(sections), now, case_sheet_id))

    conn.commit()
    conn.close()

    return {"message": "Case sheet updated successfully", "id": case_sheet_id}


@router.post("/{case_sheet_id}/approve")
def doctor_approval_sign_off(
    case_sheet_id: str,
    approval_data: Dict[str, Any] = Body(...)
):
    """
    Clinical verification and electronic sign-off by attending doctor.
    Records credentials, signature canvas image/token, verification checklist,
    computes immutable SHA-256 audit hash, and locks the record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM casesheets WHERE id = ?", (case_sheet_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Case sheet not found")

    current = dict(row)
    if current["status"] == "approved_locked":
        conn.close()
        raise HTTPException(status_code=400, detail="Case sheet has already been approved and locked.")

    doctor_id = approval_data.get("doctor_id", current["doctor_id"])
    cursor.execute("SELECT * FROM users WHERE id = ?", (doctor_id,))
    doc_row = cursor.fetchone()
    if not doc_row or doc_row["role"] != "doctor":
        conn.close()
        raise HTTPException(status_code=403, detail="Only certified Medical Doctors can approve and sign clinical records.")

    doc = dict(doc_row)
    now = datetime.now().isoformat()
    sections = json.loads(current["sections_json"])

    # Compute cryptographic hash for audit compliance
    audit_hash = compute_approval_hash(case_sheet_id, doctor_id, sections, now)

    approval_payload = {
        "is_approved": True,
        "approved_by_id": doctor_id,
        "doctor_name": doc["name"],
        "doctor_registration_number": doc.get("registration_number", "TNMC-84920"),
        "doctor_specialization": doc.get("specialization", "Internal Medicine"),
        "hospital_affiliation": doc.get("hospital_affiliation", "Apollo - MedTrust University Teaching Hospital"),
        "approval_timestamp": now,
        "electronic_signature": approval_data.get("electronic_signature", f"SIGN-{doc['id']}-{int(datetime.now().timestamp())}"),
        "verification_notes": approval_data.get("verification_notes", "Clinical findings verified. Case sheet locked."),
        "immutable_hash": audit_hash
    }

    cursor.execute("""
    UPDATE casesheets
    SET status = 'approved_locked',
        approval_json = ?,
        updated_at = ?
    WHERE id = ?
    """, (json.dumps(approval_payload), now, case_sheet_id))

    # Also update consultation record
    cursor.execute("UPDATE consultations SET is_approved = 1, status = 'completed' WHERE id = ?", (current["consultation_id"],))

    conn.commit()
    conn.close()

    return {
        "message": "Clinical case sheet verified, signed, and locked successfully.",
        "status": "approved_locked",
        "approval": approval_payload
    }


@router.get("/{case_sheet_id}/print", response_class=HTMLResponse)
def print_case_sheet_view(case_sheet_id: str):
    """
    Generates a print-ready and PDF-exportable HTML document with hospital branding,
    patient identification bar, 17 distinct sections, medications table, and signature block.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT cs.*,
           p.first_name || ' ' || p.last_name as patient_name,
           p.mrn as patient_mrn,
           p.date_of_birth,
           p.age as patient_age,
           p.gender as patient_gender,
           p.blood_group as patient_blood_group,
           p.phone as patient_phone,
           p.address as patient_address,
           u.name as doctor_name,
           u.registration_number as doctor_registration,
           u.specialization as doctor_specialization,
           s.name as student_name
    FROM casesheets cs
    JOIN patients p ON cs.patient_id = p.id
    JOIN users u ON cs.doctor_id = u.id
    LEFT JOIN users s ON cs.student_id = s.id
    WHERE cs.id = ?
    """, (case_sheet_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Case sheet not found")

    d = dict(row)
    sections = json.loads(d.get("sections_json") or "{}")
    approval = json.loads(d.get("approval_json") or "{}")
    vitals = sections.get("vitals", {})
    meds = sections.get("medications", [])

    meds_rows = "".join([
        f"""<tr>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1; font-weight: 600;">{m.get('drug_name', '')}</td>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1;">{m.get('dosage', '')}</td>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1;">{m.get('frequency', '')}</td>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1;">{m.get('route', '')}</td>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1;">{m.get('duration', '')}</td>
            <td style="padding: 6px 10px; border: 1px solid #cbd5e1; font-size: 11px;">{m.get('instructions', '')}</td>
        </tr>"""
        for m in meds
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clinical Case Sheet - {d['patient_name']} ({d['patient_mrn']})</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; margin: 30px; background: #fff; font-size: 13px; line-height: 1.5; }}
        .hospital-header {{ border-bottom: 2px solid #0f766e; padding-bottom: 12px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end; }}
        .hospital-title {{ font-size: 20px; font-weight: 800; color: #0f766e; text-transform: uppercase; letter-spacing: 0.5px; }}
        .patient-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
        .section-box {{ margin-bottom: 16px; page-break-inside: avoid; }}
        .section-num {{ display: inline-block; background: #0f766e; color: #fff; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 20px; font-size: 11px; margin-right: 6px; font-weight: bold; }}
        .section-title {{ font-size: 14px; font-weight: 700; color: #0f766e; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 6px; }}
        .section-body {{ padding-left: 26px; color: #334155; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }}
        th {{ background: #f1f5f9; text-align: left; padding: 8px 10px; border: 1px solid #cbd5e1; font-weight: 700; color: #0f766e; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #e0f2fe; color: #0369a1; margin-right: 4px; }}
        .badge-red {{ background: #fee2e2; color: #991b1b; }}
        .signature-box {{ margin-top: 30px; border-top: 1px dashed #cbd5e1; padding-top: 14px; display: flex; justify-content: space-between; align-items: flex-start; }}
        @media print {{
            body {{ margin: 15mm; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="background: #e0f2fe; border: 1px solid #bae6fd; padding: 10px 16px; border-radius: 6px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; color: #0369a1;">🖨️ MedTrust AI Print & PDF Ready Case Sheet</span>
        <button onclick="window.print()" style="background: #0f766e; color: #fff; border: none; padding: 8px 18px; border-radius: 4px; font-weight: bold; cursor: pointer;">Print / Save as PDF</button>
    </div>

    <div class="hospital-header">
        <div>
            <div class="hospital-title">Apollo - MedTrust University Teaching Hospital</div>
            <div style="font-size: 11px; color: #64748b;">Department of Internal Medicine & Academic Telehealth Clinic • Clinical Audit Standard ISO-15189</div>
        </div>
        <div style="text-align: right; font-size: 11px; color: #64748b;">
            <div><strong>MRN:</strong> {d['patient_mrn']}</div>
            <div><strong>Date:</strong> {d['created_at'][:10]}</div>
            <div><strong>Status:</strong> <span class="badge" style="background: {'#dcfce7' if approval.get('is_approved') else '#fef3c7'}; color: {'#166534' if approval.get('is_approved') else '#92400e'}">{'VERIFIED & APPROVED' if approval.get('is_approved') else 'DRAFT RECORD'}</span></div>
        </div>
    </div>

    <div class="patient-card">
        <div><strong style="color:#64748b; font-size:11px;">PATIENT NAME</strong><br><strong>{d['patient_name']}</strong></div>
        <div><strong style="color:#64748b; font-size:11px;">AGE / GENDER</strong><br>{d['patient_age']} Yrs / {d['patient_gender']}</div>
        <div><strong style="color:#64748b; font-size:11px;">BLOOD GROUP</strong><br>{d['patient_blood_group']}</div>
        <div><strong style="color:#64748b; font-size:11px;">CONTACT</strong><br>{d['patient_phone']}</div>
    </div>

    <!-- Section 1: Patient Info -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">1</span> Patient Identification & Care Team</div>
        <div class="section-body">
            Attending Physician: <strong>{d['doctor_name']} ({d['doctor_specialization']}, Reg: {d['doctor_registration']})</strong> | Medical Student: <strong>{d['student_name'] or 'None'}</strong>
        </div>
    </div>

    <!-- Section 2: Chief Complaint -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">2</span> Chief Complaint</div>
        <div class="section-body">{sections.get('chief_complaint', 'None reported')}</div>
    </div>

    <!-- Section 3: History of Present Illness -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">3</span> History of Present Illness (HPI)</div>
        <div class="section-body">{sections.get('history_of_present_illness', 'None documented')}</div>
    </div>

    <!-- Section 4: Symptoms Checklist -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">4</span> Symptoms Checklist & Severity Assessment</div>
        <div class="section-body">
            {''.join([f"<div style='margin-bottom: 3px;'>• <strong>{s.get('symptom')}</strong> - Severity: <span class='badge'>{s.get('severity')}</span> (Duration: {s.get('duration', 'N/A')}) - {s.get('notes', '')}</div>" for s in sections.get('symptoms', [])])}
        </div>
    </div>

    <!-- Section 5: Duration & Onset -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">5</span> Duration & Onset</div>
        <div class="section-body">{sections.get('duration_onset', 'Not specified')}</div>
    </div>

    <!-- Section 6: Past Medical History -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">6</span> Past Medical History</div>
        <div class="section-body">{', '.join(sections.get('past_medical_history', [])) or 'No significant prior medical illness.'}</div>
    </div>

    <!-- Section 7: Medications Table -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">7</span> Prescribed Medications & Therapeutic Regimen</div>
        <div class="section-body">
            <table>
                <thead>
                    <tr>
                        <th>Drug Name</th>
                        <th>Dosage</th>
                        <th>Frequency</th>
                        <th>Route</th>
                        <th>Duration</th>
                        <th>Instructions</th>
                    </tr>
                </thead>
                <tbody>
                    {meds_rows}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Section 8: Allergies -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">8</span> Allergies & Adverse Drug Reactions</div>
        <div class="section-body">
            {''.join([f"<span class='badge badge-red'>{a}</span>" for a in sections.get('allergies', [])]) or 'No known allergies reported.'}
        </div>
    </div>

    <!-- Section 9: Family History -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">9</span> Family History</div>
        <div class="section-body">{sections.get('family_history', 'Non-contributory')}</div>
    </div>

    <!-- Section 10: Social & Occupational History -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">10</span> Social & Occupational History</div>
        <div class="section-body">{sections.get('social_history', 'Non-smoker, non-drinker')}</div>
    </div>

    <!-- Section 11: Doctor Observations & Vitals -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">11</span> Clinical Observations & Vitals</div>
        <div class="section-body">
            <div style="margin-bottom: 6px;">{sections.get('doctor_observations', '')}</div>
            <div style="background: #f8fafc; padding: 6px 12px; border-radius: 4px; font-size: 12px; display: flex; gap: 20px;">
                <span><strong>BP:</strong> {vitals.get('blood_pressure', 'N/A')}</span>
                <span><strong>Pulse:</strong> {vitals.get('pulse_rate', 'N/A')}</span>
                <span><strong>RR:</strong> {vitals.get('respiratory_rate', 'N/A')}</span>
                <span><strong>Temp:</strong> {vitals.get('temperature', 'N/A')}</span>
                <span><strong>SpO2:</strong> {vitals.get('spo2', 'N/A')}</span>
            </div>
        </div>
    </div>

    <!-- Section 12: Investigations -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">12</span> Recommended Investigations & Laboratory Tests</div>
        <div class="section-body">
            {''.join([f"<div>• {inv}</div>" for inv in sections.get('investigations', [])])}
        </div>
    </div>

    <!-- Section 13: Assessment -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">13</span> Provisional Assessment & Clinical Impression</div>
        <div class="section-body">
            <div style="font-weight: 700; color: #0f766e; margin-bottom: 4px;">{sections.get('assessment_diagnosis', '')}</div>
            <div style="font-size: 12px; color: #64748b;"><strong>Differential Diagnoses:</strong> {', '.join(sections.get('differential_diagnoses', []))}</div>
        </div>
    </div>

    <!-- Section 14: Treatment Plan -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">14</span> Comprehensive Treatment Plan</div>
        <div class="section-body" style="white-space: pre-line;">{sections.get('treatment_plan', '')}</div>
    </div>

    <!-- Section 15: Follow-up & Red Flags -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">15</span> Follow-Up Advice & Emergency Red Flags</div>
        <div class="section-body">
            <div style="margin-bottom: 6px;">{sections.get('follow_up_instructions', '')}</div>
            <div style="color: #991b1b; font-weight: 600;">
                {''.join([f"<div>⚠️ RED FLAG: {rf}</div>" for rf in sections.get('red_flag_warnings', [])])}
            </div>
        </div>
    </div>

    <!-- Section 16: Missing Information -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">16</span> Missing / Incomplete Information Flagged by AI</div>
        <div class="section-body">
            {''.join([f"<div>🔍 {mi}</div>" for mi in sections.get('missing_information', [])]) or 'None flagged.'}
        </div>
    </div>

    <!-- Section 17: Uncertain Information -->
    <div class="section-box">
        <div class="section-title"><span class="section-num">17</span> Uncertain / Clarification Required</div>
        <div class="section-body">
            {''.join([f"<div>❓ {ui}</div>" for ui in sections.get('uncertain_information', [])]) or 'None identified.'}
        </div>
    </div>

    <!-- Doctor Electronic Sign-Off & Verification Block -->
    <div class="signature-box">
        <div>
            <div style="font-size: 11px; color: #64748b;">ELECTRONIC AUDIT HASH</div>
            <div style="font-family: monospace; font-size: 10px; color: #0f766e;">{approval.get('immutable_hash', 'RECORD_UNLOCKED_PENDING_FINAL_SIGN')}</div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">Verified under MedTrust Clinical Information System</div>
        </div>
        <div style="text-align: right;">
            <div style="font-weight: 700; color: #0f766e; font-size: 14px;">{approval.get('doctor_name', d['doctor_name'])}</div>
            <div style="font-size: 11px; color: #475569;">Reg No: {approval.get('doctor_registration_number', d['doctor_registration'])}</div>
            <div style="font-size: 11px; color: #475569;">{approval.get('doctor_specialization', d['doctor_specialization'])}</div>
            <div style="font-size: 10px; color: #166534; font-weight: bold; margin-top: 4px;">
                {'✓ DIGITALLY SIGNED & VERIFIED ON ' + approval.get('approval_timestamp', '')[:19].replace('T', ' ') if approval.get('is_approved') else 'PENDING ELECTRONIC SIGN-OFF'}
            </div>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content)
