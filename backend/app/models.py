"""
MedTrust AI - Clinical Data Models and Schemas
Defines schemas for Patient Demographics, 17-Section Clinical Case Sheet,
Google Meet Consultations, Transcripts, and Doctor Approvals.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Role & User Models ---
class UserRole:
    DOCTOR = "doctor"
    STUDENT = "student"
    PATIENT = "patient"


class UserProfile(BaseModel):
    id: str
    name: str
    role: str  # doctor, student, patient
    email: str
    avatar: Optional[str] = None
    registration_number: Optional[str] = None  # For doctors, e.g. "TNMC-84920"
    specialization: Optional[str] = None      # e.g. "General Medicine / Cardiology"
    hospital_affiliation: Optional[str] = "Apollo - MedTrust University Teaching Hospital"
    designation: Optional[str] = None        # e.g. "Senior Consultant Physician", "Final Year MBBS"


# --- Patient Models ---
class PatientBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str  # YYYY-MM-DD
    gender: str         # Male, Female, Other
    blood_group: Optional[str] = "Unknown"
    phone: str
    email: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    known_allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)


class PatientCreate(PatientBase):
    pass


class Patient(PatientBase):
    id: str
    mrn: str  # Medical Record Number, e.g. "MT-2026-0841"
    age: int  # Auto-calculated from DOB
    created_at: str
    updated_at: str


# --- Transcript & Speaker Turn Models ---
class TranscriptTurn(BaseModel):
    turn_id: str
    speaker: str  # "doctor", "student", "patient"
    speaker_name: str
    timestamp: str  # "HH:MM:SS"
    text: str
    confidence: Optional[float] = 0.98


class TranscriptSession(BaseModel):
    consultation_id: str
    turns: List[TranscriptTurn] = Field(default_factory=list)
    total_words: int = 0
    language_detected: str = "en-IN"


# --- 17-Section Case Sheet Models ---
class MedicationItem(BaseModel):
    id: str = Field(default_factory=lambda: "med-1")
    drug_name: str
    dosage: str          # e.g. "500 mg"
    frequency: str       # e.g. "1-0-1 (Twice daily after meals)"
    route: str           # e.g. "Oral", "IV", "Sublingual"
    duration: str        # e.g. "7 days"
    instructions: Optional[str] = "Take after food with plenty of water"


class SymptomItem(BaseModel):
    symptom: str
    severity: str        # Mild, Moderate, Severe, Critical
    duration: Optional[str] = None
    notes: Optional[str] = None


class Vitals(BaseModel):
    blood_pressure: Optional[str] = "120/80 mmHg"
    pulse_rate: Optional[str] = "74 bpm"
    respiratory_rate: Optional[str] = "16 /min"
    temperature: Optional[str] = "98.6 °F"
    spo2: Optional[str] = "98%"
    bmi: Optional[str] = None


class CaseSheet17Sections(BaseModel):
    # Section 1: Patient Information
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Section 2: Chief Complaint
    chief_complaint: str = ""
    
    # Section 3: History of Present Illness (HPI)
    history_of_present_illness: str = ""
    
    # Section 4: Symptoms Checklist & Severity
    symptoms: List[SymptomItem] = Field(default_factory=list)
    
    # Section 5: Duration & Onset
    duration_onset: str = ""
    
    # Section 6: Past Medical History
    past_medical_history: List[str] = Field(default_factory=list)
    
    # Section 7: Current Medications (Dynamic Table)
    medications: List[MedicationItem] = Field(default_factory=list)
    
    # Section 8: Allergies & Adverse Reactions
    allergies: List[str] = Field(default_factory=list)
    
    # Section 9: Family History
    family_history: str = ""
    
    # Section 10: Social & Occupational History
    social_history: str = ""
    
    # Section 11: Doctor Observations & Physical Examination
    doctor_observations: str = ""
    vitals: Vitals = Field(default_factory=Vitals)
    
    # Section 12: Recommended Investigations & Lab Tests
    investigations: List[str] = Field(default_factory=list)
    
    # Section 13: Provisional Assessment & Clinical Impression
    assessment_diagnosis: str = ""
    differential_diagnoses: List[str] = Field(default_factory=list)
    
    # Section 14: Treatment Plan & Prescriptions
    treatment_plan: str = ""
    
    # Section 15: Follow-up Advice & Red Flags
    follow_up_instructions: str = ""
    red_flag_warnings: List[str] = Field(default_factory=list)
    
    # Section 16: Missing / Incomplete Information
    missing_information: List[str] = Field(default_factory=list)
    
    # Section 17: Uncertain / Needs Clarification Information
    uncertain_information: List[str] = Field(default_factory=list)


# --- Multilingual Summaries ---
class MultilingualSummary(BaseModel):
    en: str = ""  # English
    ta: str = ""  # Tamil
    hi: str = ""  # Hindi
    te: str = ""  # Telugu
    ml: str = ""  # Malayalam
    kn: str = ""  # Kannada


# --- Doctor Approval & Sign-Off ---
class DoctorApproval(BaseModel):
    is_approved: bool = False
    approved_by_id: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_registration_number: Optional[str] = None
    doctor_specialization: Optional[str] = None
    approval_timestamp: Optional[str] = None
    electronic_signature: Optional[str] = None  # Base64 data URL or digital hash
    verification_notes: Optional[str] = None
    immutable_hash: Optional[str] = None


# --- Complete Case Sheet Entity ---
class CaseSheetRecord(BaseModel):
    id: str
    consultation_id: str
    patient_id: str
    doctor_id: str
    student_id: Optional[str] = None
    status: str = "draft"  # draft, under_review, approved_locked
    created_at: str
    updated_at: str
    extraction_source: str = "gemini"  # gemini, local_nlp, manual
    sections: CaseSheet17Sections
    multilingual_summary: MultilingualSummary = Field(default_factory=MultilingualSummary)
    approval: DoctorApproval = Field(default_factory=DoctorApproval)


# --- Consultation & Meeting Models ---
class GoogleMeetData(BaseModel):
    space_name: str           # e.g. "spaces/v3k-8492-910"
    meeting_uri: str          # e.g. "https://meet.google.com/xyz-abc-def"
    meeting_code: str         # e.g. "xyz-abc-def"
    calendar_event_id: Optional[str] = None
    calendar_html_link: Optional[str] = None


class ConsultationCreate(BaseModel):
    patient_id: str
    doctor_id: str
    student_id: Optional[str] = None
    scheduled_time: Optional[str] = None
    chief_complaint_hint: Optional[str] = None


class Consultation(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    student_id: Optional[str] = None
    patient_name: str
    doctor_name: str
    student_name: Optional[str] = None
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    scheduled_time: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_seconds: int = 0
    google_meet: GoogleMeetData
    transcript_turn_count: int = 0
    case_sheet_id: Optional[str] = None
    is_approved: bool = False
