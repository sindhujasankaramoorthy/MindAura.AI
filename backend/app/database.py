"""
MedTrust AI - Database layer
Manages SQLite storage, table creation, auto-migration, and seed data.
"""

import os
import json
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "medtrust.db")


def calculate_age(dob_str: str) -> int:
    """Calculates exact age in years from YYYY-MM-DD string."""
    try:
        born = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        avatar TEXT,
        registration_number TEXT,
        specialization TEXT,
        hospital_affiliation TEXT,
        designation TEXT
    );
    """)

    # 2. Patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        mrn TEXT UNIQUE NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        date_of_birth TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        blood_group TEXT,
        phone TEXT NOT NULL,
        email TEXT,
        address TEXT,
        emergency_contact_name TEXT,
        emergency_contact_phone TEXT,
        known_allergies_json TEXT,
        chronic_conditions_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # 3. Consultations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consultations (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        doctor_id TEXT NOT NULL,
        student_id TEXT,
        status TEXT NOT NULL DEFAULT 'scheduled',
        scheduled_time TEXT NOT NULL,
        started_at TEXT,
        ended_at TEXT,
        duration_seconds INTEGER DEFAULT 0,
        google_meet_json TEXT,
        case_sheet_id TEXT,
        is_approved INTEGER DEFAULT 0,
        FOREIGN KEY (patient_id) REFERENCES patients (id),
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    );
    """)

    # 4. Transcripts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL,
        speaker TEXT NOT NULL,
        speaker_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        text TEXT NOT NULL,
        confidence REAL DEFAULT 0.98,
        turn_order INTEGER NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations (id)
    );
    """)

    # 5. Case Sheets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS casesheets (
        id TEXT PRIMARY KEY,
        consultation_id TEXT UNIQUE NOT NULL,
        patient_id TEXT NOT NULL,
        doctor_id TEXT NOT NULL,
        student_id TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        extraction_source TEXT NOT NULL DEFAULT 'gemini',
        sections_json TEXT NOT NULL,
        multilingual_summary_json TEXT,
        approval_json TEXT,
        FOREIGN KEY (consultation_id) REFERENCES consultations (id),
        FOREIGN KEY (patient_id) REFERENCES patients (id),
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    );
    """)

    # 6. Video Analyses table -- multimodal (face + voice) observation
    # records, per the video-analysis pipeline (ai/video/pipeline.py).
    # Stores the complete measurable-observation JSON, not just a summary,
    # since downstream models will need the raw observations later.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_analyses (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        video_id TEXT NOT NULL,
        analysis_id TEXT UNIQUE NOT NULL,
        analysis_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        face_model TEXT,
        voice_model TEXT,
        observation_json TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)

    # 7. Text Journals table -- structured output of ai/text/pipeline.py
    # for each patient-submitted journal entry.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS text_journals (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        raw_input TEXT NOT NULL,
        analysis_json TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)

    # 8. Voice Records table -- structured output of
    # ai/video/existing_voice_adapter.py for each patient-submitted voice
    # check-in (audio only, no video).
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_records (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        analysis_json TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)

    conn.commit()
    seed_initial_data(conn)
    conn.close()


def seed_initial_data(conn: sqlite3.Connection):
    cursor = conn.cursor()

    # Check if users already seeded
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
        return

    now = datetime.now().isoformat()

    # Seed Clinical Users
    users = [
        (
            "doc-1",
            "Dr. Rajesh Sharma, MD",
            "doctor",
            "dr.sharma@medtrust.hospital.org",
            "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150",
            "TNMC-84920",
            "Internal Medicine & Cardiology",
            "Apollo - MedTrust University Teaching Hospital",
            "Senior Consultant & Clinical Professor"
        ),
        (
            "stu-1",
            "Sneha Patel",
            "student",
            "sneha.patel@student.medtrust.edu",
            "https://images.unsplash.com/photo-1594824813512-1f31f9076fdf?w=150",
            "MED-2022-092",
            "Medical Student (Final Year MBBS)",
            "MedTrust University Medical College",
            "Clinical Rotation Intern"
        ),
        (
            "pat-1",
            "K. Sundaram",
            "patient",
            "sundaram.k@gmail.com",
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
            None,
            None,
            "Apollo - MedTrust University Teaching Hospital",
            "Outpatient"
        )
    ]
    cursor.executemany("""
    INSERT INTO users (id, name, role, email, avatar, registration_number, specialization, hospital_affiliation, designation)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, users)

    # Seed Patients
    patients = [
        (
            "pat-1",
            "MT-2026-0841",
            "Sundaram",
            "Krishnamoorthy",
            "1968-05-14",
            calculate_age("1968-05-14"),
            "Male",
            "B+",
            "+91 98401 23456",
            "sundaram.k@gmail.com",
            "42 Temple View Road, Mylapore, Chennai, Tamil Nadu 600004",
            "Radha Sundaram (Spouse)",
            "+91 98401 23457",
            json.dumps(["Penicillin", "Sulfa Drugs"]),
            json.dumps(["Essential Hypertension", "Mild Dyslipidemia"]),
            now,
            now
        ),
        (
            "pat-2",
            "MT-2026-0912",
            "Lakshmi",
            "Narayanan",
            "1974-08-22",
            calculate_age("1974-08-22"),
            "Female",
            "O+",
            "+91 98412 34567",
            "lakshmi.n@gmail.com",
            "15 Anna Salai, Guindy, Chennai, Tamil Nadu 600032",
            "Narayanan S (Husband)",
            "+91 98412 34568",
            json.dumps(["Aspirin (causes stomach irritation)"]),
            json.dumps(["Type 2 Diabetes Mellitus (8 years)", "Diabetic Peripheral Neuropathy"]),
            now,
            now
        ),
        (
            "pat-3",
            "MT-2026-1004",
            "Aarav",
            "Mehra",
            "2020-03-10",
            calculate_age("2020-03-10"),
            "Male",
            "A+",
            "+91 98200 98765",
            "parents.aarav@gmail.com",
            "7B Regency Heights, Bandra West, Mumbai, Maharashtra 400050",
            "Rohan Mehra (Father)",
            "+91 98200 98765",
            json.dumps(["Peanuts"]),
            json.dumps(["Childhood Reactive Airway Disease"]),
            now,
            now
        ),
        (
            "pat-4",
            "MT-2026-1120",
            "Vikram",
            "Singh",
            "1992-11-05",
            calculate_age("1992-11-05"),
            "Male",
            "AB+",
            "+91 98110 55443",
            "vikram.singh@outlook.com",
            "204 Green Glen Layout, Bellandur, Bengaluru, Karnataka 560103",
            "Ananya Singh (Wife)",
            "+91 98110 55444",
            json.dumps(["None Known"]),
            json.dumps(["Gastroesophageal Reflux Disease (GERD)"]),
            now,
            now
        ),
        (
            "pat-5",
            "MT-2026-1234",
            "Priya",
            "Venkat",
            "1999-07-19",
            calculate_age("1999-07-19"),
            "Female",
            "B-",
            "+91 97405 11223",
            "priya.venkat@techcorp.in",
            "52 Indiranagar 100ft Road, Bengaluru, Karnataka 560038",
            "Venkat Raman (Father)",
            "+91 97405 11224",
            json.dumps(["Ciprofloxacin"]),
            json.dumps(["Generalized Anxiety Disorder", "Chronic Sleep-Onset Insomnia"]),
            now,
            now
        )
    ]

    cursor.executemany("""
    INSERT INTO patients (
        id, mrn, first_name, last_name, date_of_birth, age, gender, blood_group,
        phone, email, address, emergency_contact_name, emergency_contact_phone,
        known_allergies_json, chronic_conditions_json, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, patients)

    # Seed a Completed and Approved Consultation with full 17 sections for pat-1
    consult_id = "cons-demo-1"
    casesheet_id = "cs-demo-1"
    meet_info = {
        "space_name": "spaces/mt-cardio-9842",
        "meeting_uri": "https://meet.google.com/qam-pzjy-fkr",
        "meeting_code": "qam-pzjy-fkr",
        "calendar_event_id": "cal_evt_9981240",
        "calendar_html_link": "https://calendar.google.com/calendar/event?eid=demo"
    }

    cursor.execute("""
    INSERT INTO consultations (
        id, patient_id, doctor_id, student_id, status, scheduled_time,
        started_at, ended_at, duration_seconds, google_meet_json, case_sheet_id, is_approved
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        consult_id,
        "pat-1",
        "doc-1",
        "stu-1",
        "completed",
        "2026-09-15 10:00:00",
        "2026-09-15 10:02:15",
        "2026-09-15 10:18:45",
        990,
        json.dumps(meet_info),
        casesheet_id,
        1
    ))

    # Seed Sample Transcript Turns
    turns = [
        ("t-1", consult_id, "student", "Sneha Patel (Student)", "10:02:20", "Good morning Mr. Sundaram. My name is Sneha, final year medical student assisting Dr. Sharma today. How are you feeling?", 0.99, 1),
        ("t-2", consult_id, "patient", "K. Sundaram (Patient)", "10:02:35", "Good morning doctor. For the past 4 days I have been having a heavy sensation in the middle of my chest when climbing stairs, along with mild shortness of breath.", 0.98, 2),
        ("t-3", consult_id, "doctor", "Dr. Rajesh Sharma (Doctor)", "10:03:00", "Hello Sundaram. Does this chest tightness radiate to your left arm or jaw, and does it relieve when you rest?", 0.99, 3),
        ("t-4", consult_id, "patient", "K. Sundaram (Patient)", "10:03:15", "Yes doctor, it radiates slightly to my left shoulder. When I sit down for 5 minutes it subsides. Also I have had a dry cough since starting the new blood pressure pill last week.", 0.97, 4),
        ("t-5", consult_id, "doctor", "Dr. Rajesh Sharma (Doctor)", "10:03:40", "Understood. The dry cough is likely an adverse effect of Enalapril. Let's switch you to Telmisartan, and get an immediate 12-lead ECG, Troponin I, and Echo.", 0.99, 5)
    ]
    cursor.executemany("""
    INSERT INTO transcripts (id, consultation_id, speaker, speaker_name, timestamp, text, confidence, turn_order)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, turns)

    # Seed 17-Section Case Sheet
    sections_data = {
        "patient_info": {
            "name": "K. Sundaram",
            "mrn": "MT-2026-0841",
            "age": calculate_age("1968-05-14"),
            "gender": "Male",
            "blood_group": "B+",
            "contact": "+91 98401 23456",
            "attending_doctor": "Dr. Rajesh Sharma, MD (Reg: TNMC-84920)",
            "medical_student": "Sneha Patel (MBBS Intern)"
        },
        "chief_complaint": "Exertional retrosternal chest tightness radiating to left shoulder for 4 days, with persistent dry cough for 1 week.",
        "history_of_present_illness": "Patient reports retrosternal chest tightness onset 4 days ago, primarily provoked by stair climbing (New York Heart Association Class II). Pain character is dull, non-pleuritic, lasts 5-7 minutes, and resolves completely with rest. Accompanied by exertional dyspnea. Additionally developed an irritating non-productive dry cough after starting Enalapril 10 days ago.",
        "symptoms": [
            {"symptom": "Exertional chest tightness", "severity": "Moderate", "duration": "4 days", "notes": "Provoked by climbing 1 flight of stairs"},
            {"symptom": "Exertional breathlessness", "severity": "Mild", "duration": "4 days", "notes": "Grade II mMRC"},
            {"symptom": "Dry irritating cough", "severity": "Moderate", "duration": "7 days", "notes": "Nocturnal disturbance, secondary to ACEi"},
            {"symptom": "Left shoulder radiation", "severity": "Mild", "duration": "4 days", "notes": "Resolves with rest"}
        ],
        "duration_onset": "Acute-on-chronic presentation. Symptoms started 4 days ago with progressive exertional trigger.",
        "past_medical_history": [
            "Essential Hypertension diagnosed 6 years ago",
            "Mild Dyslipidemia on Atorvastatin",
            "No prior myocardial infarction or CABG/stenting"
        ],
        "medications": [
            {"id": "med-1", "drug_name": "Telmisartan", "dosage": "40 mg", "frequency": "1-0-0 (Once daily morning)", "route": "Oral", "duration": "30 days", "instructions": "Replacement for Enalapril; take after breakfast"},
            {"id": "med-2", "drug_name": "Aspirin (Ecosprin)", "dosage": "75 mg", "frequency": "0-1-0 (Once daily after lunch)", "route": "Oral", "duration": "30 days", "instructions": "Antiplatelet prophylaxis; take with water"},
            {"id": "med-3", "drug_name": "Atorvastatin", "dosage": "20 mg", "frequency": "0-0-1 (Once daily at bedtime)", "route": "Oral", "duration": "30 days", "instructions": "Lipid lowering; swallow whole"},
            {"id": "med-4", "drug_name": "Sorbitrate (Isosorbide Dinitrate)", "dosage": "5 mg", "frequency": "As needed (SOS)", "route": "Sublingual", "duration": "10 tablets", "instructions": "Dissolve under tongue if chest tightness persists >3 mins at rest"}
        ],
        "allergies": [
            "Penicillin (History of urticarial rash)",
            "Enalapril (ACE inhibitor induced intractable dry cough)"
        ],
        "family_history": "Father suffered Myocardial Infarction at age 62. Mother had Type 2 Diabetes.",
        "social_history": "Retired Bank Auditor. Ex-smoker (10 pack-years, quit 8 years ago). Non-alcoholic. Moderate sodium diet.",
        "doctor_observations": "Patient is conscious, alert, and oriented. Mild anxious affect. No peripheral cyanosis, clubbing, or pedal edema. JVP normal. S1, S2 heard normal, no murmurs. Chest clear bilaterally, vesicular breath sounds without wheezing.",
        "vitals": {
            "blood_pressure": "144/92 mmHg",
            "pulse_rate": "78 bpm (regular)",
            "respiratory_rate": "18 /min",
            "temperature": "98.4 °F",
            "spo2": "98% on room air",
            "bmi": "26.4 kg/m²"
        },
        "investigations": [
            "12-Lead Electrocardiogram (ECG) - STAT",
            "Serum Troponin I (High Sensitivity) - STAT",
            "2D Echocardiography with Doppler",
            "Fasting Lipid Profile, Serum Creatinine & Electrolytes",
            "Treadmill Stress Test (TMT) after stabilization"
        ],
        "assessment_diagnosis": "Angina Pectoris (Probable Exertional Coronary Artery Disease - CCS Class II) with Stage 2 Hypertension and ACE Inhibitor-Induced Cough.",
        "differential_diagnoses": [
            "Gastroesophageal Reflux Disease (GERD) with esophageal spasm",
            "Costochondritis / Musculoskeletal chest wall pain",
            "Early Unstable Angina"
        ],
        "treatment_plan": "1. Discontinue Enalapril immediately. Initiate Telmisartan 40mg OD.\n2. Dual cardioprotective therapy with Ecosprin 75mg and Atorvastatin 20mg.\n3. Prescribe sublingual Sorbitrate 5mg SOS for acute breakthrough angina.\n4. Advise strict low-sodium, heart-healthy Mediterranean diet.\n5. Avoid strenuous exertion until ECG & Echo review.",
        "follow_up_instructions": "Review in Cardiology OPD in 7 days with ECG, Echo, and Blood reports. Contact clinic immediately if frequency of chest pain increases.",
        "red_flag_warnings": [
            "Chest pain lasting longer than 15 minutes that does not resolve with rest or SOS Sorbitrate",
            "Pain radiating to jaw, back, or neck accompanied by cold sweats or vomiting",
            "Severe sudden breathlessness or dizziness/fainting spells (Call Emergency 108 immediately)"
        ],
        "missing_information": [
            "Previous baseline ECG tracing from 6 months ago not available for ST-T comparison",
            "Recent serum potassium and creatinine values pending"
        ],
        "uncertain_information": [
            "Clarify if nocturnal waking is triggered by orthopnea or pure dry cough"
        ]
    }

    multilingual_summary = {
        "en": "Patient K. Sundaram (58M) presented with chest tightness during exertion. Enalapril was stopped due to dry cough and switched to Telmisartan. Heart tests (ECG, Troponin, Echo) ordered. Prescribed Aspirin, Atorvastatin, and emergency Sorbitrate.",
        "ta": "நோயாளி திரு. சுந்தரம் (58 வயது) மாடிப்படிகள் ஏறும்போது நெஞ்சு இறுக்கம் மற்றும் இருமல் இருப்பதாகக் கூறினார். எனலாப்ரில் மருந்து நிறுத்தப்பட்டு டெல்மிசார்ட்டன் வழங்கப்பட்டது. இசிஜி மற்றும் எக்கோ பரிசோதனைகள் பரிந்துரைக்கப்பட்டுள்ளன. அவசர மாத்திரை கொடுக்கப்பட்டுள்ளது.",
        "hi": "मरीज श्री सुंदरम (58 वर्ष) को सीढ़ियां चढ़ते समय सीने में जकड़न और सूखी खांसी की शिकायत थी। पुरानी बीपी दवा बदलकर टेलमिसार्टन दी गई है। ईसीजी और इको जांच लिखी गई है। आपातकालीन दवा दी गई है।",
        "te": "రోగి కె. సుందరం (58 సం.) మెట్లు ఎక్కుతున్నప్పుడు ఛాతీలో ఒత్తిడి మరియు పొడి దగ్గుతో బాధపడుతున్నారు. ఎనలాప్రిల్ ఆపి టెల్మిసార్టన్ ఇవ్వబడింది. ఈసీజీ మరియు ఎకో పరీక్షలు సూచించబడ్డాయి.",
        "ml": "രോഗി കെ. സുന്ദരം (58 വയസ്സ്) പടികൾ കയറുമ്പോൾ നെഞ്ചിൽ ഭാരവും വരണ്ട ചുമയും ഉള്ളതായി അറിയിച്ചു. ബിപി മരുന്ന് മാറ്റി ടെൽമിസാർട്ടൻ നൽകി. ഇസിജി, എക്കോ പരിശോധനകൾ നിർദ്ദേശിച്ചു.",
        "kn": "ರೋಗಿ ಕೆ. ಸುಂದರಂ (58 ವರ್ಷ) ಮೆಟ್ಟಿಲು ಹತ್ತುವಾಗ ಎದೆ ಬಿಗಿತ ಮತ್ತು ಒಣ ಕೆಮ್ಮು ಇರುವುದಾಗಿ ತಿಳಿಸಿದ್ದಾರೆ. ಬಿಪಿ ಮಾತ್ರೆ ಬದಲಾಯಿಸಿ ಟೆಲ್ಮಿಸಾರ್ಟನ್ ನೀಡಲಾಗಿದೆ. ಇಸಿಜಿ ಮತ್ತು ಎಕೋ ಪರೀಕ್ಷೆಗಳಿಗೆ ಸೂಚಿಸಲಾಗಿದೆ."
    }

    approval_data = {
        "is_approved": True,
        "approved_by_id": "doc-1",
        "doctor_name": "Dr. Rajesh Sharma, MD",
        "doctor_registration_number": "TNMC-84920",
        "doctor_specialization": "Internal Medicine & Cardiology",
        "approval_timestamp": "2026-09-15T10:20:12+05:30",
        "electronic_signature": "SIGN-VERIFIED-DR-RAJESH-SHARMA-TNMC-84920",
        "verification_notes": "Clinical findings cross-verified against live transcription. Dosages and contraindications reviewed.",
        "immutable_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    }

    cursor.execute("""
    INSERT INTO casesheets (
        id, consultation_id, patient_id, doctor_id, student_id, status,
        created_at, updated_at, extraction_source, sections_json,
        multilingual_summary_json, approval_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        casesheet_id,
        consult_id,
        "pat-1",
        "doc-1",
        "stu-1",
        "approved_locked",
        now,
        now,
        "gemini",
        json.dumps(sections_data),
        json.dumps(multilingual_summary),
        json.dumps(approval_data)
    ))

    # Seed an active / in_progress consultation for demonstration
    active_consult_id = "cons-active-1"
    active_meet_info = {
        "space_name": "spaces/mt-tele-7721",
        "meeting_uri": "https://meet.google.com/dtz-rvcw-kmb",
        "meeting_code": "dtz-rvcw-kmb",
        "calendar_event_id": "cal_evt_7721940",
        "calendar_html_link": "https://calendar.google.com/calendar/event?eid=tele7721"
    }

    cursor.execute("""
    INSERT INTO consultations (
        id, patient_id, doctor_id, student_id, status, scheduled_time,
        started_at, ended_at, duration_seconds, google_meet_json, case_sheet_id, is_approved
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        active_consult_id,
        "pat-2",
        "doc-1",
        "stu-1",
        "in_progress",
        now,
        now,
        None,
        185,
        json.dumps(active_meet_info),
        None,
        0
    ))

    conn.commit()


# --- Database Helper Query Functions ---

def get_patients(search_term: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if search_term:
        term = f"%{search_term.lower()}%"
        cursor.execute("""
        SELECT * FROM patients
        WHERE LOWER(first_name) LIKE ? OR LOWER(last_name) LIKE ? OR LOWER(mrn) LIKE ? OR phone LIKE ?
        ORDER BY created_at DESC
        """, (term, term, term, term))
    else:
        cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d["known_allergies"] = json.loads(d.get("known_allergies_json") or "[]")
        d["chronic_conditions"] = json.loads(d.get("chronic_conditions_json") or "[]")
        result.append(d)
    return result


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["known_allergies"] = json.loads(d.get("known_allergies_json") or "[]")
    d["chronic_conditions"] = json.loads(d.get("chronic_conditions_json") or "[]")
    return d


def create_patient(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM patients")
    count = cursor.fetchone()[0] + 1
    new_id = f"pat-{count}_{int(datetime.now().timestamp())}"
    mrn = f"MT-2026-{1000 + count:04d}"
    age = calculate_age(data["date_of_birth"])
    now = datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO patients (
        id, mrn, first_name, last_name, date_of_birth, age, gender, blood_group,
        phone, email, address, emergency_contact_name, emergency_contact_phone,
        known_allergies_json, chronic_conditions_json, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        new_id,
        mrn,
        data["first_name"],
        data["last_name"],
        data["date_of_birth"],
        age,
        data["gender"],
        data.get("blood_group", "Unknown"),
        data["phone"],
        data.get("email"),
        data.get("address"),
        data.get("emergency_contact_name"),
        data.get("emergency_contact_phone"),
        json.dumps(data.get("known_allergies", [])),
        json.dumps(data.get("chronic_conditions", [])),
        now,
        now
    ))
    conn.commit()
    conn.close()
    return get_patient_by_id(new_id)


# --- Video Analysis Helper Functions ---

def save_video_analysis(patient_id: str, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists one video-analysis observation record (see
    ai/video/pipeline.analyze_video()'s output) for a patient. Stores the
    full observation JSON, not just a summary -- future models need the
    raw measurable observations, not only a final interpretation.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    record_id = f"va-{analysis_json['analysis_id']}"
    metadata = analysis_json.get("analysis_metadata", {})

    cursor.execute("""
    INSERT INTO video_analyses (
        id, patient_id, video_id, analysis_id, analysis_version,
        created_at, face_model, voice_model, observation_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_id,
        patient_id,
        analysis_json["video_id"],
        analysis_json["analysis_id"],
        analysis_json["analysis_version"],
        analysis_json["created_at"],
        metadata.get("face_model"),
        metadata.get("voice_model"),
        json.dumps(analysis_json),
    ))
    conn.commit()
    conn.close()
    return get_video_analysis(analysis_json["analysis_id"])


def get_video_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_analyses WHERE analysis_id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["observation"] = json.loads(d.pop("observation_json"))
    return d


def get_video_analyses_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM video_analyses WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d["observation"] = json.loads(d.pop("observation_json"))
        result.append(d)
    return result


# --- Text Journal Helper Functions ---

def save_text_journal(patient_id: str, raw_input: str, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    """Persists one text check-in's structured output from
    ai/text/pipeline.process_text()."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    record_id = f"tj-{int(datetime.now().timestamp() * 1000)}"

    cursor.execute("""
    INSERT INTO text_journals (id, patient_id, created_at, raw_input, analysis_json)
    VALUES (?, ?, ?, ?, ?)
    """, (record_id, patient_id, now, raw_input, json.dumps(analysis_json)))
    conn.commit()
    conn.close()
    return get_text_journals_for_patient(patient_id)[0]


def get_text_journals_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM text_journals WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d["analysis"] = json.loads(d.pop("analysis_json"))
        result.append(d)
    return result


# --- Voice Record Helper Functions ---

def save_voice_record(patient_id: str, analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    """Persists one voice check-in's structured output from
    ai/video/existing_voice_adapter.analyze_audio()."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    record_id = f"vr-{int(datetime.now().timestamp() * 1000)}"

    cursor.execute("""
    INSERT INTO voice_records (id, patient_id, created_at, analysis_json)
    VALUES (?, ?, ?, ?)
    """, (record_id, patient_id, now, json.dumps(analysis_json)))
    conn.commit()
    conn.close()
    return get_voice_records_for_patient(patient_id)[0]


def get_voice_records_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM voice_records WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d["analysis"] = json.loads(d.pop("analysis_json"))
        result.append(d)
    return result


# --- Combined Multimodal History ---

def get_patient_history(patient_id: str) -> Dict[str, Any]:
    """Returns every modality's check-in history for a patient, for the
    doctor dashboard's timeline/trends view."""
    return {
        "patient_id": patient_id,
        "text_journals": get_text_journals_for_patient(patient_id),
        "voice_records": get_voice_records_for_patient(patient_id),
        "video_analyses": get_video_analyses_for_patient(patient_id),
    }
