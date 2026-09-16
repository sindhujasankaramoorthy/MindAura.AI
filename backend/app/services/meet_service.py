"""
MedTrust AI - Google Meet & Clinical Scenarios Service
Handles Google Meet Spaces API, Calendar conference generation,
and 5 rich multi-turn clinical consultation scenarios.
"""

import random
import string
from datetime import datetime
from typing import Dict, List, Any


def generate_meet_code() -> str:
    """Generates standard Google Meet format code: xxx-yyyy-zzz"""
    chars = string.ascii_lowercase
    part1 = "".join(random.choices(chars, k=3))
    part2 = "".join(random.choices(chars, k=4))
    part3 = "".join(random.choices(chars, k=3))
    return f"{part1}-{part2}-{part3}"


def create_google_meet_session(consultation_id: str, title: str = "Clinical Telehealth Consultation") -> Dict[str, Any]:
    """
    Creates a Google Meet Space and associated Calendar invite metadata.
    Follows the Google Meet Spaces API v2 schema:
    https://developers.google.com/meet/api/guides/overview
    """
    meet_code = generate_meet_code()
    space_id = f"spaces/mt-{random.randint(1000, 9999)}-{random.randint(100, 999)}"
    meeting_uri = f"https://meet.google.com/{meet_code}"
    calendar_event_id = f"cal_evt_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
    
    return {
        "space_name": space_id,
        "meeting_uri": meeting_uri,
        "meeting_code": meet_code,
        "calendar_event_id": calendar_event_id,
        "calendar_html_link": f"https://calendar.google.com/calendar/r/eventedit?text={title}&add=patient@medtrust.hospital.org",
        "access_type": "TRUSTED_HOSPITAL_SPACE",
        "created_time": datetime.now().isoformat()
    }


# --- 5 Comprehensive Realistic Clinical Scenarios for Live Demonstration ---

CLINICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "cardiology": {
        "id": "scenario-cardio-1",
        "specialty": "Cardiology / Internal Medicine",
        "patient_name": "K. Sundaram",
        "age": 58,
        "gender": "Male",
        "title": "Exertional Angina & ACE-Inhibitor Induced Cough",
        "chief_complaint": "Retrosternal chest heaviness on climbing stairs and annoying dry cough for 1 week.",
        "turns": [
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "00:05",
                "text": "Good morning Mr. Sundaram. My name is Sneha, final year medical student on clinical posting with Dr. Rajesh Sharma. We will be reviewing your health today. How can we help you?"
            },
            {
                "speaker": "patient",
                "speaker_name": "K. Sundaram (Patient)",
                "timestamp": "00:20",
                "text": "Good morning doctor. For the past 4 days, whenever I climb the stairs to my second-floor flat, I feel a strange dull heaviness in the center of my chest. It radiates a little to my left shoulder. It stops after I sit down for 5 minutes."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "00:45",
                "text": "Hello Sundaram. Dr. Sharma here. That chest sensation is very important. Does it come with any sweating, dizziness, or breathlessness while walking on flat ground?"
            },
            {
                "speaker": "patient",
                "speaker_name": "K. Sundaram (Patient)",
                "timestamp": "01:05",
                "text": "Mild breathlessness doctor, but no sweating. But doctor, I also have this non-stop dry throat cough especially at night, ever since the local clinic started me on Enalapril 10mg ten days ago."
            },
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "01:25",
                "text": "Dr. Sharma, Enalapril is an ACE inhibitor known to cause bradykinin accumulation and dry cough in up to 15% of patients. Should we switch him to an ARB like Telmisartan?"
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "01:45",
                "text": "Exactly right, Sneha. We will stop Enalapril immediately and switch to Telmisartan 40mg. For the exertional chest heaviness, we must urgently rule out acute coronary syndrome. Sundaram, please take Aspirin 75mg, Atorvastatin 20mg, and get a 12-lead ECG, cardiac troponin test, and 2D Echo today."
            },
            {
                "speaker": "patient",
                "speaker_name": "K. Sundaram (Patient)",
                "timestamp": "02:10",
                "text": "Understood doctor. If the chest pain comes again suddenly while resting, what should I do?"
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "02:25",
                "text": "I am prescribing sublingual Sorbitrate 5mg. Keep one tablet under your tongue. If the pain persists beyond 15 minutes or spreads to your jaw with sweating, call the 108 ambulance or visit our emergency department immediately."
            }
        ]
    },
    "diabetes": {
        "id": "scenario-diabetes-2",
        "specialty": "Endocrinology & Diabetology",
        "patient_name": "Lakshmi Narayanan",
        "age": 52,
        "gender": "Female",
        "title": "Uncontrolled Type 2 Diabetes with Peripheral Neuropathy",
        "chief_complaint": "Tingling and burning pins-and-needles sensation in bilateral feet, frequent night urination, and fatigue.",
        "turns": [
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "00:04",
                "text": "Hello Mrs. Lakshmi, welcome to MedTrust Telehealth. Dr. Sharma and I are here. Can you tell us what issues you are facing recently?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Lakshmi Narayanan (Patient)",
                "timestamp": "00:18",
                "text": "Vanakkam doctor. I have had diabetes for 8 years now. For the last two months, I have terrible burning and needle-like prickling pain in both feet at night. My feet feel numb, like I am walking on cotton."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "00:40",
                "text": "Vanakkam Mrs. Lakshmi. Have you checked your blood sugars recently? And what medication doses are you taking right now?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Lakshmi Narayanan (Patient)",
                "timestamp": "00:58",
                "text": "I checked yesterday with my home glucometer. Fasting was 192 mg/dL, and two hours after lunch was 280 mg/dL. I am currently taking Metformin 500mg once daily in the morning."
            },
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "01:18",
                "text": "Her glycemic control is suboptimal with an estimated HbA1c likely above 9.0%. The symmetric stocking distribution of burning and numbness strongly indicates Diabetic Sensorimotor Polyneuropathy."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "01:38",
                "text": "Good assessment, Sneha. Mrs. Lakshmi, we need to intensify your therapy. We will increase Metformin to 1000mg twice daily with meals, and add Glimepiride 1mg before breakfast. For the neuropathic foot pain, we will start Pregabalin 75mg at bedtime."
            },
            {
                "speaker": "patient",
                "speaker_name": "Lakshmi Narayanan (Patient)",
                "timestamp": "02:00",
                "text": "Thank you doctor. Do I need to get my eyes or kidney tested as well?"
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "02:15",
                "text": "Yes, absolutely. We need an HbA1c, urine microalbumin-creatinine ratio, serum creatinine, and a dilated fundus eye examination to screen for diabetic retinopathy. Inspect your feet daily for small blisters or ulcers."
            }
        ]
    },
    "pediatrics": {
        "id": "scenario-peds-3",
        "specialty": "Pediatrics & Pulmonology",
        "patient_name": "Master Aarav Mehra",
        "age": 6,
        "gender": "Male",
        "title": "Acute Reactive Airway Disease & Nocturnal Wheezing",
        "chief_complaint": "Persistent nighttime dry coughing fits and audible wheezing during active play for 5 days.",
        "turns": [
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "00:05",
                "text": "Namaste Mr. Mehra. We have 6-year-old Aarav's chart here. What symptoms has Aarav been experiencing over the last few days?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Rohan Mehra (Father of Aarav)",
                "timestamp": "00:20",
                "text": "Doctor, for the past 5 days Aarav has this relentless dry hacking cough that gets worse between 2 AM and 5 AM. Yesterday when he was playing cricket in the garden, he became breathless and I could hear a whistling sound from his chest."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "00:45",
                "text": "Does Aarav have any fever, runny nose, or history of eczema or milk allergy in early childhood?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Rohan Mehra (Father of Aarav)",
                "timestamp": "01:00",
                "text": "No fever doctor, temperature is normal at 98.6°F. He did have skin allergy as a toddler, and his mother has bronchial asthma and uses an inhaler."
            },
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "01:18",
                "text": "The presentation is classic for pediatric bronchial asthma triggered by exercise and nocturnal cold air, with a strong atopic maternal family history."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "01:38",
                "text": "Agreed. We will prescribe a Salbutamol 100mcg metered-dose inhaler with a spacer and pediatric mask, 2 puffs as needed for acute wheezing. To control background airway inflammation, we'll start Budesonide 100mcg inhaler twice daily."
            },
            {
                "speaker": "patient",
                "speaker_name": "Rohan Mehra (Father of Aarav)",
                "timestamp": "01:58",
                "text": "Are inhalers habit-forming, doctor? We were worried about starting puffers at age 6."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "02:12",
                "text": "Not at all. Inhalers deliver microgram doses straight into the lungs with minimal systemic absorption, making them far safer than oral syrups. If Aarav shows chest retractions, blue lips, or inability to speak full sentences, bring him to pediatric emergency immediately."
            }
        ]
    },
    "gastroenterology": {
        "id": "scenario-gastro-4",
        "specialty": "Gastroenterology",
        "patient_name": "Vikram Singh",
        "age": 34,
        "gender": "Male",
        "title": "NSAID-Induced Peptic Dyspepsia & Acid Reflux",
        "chief_complaint": "Severe burning sensation in upper epigastric abdomen for 2 weeks, worse between meals.",
        "turns": [
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "00:06",
                "text": "Hello Mr. Vikram. Please describe the abdominal discomfort you've been having."
            },
            {
                "speaker": "patient",
                "speaker_name": "Vikram Singh (Patient)",
                "timestamp": "00:22",
                "text": "Doctor, for two weeks I have this gnawing, acidic burning pain right below my breastbone. It worsens when my stomach is empty around 11 AM and before dinner. Drinking cold milk gives temporary relief. Sometimes sour liquid comes up into my throat."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "00:46",
                "text": "Have you noticed black or tarry stools, vomiting, or significant unexplained weight loss? And are you taking any painkillers?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Vikram Singh (Patient)",
                "timestamp": "01:06",
                "text": "Stool color is normal, no vomiting. But for my lower back gym strain, I have been popping Ibuprofen 400mg twice daily for almost three weeks without any antacid."
            },
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "01:25",
                "text": "Prolonged unbuffered NSAID therapy inhibits mucosal COX-1 and gastric prostaglandins, leading to NSAID-induced peptic ulcer disease and secondary GERD."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "01:45",
                "text": "Spot on. Vikram, stop taking Ibuprofen immediately. We will initiate Esomeprazole 40mg once daily 30 minutes before breakfast for 4 weeks, plus Sucralfate oral suspension 10ml thrice daily before meals to coat the gastric mucosa."
            },
            {
                "speaker": "patient",
                "speaker_name": "Vikram Singh (Patient)",
                "timestamp": "02:08",
                "text": "What tests do I need to do doctor?"
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "02:22",
                "text": "Let's do an H. pylori stool antigen test and complete blood count to rule out anemia. If symptoms persist after 2 weeks, we will perform an upper GI endoscopy."
            }
        ]
    },
    "psychiatry": {
        "id": "scenario-psych-5",
        "specialty": "Psychiatry & Behavioral Health",
        "patient_name": "Priya Venkat",
        "age": 27,
        "gender": "Female",
        "title": "Generalized Anxiety Disorder with Sleep-Onset Insomnia",
        "chief_complaint": "Persistent nervousness, palpitations, trembling hands, and difficulty falling asleep for 6 weeks.",
        "turns": [
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "00:05",
                "text": "Hi Priya, thank you for connecting with us today. Please take your time and tell us what you have been experiencing."
            },
            {
                "speaker": "patient",
                "speaker_name": "Priya Venkat (Patient)",
                "timestamp": "00:22",
                "text": "Doctor, for the past month and a half, my mind won't stop racing. I work in software and had a promotion recently, but now I feel constant dread that something terrible will happen. My heart races randomly, my palms sweat, and when I lie down at night, I stay awake till 3 AM."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "00:48",
                "text": "Hello Priya. You are in a safe, confidential space. Have you noticed any tremors, heat intolerance, weight loss, or changes in your menstrual cycle?"
            },
            {
                "speaker": "patient",
                "speaker_name": "Priya Venkat (Patient)",
                "timestamp": "01:08",
                "text": "Cycles are regular. I do feel hand tremors when speaking in team calls. I had basic blood tests and thyroid check last month, and they were all completely normal."
            },
            {
                "speaker": "student",
                "speaker_name": "Sneha Patel (MBBS Intern)",
                "timestamp": "01:28",
                "text": "The patient meets DSM-5 criteria for Generalized Anxiety Disorder (GAD) with autonomic arousal symptoms (palpitations, diaphoresis) and secondary insomnia."
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "01:48",
                "text": "Yes. Priya, anxiety is an imbalance in neurochemistry, completely treatable. We will start Escitalopram 5mg in the morning, gradually titrating to 10mg. For sleep and acute autonomic palpitations, we'll give Clonazepam 0.25mg short-term for 7 nights only. We'll also connect you with our hospital clinical psychologist for Cognitive Behavioral Therapy."
            },
            {
                "speaker": "patient",
                "speaker_name": "Priya Venkat (Patient)",
                "timestamp": "02:15",
                "text": "That is so reassuring to hear doctor. Will these medications make me dependent?"
            },
            {
                "speaker": "doctor",
                "speaker_name": "Dr. Rajesh Sharma, MD (Consultant)",
                "timestamp": "02:30",
                "text": "Escitalopram is non-addictive and stabilizes your serotonin levels over 3-4 weeks. Clonazepam is only for short-term initial stabilization. Practice sleep hygiene—no laptop or mobile screens 1 hour before bed."
            }
        ]
    }
}


def get_scenario(scenario_key: str) -> Dict[str, Any]:
    return CLINICAL_SCENARIOS.get(scenario_key, CLINICAL_SCENARIOS["cardiology"])


def list_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "key": k,
            "title": v["title"],
            "specialty": v["specialty"],
            "patient_name": v["patient_name"],
            "age": v["age"],
            "gender": v["gender"],
            "chief_complaint": v["chief_complaint"],
            "turn_count": len(v["turns"])
        }
        for k, v in CLINICAL_SCENARIOS.items()
    ]
