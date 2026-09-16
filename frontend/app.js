/**
 * MedTrust AI - Clinical Telehealth & Academic Platform Client Application
 * Features:
 * - Live Webcam Preview & Web Audio API Waveform Oscillogram
 * - Google Meet Spaces API Integration & Conference Code Management
 * - Live Speech-to-Text & Diarized Turn Transcription
 * - 17-Section AI Structured Case Sheet with Gemini & Offline NLP Fallback
 * - Dynamic Medications Table (Add/Remove Rows)
 * - Multilingual Summary (EN, TA, HI, TE, ML, KN) with Text-to-Speech (TTS)
 * - Doctor Verification & Electronic Signature Sign-Off with Immutable Audit Lock
 * - Patient Registry with Real-Time DOB-to-Age Calculator
 */

// --- Global Application State ---
const state = {
  currentUser: null,
  activeConsultationId: "cons-demo-1",
  activeConsultation: null,
  activeCaseSheet: null,
  isEditMode: false,
  callTimerSeconds: 255,
  timerInterval: null,
  
  // Media & Audio
  mediaStream: null,
  audioContext: null,
  analyserNode: null,
  audioDataArray: null,
  isAudioMuted: false,
  isVideoMuted: false,
  isScreenSharing: false,
  waveformAnimationId: null,

  // Speech Recognition
  speechRecognition: null,
  isRecognizing: false,

  // Multilingual Summary
  activeLanguage: "en",
  currentSpeechUtterance: null,

  // Signature Pad
  signatureCanvas: null,
  signatureCtx: null,
  isDrawingSignature: false
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", async () => {
  await initCurrentUser();
  await loadConsultation(state.activeConsultationId);
  initAudioWaveform();
  initWebcamPreview();
  initSignaturePad();
  initSpeechRecognition();
  startCallTimer();
  await loadPatients();
  await loadConsultationArchive();
});


// --- Navigation Tabs ---
function switchTab(tabKey) {
  const tabs = ["consultation", "casesheet", "patients", "history"];
  tabs.forEach(t => {
    const view = document.getElementById(`view-${t}`);
    const tabBtn = document.getElementById(`tab-${t}`);
    if (t === tabKey) {
      view?.classList.remove("hidden");
      tabBtn?.classList.add("active");
    } else {
      view?.classList.add("hidden");
      tabBtn?.classList.remove("active");
    }
  });

  if (tabKey === "casesheet" && state.activeConsultationId) {
    loadCaseSheetForConsultation(state.activeConsultationId);
  }
}


// --- User Authentication & Role Switching ---
async function initCurrentUser() {
  try {
    const res = await fetch("/api/auth/me");
    if (res.ok) {
      state.currentUser = await res.json();
      updateUserUI();
    }
  } catch (err) {
    console.warn("Auth initialization error, using doctor default:", err);
    state.currentUser = {
      id: "doc-1",
      name: "Dr. Rajesh Sharma, MD",
      role: "doctor",
      registration_number: "TNMC-84920",
      specialization: "Internal Medicine & Cardiology"
    };
    updateUserUI();
  }
}

function updateUserUI() {
  const nameEl = document.getElementById("user-display-name");
  const roleEl = document.getElementById("user-display-role");
  const badgeEl = document.getElementById("user-role-badge");

  if (nameEl) nameEl.textContent = state.currentUser.name;
  if (roleEl) roleEl.textContent = `${state.currentUser.role.toUpperCase()} (${state.currentUser.designation || 'Specialist'})`;

  if (badgeEl) {
    badgeEl.className = "w-2.5 h-2.5 rounded-full " + 
      (state.currentUser.role === "doctor" ? "bg-teal-500" :
       state.currentUser.role === "student" ? "bg-purple-500" : "bg-emerald-500");
  }

  // Update modal doctor info
  const signDocName = document.getElementById("sign-modal-doc-name");
  const signDocReg = document.getElementById("sign-modal-doc-reg");
  const signDocSpec = document.getElementById("sign-modal-doc-spec");
  if (signDocName) signDocName.textContent = state.currentUser.name;
  if (signDocReg) signDocReg.textContent = state.currentUser.registration_number || "TNMC-84920";
  if (signDocSpec) signDocSpec.textContent = `Specialization: ${state.currentUser.specialization || 'Clinical Medicine'}`;
}

function toggleRoleDropdown() {
  const drop = document.getElementById("role-dropdown");
  drop?.classList.toggle("hidden");
}

async function selectUser(userId) {
  try {
    const res = await fetch(`/api/auth/switch-role/${userId}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      state.currentUser = data.user;
      updateUserUI();
      toggleRoleDropdown();
      showToast(`Perspective switched to ${data.user.name}`, "success");
    }
  } catch (err) {
    console.error("Failed to switch user role:", err);
  }
}


// --- Google Meet & Consultation Loading ---
async function loadConsultation(consultationId) {
  try {
    const res = await fetch(`/api/consultations/${consultationId}`);
    if (!res.ok) return;
    const data = await res.json();
    state.activeConsultation = data;
    state.activeConsultationId = consultationId;

    // Update Banner
    const spaceCode = document.getElementById("meet-space-code");
    const bannerPatient = document.getElementById("consultation-patient-banner");
    const directMeetLink = document.getElementById("direct-meet-link");

    if (spaceCode) spaceCode.textContent = data.google_meet?.space_name || "spaces/mt-cardio-9842";
    if (bannerPatient) bannerPatient.textContent = `Patient: ${data.patient_name} (${data.patient_age} Yrs)`;
    if (directMeetLink && data.google_meet?.meeting_uri) {
      directMeetLink.href = data.google_meet.meeting_uri;
    }

    // Render Transcript Turns
    renderTranscripts(data.transcripts || []);

    // Also load Case Sheet in background
    loadCaseSheetForConsultation(consultationId);
  } catch (err) {
    console.error("Error loading consultation:", err);
  }
}

function renderTranscripts(turns) {
  const container = document.getElementById("transcript-stream");
  const counter = document.getElementById("turn-counter-badge");
  if (!container) return;

  if (counter) counter.textContent = `${turns.length} Turns`;
  container.innerHTML = "";

  turns.forEach(t => {
    const div = document.createElement("div");
    const roleClass = t.speaker === "doctor" ? "badge-doctor" :
                      t.speaker === "student" ? "badge-student" : "badge-patient";
    
    div.className = `p-2.5 rounded-xl border ${roleClass} space-y-1 transition duration-150`;
    div.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="font-bold text-[11px]">${t.speaker_name || t.speaker}</span>
        <span class="text-[10px] text-slate-400 font-mono">${t.timestamp}</span>
      </div>
      <p class="text-slate-800 leading-relaxed font-sans">${t.text}</p>
    `;
    container.appendChild(div);
  });

  container.scrollTop = container.scrollHeight;
}

function copyMeetLink() {
  const uri = state.activeConsultation?.google_meet?.meeting_uri || "https://meet.google.com/qam-pzjy-fkr";
  navigator.clipboard.writeText(uri).then(() => {
    showToast("Google Meet link copied to clipboard!", "success");
  });
}


// --- Rich Demonstration Scenarios Loader ---
async function loadSelectedScenario() {
  const select = document.getElementById("scenario-select");
  const key = select ? select.value : "cardiology";
  
  showToast(`Loading clinical scenario '${key}'...`, "info");
  try {
    const res = await fetch(`/api/consultations/${state.activeConsultationId}/load-scenario/${key}`, {
      method: "POST"
    });
    if (res.ok) {
      const data = await res.json();
      await loadConsultation(state.activeConsultationId);
      showToast(`Scenario loaded: ${data.scenario.title}`, "success");
      
      // Auto-trigger case sheet generation
      await generateCaseSheetFromCurrentConsultation();
    }
  } catch (err) {
    console.error("Failed to load scenario:", err);
    showToast("Failed to load scenario.", "danger");
  }
}

function onScenarioSelectChange() {
  // Can provide quick tooltip or preview
}


// --- Live Audio Waveform Visualization ---
function initAudioWaveform() {
  const canvas = document.getElementById("waveform-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Web Audio Context
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioContextClass();
    state.analyserNode = state.audioContext.createAnalyser();
    state.analyserNode.fftSize = 64;
    const bufferLength = state.analyserNode.frequencyBinCount;
    state.audioDataArray = new Uint8Array(bufferLength);
  } catch (e) {
    console.warn("AudioContext not supported, using simulated oscillogram:", e);
  }

  // Animation Loop for Waveform
  function draw() {
    state.waveformAnimationId = requestAnimationFrame(draw);
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, width, height);

    if (state.analyserNode && !state.isAudioMuted) {
      state.analyserNode.getByteFrequencyData(state.audioDataArray);
      
      // Calculate approximate dB
      let sum = 0;
      for (let i = 0; i < state.audioDataArray.length; i++) {
        sum += state.audioDataArray[i];
      }
      const avg = sum / state.audioDataArray.length;
      const db = Math.round(-60 + (avg / 255) * 50);
      const dbEl = document.getElementById("mic-db-level");
      if (dbEl) dbEl.textContent = `${db} dB`;

      // Draw Bars
      const barWidth = (width / state.audioDataArray.length) * 1.5;
      let x = 0;

      for (let i = 0; i < state.audioDataArray.length; i++) {
        const barHeight = (state.audioDataArray[i] / 255) * height;

        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, "#0f766e");
        gradient.addColorStop(1, "#14b8a6");

        ctx.fillStyle = gradient;
        ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    } else {
      // Simulated resting oscillogram line
      ctx.beginPath();
      ctx.strokeStyle = "#14b8a6";
      ctx.lineWidth = 1.5;
      const t = Date.now() * 0.003;
      for (let i = 0; i < width; i++) {
        const y = height / 2 + Math.sin(i * 0.05 + t) * (state.isAudioMuted ? 0.5 : 4);
        if (i === 0) ctx.moveTo(i, y);
        else ctx.lineTo(i, y);
      }
      ctx.stroke();
    }
  }

  draw();
}

// --- Live Webcam & Media Preview ---
async function initWebcamPreview() {
  const videoEl = document.getElementById("local-video");
  const fallback = document.getElementById("local-avatar-fallback");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (fallback) fallback.classList.remove("hidden");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    state.mediaStream = stream;
    if (videoEl) {
      videoEl.srcObject = stream;
      videoEl.play().catch(() => {});
    }
    if (fallback) fallback.classList.add("hidden");

    // Connect audio stream to analyser
    if (state.audioContext && state.analyserNode) {
      const source = state.audioContext.createMediaStreamSource(stream);
      source.connect(state.analyserNode);
    }
  } catch (err) {
    console.warn("Webcam permission not granted or device unavailable:", err.message);
    if (fallback) fallback.classList.remove("hidden");
  }
}

function toggleMicrophone() {
  state.isAudioMuted = !state.isAudioMuted;
  if (state.mediaStream) {
    state.mediaStream.getAudioTracks().forEach(track => {
      track.enabled = !state.isAudioMuted;
    });
  }
  const btn = document.getElementById("btn-toggle-mic");
  const label = document.getElementById("audio-status-label");
  if (state.isAudioMuted) {
    btn?.classList.replace("bg-slate-100", "bg-rose-100");
    btn?.classList.replace("text-slate-700", "text-rose-700");
    if (label) label.textContent = "Mic Muted";
    showToast("Microphone muted", "info");
  } else {
    btn?.classList.replace("bg-rose-100", "bg-slate-100");
    btn?.classList.replace("text-rose-700", "text-slate-700");
    if (label) label.textContent = "Live Audio Oscillogram";
    showToast("Microphone active", "info");
  }
}

function toggleCamera() {
  state.isVideoMuted = !state.isVideoMuted;
  const videoEl = document.getElementById("local-video");
  const fallback = document.getElementById("local-avatar-fallback");

  if (state.mediaStream) {
    state.mediaStream.getVideoTracks().forEach(track => {
      track.enabled = !state.isVideoMuted;
    });
  }

  if (state.isVideoMuted) {
    videoEl?.classList.add("hidden");
    fallback?.classList.remove("hidden");
    showToast("Webcam video disabled", "info");
  } else {
    videoEl?.classList.remove("hidden");
    fallback?.classList.add("hidden");
    showToast("Webcam video enabled", "info");
  }
}

async function toggleScreenShare() {
  if (!navigator.mediaDevices.getDisplayMedia) {
    showToast("Screen sharing not supported on this browser.", "danger");
    return;
  }

  try {
    const screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
    const videoEl = document.getElementById("local-video");
    if (videoEl) {
      videoEl.srcObject = screenStream;
    }
    showToast("Screen share active in consultation theater", "success");
    screenStream.getVideoTracks()[0].onended = () => {
      if (videoEl && state.mediaStream) videoEl.srcObject = state.mediaStream;
      showToast("Screen share ended", "info");
    };
  } catch (err) {
    console.warn("Screen share cancelled or failed:", err);
  }
}

function startCallTimer() {
  state.timerInterval = setInterval(() => {
    state.callTimerSeconds++;
    const mins = String(Math.floor(state.callTimerSeconds / 60)).padStart(2, "0");
    const secs = String(state.callTimerSeconds % 60).padStart(2, "0");
    const el = document.getElementById("call-timer");
    if (el) el.textContent = `${mins}:${secs}`;
  }, 1000);
}


// --- In-Browser Speech-to-Text Recognition ---
function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.warn("Web Speech API not supported in this browser.");
    return;
  }

  state.speechRecognition = new SpeechRec();
  state.speechRecognition.continuous = true;
  state.speechRecognition.interimResults = false;
  state.speechRecognition.lang = "en-IN";

  state.speechRecognition.onresult = async (event) => {
    const lastResult = event.results[event.results.length - 1];
    if (lastResult.isFinal) {
      const text = lastResult[0].transcript.trim();
      if (text) {
        const speaker = state.currentUser?.role || "doctor";
        const speakerName = state.currentUser?.name || "Doctor";
        await sendTurnDirect(speaker, speakerName, text);
      }
    }
  };

  state.speechRecognition.onerror = (e) => {
    console.warn("Speech recognition error:", e);
    state.isRecognizing = false;
    updateSpeechBtn();
  };

  state.speechRecognition.onend = () => {
    if (state.isRecognizing) {
      try { state.speechRecognition.start(); } catch (_) {}
    }
  };
}

function startVoiceRecognition() {
  if (!state.speechRecognition) {
    showToast("Web Speech API not available. Use manual turn input below.", "warning");
    return;
  }

  if (state.isRecognizing) {
    state.isRecognizing = false;
    state.speechRecognition.stop();
    updateSpeechBtn();
    showToast("Voice transcription paused", "info");
  } else {
    state.isRecognizing = true;
    try {
      state.speechRecognition.start();
      updateSpeechBtn();
      showToast("Voice transcription active: speak now!", "success");
    } catch (e) {
      state.isRecognizing = false;
      updateSpeechBtn();
    }
  }
}

function updateSpeechBtn() {
  const btn = document.getElementById("btn-speech-recog");
  const label = document.getElementById("speech-btn-label");
  if (!btn || !label) return;

  if (state.isRecognizing) {
    btn.className = "px-3 py-1.5 rounded-lg bg-rose-100 border border-rose-300 text-rose-800 text-xs font-semibold flex items-center space-x-1";
    label.textContent = "Listening...";
  } else {
    btn.className = "px-3 py-1.5 rounded-lg bg-teal-50 hover:bg-teal-100 border border-teal-200 text-teal-800 text-xs font-semibold flex items-center space-x-1 transition";
    label.textContent = "Voice Dictate";
  }
}

async function sendManualTurn() {
  const input = document.getElementById("manual-transcript-input");
  const select = document.getElementById("quick-speaker-select");
  const text = input ? input.value.trim() : "";
  if (!text) return;

  const speaker = select ? select.value : "doctor";
  const speakerName = speaker === "doctor" ? "Dr. Rajesh Sharma (Doctor)" :
                      speaker === "student" ? "Sneha Patel (Student)" : "K. Sundaram (Patient)";

  await sendTurnDirect(speaker, speakerName, text);
  if (input) input.value = "";
}

async function sendTurnDirect(speaker, speakerName, text) {
  try {
    const timestamp = new Date().toTimeString().slice(0, 8);
    const res = await fetch(`/api/consultations/${state.activeConsultationId}/transcripts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        speaker: speaker,
        speaker_name: speakerName,
        timestamp: timestamp,
        text: text,
        confidence: 0.98
      })
    });

    if (res.ok) {
      await loadConsultation(state.activeConsultationId);
    }
  } catch (err) {
    console.error("Failed to add transcript turn:", err);
  }
}


// --- AI Case Sheet Generation & 17 Sections Studio ---
async function generateCaseSheetFromCurrentConsultation() {
  showToast("Synthesizing 17-section case sheet with AI...", "info");
  try {
    const res = await fetch(`/api/casesheets/generate/${state.activeConsultationId}`, {
      method: "POST"
    });

    if (res.ok) {
      const data = await res.json();
      state.activeCaseSheet = data;
      renderCaseSheet(data);
      switchTab("casesheet");
      showToast("17-Section Case Sheet generated successfully!", "success");
    } else {
      const err = await res.json();
      showToast(err.detail || "Generation failed", "danger");
    }
  } catch (err) {
    console.error("Error generating case sheet:", err);
    showToast("Error communicating with AI engine", "danger");
  }
}

async function loadCaseSheetForConsultation(consultationId) {
  try {
    const res = await fetch(`/api/casesheets/${consultationId}`);
    if (res.ok) {
      const data = await res.json();
      state.activeCaseSheet = data;
      renderCaseSheet(data);
    }
  } catch (err) {
    console.warn("No existing case sheet for this consultation yet.");
  }
}

function renderCaseSheet(cs) {
  if (!cs || !cs.sections) return;
  const s = cs.sections;

  // Status & Badges
  const statusBadge = document.getElementById("cs-status-badge");
  const sourceBadge = document.getElementById("cs-source-badge");
  const approvedBanner = document.getElementById("approved-lock-banner");
  const signBtn = document.getElementById("btn-doctor-sign");

  if (statusBadge) {
    statusBadge.textContent = cs.status === "approved_locked" ? "VERIFIED & LOCKED" : "DRAFT RECORD";
    statusBadge.className = cs.status === "approved_locked"
      ? "bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2.5 py-0.5 rounded-full"
      : "bg-amber-100 text-amber-800 text-[10px] font-bold px-2.5 py-0.5 rounded-full";
  }

  if (sourceBadge) {
    sourceBadge.textContent = cs.extraction_source === "gemini" ? "Gemini AI" : "Clinical NLP Engine";
  }

  if (cs.status === "approved_locked") {
    approvedBanner?.classList.remove("hidden");
    const desc = document.getElementById("approved-lock-desc");
    const hashEl = document.getElementById("immutable-audit-hash");
    if (desc) desc.textContent = `Signed by ${cs.approval?.doctor_name} (${cs.approval?.doctor_registration_number || 'TNMC-84920'}) on ${cs.approval?.approval_timestamp?.slice(0, 19).replace('T', ' ')}`;
    if (hashEl) hashEl.textContent = cs.approval?.immutable_hash || "sha256:verified";
    if (signBtn) signBtn.classList.add("hidden");
  } else {
    approvedBanner?.classList.add("hidden");
    if (signBtn) signBtn.classList.remove("hidden");
  }

  // Section 1: Patient Info
  const sec1 = document.getElementById("cs-sec-1");
  if (sec1) {
    sec1.innerHTML = `
      <div><span class="text-slate-400">MRN:</span> <strong>${cs.patient_mrn || 'MT-2026-0841'}</strong></div>
      <div><span class="text-slate-400">Name:</span> <strong>${cs.patient_name || 'Patient'}</strong></div>
      <div><span class="text-slate-400">Age/Gender:</span> <strong>${cs.patient_age || 58}Y / ${cs.patient_gender || 'M'}</strong></div>
      <div><span class="text-slate-400">Blood Group:</span> <strong>${cs.patient_blood_group || 'B+'}</strong></div>
      <div><span class="text-slate-400">Attending Doctor:</span> <strong>${cs.doctor_name || 'Dr. Sharma'}</strong></div>
      <div><span class="text-slate-400">Medical Student:</span> <strong>${cs.student_name || 'Sneha Patel'}</strong></div>
    `;
  }

  // Section 2: Chief Complaint
  renderField("chief-complaint", s.chief_complaint);

  // Section 3: HPI
  renderField("hpi", s.history_of_present_illness);

  // Section 4: Symptoms
  renderSymptoms(s.symptoms || []);

  // Section 5: Duration & Onset
  renderField("duration", s.duration_onset);

  // Section 6: Past Medical History
  const previewPmh = document.getElementById("preview-pmh");
  if (previewPmh) {
    previewPmh.innerHTML = (s.past_medical_history || []).map(p => `<div>• ${p}</div>`).join("") || "None reported";
  }

  // Section 7: Medications Table
  renderMedicationsTable(s.medications || []);

  // Section 8: Allergies
  const previewAllergies = document.getElementById("preview-allergies");
  if (previewAllergies) {
    previewAllergies.innerHTML = (s.allergies || []).map(a => `<span class="px-2 py-0.5 bg-rose-100 text-rose-800 rounded font-semibold text-[11px]">${a}</span>`).join("") || "No known drug allergies";
  }

  // Section 9: Family History
  renderField("family", s.family_history);

  // Section 10: Social History
  renderField("social", s.social_history);

  // Section 11: Doctor Observations & Vitals
  renderField("observations", s.doctor_observations);
  const previewVitals = document.getElementById("preview-vitals");
  if (previewVitals && s.vitals) {
    previewVitals.innerHTML = `
      <div>BP: <strong>${s.vitals.blood_pressure || 'N/A'}</strong></div>
      <div>Pulse: <strong>${s.vitals.pulse_rate || 'N/A'}</strong></div>
      <div>RR: <strong>${s.vitals.respiratory_rate || 'N/A'}</strong></div>
      <div>Temp: <strong>${s.vitals.temperature || 'N/A'}</strong></div>
      <div>SpO2: <strong>${s.vitals.spo2 || 'N/A'}</strong></div>
      <div>BMI: <strong>${s.vitals.bmi || '24.2'}</strong></div>
    `;
  }

  // Section 12: Recommended Investigations
  const previewInv = document.getElementById("preview-investigations");
  if (previewInv) {
    previewInv.innerHTML = (s.investigations || []).map(i => `<div>• ${i}</div>`).join("") || "None requested";
  }

  // Section 13: Assessment & Differential Diagnoses
  renderField("assessment", s.assessment_diagnosis);
  const previewDiff = document.getElementById("preview-diff-diagnoses");
  if (previewDiff) {
    previewDiff.innerHTML = (s.differential_diagnoses || []).map(d => `<div>• ${d}</div>`).join("") || "None listed";
  }

  // Section 14: Treatment Plan
  renderField("treatment", s.treatment_plan);

  // Section 15: Follow-Up & Red Flags
  renderField("followup", s.follow_up_instructions);
  const previewRed = document.getElementById("preview-redflags");
  if (previewRed) {
    previewRed.innerHTML = (s.red_flag_warnings || []).map(r => `<div>⚠️ ${r}</div>`).join("") || "None specified";
  }

  // Section 16 & 17: Clinical Audit Gaps
  const previewMissing = document.getElementById("preview-missing");
  if (previewMissing) {
    previewMissing.innerHTML = (s.missing_information || []).map(m => `<div>🔍 ${m}</div>`).join("") || "None flagged";
  }
  const previewUncertain = document.getElementById("preview-uncertain");
  if (previewUncertain) {
    previewUncertain.innerHTML = (s.uncertain_information || []).map(u => `<div>❓ ${u}</div>`).join("") || "None identified";
  }

  // Multilingual Summaries
  renderMultilingualSummary(cs.multilingual_summary);
}

function renderField(name, value) {
  const preview = document.getElementById(`preview-${name}`);
  const edit = document.getElementById(`edit-${name}`);
  if (preview) preview.textContent = value || "None documented";
  if (edit) edit.value = value || "";
}

function renderSymptoms(symptoms) {
  const preview = document.getElementById("preview-symptoms");
  if (!preview) return;
  preview.innerHTML = symptoms.map(sym => {
    const sevColor = sym.severity === "Severe" || sym.severity === "Critical" ? "bg-rose-100 text-rose-800" :
                     sym.severity === "Moderate" ? "bg-amber-100 text-amber-800" : "bg-teal-100 text-teal-800";
    return `
      <div class="flex items-center justify-between bg-slate-50 p-2 rounded">
        <div>
          <span class="font-bold text-slate-800">${sym.symptom}</span>
          <span class="text-slate-500 text-[11px]">(${sym.duration || 'N/A'}) - ${sym.notes || ''}</span>
        </div>
        <span class="px-2 py-0.5 rounded text-[10px] font-bold ${sevColor}">${sym.severity}</span>
      </div>
    `;
  }).join("") || "<div class='text-slate-400'>No symptoms recorded</div>";
}

function renderMedicationsTable(meds) {
  const tbody = document.getElementById("medications-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  meds.forEach(m => {
    const tr = document.createElement("tr");
    tr.className = "med-row border-b border-slate-100";
    tr.innerHTML = `
      <td class="py-2 px-2.5 font-bold text-teal-900">${m.drug_name}</td>
      <td class="py-2 px-2.5">${m.dosage}</td>
      <td class="py-2 px-2.5">${m.frequency}</td>
      <td class="py-2 px-2.5"><span class="px-1.5 py-0.5 bg-slate-100 rounded text-[10px] font-semibold">${m.route}</span></td>
      <td class="py-2 px-2.5">${m.duration}</td>
      <td class="py-2 px-2.5 text-slate-500 text-[11px]">${m.instructions}</td>
      <td class="py-2 px-2.5 text-right">
        <button onclick="removeMedicationRow('${m.id}')" class="btn-remove-med text-slate-400 hover:text-rose-600 ${state.isEditMode ? '' : 'hidden'}">
          <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  lucide.createIcons();
}

function addMedicationRow() {
  if (!state.activeCaseSheet) return;
  const newMed = {
    id: `med-${Date.now()}`,
    drug_name: "New Medication",
    dosage: "500 mg",
    frequency: "1-0-1",
    route: "Oral",
    duration: "7 days",
    instructions: "Take after meals"
  };
  state.activeCaseSheet.sections.medications.push(newMed);
  renderMedicationsTable(state.activeCaseSheet.sections.medications);
  showToast("Added medication row", "info");
}

function removeMedicationRow(medId) {
  if (!state.activeCaseSheet) return;
  state.activeCaseSheet.sections.medications = state.activeCaseSheet.sections.medications.filter(m => m.id !== medId);
  renderMedicationsTable(state.activeCaseSheet.sections.medications);
  showToast("Medication row removed", "info");
}

function toggleEditMode(enable) {
  if (state.activeCaseSheet?.status === "approved_locked") {
    showToast("Approved records are locked and cannot be edited.", "warning");
    return;
  }

  state.isEditMode = enable;
  const btnPrev = document.getElementById("btn-mode-preview");
  const btnEdit = document.getElementById("btn-mode-edit");
  const btnAddMed = document.getElementById("btn-add-med");
  const removeBtns = document.querySelectorAll(".btn-remove-med");

  if (enable) {
    btnEdit?.classList.replace("text-slate-600", "bg-white");
    btnEdit?.classList.add("text-slate-800", "shadow-xs");
    btnPrev?.classList.remove("bg-white", "text-slate-800", "shadow-xs");
    btnPrev?.classList.add("text-slate-600");
    btnAddMed?.classList.remove("hidden");
    removeBtns.forEach(b => b.classList.remove("hidden"));

    toggleFieldsVisibility(true);
  } else {
    // Save inline edits
    saveInlineEdits();

    btnPrev?.classList.replace("text-slate-600", "bg-white");
    btnPrev?.classList.add("text-slate-800", "shadow-xs");
    btnEdit?.classList.remove("bg-white", "text-slate-800", "shadow-xs");
    btnEdit?.classList.add("text-slate-600");
    btnAddMed?.classList.add("hidden");
    removeBtns.forEach(b => b.classList.add("hidden"));

    toggleFieldsVisibility(false);
  }
}

function toggleFieldsVisibility(showInputs) {
  const fields = ["chief-complaint", "hpi", "duration", "pmh", "allergies", "treatment"];
  fields.forEach(f => {
    const preview = document.getElementById(`preview-${f}`);
    const edit = document.getElementById(`edit-${f}`);
    if (showInputs) {
      preview?.classList.add("hidden");
      edit?.classList.remove("hidden");
    } else {
      preview?.classList.remove("hidden");
      edit?.classList.add("hidden");
    }
  });
}

async function saveInlineEdits() {
  if (!state.activeCaseSheet) return;
  const s = state.activeCaseSheet.sections;

  s.chief_complaint = document.getElementById("edit-chief-complaint")?.value || s.chief_complaint;
  s.history_of_present_illness = document.getElementById("edit-hpi")?.value || s.history_of_present_illness;
  s.duration_onset = document.getElementById("edit-duration")?.value || s.duration_onset;
  s.treatment_plan = document.getElementById("edit-treatment")?.value || s.treatment_plan;

  renderCaseSheet(state.activeCaseSheet);

  try {
    const res = await fetch(`/api/casesheets/${state.activeCaseSheet.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sections: s })
    });
    if (res.ok) {
      showToast("Case sheet edits saved", "success");
    }
  } catch (err) {
    console.error("Failed to save case sheet updates:", err);
  }
}

function openPrintableCaseSheet() {
  if (!state.activeCaseSheet) {
    showToast("Please generate or select a case sheet first", "warning");
    return;
  }
  window.open(`/api/casesheets/${state.activeCaseSheet.id}/print`, "_blank");
}


// --- Multilingual Summary & Voice Playback (TTS) ---
function renderMultilingualSummary(summaryObj) {
  if (!summaryObj) return;
  const textEl = document.getElementById("multilingual-summary-text");
  if (textEl) {
    textEl.textContent = summaryObj[state.activeLanguage] || summaryObj["en"] || "No summary available";
  }
}

function switchLanguageSummary(langCode) {
  state.activeLanguage = langCode;
  const langs = ["en", "ta", "hi", "te", "ml", "kn"];
  langs.forEach(l => {
    const btn = document.getElementById(`lang-${l}`);
    if (l === langCode) {
      btn?.classList.replace("text-slate-300", "text-white");
      btn?.classList.add("bg-teal-600", "font-bold");
    } else {
      btn?.classList.remove("bg-teal-600", "font-bold");
      btn?.classList.add("text-slate-300");
    }
  });

  if (state.activeCaseSheet?.multilingual_summary) {
    renderMultilingualSummary(state.activeCaseSheet.multilingual_summary);
  }

  // If already playing TTS, restart with new language
  if (window.speechSynthesis.speaking) {
    playSummaryTTS();
  }
}

function playSummaryTTS() {
  if (!window.speechSynthesis) {
    showToast("Text-to-Speech audio not supported in this browser.", "danger");
    return;
  }

  window.speechSynthesis.cancel(); // Stop current speech
  const textEl = document.getElementById("multilingual-summary-text");
  const text = textEl ? textEl.textContent.trim() : "";
  if (!text) return;

  const utterance = new SpeechSynthesisUtterance(text);
  
  // Map language codes
  const langMap = {
    en: "en-IN",
    ta: "ta-IN",
    hi: "hi-IN",
    te: "te-IN",
    ml: "ml-IN",
    kn: "kn-IN"
  };
  utterance.lang = langMap[state.activeLanguage] || "en-IN";
  utterance.rate = 0.95; // Slightly slower for clinical clarity
  utterance.pitch = 1.0;

  const indicator = document.getElementById("tts-audio-indicator");
  utterance.onstart = () => {
    if (indicator) indicator.classList.remove("hidden");
  };
  utterance.onend = () => {
    if (indicator) indicator.classList.add("hidden");
  };
  utterance.onerror = () => {
    if (indicator) indicator.classList.add("hidden");
  };

  state.currentSpeechUtterance = utterance;
  window.speechSynthesis.speak(utterance);
}

function stopSummaryTTS() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
    const indicator = document.getElementById("tts-audio-indicator");
    if (indicator) indicator.classList.add("hidden");
  }
}


// --- Doctor Approval & Electronic Sign-Off Workflow ---
function openDoctorApprovalModal() {
  if (state.currentUser?.role !== "doctor") {
    showToast("Access Denied: Only certified Medical Doctors can approve clinical case sheets.", "danger");
    return;
  }
  if (!state.activeCaseSheet) {
    showToast("Please generate a case sheet before approving.", "warning");
    return;
  }
  const modal = document.getElementById("modal-doctor-sign");
  modal?.classList.remove("hidden");
  clearSignatureCanvas();
}

function closeDoctorApprovalModal() {
  const modal = document.getElementById("modal-doctor-sign");
  modal?.classList.add("hidden");
}

function initSignaturePad() {
  state.signatureCanvas = document.getElementById("signature-pad");
  if (!state.signatureCanvas) return;
  state.signatureCtx = state.signatureCanvas.getContext("2d");
  state.signatureCtx.lineWidth = 2;
  state.signatureCtx.lineCap = "round";
  state.signatureCtx.strokeStyle = "#0f766e";

  const canvas = state.signatureCanvas;

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  canvas.addEventListener("mousedown", (e) => {
    state.isDrawingSignature = true;
    const pos = getPos(e);
    state.signatureCtx.beginPath();
    state.signatureCtx.moveTo(pos.x, pos.y);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!state.isDrawingSignature) return;
    const pos = getPos(e);
    state.signatureCtx.lineTo(pos.x, pos.y);
    state.signatureCtx.stroke();
  });

  window.addEventListener("mouseup", () => {
    state.isDrawingSignature = false;
  });

  // Touch Support
  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    state.isDrawingSignature = true;
    const pos = getPos(e);
    state.signatureCtx.beginPath();
    state.signatureCtx.moveTo(pos.x, pos.y);
  });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!state.isDrawingSignature) return;
    const pos = getPos(e);
    state.signatureCtx.lineTo(pos.x, pos.y);
    state.signatureCtx.stroke();
  });

  canvas.addEventListener("touchend", () => {
    state.isDrawingSignature = false;
  });
}

function clearSignatureCanvas() {
  if (state.signatureCanvas && state.signatureCtx) {
    state.signatureCtx.clearRect(0, 0, state.signatureCanvas.width, state.signatureCanvas.height);
  }
}

async function submitDoctorSignOff() {
  const checkAcc = document.getElementById("check-accuracy")?.checked;
  const checkPres = document.getElementById("check-prescriptions")?.checked;
  const checkLock = document.getElementById("check-lock")?.checked;

  if (!checkAcc || !checkPres || !checkLock) {
    showToast("Please review and check all 3 verification requirements.", "warning");
    return;
  }

  const sigDataUrl = state.signatureCanvas?.toDataURL() || `SIGN-DEFAULT-${Date.now()}`;
  const notes = document.getElementById("sign-verification-notes")?.value;

  try {
    const res = await fetch(`/api/casesheets/${state.activeCaseSheet.id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doctor_id: state.currentUser.id,
        electronic_signature: sigDataUrl,
        verification_notes: notes || "Verified and approved in hospital electronic record."
      })
    });

    if (res.ok) {
      const data = await res.json();
      closeDoctorApprovalModal();
      await loadCaseSheetForConsultation(state.activeConsultationId);
      showToast("Case sheet approved, digitally signed, and locked!", "success");
    } else {
      const err = await res.json();
      showToast(err.detail || "Approval failed", "danger");
    }
  } catch (err) {
    console.error("Sign-off error:", err);
    showToast("Error during electronic sign-off", "danger");
  }
}


// --- Patient Directory & Age Calculator ---
async function loadPatients(searchTerm = "") {
  try {
    const url = searchTerm ? `/api/patients/?search=${encodeURIComponent(searchTerm)}` : "/api/patients/";
    const res = await fetch(url);
    if (!res.ok) return;
    const patients = await res.json();
    renderPatientsTable(patients);
  } catch (err) {
    console.error("Failed to load patients:", err);
  }
}

function onPatientSearch(val) {
  loadPatients(val);
}

function renderPatientsTable(patients) {
  const tbody = document.getElementById("patients-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  patients.forEach(p => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50 transition";
    tr.innerHTML = `
      <td class="py-3 px-4 font-mono font-bold text-teal-800">${p.mrn}</td>
      <td class="py-3 px-4 font-bold text-slate-800">${p.first_name} ${p.last_name}</td>
      <td class="py-3 px-4">${p.age} Yrs / ${p.gender}</td>
      <td class="py-3 px-4"><span class="px-2 py-0.5 bg-slate-100 rounded text-slate-700 font-semibold">${p.blood_group || 'Unknown'}</span></td>
      <td class="py-3 px-4 text-slate-600">${p.phone}</td>
      <td class="py-3 px-4">
        ${(p.known_allergies || []).map(a => `<span class="px-1.5 py-0.5 bg-rose-50 text-rose-700 rounded text-[10px] mr-1">${a}</span>`).join("") || '<span class="text-slate-400">None</span>'}
      </td>
      <td class="py-3 px-4">
        ${(p.chronic_conditions || []).map(c => `<span class="px-1.5 py-0.5 bg-teal-50 text-teal-700 rounded text-[10px] mr-1">${c}</span>`).join("") || '<span class="text-slate-400">None</span>'}
      </td>
      <td class="py-3 px-4 text-right">
        <button onclick="startConsultationForPatient('${p.id}')" class="bg-teal-700 hover:bg-teal-800 text-white px-3 py-1 rounded text-xs font-bold transition">
          Start Consultation
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function openNewPatientModal() {
  const modal = document.getElementById("modal-new-patient");
  modal?.classList.remove("hidden");
}

function closeNewPatientModal() {
  const modal = document.getElementById("modal-new-patient");
  modal?.classList.add("hidden");
}

function calculateAgeLive(dobStr) {
  const badge = document.getElementById("np-computed-age-badge");
  if (!badge || !dobStr) return;
  try {
    const dob = new Date(dobStr);
    const today = new Date();
    let age = today.getFullYear() - dob.getFullYear();
    const m = today.getMonth() - dob.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
      age--;
    }
    badge.textContent = `${Math.max(0, age)} Yrs`;
  } catch (_) {
    badge.textContent = "-- Yrs";
  }
}

async function submitNewPatient(e) {
  e.preventDefault();
  const dob = document.getElementById("np-dob").value;
  const firstName = document.getElementById("np-first-name").value.trim();
  const lastName = document.getElementById("np-last-name").value.trim();
  const gender = document.getElementById("np-gender").value;
  const blood = document.getElementById("np-blood-group").value;
  const phone = document.getElementById("np-phone").value.trim();
  const email = document.getElementById("np-email").value.trim();
  const allergiesRaw = document.getElementById("np-allergies").value.trim();
  const condsRaw = document.getElementById("np-conditions").value.trim();

  const payload = {
    first_name: firstName,
    last_name: lastName,
    date_of_birth: dob,
    gender: gender,
    blood_group: blood,
    phone: phone,
    email: email || null,
    known_allergies: allergiesRaw ? allergiesRaw.split(",").map(s => s.trim()).filter(Boolean) : [],
    chronic_conditions: condsRaw ? condsRaw.split(",").map(s => s.trim()).filter(Boolean) : []
  };

  try {
    const res = await fetch("/api/patients/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const created = await res.json();
      closeNewPatientModal();
      await loadPatients();
      showToast(`Patient ${created.first_name} registered with MRN ${created.mrn}`, "success");
    } else {
      const err = await res.json();
      showToast(err.detail || "Registration failed", "danger");
    }
  } catch (err) {
    console.error("Patient creation error:", err);
    showToast("Error creating patient", "danger");
  }
}

async function startConsultationForPatient(patientId) {
  try {
    const res = await fetch("/api/consultations/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        doctor_id: state.currentUser?.id || "doc-1",
        student_id: "stu-1"
      })
    });

    if (res.ok) {
      const data = await res.json();
      state.activeConsultationId = data.id;
      await loadConsultation(data.id);
      switchTab("consultation");
      showToast(`Consultation started for ${data.patient_name}`, "success");
    }
  } catch (err) {
    console.error("Error creating consultation:", err);
  }
}


// --- Consultation Archive & Cross-Consultation Viewer ---
async function loadConsultationArchive() {
  try {
    const res = await fetch("/api/consultations/");
    if (!res.ok) return;
    const consultations = await res.json();

    const timeline = document.getElementById("consultation-timeline-list");
    if (!timeline) return;
    timeline.innerHTML = "";

    // Stats
    const totalEl = document.getElementById("stat-total-consultations");
    const appEl = document.getElementById("stat-approved-cases");
    const pendEl = document.getElementById("stat-pending-reviews");

    if (totalEl) totalEl.textContent = consultations.length;
    const approvedCount = consultations.filter(c => c.is_approved).length;
    if (appEl) appEl.textContent = approvedCount;
    if (pendEl) pendEl.textContent = consultations.length - approvedCount;

    consultations.forEach(c => {
      const item = document.createElement("div");
      item.className = "flex items-center justify-between p-3 rounded-xl border border-slate-200 bg-white hover:border-teal-400 transition cursor-pointer";
      item.onclick = () => {
        state.activeConsultationId = c.id;
        loadConsultation(c.id);
        switchTab("consultation");
      };

      const statusBadge = c.is_approved
        ? `<span class="bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2 py-0.5 rounded-full">Approved</span>`
        : `<span class="bg-amber-100 text-amber-800 text-[10px] font-bold px-2 py-0.5 rounded-full">Pending Review</span>`;

      item.innerHTML = `
        <div class="flex items-center space-x-3">
          <div class="w-9 h-9 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold text-xs">
            ${c.patient_name.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div class="text-xs font-bold text-slate-800">${c.patient_name} <span class="text-slate-400 font-mono">(${c.patient_mrn})</span></div>
            <div class="text-[11px] text-slate-500">Dr. ${c.doctor_name} • ${c.scheduled_time.slice(0, 16).replace('T', ' ')}</div>
          </div>
        </div>
        <div class="flex items-center space-x-3">
          ${statusBadge}
          <i data-lucide="chevron-right" class="w-4 h-4 text-slate-400"></i>
        </div>
      `;
      timeline.appendChild(item);
    });

    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load consultation archive:", err);
  }
}

async function endConsultationAndGenerate() {
  if (confirm("End consultation session and generate 17-section case sheet?")) {
    await fetch(`/api/consultations/${state.activeConsultationId}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "completed" })
    });
    await generateCaseSheetFromCurrentConsultation();
  }
}


// --- Toast Notification Helper ---
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  const bgClass = type === "success" ? "bg-emerald-600 text-white" :
                  type === "danger" ? "bg-rose-600 text-white" :
                  type === "warning" ? "bg-amber-500 text-white" : "bg-slate-800 text-white";

  toast.className = `${bgClass} px-4 py-2.5 rounded-xl shadow-lg text-xs font-semibold flex items-center space-x-2 transition-all transform duration-300 opacity-0 translate-y-2`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  // Trigger animation
  setTimeout(() => {
    toast.classList.remove("opacity-0", "translate-y-2");
  }, 10);

  // Auto-remove after 3.5s
  setTimeout(() => {
    toast.classList.add("opacity-0", "translate-y-2");
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
