// ============================================
// LIVE DASHBOARD v4 — SMART CLASSROOM
// Backend-integrated, no mock data, real events only.
//
//   • /api/v1/health
//   • /api/v1/vision/recognize-frame
//   • /api/v1/interaction/ask-question
//   • /api/v1/registration/{start,capture,submit,approve,reject}
//   • /logs/events  (also at /api/v1/events)
// ============================================

(function () {
    'use strict';

    // ══════════════════════════════════════
    // Storage hygiene — clear any stale state from prior sessions
    // ══════════════════════════════════════
    try {
        // Only clear keys we own to avoid trampling on third-party state.
        const PURGE_PREFIX = 'smart_classroom_';
        [localStorage, sessionStorage].forEach(store => {
            const drop = [];
            for (let i = 0; i < store.length; i++) {
                const k = store.key(i);
                if (k && k.startsWith(PURGE_PREFIX)) drop.push(k);
            }
            drop.forEach(k => store.removeItem(k));
        });
    } catch (_) { /* private mode etc. — non-fatal */ }

    // ══════════════════════════════════════
    // BACKEND CONFIG (driven by window.APP_CONFIG — see JS/config.js)
    // ══════════════════════════════════════
    const APP = window.APP_CONFIG || {};
    const SAME_ORIGIN_OK = location.protocol !== 'file:' && !!location.host
        && !/^127\.0\.0\.1:5500$|^localhost:5500$/.test(location.host);
    const FALLBACK_API = SAME_ORIGIN_OK ? location.origin : 'http://127.0.0.1:8000';
    const API = (APP.API_BASE_URL || window.SMART_CLASSROOM_API || FALLBACK_API).replace(/\/+$/, '');
    const API_V1 = `${API}/api/v1`;

    // ── Cadence (self-paced; each value is a floor) ───────────────
    // The recognize loop schedules its next call AFTER the current
    // response returns — we never queue requests on a fixed interval.
    // Health: slow heartbeat when backend is reachable, fast retry
    // only when offline (so the UI recovers quickly on reconnect).
    const RECOGNIZE_INTERVAL_MS       = APP.RECOGNIZE_INTERVAL_MS       || 200;
    const HEALTH_INTERVAL_MS          = APP.HEALTH_INTERVAL_MS          || 30000;
    const HEALTH_OFFLINE_INTERVAL_MS  = APP.HEALTH_OFFLINE_INTERVAL_MS  || 2000;
    const EVENT_POLL_MS               = APP.EVENT_POLL_MS               || 1500;
    const FRAME_SEND_QUALITY    = 0.65;
    const FRAME_SEND_WIDTH      = 480;
    const ADMIN_PASSWORD        = 'aiu';   // frontend gate; backend re-validates

    // ══════════════════════════════════════
    // STATE STORE — starts EMPTY. Only populated by real backend events.
    // ══════════════════════════════════════
    const store = {
        backendOnline: false,
        cameraActive: false,
        cameraStream: null,
        cameraResumeAfterReg: false,   // remember whether to restart cam after reg ends
        micState: 'idle', // idle | preparing | listening | processing | completed
        currentStudent: null,
        students: {},                   // ONLY populated from real backend events
        eventCursor: 0,                 // next event index to consume (skips prior log on first connect)
        firstEventConnect: true,        // skip past events on initial connect
        recognizeTimer: null,
        healthTimer: null,
        eventTimer: null,
        registration: {
            sessionId: null,
            captureCount: 0,
            pendingName: null,
            inFlight: false,
            adminVisible: false,
        },
        // Guest mode — when active, guestId is the backend-allocated
        // Guest_NNN identity that owns the next mic press.
        guestMode: false,
        guestId: null,
    };

    // ══════════════════════════════════════
    // DOM HELPERS
    // ══════════════════════════════════════
    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    const dom = {
        section:        $('#demo'),
        statusDot:      $('.status-dot'),
        statusLabel:    $('.status-label'),
        cameraWrapper:  $('.camera-feed-wrapper'),
        cameraVideo:    $('#dash-camera-video'),
        cameraBtn:      $('#dash-camera-btn'),
        downloadCsvBtn: $('#dash-download-csv-btn'),
        cameraBadge:    $('#camera-badge'),
        offlineMsg:     $('.offline-msg'),
        identityCard:   $('.identity-card'),
        idAvatar:       $('.id-avatar'),
        idName:         $('.id-name'),
        idAttendance:   $('.id-attendance'),
        idEmotion:      $('.id-emotion'),
        questionOvr:    $('.question-overlay'),
        qoText:         $('.qo-text'),
        qoTopic:        $('.qo-topic'),
        micOrb:         $('#dash-mic-orb'),
        micLabel:       $('#dash-mic-label'),
        waveBars:       $('.wave-bars'),
        actionBtns:     $('.action-buttons'),
        registerBtn:    $('#dash-register-btn'),
        guestBtn:       $('#dash-guest-btn'),
        logFeed:        $('#log-feed'),
        questionFeed:   $('#question-feed'),
        summaryGrid:    $('#summary-grid'),

        // Mic countdown overlay (inside camera wrapper)
        micCountdown:        $('#mic-countdown'),
        micCountdownLabel:   $('#mic-countdown-label'),
        micCountdownNumber:  $('#mic-countdown-number'),

        // Inline admin panel
        adminPanel:     $('#admin-panel'),
        adminTarget:    $('#admin-target-name'),
        adminPass:      $('#admin-pass-input'),
        adminHint:      $('#admin-panel-hint'),
        adminApprove:   $('#admin-approve-btn'),
        adminReject:    $('#admin-reject-btn'),

        // Toasts
        toastRoot:      $('#toast-container'),

        // Capture/name modal (admin step removed)
        modal:          $('#registration-modal'),
        modalClose:     $('.modal-close'),
        modalSteps:     $$('.modal-step'),
        stepDots:       $$('.step-dot'),
        captureVideo:   $('#capture-preview-video'),
        captureCount:   $('#capture-count'),
        captureBtn:     $('#modal-capture-btn'),
        nameInput:      $('#reg-name-input'),
        nameError:      $('#name-error'),
        nameNextBtn:    $('#name-next-btn'),
    };

    // Hidden canvas reused for frame uploads
    const uploadCanvas = document.createElement('canvas');
    const uploadCtx = uploadCanvas.getContext('2d');

    // ══════════════════════════════════════
    // UTILITIES
    // ══════════════════════════════════════
    function timeNow() {
        return new Date().toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    }
    function getInitials(name) {
        if (!name || name === 'Guest' || name === 'Unknown') return '?';
        if (name.startsWith('Guest_')) return 'G';
        return name.split('_').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    }
    const EMOJIS = {
        happy:'😊', neutral:'😐', tired:'😴', sad:'😢',
        angry:'😠', surprised:'😲', anxious:'😨', uncomfortable:'🤢'
    };
    const MAX_LOG_CARDS = 30;
    function emotionStr(e) {
        const k = (e || '').toLowerCase();
        return `${EMOJIS[k] || '😐'} ${e ? e.charAt(0).toUpperCase() + e.slice(1) : 'Neutral'}`;
    }

    async function api(method, path, body) {
        const url = path.startsWith('http') ? path : `${API_V1}${path}`;
        const opts = {
            method,
            headers: body ? { 'Content-Type': 'application/json' } : undefined,
            body: body ? JSON.stringify(body) : undefined,
        };
        const resp = await fetch(url, opts);
        if (!resp.ok) {
            let detail;
            try { detail = (await resp.json()).detail; } catch (_) {}
            const err = new Error(detail || `${resp.status} ${resp.statusText}`);
            err.status = resp.status;
            throw err;
        }
        return resp.status === 204 ? null : resp.json();
    }

    // ══════════════════════════════════════
    // TOAST NOTIFICATIONS
    // ══════════════════════════════════════
    function toast({ kind = 'info', title, message, timeout = 4000 } = {}) {
        if (!dom.toastRoot) {
            console[kind === 'error' ? 'error' : 'log'](title, message || '');
            return;
        }
        const icons = { success: '✓', error: '⚠', info: 'ℹ' };
        const el = document.createElement('div');
        el.className = `toast ${kind}`;
        el.innerHTML = `
            <div class="toast-icon">${icons[kind] || icons.info}</div>
            <div class="toast-body">
                <div class="toast-title"></div>
                ${message ? '<div class="toast-message"></div>' : ''}
            </div>`;
        el.querySelector('.toast-title').textContent = title || '';
        if (message) el.querySelector('.toast-message').textContent = message;
        dom.toastRoot.appendChild(el);
        // Trigger enter animation
        requestAnimationFrame(() => el.classList.add('visible'));
        // Auto-dismiss
        const remove = () => {
            el.classList.remove('visible');
            setTimeout(() => el.remove(), 400);
        };
        const t = setTimeout(remove, timeout);
        el.addEventListener('click', () => { clearTimeout(t); remove(); });
    }

    // ══════════════════════════════════════
    // BACKEND HEALTH
    // ══════════════════════════════════════
    async function checkHealth() {
        try {
            const healthPath = (APP.HEALTH_PATH || '/health').replace(/^\/?/, '/');
            await api('GET', `${API}${healthPath}`);
            if (!store.backendOnline) {
                setStatus(true, store.cameraActive ? 'Camera Active — Scanning' : 'Backend Online');
                // First time we connect: skip everything that's already in the log.
                if (store.firstEventConnect) {
                    seedEventCursor();
                }
            }
        } catch (_) {
            if (store.backendOnline) setStatus(false, 'Backend Offline');
            store.backendOnline = false;
        }
    }
    // Self-pacing health loop.  Re-schedules with setTimeout so the
    // wait is always SLOW (30 s) when the backend is online and FAST
    // (2 s) only while we're trying to reconnect — never a steady
    // 5 s ping flood like the old setInterval pattern.
    function startHealthLoop() {
        clearTimeout(store.healthTimer);
        const tick = async () => {
            await checkHealth();
            const next = store.backendOnline
                ? HEALTH_INTERVAL_MS
                : HEALTH_OFFLINE_INTERVAL_MS;
            store.healthTimer = setTimeout(tick, next);
        };
        tick();
    }

    // First connect: auto-populate the dashboard with the existing log history.
    async function seedEventCursor() {
        store.firstEventConnect = false;
        try {
            const data = await api('GET', `${API}/logs/events`);
            const events = data.events || [];
            resetDashboardState();
            events.forEach(processBackendEvent);
            const total = (typeof data.total === 'number') ? data.total : events.length;
            store.eventCursor = total;
            toast({
                kind: 'success',
                title: 'Attendance loaded',
                message: `${events.length} event(s) replayed automatically`,
            });
        } catch (_) {
            store.eventCursor = 0;
        }
    }

    // ══════════════════════════════════════
    // STATUS
    // ══════════════════════════════════════
    function setStatus(online, msg) {
        store.backendOnline = online;
        if (dom.statusDot)   dom.statusDot.classList.toggle('offline', !online);
        if (dom.statusLabel) dom.statusLabel.textContent = msg || (online ? 'System Online' : 'System Offline');
        if (dom.offlineMsg)  dom.offlineMsg.classList.toggle('visible', !online && !store.cameraActive);
    }

    // ══════════════════════════════════════
    // LAUNCH
    // ══════════════════════════════════════
    let launched = false;
    $$('.launch-demo-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.preventDefault();
            if (launched) { dom.section.scrollIntoView({ behavior: 'smooth' }); return; }
            launched = true;
            dom.section.classList.add('dashboard-active');
            if (window.gsap) {
                gsap.from(dom.section, { opacity: 0, y: 50, duration: .9, ease: 'power3.out' });
                gsap.from('.dash-card', { y: 40, opacity: 0, duration: .7, stagger: .12, ease: 'power2.out', delay: .25 });
            }
            setTimeout(() => dom.section.scrollIntoView({ behavior: 'smooth' }), 120);
            setTimeout(() => { if (window.startModelTransition) window.startModelTransition(); }, 400);
        });
    });

    // ══════════════════════════════════════
    // CAMERA — browser webcam → backend recognize-frame
    // ══════════════════════════════════════
    async function openCamera() {
        try {
            store.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
            dom.cameraVideo.srcObject = store.cameraStream;
            dom.cameraVideo.style.display = 'block';
            dom.cameraVideo.style.transform = 'scaleX(-1)';
            dom.cameraWrapper.classList.add('active');
            dom.cameraBtn.textContent = 'Stop Camera';
            dom.cameraBtn.classList.add('camera-toggle-active');
            dom.cameraBadge.textContent = 'LIVE';
            dom.cameraBadge.className = 'dash-card-badge recording';
            store.cameraActive = true;
            setStatus(store.backendOnline, 'Camera Active — Scanning');
            startRecognitionLoop();
            return true;
        } catch (err) {
            console.error('Camera error:', err);
            toast({ kind: 'error', title: 'Camera unavailable', message: err.message || 'Permission denied' });
            return false;
        }
    }

    if (dom.cameraBtn) {
        dom.cameraBtn.addEventListener('click', async () => {
            if (!store.cameraActive) {
                await openCamera();
            } else {
                stopCamera();
            }
        });
    }

    function stopCamera() {
        stopRecognitionLoop();
        if (store.cameraStream) {
            store.cameraStream.getTracks().forEach(t => t.stop());
            store.cameraStream = null;
        }
        if (dom.cameraVideo) {
            dom.cameraVideo.srcObject = null;
            dom.cameraVideo.style.display = 'none';
        }
        dom.cameraWrapper.classList.remove('active');
        dom.cameraBtn.textContent = 'Start Camera';
        dom.cameraBtn.classList.remove('camera-toggle-active');
        dom.cameraBadge.textContent = 'STANDBY';
        dom.cameraBadge.className = 'dash-card-badge';
        if (dom.identityCard) dom.identityCard.classList.remove('visible');
        hideQuestionOverlay();
        store.cameraActive = false;
        store.currentStudent = null;
        store.guestMode = false;
        store.guestId = null;
        setStatus(store.backendOnline, store.backendOnline ? 'Backend Online' : 'System Offline');
        if (dom.actionBtns) dom.actionBtns.style.display = 'none';
    }

    function captureFrameDataURL(videoEl) {
        const v = videoEl || dom.cameraVideo;
        if (!v || v.readyState < 2) return null;
        const w = FRAME_SEND_WIDTH;
        const h = Math.round((v.videoHeight || 480) * (w / (v.videoWidth || 640)));
        uploadCanvas.width = w; uploadCanvas.height = h;
        uploadCtx.save();
        uploadCtx.translate(w, 0);
        uploadCtx.scale(-1, 1);
        uploadCtx.drawImage(v, 0, 0, w, h);
        uploadCtx.restore();
        return uploadCanvas.toDataURL('image/jpeg', FRAME_SEND_QUALITY);
    }

    // Self-pacing recognition loop.
    //
    // The backend now answers /vision/recognize-frame in ~25-50 ms
    // (lock-cache hit) or ~150-300 ms (cold encode).  A fixed-interval
    // setInterval would either under-utilise the backend on the fast
    // path or pile up overlapping requests on the slow one.
    //
    // Solution: schedule the NEXT call only AFTER the current one
    // returns, with a small floor (RECOGNIZE_INTERVAL_MS) so we never
    // hammer the backend back-to-back when responses are sub-1 ms.
    // This naturally adapts:
    //   • Fast backend → ~5 fps recognize cadence.
    //   • Slow backend → 1-2 fps, no queue, no UI lag.
    let recognizeBusy = false;
    async function recognizeFrameOnce() {
        if (!store.cameraActive || recognizeBusy) return;
        const dataUrl = captureFrameDataURL();
        if (!dataUrl) return;
        recognizeBusy = true;
        try {
            const data = await api('POST', '/vision/recognize-frame', {
                image_base64: dataUrl,
                mark_attendance: true,
            });
            handleRecognitionResult(data);
        } catch (err) {
            if (err.status === undefined) setStatus(false, 'Backend Offline');
        } finally {
            recognizeBusy = false;
        }
    }
    function startRecognitionLoop() {
        clearTimeout(store.recognizeTimer);
        const tick = async () => {
            if (!store.cameraActive) {
                store.recognizeTimer = null;
                return;
            }
            await recognizeFrameOnce();
            // Re-schedule with a floor; a long backend response means
            // the tick itself was already long, and we still wait the
            // same floor before queuing the next one.
            if (store.cameraActive) {
                store.recognizeTimer = setTimeout(tick, RECOGNIZE_INTERVAL_MS);
            }
        };
        tick();
    }
    function stopRecognitionLoop() {
        clearTimeout(store.recognizeTimer);
        store.recognizeTimer = null;
    }

    function handleRecognitionResult(data) {
        const results = (data && data.results) || [];
        if (results.length === 0) return;
        const best = results.reduce((acc, r) => {
            if (!r.location) return acc;
            const [t, rg, b, l] = r.location;
            const area = Math.max(0, (b - t) * (rg - l));
            return (!acc || area > acc.area) ? { area, r } : acc;
        }, null);
        if (!best) return;

        const r = best.r;
        const name = r.registered ? r.name : 'Unknown';
        const emotion = r.emotion || 'Neutral';
        const attendance = r.registered ? 'Present' : 'Unregistered';
        const emotionStable = !!r.emotion_stable;
        const emotionSamples = r.emotion_samples || 0;
        showIdentity(name, attendance, emotion, r.registered, emotionStable, emotionSamples);

        if (r.registered) {
            // A registered face takes over the active mic context; any
            // prior guest session is dropped so the next mic press is
            // attributed to the recognised student.
            store.guestMode = false;
            store.guestId = null;
            store.currentStudent = { name, attendance, emotion, registered: true };
        } else if (!store.guestMode) {
            store.currentStudent = { name: 'Unknown', attendance: 'Unregistered', emotion, registered: false };
        }

        // Immediately fetch new events when attendance was JUST marked
        // (fires exactly once per student, not on every subsequent frame).
        if (results.some(res => res.newly_marked)) {
            pollEventsOnce();
        }
    }

    // ══════════════════════════════════════
    // IDENTITY CARD
    // ══════════════════════════════════════
    function showIdentity(name, attendance, emotion, registered, emotionStable, emotionSamples) {
        if (!dom.identityCard) return;
        // Determine what to show for emotion:
        //   - If emotion averaging is NOT done yet → show "Detecting emotion..."
        //   - If emotion averaging IS done → show the final averaged label
        const emotionDisplay = (emotionStable)
            ? emotionStr(emotion)
            : `⏳ Detecting emotion... (${emotionSamples || 0}/5)`;

        if (registered) {
            dom.idAvatar.textContent = getInitials(name);
            dom.idName.textContent = name.replace(/_/g, ' ');
            dom.idAttendance.textContent = attendance || 'Present';
            dom.idAttendance.classList.toggle('absent', attendance === 'Absent');
            dom.idEmotion.textContent = emotionDisplay;
            if (dom.actionBtns) dom.actionBtns.style.display = 'none';
        } else if (store.guestMode) {
            dom.idAvatar.textContent = 'G';
            dom.idName.textContent = (store.guestId || 'Guest Visitor').replace(/_/g, ' ');
            dom.idAttendance.textContent = 'Guest';
            dom.idAttendance.classList.add('absent');
            dom.idEmotion.textContent = emotionDisplay;
            if (dom.actionBtns) dom.actionBtns.style.display = 'none';
        } else {
            dom.idAvatar.textContent = '?';
            dom.idName.textContent = 'Unknown Visitor';
            dom.idAttendance.textContent = 'Unregistered';
            dom.idAttendance.classList.add('absent');
            dom.idEmotion.textContent = emotionDisplay;
            if (dom.actionBtns) dom.actionBtns.style.display = 'flex';
        }
        dom.identityCard.classList.add('visible');
    }

    // ══════════════════════════════════════
    // QUESTION OVERLAY
    // ══════════════════════════════════════
    function showQuestionOverlay(question, topic) {
        if (!dom.questionOvr) return;
        dom.qoText.textContent = `"${question}"`;
        dom.qoTopic.innerHTML = `Topic: <strong>${topic}</strong>`;
        dom.questionOvr.classList.add('visible');
        clearTimeout(showQuestionOverlay._t);
        showQuestionOverlay._t = setTimeout(hideQuestionOverlay, 6000);
    }
    function hideQuestionOverlay() {
        if (dom.questionOvr) dom.questionOvr.classList.remove('visible');
    }

    // ══════════════════════════════════════
    // MICROPHONE — toggle with countdown UX
    // States: idle → preparing → listening → processing → completed → idle
    // ══════════════════════════════════════
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let prepareTimers = [];

    let _recognitionGotResult = false;

    if (SR) {
        recognition = new SR();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            // Note: state was already set to 'listening' by startPushToTalk()
            _recognitionGotResult = false;
            setMicState('listening');
        };

        recognition.onresult = async ev => {
            _recognitionGotResult = true;
            const transcript = ev.results[0][0].transcript;
            setMicState('processing');
            try {
                const studentName = pickStudentForQuestion();
                const data = await api('POST', '/interaction/ask-question', {
                    student: studentName,
                    text: transcript,
                });
                showQuestionOverlay(data.question, data.topic);
                dom.micLabel.textContent = `"${data.question}"`;
                setMicState('completed');
                setTimeout(() => { setMicState('idle'); dom.micLabel.textContent = 'Ask a Question'; }, 3000);
            } catch (err) {
                console.error('ask-question failed:', err);
                toast({ kind: 'error', title: 'Question failed', message: err.message || 'Backend error' });
                setMicState('idle');
                dom.micLabel.textContent = 'Ask a Question';
            }
        };

        recognition.onerror = async ev => {
            console.error('Speech error:', ev.error);
            // Translate browser SpeechRecognition error codes into messages
            // a user can act on.  Never surface raw codes like "network".
            const SR_ERROR_MAP = {
                'no-speech':       { title: 'No speech detected',          msg: 'Click the mic and try again.' },
                'audio-capture':   { title: 'Microphone unavailable',      msg: 'No working microphone was found.' },
                'not-allowed':     { title: 'Microphone permission denied', msg: 'Allow microphone access in your browser settings.' },
                'service-not-allowed': { title: 'Speech service blocked',  msg: 'Browser blocked the speech service.' },
                'network':         { title: 'Speech service unavailable',  msg: 'Could not reach the online speech recognizer.' },
                'language-not-supported': { title: 'Language unsupported', msg: 'en-US is not available on this device.' },
                'bad-grammar':     { title: 'Could not understand speech', msg: 'Please rephrase your question.' },
            };

            // Browser speech can't reach Google → try the backend MediaRecorder fallback
            if (ev.error === 'network' && !ev.__fallbackTried && store.backendOnline) {
                ev.__fallbackTried = true;
                hideMicCountdown();
                clearPrepareTimers();
                setMicState('processing');
                dom.micLabel.textContent = 'Falling back to server mic…';
                try {
                    await startMediaRecorderFallback();
                    return;
                } catch (err) {
                    console.warn('Backend speech fallback failed:', err);
                    // fall through to surface a clean error
                }
            }
            if (ev.error && ev.error !== 'aborted') {
                const mapped = SR_ERROR_MAP[ev.error] || { title: 'Microphone error', msg: 'Please try again.' };
                toast({ kind: 'error', title: mapped.title, message: mapped.msg });
            }
            hideMicCountdown();
            clearPrepareTimers();
            dom.micLabel.textContent = 'Ask a Question';
            setMicState('idle');
        };

        recognition.onend = () => {
            // If we ended without producing a result (timeout, or manual stop
            // with no audio captured), reset back to idle.  This also catches
            // the case where stopPushToTalk() bumped us to 'processing' but
            // onresult never fired — without this, the mic would hang forever.
            const stuck = store.micState === 'listening' ||
                          (store.micState === 'processing' && !_recognitionGotResult);
            if (stuck) {
                setMicState('idle');
                dom.micLabel.textContent = 'Ask a Question';
            }
            _recognitionGotResult = false;
        };
    }

    function pickStudentForQuestion() {
        // Guest mode → use the backend-allocated Guest_NNN id so the
        // question lands on the right guest in the log.
        if (store.guestMode && store.guestId) return store.guestId;
        if (store.currentStudent && store.currentStudent.registered) return store.currentStudent.name;
        // Empty signals "let the backend resolve from the active
        // vision-session identity (last seen face within ~8s)".
        return '';
    }

    let mediaRecorder = null;
    let audioChunks = [];
    let mediaRecorderStream = null;

    async function startMediaRecorderFallback() {
        setMicState('listening');
        dom.micLabel.textContent = 'Listening (Fallback)…';
        
        try {
            mediaRecorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(mediaRecorderStream);
            audioChunks = [];
            
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener("stop", async () => {
                setMicState('processing');
                dom.micLabel.textContent = 'Processing audio…';
                mediaRecorderStream.getTracks().forEach(track => track.stop());
                
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                if (audioBlob.size === 0) {
                    toast({ kind: 'error', title: 'Audio Error', message: 'Payload empty: frontend bug.' });
                    setMicState('idle');
                    dom.micLabel.textContent = 'Ask a Question';
                    return;
                }
                
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.webm");
                formData.append("language", "en-US");
                
                try {
                    // Send to the debug endpoint first if you want to inspect, or directly to transcribe-audio
                    // For now, let's use the debug endpoint to get detailed diagnostics.
                    // Wait, the debug endpoint doesn't process the question natively. 
                    // Let's use the transcribe-audio endpoint, but log the exact error if it fails!
                    
                    const fetchUrl = `${API_V1}/speech/transcribe-audio`;
                    const resp = await fetch(fetchUrl, {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!resp.ok) {
                        let detail;
                        try { detail = (await resp.json()).detail; } catch (_) {}
                        throw new Error(detail || `Backend returned ${resp.status}`);
                    }
                    
                    const speech = await resp.json();
                    const transcript = (speech && speech.text) || '';
                    if (!transcript) {
                        throw new Error('Empty transcript from server');
                    }
                    
                    const studentName = pickStudentForQuestion();
                    const data = await api('POST', '/interaction/ask-question', {
                        student: studentName,
                        text: transcript,
                    });
                    
                    showQuestionOverlay(data.question, data.topic);
                    dom.micLabel.textContent = `"${data.question}"`;
                    setMicState('completed');
                    setTimeout(() => { setMicState('idle'); dom.micLabel.textContent = 'Ask a Question'; }, 3000);
                    pollEventsOnce();
                } catch (err) {
                    console.error('Backend audio upload failed:', err);
                    toast({
                        kind: 'error',
                        title: 'Backend Speech Error',
                        message: err.message || 'Unknown backend error'
                    });
                    setMicState('idle');
                    dom.micLabel.textContent = 'Ask a Question';
                }
            });
            
            mediaRecorder.start();
        } catch (err) {
            console.error('Failed to start MediaRecorder:', err);
            toast({ kind: 'error', title: 'Microphone Error', message: 'Could not access microphone for fallback.' });
            setMicState('idle');
            dom.micLabel.textContent = 'Ask a Question';
        }
    }
    
    // Legacy runBackendSpeechFlow is replaced by startMediaRecorderFallback above.
    // The previous error map is no longer needed since we surface the exact message.

    function clearPrepareTimers() {
        prepareTimers.forEach(clearTimeout);
        prepareTimers = [];
    }

    function showMicCountdown(label, number, extraClass) {
        if (!dom.micCountdown) return;
        dom.micCountdown.hidden = false;
        dom.micCountdown.classList.toggle('go', extraClass === 'go');
        if (label  !== undefined) dom.micCountdownLabel.textContent  = label;
        if (number !== undefined) dom.micCountdownNumber.textContent = number;
        // Re-trigger pop animation
        dom.micCountdownNumber.style.animation = 'none';
        // eslint-disable-next-line no-unused-expressions
        dom.micCountdownNumber.offsetWidth;
        dom.micCountdownNumber.style.animation = '';
    }
    function hideMicCountdown() {
        if (!dom.micCountdown) return;
        dom.micCountdown.hidden = true;
        dom.micCountdown.classList.remove('go');
    }

    function startPushToTalk() {
        if (store.micState !== 'idle') return;
        setMicState('preparing');
        dom.micLabel.textContent = 'Preparing microphone…';
        showMicCountdown('Preparing microphone…', '3');
        clearPrepareTimers();
        prepareTimers.push(setTimeout(() => {
            if (store.micState !== 'preparing') return;
            showMicCountdown('Speak in', '3');
        }, 50));
        prepareTimers.push(setTimeout(() => store.micState === 'preparing' && showMicCountdown('Speak in', '2'), 1050));
        prepareTimers.push(setTimeout(() => store.micState === 'preparing' && showMicCountdown('Speak in', '1'), 2050));
        prepareTimers.push(setTimeout(async () => {
            if (store.micState !== 'preparing') return;
            showMicCountdown('Speak now', 'GO', 'go');
            dom.micLabel.textContent = 'Listening…';
            setMicState('listening');

            if (recognition) {
                try {
                    recognition.start();
                } catch (e) {
                    console.warn(e);
                    hideMicCountdown();
                    setMicState('idle');
                    dom.micLabel.textContent = 'Ask a Question';
                }
            } else {
                // Backend mic mode (browser has no Web Speech API)
                await startMediaRecorderFallback();
            }
            prepareTimers.push(setTimeout(hideMicCountdown, 700));
        }, 3050));
    }

    function stopPushToTalk() {
        clearPrepareTimers();
        hideMicCountdown();
        if (store.micState === 'preparing') {
            setMicState('idle');
            dom.micLabel.textContent = 'Ask a Question';
            return;
        }
        if (store.micState === 'listening') {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            } else if (recognition) {
                try { recognition.stop(); } catch (_) {}
                setMicState('processing'); // Transition to processing as it will trigger onresult if audio was caught
            } else {
                toast({ kind: 'info', title: 'Processing', message: 'Waiting for server microphone to finish...' });
            }
        }
    }

    if (dom.micOrb) {
        dom.micOrb.addEventListener('click', () => {
            // Toggle behaviour: idle → start; preparing/listening → stop.
            if (store.micState === 'idle' || store.micState === 'completed') {
                startPushToTalk();
            } else if (store.micState === 'preparing' || store.micState === 'listening') {
                stopPushToTalk();
            }
            // Ignore clicks during 'processing' so we don't fire multiple requests.
        });
    }

    function setMicState(s) {
        store.micState = s;
        if (!dom.micOrb) return;
        dom.micOrb.classList.remove('preparing', 'listening', 'processing', 'completed');
        if (dom.waveBars) dom.waveBars.classList.remove('active');
        if (s === 'preparing') {
            dom.micOrb.classList.add('preparing');
        } else if (s === 'listening') {
            dom.micOrb.classList.add('listening');
            if (dom.waveBars) dom.waveBars.classList.add('active');
            dom.micLabel.textContent = 'Listening now…';
        } else if (s === 'processing') {
            dom.micOrb.classList.add('processing');
            dom.micLabel.textContent = 'Processing question…';
        } else if (s === 'completed') {
            dom.micOrb.classList.add('completed');
        }
    }

    // ══════════════════════════════════════
    // EVENT FEED RENDERING (real backend events only)
    // ══════════════════════════════════════
    function renderLogCard(data, container) {
        const empty = $('.panel-empty', container);
        if (empty) empty.remove();

        let iconClass, emoji, label;
        switch (data.event) {
            case 'attendance':              iconClass='attendance'; emoji='🟢'; label='Attendance'; break;
            case 'question':                iconClass='question';   emoji='🔵'; label='Question';   break;
            case 'guest':                   iconClass='guest';      emoji='🟠'; label='Guest';      break;
            case 'registration_approved':   iconClass='attendance'; emoji='✅'; label='Approved';   break;
            case 'registration_rejected':   iconClass='guest';      emoji='❌'; label='Rejected';   break;
            default:                        iconClass='system';     emoji='⚪'; label='System';
        }

        let detail = '';
        if (data.event === 'attendance') {
            detail = `${data.attendance || 'Present'} · ${emotionStr(data.emotion)}`;
        } else if (data.event === 'question') {
            detail = `<em>"${data.question}"</em><br>Topic: ${data.topic}`;
        } else if (data.event === 'guest') {
            detail = `Topic: ${data.topic || 'General'}`;
        } else if (data.event === 'registration_approved') {
            detail = 'Student added to recognition database';
        } else if (data.event === 'registration_rejected') {
            detail = 'Registration declined';
        } else {
            detail = data.message || '';
        }

        const time = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : (data.time || timeNow());
        const card = document.createElement('div');
        card.className = 'log-card';
        card.innerHTML = `
            <div class="log-icon ${iconClass}">${emoji}</div>
            <div class="log-body">
                <span class="log-event-type ${iconClass}">${label}</span>
                <span class="log-student">${data.student || 'System'}</span>
                <span class="log-detail">${detail}</span>
                <span class="log-time">${time}</span>
            </div>`;
        container.prepend(card);
        const cards = container.children;
        while (cards.length > MAX_LOG_CARDS) cards[cards.length - 1].remove();
    }

    function addQuestion(data) {
        const empty = $('.panel-empty', dom.questionFeed);
        if (empty) empty.remove();
        const card = document.createElement('div');
        card.className = 'q-card';
        const student = (data.student || 'Guest').replace(/_/g, ' ');
        card.innerHTML = `
            <div class="q-card-student">${student}</div>
            <div class="q-card-text">"${data.question}"</div>
            <div class="q-card-topic">${data.topic}</div>`;
        dom.questionFeed.prepend(card);
    }

    function isTrackable(name) {
        // Render summaries for registered students AND guests
        // (Guest_001, Guest_002, …) so the dashboard mirrors what
        // landed in the log.  "Unknown" / empty names stay filtered.
        return Boolean(name) && name !== 'Guest' && name !== 'Unknown';
    }

    function upsertStudent(name, attendance, emotion, isGuest = false) {
        if (!isTrackable(name)) return;
        if (!store.students[name]) {
            store.students[name] = {
                attendance,
                emotion,
                questions: [],
                isGuest: !!isGuest || name.startsWith('Guest_'),
            };
        } else {
            store.students[name].attendance = attendance;
            store.students[name].emotion = emotion;
        }
        renderSummaries();
    }
    function upsertStudentQuestion(student, question, topic) {
        if (!isTrackable(student)) return;
        if (!store.students[student]) {
            store.students[student] = {
                attendance: student.startsWith('Guest_') ? 'Guest' : 'Present',
                emotion: 'Neutral',
                questions: [],
                isGuest: student.startsWith('Guest_'),
            };
        }
        store.students[student].questions.push({ question, topic });
        renderSummaries();
    }
    function renderSummaries() {
        const names = Object.keys(store.students);
        if (names.length === 0) {
            dom.summaryGrid.innerHTML = '<div class="summary-empty">No students detected yet.</div>';
            return;
        }
        const frag = document.createDocumentFragment();
        names.forEach(name => {
            const s = store.students[name];
            const card = document.createElement('div');
            card.className = 'summary-card';
            let qHTML = '';
            if (s.questions.length > 0) {
                qHTML = `<div class="summary-questions">
                    <div class="sq-title">Questions (${s.questions.length})</div>
                    ${s.questions.map(q => `
                        <div class="sq-item">
                            <span class="sq-item-text">"${q.question}"</span>
                            <span class="sq-item-topic">${q.topic}</span>
                        </div>`).join('')}
                </div>`;
            }
            card.innerHTML = `
                <div class="summary-card-header">
                    <div class="summary-avatar">${getInitials(name)}</div>
                    <div class="summary-info">
                        <span class="summary-name">${name.replace(/_/g, ' ')}</span>
                        <div class="summary-meta">
                            <span class="summary-attendance">${s.attendance}</span>
                            <span class="summary-emotion">${emotionStr(s.emotion)}</span>
                        </div>
                    </div>
                </div>
                ${qHTML}`;
            frag.appendChild(card);
        });
        dom.summaryGrid.replaceChildren(frag);
    }

    function resetDashboardState() {
        store.students = {};
        if (dom.logFeed) {
            dom.logFeed.innerHTML = `
                <div class="panel-empty">
                    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span>Waiting for live events…</span>
                </div>`;
        }
        if (dom.questionFeed) {
            dom.questionFeed.innerHTML = `
                <div class="panel-empty">
                    <svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
                    <span>No questions yet.</span>
                </div>`;
        }
        renderSummaries();
    }

    // ══════════════════════════════════════
    // EVENTS POLL — only NEW events; first connect skips history
    // ══════════════════════════════════════
    let pollBusy = false;
    async function pollEventsOnce() {
        if (pollBusy || !store.backendOnline || store.firstEventConnect) return;
        pollBusy = true;
        try {
            const cursorNow = store.eventCursor;
            const data = await api('GET', `${API}/logs/events?since=${cursorNow}`);
            const events = data.events || [];
            if (events.length && store.eventCursor === cursorNow) {
                events.forEach(processBackendEvent);
                store.eventCursor = cursorNow + events.length;
            }
        } catch (_) { /* silent */ }
        finally { pollBusy = false; }
    }
    // Self-pacing event loop.  Same pattern as the recognize loop:
    // schedule the next poll AFTER the current one returns, never
    // overlap.  recognize-frame triggers an immediate one-shot poll
    // on `newly_marked`, so this is just the idle refresh tick — the
    // dashboard already updates instantly when attendance is marked.
    function startEventLoop() {
        clearTimeout(store.eventTimer);
        const tick = async () => {
            await pollEventsOnce();
            store.eventTimer = setTimeout(tick, EVENT_POLL_MS);
        };
        tick();
    }

    function processBackendEvent(evt) {
        if (!evt || !evt.event) return;
        renderLogCard(evt, dom.logFeed);
        if (evt.event === 'attendance') {
            const isGuest = (evt.student || '').startsWith('Guest_');
            if (evt.registered) {
                upsertStudent(evt.student, evt.attendance, evt.emotion || 'Neutral');
            } else if (isGuest) {
                upsertStudent(evt.student, evt.attendance || 'Guest', evt.emotion || 'Neutral', true);
            }
        } else if (evt.event === 'question') {
            addQuestion(evt);
            if (isTrackable(evt.student)) {
                upsertStudentQuestion(evt.student, evt.question, evt.topic);
            }
            if (store.cameraActive) showQuestionOverlay(evt.question, evt.topic);
        }
    }

    // ══════════════════════════════════════
    // DOWNLOAD CSV
    // ══════════════════════════════════════
    if (dom.downloadCsvBtn) {
        dom.downloadCsvBtn.addEventListener('click', () => {
            if (!store.backendOnline) {
                toast({ kind: 'error', title: 'Backend offline', message: 'Cannot download CSV.' });
                return;
            }
            window.location.href = `${API}/logs/attendance-csv`;
        });
    }

    // ══════════════════════════════════════
    // GUEST FLOW — backend allocates a Guest_NNN identity so attendance
    // and follow-up questions land in the unified log with proper
    // attribution.
    // ══════════════════════════════════════
    if (dom.guestBtn) {
        dom.guestBtn.addEventListener('click', async () => {
            if (!store.backendOnline) {
                toast({ kind: 'error', title: 'Backend offline', message: 'Cannot start guest session.' });
                return;
            }
            try {
                const data = await api('POST', '/interaction/guest-session');
                store.guestMode = true;
                store.guestId = data.student;
                store.currentStudent = {
                    name: data.student,
                    attendance: 'Guest',
                    emotion: 'Neutral',
                    registered: false,
                    isGuest: true,
                };
                if (dom.actionBtns) dom.actionBtns.style.display = 'none';
                if (dom.idAvatar) dom.idAvatar.textContent = 'G';
                if (dom.idName) dom.idName.textContent = data.student.replace(/_/g, ' ');
                if (dom.idAttendance) {
                    dom.idAttendance.textContent = 'Guest';
                    dom.idAttendance.classList.add('absent');
                }
                toast({
                    kind: 'info',
                    title: `Continuing as ${data.student.replace(/_/g, ' ')}`,
                    message: 'Questions are logged with this guest identity.',
                });
                // Pull the freshly-logged guest attendance event so the
                // live feed and summary update immediately.
                pollEventsOnce();
            } catch (err) {
                console.error('guest session failed:', err);
                toast({ kind: 'error', title: 'Could not start guest session', message: err.message || 'Backend error' });
            }
        });
    }

    // ══════════════════════════════════════
    // REGISTRATION FLOW — capture + name in modal, admin INLINE
    // ══════════════════════════════════════
    if (dom.registerBtn) dom.registerBtn.addEventListener('click', openCaptureModal);
    if (dom.modalClose)  dom.modalClose.addEventListener('click', cancelCaptureModal);
    if (dom.modal) dom.modal.addEventListener('click', e => { if (e.target === dom.modal) cancelCaptureModal(); });

    async function openCaptureModal() {
        if (!store.backendOnline) {
            toast({ kind: 'error', title: 'Backend offline', message: 'Registration unavailable.' });
            return;
        }
        // Stop main recognition loop while we register, but reuse the same camera stream
        // (so the modal preview can show the live feed). We'll resume after.
        store.cameraResumeAfterReg = store.cameraActive;
        if (store.cameraActive) stopRecognitionLoop();

        // Make sure we have a camera stream for the capture preview.
        if (!store.cameraStream) {
            try {
                store.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
            } catch (err) {
                toast({ kind: 'error', title: 'Camera unavailable', message: err.message || 'Permission denied' });
                store.cameraResumeAfterReg = false;
                return;
            }
        }

        store.registration = { sessionId: null, captureCount: 0, pendingName: null, inFlight: false, adminVisible: false };
        updateCaptureCount();
        showStep(0);
        if (dom.modal) dom.modal.classList.add('visible');
        if (dom.captureVideo) dom.captureVideo.srcObject = store.cameraStream;

        try {
            const info = await api('POST', '/registration/start');
            store.registration.sessionId = info.session_id;
        } catch (err) {
            toast({ kind: 'error', title: 'Could not start registration', message: err.message });
            cancelCaptureModal();
        }
    }

    async function cancelCaptureModal() {
        // Hide modal and ABORT any pending session.
        if (dom.modal) dom.modal.classList.remove('visible');
        if (dom.captureVideo) dom.captureVideo.srcObject = null;
        const sid = store.registration.sessionId;
        if (sid && !store.registration.inFlight) {
            store.registration.inFlight = true;
            try { await api('POST', '/registration/reject', { session_id: sid, delete_files: true }); } catch (_) {}
            store.registration.sessionId = null;
            store.registration.inFlight = false;
        }
        // If we were running recognition before opening, resume it.
        if (store.cameraResumeAfterReg && store.cameraActive) {
            startRecognitionLoop();
        }
        store.cameraResumeAfterReg = false;
    }

    function showStep(n) {
        dom.modalSteps.forEach((s, i) => s.classList.toggle('active', i === n));
        dom.stepDots.forEach((d, i) => {
            d.classList.remove('active', 'done');
            if (i < n) d.classList.add('done');
            if (i === n) d.classList.add('active');
        });
    }
    function updateCaptureCount() {
        if (dom.captureCount) {
            dom.captureCount.innerHTML = `<strong>${store.registration.captureCount}</strong> / 10 captured`;
        }
    }

    function captureModalFrame() {
        return captureFrameDataURL(dom.captureVideo);
    }

    if (dom.captureBtn) {
        dom.captureBtn.addEventListener('click', async () => {
            const sid = store.registration.sessionId;
            if (!sid) { toast({ kind: 'error', title: 'No active session' }); return; }
            const dataUrl = captureModalFrame();
            if (!dataUrl) { toast({ kind: 'error', title: 'Camera not ready' }); return; }
            try {
                const res = await api('POST', '/registration/capture', { session_id: sid, image_base64: dataUrl });
                store.registration.captureCount = res.image_count;
                updateCaptureCount();
                if (window.gsap) {
                    gsap.fromTo(dom.captureBtn, { scale: .9 }, { scale: 1, duration: .3, ease: 'back.out(2)' });
                }
                if (res.ready_for_submit && res.image_count >= 5) {
                    setTimeout(() => showStep(1), 600);
                }
            } catch (err) {
                toast({ kind: 'error', title: 'Capture failed', message: err.message });
            }
        });
    }

    if (dom.nameNextBtn) {
        dom.nameNextBtn.addEventListener('click', async () => {
            const val = dom.nameInput.value.trim();
            if (!/^[A-Za-z]+_[A-Za-z]+$/.test(val)) {
                dom.nameError.classList.add('visible');
                dom.nameInput.style.borderColor = 'rgba(248,113,113,.5)';
                return;
            }
            dom.nameError.classList.remove('visible');
            dom.nameInput.style.borderColor = '';
            const sid = store.registration.sessionId;
            if (!sid) return;
            try {
                await api('POST', '/registration/submit', { session_id: sid, name: val });
                store.registration.pendingName = val;
                // Close modal, surface the inline admin panel.
                if (dom.modal) dom.modal.classList.remove('visible');
                if (dom.captureVideo) dom.captureVideo.srcObject = null;
                openAdminPanel(val);
            } catch (err) {
                dom.nameError.textContent = '⚠ ' + err.message;
                dom.nameError.classList.add('visible');
            }
        });
    }

    if (dom.nameInput) {
        dom.nameInput.addEventListener('input', () => {
            dom.nameError.classList.remove('visible');
            dom.nameError.textContent = '⚠ Invalid format. Use Firstname_Lastname';
            dom.nameInput.style.borderColor = '';
        });
    }

    // ── Inline admin panel ──────────────────────────────────────

    function openAdminPanel(name) {
        store.registration.adminVisible = true;
        dom.adminPanel.hidden = false;
        dom.adminPanel.classList.add('visible');
        if (dom.adminTarget) dom.adminTarget.textContent = (name || '').replace(/_/g, ' ') || 'Pending student';
        if (dom.adminPass) {
            dom.adminPass.value = '';
            dom.adminPass.classList.remove('invalid');
            dom.adminPass.focus();
        }
        if (dom.adminHint) dom.adminHint.textContent = 'Type the admin password to enable approval.';
        setAdminButtons(false);
    }

    function closeAdminPanel() {
        store.registration.adminVisible = false;
        if (dom.adminPanel) {
            dom.adminPanel.classList.remove('visible');
            dom.adminPanel.hidden = true;
        }
        if (dom.adminPass) dom.adminPass.value = '';
        setAdminButtons(false);
        // Resume the live recognition loop if it was running before.
        if (store.cameraResumeAfterReg && store.cameraActive) {
            startRecognitionLoop();
        }
        store.cameraResumeAfterReg = false;
    }

    function setAdminButtons(enabled) {
        if (dom.adminApprove) dom.adminApprove.disabled = !enabled;
        if (dom.adminReject)  dom.adminReject.disabled  = !enabled;
    }

    if (dom.adminPass) {
        dom.adminPass.addEventListener('input', () => {
            const v = dom.adminPass.value || '';
            const ok = v === ADMIN_PASSWORD;
            setAdminButtons(ok);
            dom.adminPass.classList.toggle('invalid', v.length > 0 && !ok);
            if (dom.adminHint) {
                dom.adminHint.textContent = ok
                    ? 'Password accepted — Approve or Reject below.'
                    : 'Type the admin password to enable approval.';
            }
        });
    }

    async function submitAdminDecision(approved) {
        if (store.registration.inFlight) return;
        const sid = store.registration.sessionId;
        if (!sid) return;
        const code = (dom.adminPass && dom.adminPass.value || '').trim();
        if (!code) {
            dom.adminPass.classList.add('invalid');
            return;
        }
        store.registration.inFlight = true;
        setAdminButtons(false);
        try {
            const path = approved ? '/registration/approve' : '/registration/reject';
            const body = approved
                ? { session_id: sid, codeword: code }
                : { session_id: sid, delete_files: true };
            // Reject doesn't require codeword on the backend, but we still
            // gate the UI button on the password being correct so an
            // unauthorised user can't reject either.
            const res = await api('POST', path, body);
            store.registration.sessionId = null;
            const studentName = res.student || store.registration.pendingName || '';
            if (approved) {
                toast({
                    kind: 'success',
                    title: 'Student added successfully',
                    message: studentName ? studentName.replace(/_/g, ' ') : '',
                });
            } else {
                toast({
                    kind: 'error',
                    title: 'Registration rejected',
                    message: studentName ? studentName.replace(/_/g, ' ') : '',
                });
            }
            closeAdminPanel();
        } catch (err) {
            if (err.status === 401) {
                dom.adminPass.classList.add('invalid');
                toast({ kind: 'error', title: 'Invalid admin password', message: 'Backend rejected the codeword.' });
                setAdminButtons(false);
            } else {
                toast({ kind: 'error', title: approved ? 'Approval failed' : 'Reject failed',
                        message: err.message || 'Backend error' });
                setAdminButtons(true);
            }
        } finally {
            store.registration.inFlight = false;
        }
    }

    if (dom.adminApprove) dom.adminApprove.addEventListener('click', () => submitAdminDecision(true));
    if (dom.adminReject)  dom.adminReject.addEventListener('click', () => submitAdminDecision(false));

    document.addEventListener('keydown', e => {
        if (e.key !== 'Escape') return;
        if (store.registration.adminVisible) {
            // Treat ESC during admin approval as a cancel — but only
            // when no decision is in-flight.
            if (!store.registration.inFlight) submitAdminDecision(false);
        } else if (dom.modal?.classList.contains('visible')) {
            cancelCaptureModal();
        }
    });

    // ══════════════════════════════════════
    // Public hooks (kept for compatibility with prior callers)
    // ══════════════════════════════════════
    window.dashShowIdentity = function (name, attendance, emotion) {
        showIdentity(name, attendance, emotion, !!name && name !== 'Unknown', true, 5);
        if (name && name !== 'Unknown') upsertStudent(name, attendance, emotion);
    };
    window.dashAddLog = renderLogCard.bind(null);
    window.dashProcessEvent = processBackendEvent;
    window.dashSetStatus = setStatus;
    window.dashToast = toast;

    // ══════════════════════════════════════
    // BOOT
    // ══════════════════════════════════════
    function boot() {
        resetDashboardState();
        startHealthLoop();
        startEventLoop();
    }
    if (document.readyState !== 'loading') boot();
    else document.addEventListener('DOMContentLoaded', boot);

    // ══════════════════════════════════════
    // CLEANUP — clear transient state when the tab closes / reloads
    // ══════════════════════════════════════
    window.addEventListener('beforeunload', () => {
        try {
            const PURGE_PREFIX = 'smart_classroom_';
            [localStorage, sessionStorage].forEach(s => {
                const drop = [];
                for (let i = 0; i < s.length; i++) {
                    const k = s.key(i);
                    if (k && k.startsWith(PURGE_PREFIX)) drop.push(k);
                }
                drop.forEach(k => s.removeItem(k));
            });
        } catch (_) {}
        if (store.cameraStream) {
            try { store.cameraStream.getTracks().forEach(t => t.stop()); } catch (_) {}
        }
    });

})();
