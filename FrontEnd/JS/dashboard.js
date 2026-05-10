// ============================================
// LIVE DASHBOARD — SMART CLASSROOM
// Premium interactive dashboard controller
// ============================================

(function () {
    'use strict';

    // ── State ──
    const state = {
        cameraActive: false,
        cameraStream: null,
        micActive: false,
        recognized: false,
        studentName: null,
        attendance: null,
        emotion: null,
        logs: [],
        registrationStep: 0,
        captureCount: 0,
    };

    // ── DOM Cache ──
    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    // Dashboard elements
    const dashSection = $('#demo');
    const cameraWrapper = $('.camera-feed-wrapper');
    const cameraVideo = $('#dash-camera-video');
    const camPlaceholder = $('.cam-placeholder');
    const cameraBtn = $('#dash-camera-btn');
    const cameraBadge = $('#camera-badge');
    const identityCard = $('.identity-card');
    const idAvatar = $('.id-avatar');
    const idName = $('.id-name');
    const idAttendance = $('.id-attendance');
    const idEmotion = $('.id-emotion');
    const micOrb = $('#dash-mic-orb');
    const micLabel = $('#dash-mic-label');
    const waveBars = $('.wave-bars');
    const registerBtn = $('#dash-register-btn');
    const guestBtn = $('#dash-guest-btn');
    const actionBtns = $('.action-buttons');
    const logFeed = $('.log-feed');
    const modalOverlay = $('#registration-modal');
    const statusDot = $('.status-dot');
    const statusLabel = $('.status-label');

    // Modal elements
    const modalSteps = $$('.modal-step');
    const stepDots = $$('.step-dot');
    const capturePreviewVideo = $('#capture-preview-video');
    const captureCountEl = $('#capture-count');
    const captureBtn = $('#modal-capture-btn');
    const nameInput = $('#reg-name-input');
    const nameError = $('#name-error');
    const nameNextBtn = $('#name-next-btn');
    const adminInput = $('#admin-code-input');
    const approveBtn = $('#modal-approve-btn');
    const cancelRegBtn = $('#modal-cancel-btn');
    const modalClose = $('.modal-close');
    const regResult = $('#reg-result');
    const regResultIcon = $('#reg-result-icon');
    const regResultTitle = $('#reg-result-title');
    const regResultMsg = $('#reg-result-msg');
    const regDoneBtn = $('#reg-done-btn');

    // ── Utilities ──
    function timeNow() {
        return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function getInitials(name) {
        if (!name) return '?';
        return name.split('_').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    }

    const emotionEmojis = {
        happy: '😊', neutral: '😐', tired: '😴', sad: '😢',
        angry: '😠', surprised: '😲', fearful: '😨', disgusted: '🤢'
    };

    function getEmotionDisplay(emotion) {
        const e = (emotion || '').toLowerCase();
        const emoji = emotionEmojis[e] || '😐';
        const label = emotion ? emotion.charAt(0).toUpperCase() + emotion.slice(1) : 'Neutral';
        return `${emoji} ${label}`;
    }

    // ── Launch Logic ──
    const launchBtns = $$('.launch-demo-btn');
    let dashboardLaunched = false;

    launchBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            if (dashboardLaunched) {
                dashSection.scrollIntoView({ behavior: 'smooth' });
                return;
            }
            dashboardLaunched = true;
            dashSection.classList.add('dashboard-active');

            // Animate in
            gsap.from(dashSection, {
                opacity: 0, y: 50, duration: .9, ease: 'power3.out',
                onComplete: () => { dashSection.style.pointerEvents = 'auto'; }
            });

            // Stagger child cards
            gsap.from('.dash-card', {
                y: 40, opacity: 0, duration: .7, stagger: .15, ease: 'power2.out', delay: .3
            });

            setTimeout(() => dashSection.scrollIntoView({ behavior: 'smooth' }), 120);

            // Trigger 3D transition if available
            setTimeout(() => {
                if (window.startModelTransition) window.startModelTransition();
            }, 400);
        });
    });

    // ── Camera ──
    if (cameraBtn) {
        cameraBtn.addEventListener('click', async () => {
            if (!state.cameraActive) {
                try {
                    state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    cameraVideo.srcObject = state.cameraStream;
                    cameraVideo.style.display = 'block';
                    cameraWrapper.classList.add('active');
                    cameraBtn.textContent = 'Stop Camera';
                    cameraBtn.classList.add('camera-toggle-active');
                    cameraBadge.textContent = 'LIVE';
                    cameraBadge.className = 'dash-card-badge recording';
                    statusDot.classList.remove('offline');
                    statusLabel.textContent = 'System Online — Camera Active';
                    state.cameraActive = true;

                    addLog({ event: 'system', message: 'Camera activated' });

                    // Simulate recognition after delay (mock)
                    setTimeout(() => simulateRecognition(), 2500);
                } catch (err) {
                    console.error('Camera error:', err);
                    addLog({ event: 'system', message: 'Camera access denied' });
                }
            } else {
                stopCamera();
            }
        });
    }

    function stopCamera() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(t => t.stop());
            state.cameraStream = null;
        }
        cameraVideo.srcObject = null;
        cameraVideo.style.display = 'none';
        cameraWrapper.classList.remove('active');
        cameraBtn.textContent = 'Start Camera';
        cameraBtn.classList.remove('camera-toggle-active');
        cameraBadge.textContent = 'STANDBY';
        cameraBadge.className = 'dash-card-badge';
        identityCard.classList.remove('visible');
        statusDot.classList.add('offline');
        statusLabel.textContent = 'System Standby';
        state.cameraActive = false;
        state.recognized = false;
    }

    // ── Face Recognition (mock / backend hook) ──
    function simulateRecognition() {
        // This is the integration hook.
        // Replace with: fetch('/api/recognize').then(...)
        // For demo, simulate a known student
        const mockKnown = true;
        if (mockKnown) {
            showIdentity('Mohammed_Ayman', 'Present', 'Happy');
            addLog({
                event: 'attendance',
                student: 'Mohammed_Ayman',
                attendance: 'Present',
                emotion: 'Happy'
            });
        } else {
            showIdentity(null, null, null); // Unknown
        }
    }

    // ── Backend Integration Hooks ──
    // Call these from your real backend websocket/polling

    /** Show recognized student identity overlay */
    window.dashShowIdentity = function (name, attendance, emotion) {
        showIdentity(name, attendance, emotion);
    };

    /** Add a log event */
    window.dashAddLog = function (logData) {
        addLog(logData);
    };

    /** Update system status */
    window.dashSetStatus = function (online, message) {
        if (online) { statusDot.classList.remove('offline'); }
        else { statusDot.classList.add('offline'); }
        statusLabel.textContent = message || (online ? 'System Online' : 'System Offline');
    };

    // ── Identity Card ──
    function showIdentity(name, attendance, emotion) {
        state.studentName = name;
        state.attendance = attendance;
        state.emotion = emotion;
        state.recognized = !!name;

        if (name) {
            idAvatar.textContent = getInitials(name);
            idName.textContent = name.replace(/_/g, ' ');
            idAttendance.textContent = attendance || 'Present';
            idAttendance.classList.toggle('absent', attendance === 'Absent');
            idEmotion.textContent = getEmotionDisplay(emotion);
            actionBtns.style.display = 'none';
        } else {
            idAvatar.textContent = '?';
            idName.textContent = 'Unknown Visitor';
            idAttendance.textContent = 'Unregistered';
            idAttendance.classList.add('absent');
            idEmotion.textContent = getEmotionDisplay(emotion);
            actionBtns.style.display = 'flex';
        }

        identityCard.classList.add('visible');

        gsap.fromTo(identityCard,
            { opacity: 0, y: 12 },
            { opacity: 1, y: 0, duration: .45, ease: 'power2.out' }
        );
    }

    // ── Microphone ──
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
            state.micActive = true;
            micOrb.classList.add('recording');
            micLabel.textContent = 'Listening...';
            waveBars.classList.add('active');
        };

        recognition.onresult = (ev) => {
            const transcript = ev.results[0][0].transcript;
            micLabel.textContent = `"${transcript}"`;

            // Mock topic classification — replace with backend call
            const topic = classifyTopic(transcript);

            addLog({
                event: 'question',
                student: state.studentName || 'Guest',
                question: transcript,
                topic: topic
            });
        };

        recognition.onerror = (ev) => {
            console.error('Speech error:', ev.error);
            micLabel.textContent = 'Error — try again';
            stopMic();
        };

        recognition.onend = () => { stopMic(); };
    }

    if (micOrb) {
        micOrb.addEventListener('click', () => {
            if (!recognition) { micLabel.textContent = 'Not supported'; return; }
            if (!state.micActive) {
                try { recognition.start(); } catch (e) { console.warn(e); }
            } else {
                recognition.stop();
                stopMic();
            }
        });
    }

    function stopMic() {
        state.micActive = false;
        micOrb.classList.remove('recording');
        waveBars.classList.remove('active');
        if (micLabel.textContent === 'Listening...') {
            micLabel.textContent = 'Ask a Question';
        }
    }

    function classifyTopic(text) {
        // Integration hook: replace with fetch('/api/classify', { body: text })
        const keywords = {
            'TCP': 'Computer Networks', 'UDP': 'Computer Networks', 'IP': 'Computer Networks',
            'neural': 'Deep Learning', 'CNN': 'Deep Learning', 'layer': 'Deep Learning',
            'SQL': 'Databases', 'database': 'Databases', 'query': 'Databases',
            'sort': 'Algorithms', 'algorithm': 'Algorithms', 'complexity': 'Algorithms',
            'OS': 'Operating Systems', 'thread': 'Operating Systems', 'process': 'Operating Systems',
        };
        for (const [kw, topic] of Object.entries(keywords)) {
            if (text.toLowerCase().includes(kw.toLowerCase())) return topic;
        }
        return 'General';
    }

    // ── Log System ──
    function addLog(data) {
        data.time = timeNow();
        state.logs.unshift(data);
        renderLog(data);
    }

    function renderLog(data) {
        // Remove empty state
        const empty = $('.log-empty', logFeed);
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'log-card';

        let iconClass, iconEmoji, typeLabel;
        switch (data.event) {
            case 'attendance':
                iconClass = 'attendance'; iconEmoji = '🟢'; typeLabel = 'Attendance';
                break;
            case 'question':
                iconClass = 'question'; iconEmoji = '🔵'; typeLabel = 'Question';
                break;
            case 'guest':
                iconClass = 'guest'; iconEmoji = '🟠'; typeLabel = 'Guest Interaction';
                break;
            default:
                iconClass = 'system'; iconEmoji = '⚪'; typeLabel = 'System';
        }

        let detailHTML = '';
        if (data.event === 'attendance') {
            detailHTML = `${data.attendance} · ${getEmotionDisplay(data.emotion)}`;
        } else if (data.event === 'question') {
            detailHTML = `<em>"${data.question}"</em><br>Topic: ${data.topic}`;
        } else if (data.event === 'guest') {
            detailHTML = `Topic: ${data.topic || 'General'}`;
        } else {
            detailHTML = data.message || '';
        }

        card.innerHTML = `
            <div class="log-icon ${iconClass}">${iconEmoji}</div>
            <div class="log-body">
                <span class="log-event-type ${iconClass}">${typeLabel}</span>
                <span class="log-student">${data.student || 'System'}</span>
                <span class="log-detail">${detailHTML}</span>
                <span class="log-time">${data.time}</span>
            </div>
        `;

        logFeed.prepend(card);

        // Keep max 50 cards
        const cards = $$('.log-card', logFeed);
        if (cards.length > 50) cards[cards.length - 1].remove();
    }

    // ── Guest Button ──
    if (guestBtn) {
        guestBtn.addEventListener('click', () => {
            actionBtns.style.display = 'none';
            idName.textContent = 'Guest Student';
            idAttendance.textContent = 'Guest';
            state.studentName = 'Guest';
            addLog({ event: 'guest', student: 'Guest', topic: 'General' });
        });
    }

    // ── Registration Modal ──
    if (registerBtn) {
        registerBtn.addEventListener('click', () => {
            openModal();
        });
    }

    if (modalClose) {
        modalClose.addEventListener('click', closeModal);
    }

    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) closeModal();
        });
    }

    function openModal() {
        state.registrationStep = 0;
        state.captureCount = 0;
        updateCaptureCount();
        showStep(0);
        modalOverlay.classList.add('visible');

        // Mirror camera to capture preview
        if (state.cameraStream && capturePreviewVideo) {
            capturePreviewVideo.srcObject = state.cameraStream;
        }
    }

    function closeModal() {
        modalOverlay.classList.remove('visible');
        if (capturePreviewVideo) capturePreviewVideo.srcObject = null;
    }

    function showStep(n) {
        modalSteps.forEach((s, i) => {
            s.classList.toggle('active', i === n);
        });
        stepDots.forEach((d, i) => {
            d.classList.remove('active', 'done');
            if (i < n) d.classList.add('done');
            if (i === n) d.classList.add('active');
        });
    }

    // Step 1: Capture
    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            if (state.captureCount >= 10) return;
            state.captureCount++;
            updateCaptureCount();

            // Flash effect
            gsap.fromTo(captureBtn, { scale: .9 }, { scale: 1, duration: .3, ease: 'back.out(2)' });

            if (state.captureCount >= 5) {
                // Allow proceeding to step 2
                setTimeout(() => showStep(1), 600);
            }
        });
    }

    function updateCaptureCount() {
        if (captureCountEl) {
            captureCountEl.innerHTML = `<strong>${state.captureCount}</strong> / 10 captured`;
        }
    }

    // Step 2: Name entry
    if (nameNextBtn) {
        nameNextBtn.addEventListener('click', () => {
            const val = nameInput.value.trim();
            if (!/^[A-Za-z]+_[A-Za-z]+$/.test(val)) {
                nameError.classList.add('visible');
                nameInput.style.borderColor = 'rgba(248,113,113,.5)';
                return;
            }
            nameError.classList.remove('visible');
            nameInput.style.borderColor = '';
            state.pendingName = val;
            showStep(2);
        });
    }

    if (nameInput) {
        nameInput.addEventListener('input', () => {
            nameError.classList.remove('visible');
            nameInput.style.borderColor = '';
        });
    }

    // Step 3: Admin approval
    if (approveBtn) {
        approveBtn.addEventListener('click', () => {
            const code = adminInput ? adminInput.value.trim() : '';
            // Integration hook: send to backend for validation
            // For demo, accept any non-empty code
            if (!code) {
                adminInput.style.borderColor = 'rgba(248,113,113,.5)';
                return;
            }

            // Simulate approval
            showRegistrationResult(true, state.pendingName);
        });
    }

    if (cancelRegBtn) {
        cancelRegBtn.addEventListener('click', () => {
            showRegistrationResult(false, null);
        });
    }

    function showRegistrationResult(approved, name) {
        showStep(3);
        if (approved) {
            regResultIcon.textContent = '✅';
            regResultTitle.textContent = 'Registration Approved';
            regResultMsg.textContent = `Welcome ${(name || '').replace(/_/g, ' ')}`;
            // Update identity
            showIdentity(name, 'Present', state.emotion);
            addLog({ event: 'attendance', student: name, attendance: 'Present', emotion: state.emotion || 'Neutral' });
        } else {
            regResultIcon.textContent = '❌';
            regResultTitle.textContent = 'Registration Rejected';
            regResultMsg.textContent = 'Admin did not approve this registration.';
        }
    }

    if (regDoneBtn) {
        regDoneBtn.addEventListener('click', closeModal);
    }

    // ── Keyboard shortcut: ESC closes modal ──
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modalOverlay && modalOverlay.classList.contains('visible')) {
            closeModal();
        }
    });

})();
