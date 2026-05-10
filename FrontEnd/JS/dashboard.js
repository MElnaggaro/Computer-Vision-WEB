// ============================================
// LIVE DASHBOARD v2 — SMART CLASSROOM
// State-driven multi-panel dashboard controller
// ============================================

(function () {
    'use strict';

    // ══════════════════════════════════════
    // STATE STORE
    // ══════════════════════════════════════
    const store = {
        backendOnline: false,
        cameraActive: false,
        cameraStream: null,
        micState: 'idle', // idle | listening | processing | completed
        currentStudent: null, // { name, attendance, emotion }
        logs: [],
        questions: [],
        students: {}, // { name: { attendance, emotion, questions:[] } }
        registrationStep: 0,
        captureCount: 0,
        pendingName: null,
    };

    // ══════════════════════════════════════
    // DOM HELPERS
    // ══════════════════════════════════════
    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    // ══════════════════════════════════════
    // DOM CACHE
    // ══════════════════════════════════════
    const dom = {
        section:       $('#demo'),
        statusDot:     $('.status-dot'),
        statusLabel:   $('.status-label'),
        cameraWrapper: $('.camera-feed-wrapper'),
        cameraVideo:   $('#dash-camera-video'),
        cameraBtn:     $('#dash-camera-btn'),
        cameraBadge:   $('#camera-badge'),
        offlineMsg:    $('.offline-msg'),
        identityCard:  $('.identity-card'),
        idAvatar:      $('.id-avatar'),
        idName:        $('.id-name'),
        idAttendance:  $('.id-attendance'),
        idEmotion:     $('.id-emotion'),
        questionOvr:   $('.question-overlay'),
        qoText:        $('.qo-text'),
        qoTopic:       $('.qo-topic'),
        micOrb:        $('#dash-mic-orb'),
        micLabel:      $('#dash-mic-label'),
        waveBars:      $('.wave-bars'),
        actionBtns:    $('.action-buttons'),
        registerBtn:   $('#dash-register-btn'),
        guestBtn:      $('#dash-guest-btn'),
        logFeed:       $('#log-feed'),
        questionFeed:  $('#question-feed'),
        summaryGrid:   $('#summary-grid'),
        modal:         $('#registration-modal'),
        modalClose:    $('.modal-close'),
        modalSteps:    $$('.modal-step'),
        stepDots:      $$('.step-dot'),
        captureVideo:  $('#capture-preview-video'),
        captureCount:  $('#capture-count'),
        captureBtn:    $('#modal-capture-btn'),
        nameInput:     $('#reg-name-input'),
        nameError:     $('#name-error'),
        nameNextBtn:   $('#name-next-btn'),
        adminInput:    $('#admin-code-input'),
        approveBtn:    $('#modal-approve-btn'),
        cancelBtn:     $('#modal-cancel-btn'),
        regResultIcon: $('#reg-result-icon'),
        regResultTitle:$('#reg-result-title'),
        regResultMsg:  $('#reg-result-msg'),
        regDoneBtn:    $('#reg-done-btn'),
    };

    // ══════════════════════════════════════
    // UTILITIES
    // ══════════════════════════════════════
    function timeNow() {
        return new Date().toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    }

    function getInitials(name) {
        if (!name || name === 'Guest') return '?';
        return name.split('_').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    }

    const EMOJIS = {
        happy:'😊', neutral:'😐', tired:'😴', sad:'😢',
        angry:'😠', surprised:'😲', fearful:'😨', disgusted:'🤢'
    };

    function emotionStr(e) {
        const k = (e || '').toLowerCase();
        return `${EMOJIS[k] || '😐'} ${e ? e.charAt(0).toUpperCase() + e.slice(1) : 'Neutral'}`;
    }

    function classifyTopic(text) {
        const map = {
            'tcp':'Computer Networks','udp':'Computer Networks','ip ':'Computer Networks',
            'handshake':'Computer Networks','router':'Computer Networks','subnet':'Computer Networks',
            'neural':'Deep Learning','cnn':'Deep Learning','layer':'Deep Learning',
            'sql':'Databases','database':'Databases','query':'Databases',
            'sort':'Algorithms','algorithm':'Algorithms','complexity':'Algorithms',
            'semaphore':'Operating Systems','thread':'Operating Systems','process':'Operating Systems',
            'deadlock':'Operating Systems','stack':'Programming and Data Structures',
            'queue':'Programming and Data Structures','linked list':'Programming and Data Structures',
        };
        const lower = text.toLowerCase();
        for (const [kw, topic] of Object.entries(map)) {
            if (lower.includes(kw)) return topic;
        }
        return 'General';
    }

    // ══════════════════════════════════════
    // STATUS
    // ══════════════════════════════════════
    function setStatus(online, msg) {
        store.backendOnline = online;
        dom.statusDot.classList.toggle('offline', !online);
        dom.statusLabel.textContent = msg || (online ? 'System Online' : 'System Offline');
        if (dom.offlineMsg) {
            dom.offlineMsg.classList.toggle('visible', !online && !store.cameraActive);
        }
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
            gsap.from(dom.section, { opacity: 0, y: 50, duration: .9, ease: 'power3.out' });
            gsap.from('.dash-card', { y: 40, opacity: 0, duration: .7, stagger: .12, ease: 'power2.out', delay: .25 });
            setTimeout(() => dom.section.scrollIntoView({ behavior: 'smooth' }), 120);
            setTimeout(() => { if (window.startModelTransition) window.startModelTransition(); }, 400);
        });
    });

    // ══════════════════════════════════════
    // CAMERA
    // ══════════════════════════════════════
    if (dom.cameraBtn) {
        dom.cameraBtn.addEventListener('click', async () => {
            if (!store.cameraActive) {
                try {
                    store.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    dom.cameraVideo.srcObject = store.cameraStream;
                    dom.cameraVideo.style.display = 'block';
                    dom.cameraWrapper.classList.add('active');
                    dom.cameraBtn.textContent = 'Stop Camera';
                    dom.cameraBtn.classList.add('camera-toggle-active');
                    dom.cameraBadge.textContent = 'LIVE';
                    dom.cameraBadge.className = 'dash-card-badge recording';
                    store.cameraActive = true;
                    setStatus(true, 'Camera Active — Scanning');
                    addLog({ event: 'system', message: 'Camera activated' });
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
        if (store.cameraStream) {
            store.cameraStream.getTracks().forEach(t => t.stop());
            store.cameraStream = null;
        }
        dom.cameraVideo.srcObject = null;
        dom.cameraVideo.style.display = 'none';
        dom.cameraWrapper.classList.remove('active');
        dom.cameraBtn.textContent = 'Start Camera';
        dom.cameraBtn.classList.remove('camera-toggle-active');
        dom.cameraBadge.textContent = 'STANDBY';
        dom.cameraBadge.className = 'dash-card-badge';
        dom.identityCard.classList.remove('visible');
        hideQuestionOverlay();
        store.cameraActive = false;
        store.currentStudent = null;
        setStatus(false, 'System Standby');
        dom.actionBtns.style.display = 'none';
    }

    // ══════════════════════════════════════
    // IDENTITY
    // ══════════════════════════════════════
    function showIdentity(name, attendance, emotion) {
        store.currentStudent = name ? { name, attendance, emotion } : null;

        if (name) {
            dom.idAvatar.textContent = getInitials(name);
            dom.idName.textContent = name.replace(/_/g, ' ');
            dom.idAttendance.textContent = attendance || 'Present';
            dom.idAttendance.classList.toggle('absent', !attendance || attendance === 'Absent');
            dom.idEmotion.textContent = emotionStr(emotion);
            dom.actionBtns.style.display = 'none';
        } else {
            dom.idAvatar.textContent = '?';
            dom.idName.textContent = 'Unknown Visitor';
            dom.idAttendance.textContent = 'Unregistered';
            dom.idAttendance.classList.add('absent');
            dom.idEmotion.textContent = emotionStr(emotion);
            dom.actionBtns.style.display = 'flex';
        }

        dom.identityCard.classList.add('visible');
        gsap.fromTo(dom.identityCard,
            { opacity: 0, y: 12 },
            { opacity: 1, y: 0, duration: .45, ease: 'power2.out' }
        );
    }

    // ══════════════════════════════════════
    // QUESTION OVERLAY ON CAMERA
    // ══════════════════════════════════════
    function showQuestionOverlay(question, topic) {
        dom.qoText.textContent = `"${question}"`;
        dom.qoTopic.innerHTML = `Topic: <strong>${topic}</strong>`;
        dom.questionOvr.classList.add('visible');
        gsap.fromTo(dom.questionOvr,
            { opacity: 0, y: -10, scale: .95 },
            { opacity: 1, y: 0, scale: 1, duration: .4, ease: 'power2.out' }
        );
        // Auto-dismiss after 6s
        clearTimeout(showQuestionOverlay._timer);
        showQuestionOverlay._timer = setTimeout(hideQuestionOverlay, 6000);
    }

    function hideQuestionOverlay() {
        gsap.to(dom.questionOvr, {
            opacity: 0, y: -8, duration: .3, ease: 'power2.in',
            onComplete: () => dom.questionOvr.classList.remove('visible')
        });
    }

    // ══════════════════════════════════════
    // MICROPHONE
    // ══════════════════════════════════════
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;

    if (SR) {
        recognition = new SR();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => setMicState('listening');

        recognition.onresult = ev => {
            const transcript = ev.results[0][0].transcript;
            setMicState('processing');
            dom.micLabel.textContent = 'Processing...';

            // Simulate brief processing delay
            setTimeout(() => {
                const topic = classifyTopic(transcript);
                const student = store.currentStudent ? store.currentStudent.name : 'Guest';

                // Show overlay on camera
                showQuestionOverlay(transcript, topic);

                // Add to logs + questions
                addLog({ event: 'question', student, question: transcript, topic });
                addQuestion({ student, question: transcript, topic });
                upsertStudentQuestion(student, transcript, topic);

                dom.micLabel.textContent = `"${transcript}"`;
                setMicState('completed');
                setTimeout(() => { setMicState('idle'); dom.micLabel.textContent = 'Ask a Question'; }, 3000);
            }, 800);
        };

        recognition.onerror = ev => {
            console.error('Speech error:', ev.error);
            dom.micLabel.textContent = 'Error — try again';
            setMicState('idle');
        };

        recognition.onend = () => {
            if (store.micState === 'listening') setMicState('idle');
        };
    }

    if (dom.micOrb) {
        dom.micOrb.addEventListener('click', () => {
            if (!recognition) { dom.micLabel.textContent = 'Not supported'; return; }
            if (store.micState === 'idle') {
                try { recognition.start(); } catch (e) { console.warn(e); }
            } else if (store.micState === 'listening') {
                recognition.stop();
            }
        });
    }

    function setMicState(s) {
        store.micState = s;
        dom.micOrb.classList.remove('listening', 'processing', 'completed');
        dom.waveBars.classList.remove('active');

        if (s === 'listening') {
            dom.micOrb.classList.add('listening');
            dom.waveBars.classList.add('active');
            dom.micLabel.textContent = 'Listening...';
        } else if (s === 'processing') {
            dom.micOrb.classList.add('processing');
        } else if (s === 'completed') {
            dom.micOrb.classList.add('completed');
        } else {
            // idle — label reset handled by caller
        }
    }

    // ══════════════════════════════════════
    // LOG SYSTEM
    // ══════════════════════════════════════
    function addLog(data) {
        data.time = timeNow();
        store.logs.unshift(data);
        renderLogCard(data, dom.logFeed);
        if (store.logs.length > 50) store.logs.pop();
    }

    function renderLogCard(data, container) {
        const empty = $('.panel-empty', container);
        if (empty) empty.remove();

        let iconClass, emoji, label;
        switch (data.event) {
            case 'attendance': iconClass='attendance'; emoji='🟢'; label='Attendance'; break;
            case 'question':   iconClass='question';   emoji='🔵'; label='Question';   break;
            case 'guest':      iconClass='guest';      emoji='🟠'; label='Guest';      break;
            default:           iconClass='system';     emoji='⚪'; label='System';
        }

        let detail = '';
        if (data.event === 'attendance') detail = `${data.attendance} · ${emotionStr(data.emotion)}`;
        else if (data.event === 'question') detail = `<em>"${data.question}"</em><br>Topic: ${data.topic}`;
        else if (data.event === 'guest') detail = `Topic: ${data.topic || 'General'}`;
        else detail = data.message || '';

        const card = document.createElement('div');
        card.className = 'log-card';
        card.innerHTML = `
            <div class="log-icon ${iconClass}">${emoji}</div>
            <div class="log-body">
                <span class="log-event-type ${iconClass}">${label}</span>
                <span class="log-student">${data.student || 'System'}</span>
                <span class="log-detail">${detail}</span>
                <span class="log-time">${data.time}</span>
            </div>`;
        container.prepend(card);

        const cards = $$('.log-card', container);
        if (cards.length > 50) cards[cards.length - 1].remove();
    }

    // ══════════════════════════════════════
    // QUESTION HISTORY
    // ══════════════════════════════════════
    function addQuestion(data) {
        data.time = timeNow();
        store.questions.unshift(data);

        const empty = $('.panel-empty', dom.questionFeed);
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'q-card';
        card.innerHTML = `
            <div class="q-card-student">${(data.student || 'Guest').replace(/_/g, ' ')}</div>
            <div class="q-card-text">"${data.question}"</div>
            <div class="q-card-topic">${data.topic}</div>`;
        dom.questionFeed.prepend(card);
    }

    // ══════════════════════════════════════
    // STUDENT SUMMARY AGGREGATION
    // ══════════════════════════════════════
    function upsertStudent(name, attendance, emotion) {
        if (!name || name === 'Guest') return;
        if (!store.students[name]) {
            store.students[name] = { attendance, emotion, questions: [] };
        } else {
            store.students[name].attendance = attendance;
            store.students[name].emotion = emotion;
        }
        renderSummaries();
    }

    function upsertStudentQuestion(student, question, topic) {
        if (!student || student === 'Guest') return;
        if (!store.students[student]) {
            store.students[student] = { attendance: 'Present', emotion: 'Neutral', questions: [] };
        }
        store.students[student].questions.push({ question, topic });
        renderSummaries();
    }

    function renderSummaries() {
        const names = Object.keys(store.students);
        if (names.length === 0) {
            dom.summaryGrid.innerHTML = '<div class="summary-empty">No students detected yet</div>';
            return;
        }
        dom.summaryGrid.innerHTML = '';
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
            dom.summaryGrid.appendChild(card);
        });
    }

    // ══════════════════════════════════════
    // GUEST
    // ══════════════════════════════════════
    if (dom.guestBtn) {
        dom.guestBtn.addEventListener('click', () => {
            dom.actionBtns.style.display = 'none';
            dom.idName.textContent = 'Guest Student';
            dom.idAttendance.textContent = 'Guest';
            store.currentStudent = { name: 'Guest', attendance: 'Guest', emotion: 'Neutral' };
            addLog({ event: 'guest', student: 'Guest', topic: 'General' });
        });
    }

    // ══════════════════════════════════════
    // REGISTRATION MODAL
    // ══════════════════════════════════════
    if (dom.registerBtn) dom.registerBtn.addEventListener('click', openModal);
    if (dom.modalClose) dom.modalClose.addEventListener('click', closeModal);
    if (dom.modal) dom.modal.addEventListener('click', e => { if (e.target === dom.modal) closeModal(); });

    function openModal() {
        store.registrationStep = 0;
        store.captureCount = 0;
        updateCaptureCount();
        showStep(0);
        dom.modal.classList.add('visible');
        if (store.cameraStream && dom.captureVideo) dom.captureVideo.srcObject = store.cameraStream;
    }

    function closeModal() {
        dom.modal.classList.remove('visible');
        if (dom.captureVideo) dom.captureVideo.srcObject = null;
    }

    function showStep(n) {
        dom.modalSteps.forEach((s, i) => s.classList.toggle('active', i === n));
        dom.stepDots.forEach((d, i) => {
            d.classList.remove('active', 'done');
            if (i < n) d.classList.add('done');
            if (i === n) d.classList.add('active');
        });
    }

    if (dom.captureBtn) {
        dom.captureBtn.addEventListener('click', () => {
            if (store.captureCount >= 10) return;
            store.captureCount++;
            updateCaptureCount();
            gsap.fromTo(dom.captureBtn, { scale: .9 }, { scale: 1, duration: .3, ease: 'back.out(2)' });
            if (store.captureCount >= 5) setTimeout(() => showStep(1), 600);
        });
    }

    function updateCaptureCount() {
        if (dom.captureCount) dom.captureCount.innerHTML = `<strong>${store.captureCount}</strong> / 10 captured`;
    }

    if (dom.nameNextBtn) {
        dom.nameNextBtn.addEventListener('click', () => {
            const val = dom.nameInput.value.trim();
            if (!/^[A-Za-z]+_[A-Za-z]+$/.test(val)) {
                dom.nameError.classList.add('visible');
                dom.nameInput.style.borderColor = 'rgba(248,113,113,.5)';
                return;
            }
            dom.nameError.classList.remove('visible');
            dom.nameInput.style.borderColor = '';
            store.pendingName = val;
            showStep(2);
        });
    }

    if (dom.nameInput) {
        dom.nameInput.addEventListener('input', () => {
            dom.nameError.classList.remove('visible');
            dom.nameInput.style.borderColor = '';
        });
    }

    if (dom.approveBtn) {
        dom.approveBtn.addEventListener('click', () => {
            const code = dom.adminInput ? dom.adminInput.value.trim() : '';
            if (!code) { dom.adminInput.style.borderColor = 'rgba(248,113,113,.5)'; return; }
            showRegResult(true, store.pendingName);
        });
    }

    if (dom.cancelBtn) dom.cancelBtn.addEventListener('click', () => showRegResult(false, null));

    function showRegResult(approved, name) {
        showStep(3);
        if (approved) {
            dom.regResultIcon.textContent = '✅';
            dom.regResultTitle.textContent = 'Registration Approved';
            dom.regResultMsg.textContent = `Welcome ${(name || '').replace(/_/g, ' ')}`;
            showIdentity(name, 'Present', store.currentStudent?.emotion || 'Neutral');
            upsertStudent(name, 'Present', store.currentStudent?.emotion || 'Neutral');
            addLog({ event: 'attendance', student: name, attendance: 'Present', emotion: 'Neutral' });
        } else {
            dom.regResultIcon.textContent = '❌';
            dom.regResultTitle.textContent = 'Registration Rejected';
            dom.regResultMsg.textContent = 'Admin did not approve this registration.';
        }
    }

    if (dom.regDoneBtn) dom.regDoneBtn.addEventListener('click', closeModal);
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && dom.modal?.classList.contains('visible')) closeModal();
    });

    // ══════════════════════════════════════
    // PUBLIC API (Backend Integration Hooks)
    // ══════════════════════════════════════
    /** Show identity overlay — call from backend */
    window.dashShowIdentity = function (name, attendance, emotion) {
        showIdentity(name, attendance, emotion);
        if (name) upsertStudent(name, attendance, emotion);
    };

    /** Add a raw log event — call from backend */
    window.dashAddLog = function (data) { addLog(data); };

    /** Process a full event from backend */
    window.dashProcessEvent = function (evt) {
        addLog(evt);
        if (evt.event === 'attendance') {
            showIdentity(evt.student, evt.attendance, evt.emotion);
            upsertStudent(evt.student, evt.attendance, evt.emotion);
        } else if (evt.event === 'question') {
            addQuestion(evt);
            upsertStudentQuestion(evt.student, evt.question, evt.topic);
            showQuestionOverlay(evt.question, evt.topic);
        }
    };

    /** Batch-load events (e.g. on reconnect) */
    window.dashLoadEvents = function (events) {
        events.forEach(evt => window.dashProcessEvent(evt));
    };

    /** Update system status */
    window.dashSetStatus = function (online, msg) { setStatus(online, msg); };

})();
