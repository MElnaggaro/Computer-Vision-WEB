// ============================================
// INTERACTIVE DEMO SYSTEM (UNIFIED DASHBOARD)
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    const launchBtns = document.querySelectorAll('.launch-demo-btn');
    const demoSection = document.getElementById('demo');
    
    // Camera Elements
    const cameraToggleBtn = document.getElementById('camera-toggle-btn');
    const videoElement = document.getElementById('demo-video');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    
    // Voice Elements
    const micToggleBtn = document.getElementById('mic-toggle-btn');
    const micBtn = document.getElementById('demo-mic-btn'); // icon button
    const voiceStatusText = document.getElementById('voice-status-text');
    
    // Result Elements
    const finalQuestionText = document.getElementById('final-question-text');
    const finalTopicText = document.getElementById('final-topic-text');
    
    let demoActivated = false;

    // 1. Launch Demo Flow — triggers 3D model transition + reveals demo
    launchBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            
            if (demoActivated) {
                demoSection.scrollIntoView({ behavior: 'smooth' });
                return;
            }
            demoActivated = true;
            
            // Show the demo section
            demoSection.style.display = "block";
            
            // Reveal with GSAP animation
            gsap.from(demoSection, { 
                opacity: 0, 
                y: 60, 
                duration: 1, 
                ease: 'power3.out', 
                onComplete: () => {
                    demoSection.style.pointerEvents = "auto";
                }
            });
            
            // Scroll to the demo section
            setTimeout(() => {
                demoSection.scrollIntoView({ behavior: 'smooth' });
            }, 100);

            // ─── TRIGGER 3D MODEL TRANSITION ───
            // Fires the cinematic swap: AIU exits left → Camera enters right
            setTimeout(() => {
                if (window.startModelTransition) {
                    window.startModelTransition();
                }
            }, 400); // Slight delay so scroll begins first
        });
    });

    // 2. Camera Toggle Logic
    let cameraStream = null;
    let isCameraRunning = false;

    cameraToggleBtn.addEventListener('click', async () => {
        if (!isCameraRunning) {
            // Start Camera
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoElement.srcObject = cameraStream;
                videoElement.style.display = 'block';
                videoElement.style.transform = 'scaleX(-1)'; // natural mirror fix
                cameraPlaceholder.style.display = 'none';
                
                cameraToggleBtn.textContent = 'Stop Camera';
                cameraToggleBtn.classList.replace('btn-secondary', 'btn-primary');
                isCameraRunning = true;
            } catch (error) {
                console.error('Camera access denied or error:', error);
                alert('Camera access is required for this demo. Please allow camera permissions.');
            }
        } else {
            // Stop Camera
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                videoElement.srcObject = null;
                cameraStream = null;
            }
            videoElement.style.display = 'none';
            cameraPlaceholder.style.display = 'flex';
            
            cameraToggleBtn.textContent = 'Start Camera';
            cameraToggleBtn.classList.replace('btn-primary', 'btn-secondary');
            isCameraRunning = false;
        }
    });

    // 3. Web Speech API Toggle Logic
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isRecording = false;
    
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        
        recognition.onstart = () => {
            micBtn.classList.add('recording');
            voiceStatusText.textContent = "Listening...";
            voiceStatusText.style.color = "var(--accent-pink)";
            isRecording = true;
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            voiceStatusText.textContent = `"${transcript}"`;
            voiceStatusText.style.color = "var(--text-secondary)";
            
            // Update Results Card
            finalQuestionText.textContent = `"${transcript}"`;
            finalTopicText.textContent = "Computer Vision";
            
            // Animate highlight box to show it updated
            const highlightBox = document.querySelector('.highlight-box');
            gsap.fromTo(highlightBox, 
                { backgroundColor: 'rgba(236, 72, 153, 0.2)' },
                { backgroundColor: 'rgba(79, 70, 229, 0.05)', duration: 1, ease: 'power2.out' }
            );
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            voiceStatusText.textContent = "Error listening. Try again.";
            voiceStatusText.style.color = "var(--text-secondary)";
            stopRecordingUI();
        };
        
        recognition.onend = () => {
            stopRecordingUI();
        };
    } else {
        voiceStatusText.textContent = "Speech recognition not supported in this browser.";
    }
    
    // Wire up both the text button and the icon button
    micToggleBtn.addEventListener('click', toggleRecording);
    micBtn.addEventListener('click', toggleRecording);
    
    function toggleRecording() {
        if (!recognition) return;
        
        if (!isRecording) {
            // Start Recording
            try {
                recognition.start();
                micToggleBtn.textContent = 'Stop Recording';
                micToggleBtn.classList.replace('btn-secondary', 'btn-primary');
            } catch(e) {
                console.error("Recognition already started", e);
            }
        } else {
            // Stop Recording
            recognition.stop();
            stopRecordingUI();
        }
    }
    
    function stopRecordingUI() {
        micBtn.classList.remove('recording');
        micToggleBtn.textContent = 'Start Recording';
        micToggleBtn.classList.replace('btn-primary', 'btn-secondary');
        isRecording = false;
        
        if (voiceStatusText.textContent === "Listening...") {
             voiceStatusText.textContent = "Ready to record";
             voiceStatusText.style.color = "var(--text-secondary)";
        }
    }
});
