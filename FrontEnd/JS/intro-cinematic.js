// ============================================
// CINEMATIC INTRO SYSTEM (MINIMAL & CLEAN)
// ============================================

const IntroSystem = (() => {
    // ─── DOM References ───
    const introScreen   = document.getElementById('intro-screen');
    const introCanvas   = document.getElementById('intro-particles');
    const introStatus   = document.getElementById('intro-status');
    const introCtx      = introCanvas.getContext('2d');
    const mainWrapper   = document.getElementById('smooth-wrapper');

    // ─── State ───
    let modelLoaded  = false;
    let introExited  = false;
    let particles = [];
    let isClicked = false;
    
    // Initial State of Main Website
    gsap.set(mainWrapper, { opacity: 0, y: 20 });

    // Lock scroll during intro
    document.body.classList.add('intro-active');

    // Mouse tracking for background parallax
    let mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    // ─── Background Particle System ───
    const INTRO_COLORS = [
        { h: 197, s: 92, l: 60 },
        { h: 231, s: 92, l: 74 },
        { h: 263, s: 93, l: 77 },
        { h: 190, s: 82, l: 65 },
    ];

    class Particle {
        constructor() {
            this.x = Math.random() * introCanvas.width;
            this.y = Math.random() * introCanvas.height;
            this.size = Math.random() * 1.8 + 0.5;
            this.vx = (Math.random() - 0.5) * 0.15;
            this.vy = (Math.random() - 0.5) * 0.15;
            this.baseAlpha = Math.random() * 0.35 + 0.05;
            const c = INTRO_COLORS[Math.floor(Math.random() * INTRO_COLORS.length)];
            this.h = c.h; this.s = c.s; this.l = c.l;
        }
        update() {
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            if (dist < 150) {
                this.x -= dx * 0.0003;
                this.y -= dy * 0.0003;
            }

            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0) this.x = introCanvas.width;
            if (this.x > introCanvas.width) this.x = 0;
            if (this.y < 0) this.y = introCanvas.height;
            if (this.y > introCanvas.height) this.y = 0;
        }
        draw() {
            // Glow
            introCtx.beginPath();
            introCtx.arc(this.x, this.y, this.size * 4, 0, Math.PI * 2);
            introCtx.fillStyle = `hsla(${this.h}, ${this.s}%, ${this.l}%, ${this.baseAlpha * 0.08})`;
            introCtx.fill();
            // Core
            introCtx.beginPath();
            introCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            introCtx.fillStyle = `hsla(${this.h}, ${this.s}%, ${this.l}%, ${this.baseAlpha})`;
            introCtx.fill();
        }
    }

    const initParticles = () => {
        particles = [];
        for (let i = 0; i < 60; i++) {
            particles.push(new Particle());
        }
    };

    const resize = () => {
        if(introCanvas) {
            introCanvas.width = window.innerWidth;
            introCanvas.height = window.innerHeight;
        }
    };
    window.addEventListener('resize', resize);
    resize();
    initParticles();

    let animId;
    const animate = () => {
        if (introExited) return;
        introCtx.clearRect(0, 0, introCanvas.width, introCanvas.height);
        
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        
        animId = requestAnimationFrame(animate);
    };
    animate();

    // ─── Public: Signal that the 3D model has loaded ───
    const onModelLoaded = () => {
        modelLoaded = true;
    };

    // ─── Master Timeline ───
    const masterTl = gsap.timeline({ defaults: { ease: "power3.out" } });

    // Initial setups
    gsap.set(introStatus, { opacity: 0 });
    gsap.set(mainWrapper, { opacity: 0, y: 30 });

    // PHASE 1 — LOADING
    masterTl.to(introStatus, { opacity: 1, duration: 1.2 });

    // PHASE 2 — READY STATE
    masterTl.call(() => {
        // Pause until model is fully loaded if it isn't already
        if (!modelLoaded) masterTl.pause();
        
        // Wait interval to periodically check
        const checkLoad = setInterval(() => {
            if (modelLoaded) {
                clearInterval(checkLoad);
                masterTl.resume();
            }
        }, 100);
    });

    masterTl.to(introStatus, { opacity: 0, duration: 0.4 })
            .call(() => {
                introStatus.textContent = '[SYSTEM READY — CLICK TO ENTER]';
                introStatus.classList.add('ready');
            })
            .to(introStatus, { opacity: 1, duration: 0.4 });

    // WAIT FOR USER CLICK
    masterTl.call(() => {
        console.log("Intro loaded - waiting for click");
        introScreen.addEventListener('click', startExperience);
    });

    // ─── Click-to-Start Sequence ───
    const startExperience = () => {
        console.log("Intro clicked - startExperience executing");
        if (isClicked) return;
        isClicked = true; // Lock input

        const clickTl = gsap.timeline({ defaults: { ease: "power3.out" } });

        // Trigger Model Fade In & Glow
        clickTl.call(() => {
            window.dispatchEvent(new CustomEvent('intro:reveal-model'));
        });

        // STEP 1: Fade out intro
        clickTl.to(introScreen, {
            opacity: 0,
            duration: 1.2,
            ease: 'power2.out'
        });

        // STEP 2: Main hero container reveal (OVERLAP)
        clickTl.to(mainWrapper, {
            opacity: 1,
            y: 0,
            duration: 1.6,
            ease: 'power3.out'
        }, "-=0.8");

        // STEP 3: Delay and Trigger Hero Animation
        clickTl.call(() => {
            setTimeout(() => {
                if (window.playHeroAnimation) {
                    window.playHeroAnimation();
                }
            }, 400);
        });

        // STEP 4: Remove intro from DOM
        clickTl.call(() => {
            introExited = true;
            cancelAnimationFrame(animId);
            introScreen.remove();
            document.body.classList.remove('intro-active');
            window.dispatchEvent(new CustomEvent('intro:complete'));
        });
    };

    // ─── Public API ───
    return {
        onModelLoaded
    };

})();

// Expose globally so three-scene.js can call it
window.IntroSystem = IntroSystem;
