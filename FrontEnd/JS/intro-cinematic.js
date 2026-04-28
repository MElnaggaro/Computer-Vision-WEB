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
    class Particle {
        constructor() {
            this.x = Math.random() * introCanvas.width;
            this.y = Math.random() * introCanvas.height;
            this.size = Math.random() * 1.5 + 0.5;
            this.vx = (Math.random() - 0.5) * 0.15;
            this.vy = (Math.random() - 0.5) * 0.15;
            this.baseAlpha = Math.random() * 0.3 + 0.05;
        }
        update() {
            // Subtle reaction to mouse
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
            introCtx.beginPath();
            introCtx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            introCtx.fillStyle = `rgba(255, 255, 255, ${this.baseAlpha})`;
            introCtx.fill();
        }
    }

    const initParticles = () => {
        particles = [];
        for (let i = 0; i < 50; i++) {
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

        // Transition from STATE 1 -> STATE 2
        gsap.to(introStatus, {
            opacity: 0,
            duration: 0.6,
            ease: 'power2.inOut',
            onComplete: () => {
                introStatus.textContent = '[SYSTEM READY — CLICK TO ENTER]';
                introStatus.classList.add('ready');
                gsap.to(introStatus, { opacity: 1, duration: 0.8, ease: 'power2.inOut' });
                introScreen.style.cursor = 'pointer';
                introScreen.addEventListener('click', startExperience, { once: true });
            }
        });
    };

    // ─── Click-to-Start Sequence ───
    const startExperience = () => {
        if (isClicked) return;
        isClicked = true; // Lock input

        const tl = gsap.timeline();

        // Dispatch reveal model so Three.js can start
        window.dispatchEvent(new CustomEvent('intro:reveal-model'));

        // Step 1: Fade out intro overlay slowly
        tl.to(introScreen, {
            opacity: 0,
            duration: 1.5,
            ease: 'power2.out'
        }, 0);

        // Step 2: Gradually fade IN main website
        tl.to(mainWrapper, {
            opacity: 1,
            y: 0,
            duration: 1.8,
            ease: 'power3.out'
        }, 0.2); // parallel, slightly offset

        tl.call(() => {
            introExited = true;
            cancelAnimationFrame(animId);
            // Step 3: Remove from DOM
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
