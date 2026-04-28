// ============================================
// MAIN APPLICATION SCRIPT
// ============================================
// Systems:
//   1. Lenis smooth scrolling
//   2. Custom cursor (dot + outline)
//   3. Interactive background particles
//   4. GSAP scroll animations
// ============================================

// Register GSAP Plugins
gsap.registerPlugin(ScrollTrigger);


// ============================================
// 1. LENIS — SMOOTH SCROLLING
// ============================================
const lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    direction: 'vertical',
    gestureDirection: 'vertical',
    smooth: true,
    smoothTouch: false,
    touchMultiplier: 2,
});

// Sync with GSAP ScrollTrigger
lenis.on('scroll', ScrollTrigger.update);

gsap.ticker.add((time) => {
    lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0, 0);


// ============================================
// 2. CUSTOM CURSOR
// ============================================
const cursorDot     = document.querySelector('.cursor-dot');
const cursorOutline = document.querySelector('.cursor-outline');
const hoverElements = document.querySelectorAll('[data-cursor="hover"]');

// GSAP quickSetter for performance
const setDotX     = gsap.quickSetter(cursorDot, "x", "px");
const setDotY     = gsap.quickSetter(cursorDot, "y", "px");
const setOutlineX = gsap.quickSetter(cursorOutline, "x", "px");
const setOutlineY = gsap.quickSetter(cursorOutline, "y", "px");

let mouse   = { x: 0, y: 0 };
let outline = { x: 0, y: 0 };

window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    setDotX(mouse.x);
    setDotY(mouse.y);
});

// Smooth outline follow (lerp)
const updateCursorOutline = () => {
    const lerpFactor = 0.15;
    outline.x += (mouse.x - outline.x) * lerpFactor;
    outline.y += (mouse.y - outline.y) * lerpFactor;
    setOutlineX(outline.x);
    setOutlineY(outline.y);
    requestAnimationFrame(updateCursorOutline);
};
updateCursorOutline();

// Click scale effect
window.addEventListener('mousedown', () => {
    gsap.to(cursorDot, { scale: 0.6, duration: 0.15, ease: 'power2.out' });
    gsap.to(cursorOutline, { scale: 0.8, duration: 0.15, ease: 'power2.out' });
});
window.addEventListener('mouseup', () => {
    gsap.to(cursorDot, { scale: 1, duration: 0.15, ease: 'power2.out' });
    gsap.to(cursorOutline, { scale: 1, duration: 0.15, ease: 'power2.out' });
});

// Hover effects
hoverElements.forEach(el => {
    el.addEventListener('mouseenter', () => {
        cursorOutline.classList.add('hover-active');
        gsap.to(cursorDot, { scale: 0, duration: 0.2 });
    });
    el.addEventListener('mouseleave', () => {
        cursorOutline.classList.remove('hover-active');
        gsap.to(cursorDot, { scale: 1, duration: 0.2 });
    });
});

// Hide cursor when leaving window
document.addEventListener('mouseleave', () => {
    gsap.to([cursorDot, cursorOutline], { opacity: 0, duration: 0.3 });
});
document.addEventListener('mouseenter', () => {
    gsap.to([cursorDot, cursorOutline], { opacity: 1, duration: 0.3 });
});


// ============================================
// 3. INTERACTIVE BACKGROUND PARTICLES
// ============================================
const canvas = document.getElementById('bg-canvas');
const ctx    = canvas.getContext('2d');
let bgParticles = [];
const BG_PARTICLE_COUNT = 70;

// Mouse position for particle interaction
let bgMouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
window.addEventListener('mousemove', (e) => {
    bgMouse.x = e.clientX;
    bgMouse.y = e.clientY;
});

const resizeCanvas = () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
};
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

class BgParticle {
    constructor() {
        this.x      = Math.random() * canvas.width;
        this.y      = Math.random() * canvas.height;
        this.baseX  = this.x;
        this.baseY  = this.y;
        this.size   = Math.random() * 2 + 0.5;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * -0.5 - 0.2;
        this.opacity = Math.random() * 0.5 + 0.1;
        // Blue/purple palette
        this.hue = Math.random() > 0.6 ? 270 : 240;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        // Mouse attraction — subtle distortion
        const dx = bgMouse.x - this.x;
        const dy = bgMouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const maxDist = 200;

        if (dist < maxDist) {
            const force = (maxDist - dist) / maxDist;
            this.x += dx * force * 0.008;
            this.y += dy * force * 0.008;
        }

        // Wrap around
        if (this.y < 0) this.y = canvas.height;
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
    }

    draw() {
        // Core dot
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${this.hue}, 70%, 60%, ${this.opacity})`;
        ctx.fill();

        // Glow halo
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 4, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${this.hue}, 70%, 60%, ${this.opacity * 0.08})`;
        ctx.fill();
    }
}

const initBgParticles = () => {
    bgParticles = [];
    for (let i = 0; i < BG_PARTICLE_COUNT; i++) {
        bgParticles.push(new BgParticle());
    }
};

// Draw subtle connections between nearby particles
const drawBgConnections = () => {
    for (let i = 0; i < bgParticles.length; i++) {
        for (let j = i + 1; j < bgParticles.length; j++) {
            const dx = bgParticles[i].x - bgParticles[j].x;
            const dy = bgParticles[i].y - bgParticles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 150) {
                ctx.beginPath();
                ctx.moveTo(bgParticles[i].x, bgParticles[i].y);
                ctx.lineTo(bgParticles[j].x, bgParticles[j].y);
                ctx.strokeStyle = `rgba(79, 70, 229, ${0.04 * (1 - dist / 150)})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
};

const animateBgParticles = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Central radial gradient
    const gradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, canvas.width * 0.7
    );
    gradient.addColorStop(0, 'rgba(79, 70, 229, 0.04)');
    gradient.addColorStop(0.5, 'rgba(147, 51, 234, 0.02)');
    gradient.addColorStop(1, 'rgba(5, 5, 5, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    bgParticles.forEach(p => {
        p.update();
        p.draw();
    });

    drawBgConnections();

    requestAnimationFrame(animateBgParticles);
};

initBgParticles();
animateBgParticles();


// ============================================
// 4. GSAP SCROLL ANIMATIONS
// ============================================

// ─── Controlled Hero Animation ───
let heroPlayed = false;

window.playHeroAnimation = () => {
    if (heroPlayed) return;
    heroPlayed = true;

    const tl = gsap.timeline();

    tl.fromTo(".hero .badge", {
        opacity: 0,
        y: 20
    }, {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "power3.out"
    })
    .fromTo(".hero .hero-title", {
        opacity: 0,
        y: 30
    }, {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: "power3.out"
    }, "-=0.4")
    .fromTo(".hero .hero-subtitle", {
        opacity: 0,
        y: 30
    }, {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: "power3.out"
    }, "-=0.5")
    .fromTo(".hero .hero-actions", {
        opacity: 0,
        y: 20
    }, {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "power3.out"
    }, "-=0.4")
    .fromTo(".hero .scroll-indicator", {
        opacity: 0,
        y: 20
    }, {
        opacity: 0.6,
        y: 0,
        duration: 0.8,
        ease: "power3.out"
    }, "-=0.4");
};

// Animations are deferred until the intro completes

window.addEventListener('intro:complete', () => {

    // ─── Features Section ───
    gsap.fromTo('.feature-card',
        { y: 60, opacity: 0 },
        {
            y: 0,
            opacity: 1,
            duration: 0.8,
            stagger: 0.2,
            ease: 'power2.out',
            scrollTrigger: {
                trigger: '.features',
                start: 'top 75%',
                toggleActions: 'play none none reverse'
            }
        }
    );

    // ─── Workflow Pipeline ───
    const workflowTl = gsap.timeline({
        scrollTrigger: {
            trigger: '.workflow',
            start: 'top 60%',
            toggleActions: 'play none none reverse'
        }
    });

    workflowTl.fromTo('.workflow-node',
        { scale: 0.8, opacity: 0, y: 20 },
        { scale: 1, opacity: 1, y: 0, duration: 0.6, stagger: 0.3, ease: 'back.out(1.5)' }
    )
    .to('.connector-line',
        { width: '100%', duration: 1, ease: 'power2.inOut', stagger: 0.3 },
        '-=0.9'
    )
    .to('.connector-particle',
        {
            opacity: 1,
            left: '100%',
            duration: 1.5,
            repeat: -1,
            stagger: 0.5,
            ease: 'none'
        },
        '-=0.5'
    );

    // ─── Impact Section ───
    gsap.fromTo('.impact-item',
        { x: -40, opacity: 0 },
        {
            x: 0,
            opacity: 1,
            duration: 0.8,
            stagger: 0.15,
            ease: 'power2.out',
            scrollTrigger: {
                trigger: '.impact',
                start: 'top 70%',
                toggleActions: 'play none none reverse'
            }
        }
    );

    // ─── CTA / Final Section ───
    gsap.fromTo('.cta-box',
        { scale: 0.95, opacity: 0, y: 40 },
        {
            scale: 1,
            opacity: 1,
            y: 0,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.cta',
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        }
    );

    // ─── Section titles reveal ───
    gsap.utils.toArray('.section-title').forEach(title => {
        gsap.fromTo(title,
            { y: 30, opacity: 0 },
            {
                y: 0,
                opacity: 1,
                duration: 0.8,
                ease: 'power2.out',
                scrollTrigger: {
                    trigger: title,
                    start: 'top 85%',
                    toggleActions: 'play none none reverse'
                }
            }
        );
    });

    gsap.utils.toArray('.section-subtitle').forEach(sub => {
        gsap.fromTo(sub,
            { y: 20, opacity: 0 },
            {
                y: 0,
                opacity: 1,
                duration: 0.8,
                delay: 0.15,
                ease: 'power2.out',
                scrollTrigger: {
                    trigger: sub,
                    start: 'top 85%',
                    toggleActions: 'play none none reverse'
                }
            }
        );
    });

});
