// ============================================
// MAIN APPLICATION SCRIPT — OPTIMIZED
// ============================================
// Systems:
//   1. Lenis smooth scrolling
//   2. Custom cursor (dot + outline)
//   3. Interactive background particles (optimized)
//   4. GSAP scroll animations
//   5. Visibility API — pause when hidden
// ============================================

// Register GSAP Plugins
gsap.registerPlugin(ScrollTrigger);

// ============================================
// 0. GLOBAL MOUSE STATE (single listener)
// ============================================
const gMouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
window.addEventListener('mousemove', (e) => {
    gMouse.x = e.clientX;
    gMouse.y = e.clientY;
}, { passive: true });


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

lenis.on('scroll', ScrollTrigger.update);

gsap.ticker.add((time) => {
    lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0, 0);


// ============================================
// 2. CUSTOM CURSOR (merged into GSAP ticker)
// ============================================
const cursorDot     = document.querySelector('.cursor-dot');
const cursorOutline = document.querySelector('.cursor-outline');
const hoverElements = document.querySelectorAll('[data-cursor="hover"]');

const setDotX     = gsap.quickSetter(cursorDot, "x", "px");
const setDotY     = gsap.quickSetter(cursorDot, "y", "px");
const setOutlineX = gsap.quickSetter(cursorOutline, "x", "px");
const setOutlineY = gsap.quickSetter(cursorOutline, "y", "px");

let outline = { x: 0, y: 0 };

// Use GSAP ticker instead of separate rAF loop for cursor
gsap.ticker.add(() => {
    // Dot follows instantly (set in mousemove)
    setDotX(gMouse.x);
    setDotY(gMouse.y);
    // Outline lerps
    outline.x += (gMouse.x - outline.x) * 0.15;
    outline.y += (gMouse.y - outline.y) * 0.15;
    setOutlineX(outline.x);
    setOutlineY(outline.y);
});

// Click scale effect
window.addEventListener('mousedown', () => {
    gsap.to(cursorDot, { scale: 0.6, duration: 0.15, ease: 'power2.out' });
    gsap.to(cursorOutline, { scale: 0.8, duration: 0.15, ease: 'power2.out' });
}, { passive: true });
window.addEventListener('mouseup', () => {
    gsap.to(cursorDot, { scale: 1, duration: 0.15, ease: 'power2.out' });
    gsap.to(cursorOutline, { scale: 1, duration: 0.15, ease: 'power2.out' });
}, { passive: true });

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
// 3. INTERACTIVE BACKGROUND — OPTIMIZED
// ============================================
const canvas = document.getElementById('bg-canvas');
const ctx    = canvas.getContext('2d', { alpha: true });
let bgParticles = [];
const BG_PARTICLE_COUNT = 55; // reduced from 90

let canvasW = 0, canvasH = 0;
const resizeCanvas = () => {
    canvasW = canvas.width  = window.innerWidth;
    canvasH = canvas.height = window.innerHeight;
};
window.addEventListener('resize', resizeCanvas, { passive: true });
resizeCanvas();

// Pre-computed color strings for particles (avoid per-frame string concat)
const PARTICLE_COLORS = [
    { h: 197, s: 92, l: 60 },
    { h: 231, s: 92, l: 74 },
    { h: 263, s: 93, l: 77 },
    { h: 190, s: 82, l: 65 },
    { h: 330, s: 86, l: 70 },
];

class BgParticle {
    constructor() {
        this.x      = Math.random() * canvasW;
        this.y      = Math.random() * canvasH;
        this.size   = Math.random() * 2.2 + 0.4;
        this.speedX = Math.random() * 0.4 - 0.2;
        this.speedY = Math.random() * -0.45 - 0.15;
        this.opacity = Math.random() * 0.55 + 0.08;
        this.pulseOffset = Math.random() * Math.PI * 2;
        const c = PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)];
        this.h = c.h; this.s = c.s; this.l = c.l;
        // Pre-build color base strings
        this._colorBase = `${c.h}, ${c.s}%, ${c.l}%`;
        this.currentOpacity = this.opacity;
    }

    update(time) {
        this.x += this.speedX;
        this.y += this.speedY;

        // Breathing (simplified — every 2nd frame skip is handled externally)
        this.currentOpacity = this.opacity * (Math.sin(time * 0.001 + this.pulseOffset) * 0.15 + 1);

        // Mouse attraction (use squared distance to avoid sqrt)
        const dx = gMouse.x - this.x;
        const dy = gMouse.y - this.y;
        const distSq = dx * dx + dy * dy;
        if (distSq < 48400) { // 220^2
            const force = (48400 - distSq) / 48400 * 0.01;
            this.x += dx * force;
            this.y += dy * force;
        }

        // Wrap around
        if (this.y < -10) this.y = canvasH + 10;
        if (this.x < -10) this.x = canvasW + 10;
        if (this.x > canvasW + 10) this.x = -10;
    }

    draw() {
        const op = this.currentOpacity;
        // Glow halo
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 5, 0, 6.2832);
        ctx.fillStyle = `hsla(${this._colorBase}, ${(op * 0.06).toFixed(3)})`;
        ctx.fill();
        // Core dot
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, 6.2832);
        ctx.fillStyle = `hsla(${this._colorBase}, ${op.toFixed(3)})`;
        ctx.fill();
    }
}

const initBgParticles = () => {
    bgParticles = [];
    for (let i = 0; i < BG_PARTICLE_COUNT; i++) {
        bgParticles.push(new BgParticle());
    }
};

// Spatial grid for O(n) connection checks instead of O(n²)
const GRID_SIZE = 160;
const drawBgConnections = () => {
    const cols = Math.ceil(canvasW / GRID_SIZE) + 1;
    const rows = Math.ceil(canvasH / GRID_SIZE) + 1;
    const grid = new Array(cols * rows);

    // Place particles in grid cells
    for (let i = 0; i < bgParticles.length; i++) {
        const p = bgParticles[i];
        const col = Math.floor(p.x / GRID_SIZE);
        const row = Math.floor(p.y / GRID_SIZE);
        const idx = row * cols + col;
        if (idx >= 0 && idx < grid.length) {
            if (!grid[idx]) grid[idx] = [];
            grid[idx].push(p);
        }
    }

    // Check only neighboring cells
    ctx.lineWidth = 0.5;
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const cell = grid[row * cols + col];
            if (!cell) continue;

            // Check this cell + right + bottom + bottom-right neighbors
            const neighbors = [
                cell,
                col + 1 < cols ? grid[row * cols + col + 1] : null,
                row + 1 < rows ? grid[(row + 1) * cols + col] : null,
                (col + 1 < cols && row + 1 < rows) ? grid[(row + 1) * cols + col + 1] : null,
            ];

            for (let ni = 0; ni < neighbors.length; ni++) {
                const ncell = neighbors[ni];
                if (!ncell) continue;
                const startJ = (ncell === cell) ? 0 : 0;
                for (let ci = 0; ci < cell.length; ci++) {
                    const p1 = cell[ci];
                    const jStart = (ncell === cell) ? ci + 1 : 0;
                    for (let cj = jStart; cj < ncell.length; cj++) {
                        const p2 = ncell[cj];
                        const dx = p1.x - p2.x;
                        const dy = p1.y - p2.y;
                        const distSq = dx * dx + dy * dy;
                        if (distSq < 25600) { // 160^2
                            const dist = Math.sqrt(distSq);
                            const alpha = 0.05 * (1 - dist / 160);
                            ctx.beginPath();
                            ctx.moveTo(p1.x, p1.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.strokeStyle = `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
                            ctx.stroke();
                        }
                    }
                }
            }
        }
    }
};

// Visibility API — pause animation when tab is hidden
let bgPaused = false;
document.addEventListener('visibilitychange', () => {
    bgPaused = document.hidden;
    if (!bgPaused) requestAnimationFrame(animateBgParticles);
});

let bgFrame = 0;
const animateBgParticles = () => {
    if (bgPaused) return;

    const now = performance.now();
    ctx.clearRect(0, 0, canvasW, canvasH);

    // Aurora layer 1 — only redraw every 2nd frame for performance
    bgFrame++;
    if (bgFrame % 2 === 0) {
        const g1 = ctx.createRadialGradient(
            canvasW * 0.5, canvasH * 0.15, 0,
            canvasW * 0.5, canvasH * 0.15, canvasW * 0.6
        );
        g1.addColorStop(0, 'rgba(56, 189, 248, 0.05)');
        g1.addColorStop(0.4, 'rgba(129, 140, 248, 0.03)');
        g1.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g1;
        ctx.fillRect(0, 0, canvasW, canvasH);

        // Aurora layer 2
        const shift = Math.sin(now * 0.0003) * 0.1 + 0.75;
        const g2 = ctx.createRadialGradient(
            canvasW * shift, canvasH * 0.85, 0,
            canvasW * shift, canvasH * 0.85, canvasW * 0.5
        );
        g2.addColorStop(0, 'rgba(167, 139, 250, 0.04)');
        g2.addColorStop(0.5, 'rgba(129, 140, 248, 0.02)');
        g2.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g2;
        ctx.fillRect(0, 0, canvasW, canvasH);
    }

    // Cursor spotlight — every frame (cheap single gradient)
    const g3 = ctx.createRadialGradient(
        gMouse.x, gMouse.y, 0, gMouse.x, gMouse.y, 280
    );
    g3.addColorStop(0, 'rgba(56, 189, 248, 0.04)');
    g3.addColorStop(0.5, 'rgba(129, 140, 248, 0.015)');
    g3.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g3;
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Update and draw particles
    for (let i = 0; i < bgParticles.length; i++) {
        bgParticles[i].update(now);
        bgParticles[i].draw();
    }

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

// Animations deferred until intro completes
window.addEventListener('intro:complete', () => {

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

// Expose gMouse for other scripts
window.gMouse = gMouse;
