// Register GSAP Plugins
gsap.registerPlugin(ScrollTrigger);

// ==========================================
// 1. Initialize Lenis (Smooth Scrolling)
// ==========================================
const lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    direction: 'vertical',
    gestureDirection: 'vertical',
    smooth: true,
    smoothTouch: false,
    touchMultiplier: 2,
});

// Sync Lenis with GSAP ScrollTrigger
lenis.on('scroll', ScrollTrigger.update);

gsap.ticker.add((time) => {
    lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0, 0);

// ==========================================
// 2. Custom Cursor
// ==========================================
const cursorDot = document.querySelector('.cursor-dot');
const cursorOutline = document.querySelector('.cursor-outline');
const hoverElements = document.querySelectorAll('[data-cursor="hover"]');

// Use GSAP quickSetter for performant cursor updates
const setDotX = gsap.quickSetter(cursorDot, "x", "px");
const setDotY = gsap.quickSetter(cursorDot, "y", "px");
const setOutlineX = gsap.quickSetter(cursorOutline, "x", "px");
const setOutlineY = gsap.quickSetter(cursorOutline, "y", "px");

let mouse = { x: 0, y: 0 };
let outline = { x: 0, y: 0 };

window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    // Dot follows immediately
    setDotX(mouse.x);
    setDotY(mouse.y);
});

// Interpolation for smooth outline following
const updateCursorOutline = () => {
    // Lerp (Linear Interpolation) factor for smoothness
    const lerpFactor = 0.15;
    
    outline.x += (mouse.x - outline.x) * lerpFactor;
    outline.y += (mouse.y - outline.y) * lerpFactor;
    
    setOutlineX(outline.x);
    setOutlineY(outline.y);
    
    requestAnimationFrame(updateCursorOutline);
};
updateCursorOutline();

// Hover Effects
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

// ==========================================
// 3. Dynamic Canvas Background (Subtle Particles)
// ==========================================
const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');
let particles = [];
const particleCount = 60; // Keep it low for performance

const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
};

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * -0.5 - 0.2; // Move upwards slightly
        this.opacity = Math.random() * 0.5 + 0.1;
    }
    
    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        
        // Wrap around
        if (this.y < 0) this.y = canvas.height;
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
    }
    
    draw() {
        ctx.fillStyle = `rgba(147, 51, 234, ${this.opacity})`; // Accent purple glow
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

const initParticles = () => {
    particles = [];
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
};

const animateParticles = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Add subtle gradient to background
    const gradient = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 0, canvas.width/2, canvas.height/2, canvas.width);
    gradient.addColorStop(0, 'rgba(79, 70, 229, 0.05)'); // Blue hint
    gradient.addColorStop(1, 'rgba(5, 5, 5, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    particles.forEach(p => {
        p.update();
        p.draw();
    });
    
    requestAnimationFrame(animateParticles);
};

initParticles();
animateParticles();

// ==========================================
// 4. GSAP Scroll Animations
// ==========================================

// Initial Load Animation
const initLoad = () => {
    const tl = gsap.timeline();
    
    tl.fromTo('.hero .fade-up', 
        { y: 50, opacity: 0 },
        { y: 0, opacity: 1, duration: 1, stagger: 0.15, ease: "power3.out", delay: 0.2 }
    );
};
initLoad();

// Features Section
gsap.fromTo('.feature-card',
    { y: 60, opacity: 0 },
    {
        y: 0,
        opacity: 1,
        duration: 0.8,
        stagger: 0.2,
        ease: "power2.out",
        scrollTrigger: {
            trigger: '.features',
            start: "top 75%",
            toggleActions: "play none none reverse"
        }
    }
);

// Workflow Pipeline Section
const workflowTl = gsap.timeline({
    scrollTrigger: {
        trigger: '.workflow',
        start: "top 60%",
        toggleActions: "play none none reverse"
    }
});

workflowTl.fromTo('.workflow-node',
    { scale: 0.8, opacity: 0, y: 20 },
    { scale: 1, opacity: 1, y: 0, duration: 0.6, stagger: 0.3, ease: "back.out(1.5)" }
)
.to('.connector-line', 
    { width: "100%", duration: 1, ease: "power2.inOut", stagger: 0.3 }, 
    "-=0.9" // Start while nodes are still appearing
)
.to('.connector-particle',
    { 
        opacity: 1,
        left: "100%", 
        duration: 1.5, 
        repeat: -1, 
        stagger: 0.5, 
        ease: "none" 
    },
    "-=0.5"
);

// Impact Section
gsap.fromTo('.impact-item',
    { x: -40, opacity: 0 },
    {
        x: 0,
        opacity: 1,
        duration: 0.8,
        stagger: 0.15,
        ease: "power2.out",
        scrollTrigger: {
            trigger: '.impact',
            start: "top 70%",
            toggleActions: "play none none reverse"
        }
    }
);

// CTA Section
gsap.fromTo('.cta-box',
    { scale: 0.95, opacity: 0, y: 40 },
    {
        scale: 1,
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
        scrollTrigger: {
            trigger: '.cta',
            start: "top 80%",
            toggleActions: "play none none reverse"
        }
    }
);
