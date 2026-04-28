// ============================================
// THREE.JS CINEMATIC 3D SYSTEM
// ============================================
// Full-featured 3D layer with:
//   1. Scene, Camera, Renderer setup
//   2. AI-themed lighting (blue/purple)
//   3. GLB model loading + intro integration
//   4. Base idle animation (rotation + float)
//   5. Mouse parallax interaction (lerp)
//   6. Scroll-driven section transitions (GSAP)
//   7. Fog + depth atmosphere
//   8. Responsiveness & performance
// ============================================

(() => {
    'use strict';

    // ─── DOM ───
    const container = document.getElementById('three-container');

    // ─── Scene ───
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050510, 0.06); // Subtle depth fog

    // ─── Camera ───
    const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        100
    );
    camera.position.z = 6;

    // ─── Renderer ───
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);


    // ============================================
    // LIGHTING — AI/FUTURISTIC STYLE
    // ============================================

    // Low ambient for base visibility
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
    scene.add(ambientLight);

    // Main directional (key light)
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);

    // Blue accent — left/top
    const bluePoint = new THREE.PointLight(0x4F46E5, 2.5, 20);
    bluePoint.position.set(-4, 3, 4);
    scene.add(bluePoint);

    // Purple accent — right/bottom
    const purplePoint = new THREE.PointLight(0x9333EA, 2.5, 20);
    purplePoint.position.set(4, -2, 3);
    scene.add(purplePoint);

    // Pink rim light — behind
    const rimLight = new THREE.PointLight(0xEC4899, 1.5, 15);
    rimLight.position.set(0, 0, -5);
    scene.add(rimLight);

    // Subtle fill from below for depth
    const fillLight = new THREE.PointLight(0x4F46E5, 0.8, 12);
    fillLight.position.set(0, -4, 2);
    scene.add(fillLight);


    // ============================================
    // MODEL LOADING
    // ============================================
    let model = null;
    let mixer = null;
    const clock = new THREE.Clock();

    // ModelGroup separates scroll transforms from idle animations
    const modelGroup = new THREE.Group();
    const parallaxGroup = new THREE.Group();
    modelGroup.add(parallaxGroup);
    
    modelGroup.visible = false;   // Hidden until intro reveal
    scene.add(modelGroup);

    const loader = new THREE.GLTFLoader();

    loader.load(
        'model/aiu 3d.glb',
        (gltf) => {
            model = gltf.scene;

            // Center geometry
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            model.position.set(-center.x, -center.y, -center.z);

            // Scale up for visual impact (increased by ~20% for hero presence)
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const desiredScale = (2.5 / maxDim) * 1.2; 
            model.scale.setScalar(desiredScale);

            parallaxGroup.add(model);

            // Play embedded animations
            if (gltf.animations && gltf.animations.length > 0) {
                mixer = new THREE.AnimationMixer(model);
                gltf.animations.forEach((clip) => {
                    mixer.clipAction(clip).play();
                });
            }

            // Set initial transform to hero state
            const heroState = states[0];
            gsap.set(modelGroup.position, heroState.position);
            gsap.set(modelGroup.rotation, heroState.rotation);

            // Start fully transparent (for intro fade-in)
            setModelOpacity(0);

            // Notify the intro system
            if (window.IntroSystem) {
                window.IntroSystem.onModelLoaded();
            }
        },
        (xhr) => {
            // Optional: progress callback
        },
        (error) => {
            console.error('3D Model load error:', error);
            // Still allow intro to proceed
            if (window.IntroSystem) {
                window.IntroSystem.onModelLoaded();
            }
        }
    );


    // ─── Helper: Set model opacity (for fade-in) ───
    const setModelOpacity = (opacity) => {
        if (!model) return;
        model.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.transparent = true;
                child.material.opacity = opacity;
            }
        });
    };


    // ============================================
    // INTRO INTEGRATION — MODEL REVEAL
    // ============================================

    // Listen for the reveal event from intro-cinematic.js
    window.addEventListener('intro:reveal-model', () => {
        modelGroup.visible = true;

        // Fade in model over 2s
        const fadeObj = { opacity: 0 };
        gsap.to(fadeObj, {
            opacity: 1,
            duration: 2,
            ease: 'power2.out',
            onUpdate: () => setModelOpacity(fadeObj.opacity)
        });

        // Glow pulse — animate lights intensity (yoyo)
        gsap.to(bluePoint, {
            intensity: 4,
            duration: 1.2,
            yoyo: true,
            repeat: 1,
            ease: 'power2.inOut'
        });
        gsap.to(purplePoint, {
            intensity: 4,
            duration: 1.2,
            yoyo: true,
            repeat: 1,
            ease: 'power2.inOut',
            delay: 0.2
        });
    });


    // ============================================
    // TRANSFORM STATES — SCROLL-DRIVEN POSITIONS
    // ============================================
    // Model alternates LEFT ↔ RIGHT between sections

    const states = [
        {
            id: "hero",
            position: { x: -1.8, y: 0, z: 2 }, // LEFT side
            rotation: { x: 0, y: 0.5, z: 0 }
        },
        {
            id: "features",
            position: { x: 1.5, y: 0.2, z: 2.5 },
            rotation: { x: 0.2, y: 1.8, z: 0 }
        },
        {
            id: "workflow",
            position: { x: -1.5, y: -0.2, z: 2.2 },
            rotation: { x: 0, y: 3.2, z: 0.1 }
        },
        {
            id: "impact",
            position: { x: 1.5, y: 0.3, z: 1.8 },
            rotation: { x: 0.3, y: 4.5, z: 0 }
        },
        {
            id: "final",
            position: { x: 0, y: 0, z: 1.6 },
            rotation: { x: 0, y: 6.2, z: 0 }
        }
    ];


    // ============================================
    // SCROLL DETECTION + GSAP TRANSITIONS
    // ============================================
    let currentStateId = "hero";
    let scrollEnabled = false;

    // Enable scroll tracking after intro completes
    window.addEventListener('intro:complete', () => {
        scrollEnabled = true;
    });

    window.addEventListener('scroll', () => {
        if (!scrollEnabled || !modelGroup) return;

        let activeStateId = currentStateId;
        let minDistance = Infinity;
        const centerY = window.innerHeight / 2;

        // Find the closest section to center of viewport
        states.forEach(state => {
            const el = document.getElementById(state.id);
            if (el) {
                const rect = el.getBoundingClientRect();
                const elCenter = rect.top + rect.height / 2;
                const dist = Math.abs(centerY - elCenter);

                if (dist < minDistance) {
                    minDistance = dist;
                    activeStateId = state.id;
                }
            }
        });

        // Animate only when state changes (no repeated triggers)
        if (activeStateId !== currentStateId) {
            currentStateId = activeStateId;
            const target = states.find(s => s.id === currentStateId);

            if (target) {
                // Cinematic position transition
                gsap.to(modelGroup.position, {
                    x: target.position.x,
                    y: target.position.y,
                    z: target.position.z,
                    duration: 1.5,
                    ease: 'power3.out',
                    overwrite: 'auto'
                });

                // Cinematic rotation transition
                gsap.to(modelGroup.rotation, {
                    x: target.rotation.x,
                    y: target.rotation.y,
                    z: target.rotation.z,
                    duration: 1.5,
                    ease: 'power3.out',
                    overwrite: 'auto'
                });
            }
        }
    });


    // ============================================
    // MOUSE INTERACTION — PARALLAX
    // ============================================
    const mouseTarget  = { x: 0, y: 0 };
    const mouseCurrent = { x: 0, y: 0 };
    const MOUSE_LERP   = 0.06;  // Smooth lerp factor
    const MOUSE_RANGE  = 0.3;   // Max rotation range in radians

    window.addEventListener('mousemove', (e) => {
        // Normalize mouse to -1..1
        mouseTarget.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouseTarget.y = (e.clientY / window.innerHeight) * 2 - 1;
    });


    // ============================================
    // ANIMATION LOOP
    // ============================================
    const idleTime = { value: 0 };

    const animate = () => {
        requestAnimationFrame(animate);

        const delta = clock.getDelta();
        idleTime.value += delta;

        // Update embedded animations
        if (mixer) {
            mixer.update(delta);
        }

        if (model) {
            // ─── Base Idle: Slow continuous Y rotation ───
            model.rotation.y += 0.003;

            // ─── Base Idle: Floating sine wave ───
            model.position.y += Math.sin(idleTime.value * 1.5) * 0.001;
        }

        // ─── Mouse Parallax (smooth lerp) ───
        mouseCurrent.x += (mouseTarget.x - mouseCurrent.x) * MOUSE_LERP;
        mouseCurrent.y += (mouseTarget.y - mouseCurrent.y) * MOUSE_LERP;

        if (parallaxGroup && modelGroup.visible) {
            // Apply mouse rotation offset to the parallax group
            // This is additive to the continuous rotation on the model
            parallaxGroup.rotation.x += (mouseCurrent.y * MOUSE_RANGE - parallaxGroup.rotation.x) * 0.1;
            parallaxGroup.rotation.y += (mouseCurrent.x * MOUSE_RANGE - parallaxGroup.rotation.y) * 0.1;
        }

        // ─── Animate accent lights subtly ───
        const lightTime = idleTime.value * 0.5;
        bluePoint.position.x  = -4 + Math.sin(lightTime) * 0.5;
        bluePoint.position.y  = 3 + Math.cos(lightTime * 0.8) * 0.3;
        purplePoint.position.x = 4 + Math.cos(lightTime * 0.7) * 0.5;
        purplePoint.position.y = -2 + Math.sin(lightTime * 0.9) * 0.3;

        renderer.render(scene, camera);
    };

    animate();


    // ============================================
    // RESPONSIVENESS
    // ============================================
    const handleResize = () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Adjust scale for mobile
        if (window.innerWidth < 768) {
            modelGroup.scale.set(0.6, 0.6, 0.6);
        } else if (window.innerWidth < 1024) {
            modelGroup.scale.set(0.8, 0.8, 0.8);
        } else {
            modelGroup.scale.set(1, 1, 1);
        }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

})();
