// ============================================
// THREE.JS CINEMATIC 3D SYSTEM — DUAL MODEL
// ============================================
// Full-featured 3D layer with:
//   1. Scene, Camera, Renderer setup
//   2. AI-themed lighting (blue/purple)
//   3. TWO GLB models: AIU + 1930s Camera
//   4. Cinematic model swap on button click
//   5. Base idle animation (rotation + float)
//   6. Mouse parallax interaction (lerp)
//   7. Scroll-driven section transitions (GSAP)
//   8. Demo sub-section scroll states for camera model
//   9. Fog + depth atmosphere
//  10. Responsiveness & performance
// ============================================

(() => {
    'use strict';

    // ─── DOM ───
    const container = document.getElementById('three-container');

    // ─── Scene ───
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050510, 0.06);

    // ─── Camera ───
    const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        100
    );
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);

    // ─── Renderer (optimized) ───
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false, // big GPU savings
        powerPreference: 'high-performance'
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // capped DPR
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.domElement.style.zIndex = "0";
    container.appendChild(renderer.domElement);


    // ============================================
    // LIGHTING — OPTIMIZED (5 lights instead of 9)
    // ============================================

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight.position.set(3, 4, 5);
    scene.add(dirLight);

    const bluePoint = new THREE.PointLight(0x38BDF8, 2.5, 20);
    bluePoint.position.set(-4, 3, 4);
    scene.add(bluePoint);

    const purplePoint = new THREE.PointLight(0x818CF8, 2.5, 20);
    purplePoint.position.set(4, -2, 3);
    scene.add(purplePoint);

    // Extra highlight for camera model (starts dim, activates on swap)
    const cameraHighlight = new THREE.SpotLight(0xFFE4B5, 0, 18, Math.PI / 6, 0.5, 1);
    cameraHighlight.position.set(0, 5, 5);
    scene.add(cameraHighlight);
    scene.add(cameraHighlight.target);

    // Warm rim reused for camera model
    const warmRim = new THREE.PointLight(0xFFA726, 0, 14);
    warmRim.position.set(-3, 2, -3);
    scene.add(warmRim);


    // ============================================
    // MODEL LOADING — BOTH MODELS
    // ============================================
    let aiuModel = null;
    let cameraModel = null;
    let aiuMixer = null;
    let cameraMixer = null;
    const clock = new THREE.Clock();

    // Track loading state for robust transition
    let aiuLoaded = false;
    let cameraLoaded = false;

    // ─── AIU Model Group ───
    const aiuGroup = new THREE.Group();
    const aiuParallaxGroup = new THREE.Group();
    aiuGroup.add(aiuParallaxGroup);
    aiuGroup.visible = false;
    scene.add(aiuGroup);

    // ─── Camera Model Group ───
    const cameraGroup = new THREE.Group();
    const cameraParallaxGroup = new THREE.Group();
    cameraGroup.add(cameraParallaxGroup);
    cameraGroup.visible = false;
    scene.add(cameraGroup);

    // Transition flag
    let switched = false;
    let transitionInProgress = false;

    const loader = new THREE.GLTFLoader();

    // ─── Helper: Set model opacity (traverse all meshes) ───
    // IMPORTANT: Materials are cloned per-mesh during load to prevent
    // shared-material conflicts when animating opacity independently.
    const setModelOpacity = (targetModel, opacity) => {
        if (!targetModel) return;
        targetModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.transparent = true;
                child.material.opacity = opacity;
                // Ensure proper depth sorting for transparent objects
                child.material.depthWrite = opacity > 0.99;
            }
        });
    };

    // ─── Helper: Clone materials so each mesh has its own instance ───
    // This prevents shared-material bugs where changing opacity on one
    // mesh affects another mesh using the same material reference.
    const cloneMaterials = (targetModel) => {
        if (!targetModel) return;
        targetModel.traverse((child) => {
            if (child.isMesh && child.material) {
                if (Array.isArray(child.material)) {
                    child.material = child.material.map(m => m.clone());
                } else {
                    child.material = child.material.clone();
                }
                child.material.transparent = true;
                child.material.needsUpdate = true;
            }
        });
    };

    // ─── Helper: Enhance metalness/roughness for camera model ───
    const enhanceCameraReflections = (targetModel) => {
        if (!targetModel) return;
        targetModel.traverse((child) => {
            if (child.isMesh && child.material) {
                if (child.material.metalness !== undefined) {
                    child.material.metalness = Math.min(child.material.metalness + 0.15, 1.0);
                    child.material.roughness = Math.max(child.material.roughness - 0.1, 0.1);
                }
                child.material.envMapIntensity = 1.5;
            }
        });
    };


    // ──────────────────────────────────────
    // LOAD MODEL 1: AIU 3D
    // ──────────────────────────────────────
    let modelsLoaded = 0;
    const totalModels = 2;

    const checkAllModelsLoaded = () => {
        modelsLoaded++;
        if (modelsLoaded >= totalModels) {
            if (window.IntroSystem) {
                window.IntroSystem.onModelLoaded();
            }
        }
    };

    loader.load(
        'model/aiu%203d.glb',
        (gltf) => {
            aiuModel = gltf.scene;

            const box = new THREE.Box3().setFromObject(aiuModel);
            const center = box.getCenter(new THREE.Vector3());
            aiuModel.position.set(-center.x, -center.y, -center.z);
            aiuModel.position.x = 0.5;

            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const desiredScale = (2.5 / maxDim) * 0.85;
            aiuModel.scale.setScalar(desiredScale);

            // Clone materials for independent opacity control
            cloneMaterials(aiuModel);

            aiuParallaxGroup.add(aiuModel);

            if (gltf.animations && gltf.animations.length > 0) {
                aiuMixer = new THREE.AnimationMixer(aiuModel);
                gltf.animations.forEach((clip) => {
                    aiuMixer.clipAction(clip).play();
                });
            }

            // Set initial transform to hero state
            const heroState = aiuStates[0];
            gsap.set(aiuGroup.position, heroState.position);
            gsap.set(aiuGroup.rotation, heroState.rotation);

            setModelOpacity(aiuModel, 0);

            aiuLoaded = true;
            console.log('[3D] ✅ AIU model loaded successfully');

            checkAllModelsLoaded();
        },
        (xhr) => {
            if (xhr.total) {
                const pct = Math.round((xhr.loaded / xhr.total) * 100);
                if (pct % 25 === 0) console.log(`[3D] AIU model loading: ${pct}%`);
            }
        },
        (error) => {
            console.error('[3D] ❌ AIU Model load error:', error);
            checkAllModelsLoaded();
        }
    );


    // ──────────────────────────────────────
    // LOAD MODEL 2: 1930s MOVIE CAMERA
    // ──────────────────────────────────────
    loader.load(
        'model/1930s_movie_camera.glb',
        (gltf) => {
            console.log("MODEL LOADED");
            cameraModel = gltf.scene;

            // ─── DEBUG: Match scale and position ───
            scene.add(cameraModel);

            // Cinematic Scale Adjustment
            cameraModel.scale.set(0.020, 0.020, 0.020);

            // Move closer to camera
            cameraModel.position.set(0, 0, 1.3);

            // Tilt model (~45 degrees)
            cameraModel.rotation.z = Math.PI / 4;
            cameraModel.rotation.y = 0.5;

            cameraModel.visible = false;

            // ─── DEBUG: Fix material ───
            cameraModel.traverse((child) => {
              if (child.isMesh) {
                child.material.transparent = true;
                child.material.opacity = 1;
              }
            });

            if (gltf.animations && gltf.animations.length > 0) {
                cameraMixer = new THREE.AnimationMixer(cameraModel);
                gltf.animations.forEach((clip) => {
                    cameraMixer.clipAction(clip).play();
                });
            }

            cameraLoaded = true;
            checkAllModelsLoaded();
        },
        undefined,
        (error) => {
            console.error("ERROR LOADING MODEL", error);
            checkAllModelsLoaded();
        }
    );


    // ============================================
    // INTRO INTEGRATION — MODEL REVEAL
    // ============================================

    window.addEventListener('intro:reveal-model', () => {
        aiuGroup.visible = true;

        const fadeObj = { opacity: 0 };
        gsap.to(fadeObj, {
            opacity: 1,
            duration: 1.8,
            ease: 'power2.out',
            onUpdate: () => setModelOpacity(aiuModel, fadeObj.opacity)
        });

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
    // TRANSFORM STATES — AIU MODEL (SCROLL-DRIVEN)
    // ============================================

    const aiuStates = [
        {
            id: "hero",
            position: { x: -1.8, y: 0, z: 2 },
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
    // TRANSFORM STATES — CAMERA MODEL (DEMO SUB-SECTIONS)
    // ============================================

    const demoStates = [
        {
            id: "camera-section",
            position: { x: 1.5 },   // RIGHT
            rotationY: 1.5
        },
        {
            id: "voice-section",
            position: { x: -1.5 },  // LEFT
            rotationY: 3.0
        },
        {
            id: "result-section",
            position: { x: 1.5 },   // RIGHT AGAIN
            rotationY: 4.5
        }
    ];


    // ============================================
    // CINEMATIC TRANSITION — CLICK-TRIGGERED
    // ============================================
    // Exposed as window.startModelTransition()
    // Called from interactive-demo.js on button click

    const startModelTransition = () => {
        if (switched || transitionInProgress) {
            console.log('[3D] Transition skipped — already switched or in progress');
            return;
        }

        // ─── GUARD: Ensure camera model is loaded ───
        if (!cameraLoaded || !cameraModel) {
            console.warn('[3D] ⚠ Camera model not loaded yet — deferring transition');
            // Retry every 200ms until model is loaded (max 15s)
            let attempts = 0;
            const maxAttempts = 75;
            const retryInterval = setInterval(() => {
                attempts++;
                if (cameraLoaded && cameraModel) {
                    clearInterval(retryInterval);
                    console.log('[3D] Camera model now loaded — executing transition');
                    executeTransition();
                } else if (attempts >= maxAttempts) {
                    clearInterval(retryInterval);
                    console.error('[3D] ❌ Camera model failed to load after 15s — transition aborted');
                }
            }, 200);
            return;
        }

        executeTransition();
    };

    const executeTransition = () => {
        transitionInProgress = true;
        console.log('[3D] 🎬 Starting cinematic model transition...');

        // ─── 1) AIU MODEL EXIT (LEFT + FADE) ───
        gsap.to(aiuGroup.position, {
            x: -4,
            duration: 1.2,
            ease: "power3.inOut"
        });

        const aiuFade = { opacity: 1 };
        gsap.to(aiuFade, {
            opacity: 0,
            duration: 1,
            onUpdate: () => setModelOpacity(aiuModel, aiuFade.opacity),
            onComplete: () => {
                aiuGroup.visible = false;
                console.log('[3D] AIU model hidden');
            }
        });

        // ─── 2) CAMERA MODEL ENTER (RIGHT → CENTER) ───
        cameraModel.visible = true;
        cameraModel.position.set(3.5, 0, 1.3);

        gsap.to(cameraModel.position, {
            x: 0,
            duration: 1.2,
            ease: "power3.out"
        });

        // ─── 3) SET FLAGS ───
        switched = true;
        transitionInProgress = false;
        console.log('[3D] ✅ Camera model fully visible — transition complete');

        // ─── Enhanced lighting for camera model ───
        gsap.to(cameraHighlight, {
            intensity: 2.0,
            duration: 1.5,
            delay: 0.5,
            ease: "power2.out"
        });

        gsap.to(warmRim, {
            intensity: 1.2,
            duration: 1.5,
            delay: 0.6,
            ease: "power2.out"
        });

        // Shift existing lights for warm vintage tone
        gsap.to(bluePoint, {
            intensity: 1.8,
            duration: 1.5,
            delay: 0.5,
            ease: "power2.out"
        });

        gsap.to(purplePoint, {
            intensity: 1.8,
            duration: 1.5,
            delay: 0.5,
            ease: "power2.out"
        });
    };

    // ─── EXPOSE globally so interactive-demo.js can call it ───
    window.startModelTransition = startModelTransition;


    // ============================================
    // SCROLL DETECTION + GSAP TRANSITIONS
    // ============================================
    let currentStateId = "hero";
    let currentDemoStateId = null;
    let scrollEnabled = false;

    // Throttled scroll handler using rAF
    let scrollTicking = false;
    window.addEventListener('scroll', () => {
        if (!scrollEnabled || scrollTicking) return;
        scrollTicking = true;
        requestAnimationFrame(() => {
            handleScroll();
            scrollTicking = false;
        });
    }, { passive: true });

    function handleScroll() {

        // ─── PRE-SWITCH: AIU model follows main page sections ───
        if (!switched && !transitionInProgress) {
            let activeStateId = currentStateId;
            let minDistance = Infinity;
            const centerY = window.innerHeight / 2;

            aiuStates.forEach(state => {
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

            // Animate only when state changes
            if (activeStateId !== currentStateId) {
                currentStateId = activeStateId;
                const target = aiuStates.find(s => s.id === currentStateId);

                if (target) {
                    gsap.to(aiuGroup.position, {
                        x: target.position.x,
                        y: target.position.y,
                        z: target.position.z,
                        duration: 1.5,
                        ease: 'power3.out',
                        overwrite: 'auto'
                    });

                    gsap.to(aiuGroup.rotation, {
                        x: target.rotation.x,
                        y: target.rotation.y,
                        z: target.rotation.z,
                        duration: 1.5,
                        ease: 'power3.out',
                        overwrite: 'auto'
                    });
                }
            }
        }

        // ─── POST-SWITCH: Camera model follows demo sub-sections ───
        if (switched && !transitionInProgress && cameraModel && cameraModel.visible) {
            const centerY = window.innerHeight / 2;
            let closestDemoId = null;
            let minDemoDist = Infinity;

            demoStates.forEach(state => {
                const el = document.getElementById(state.id);
                if (el) {
                    const rect = el.getBoundingClientRect();
                    const elCenter = rect.top + rect.height / 2;
                    const dist = Math.abs(centerY - elCenter);

                    if (dist < minDemoDist) {
                        minDemoDist = dist;
                        closestDemoId = state.id;
                    }
                }
            });

            if (closestDemoId && closestDemoId !== currentDemoStateId) {
                currentDemoStateId = closestDemoId;
                const target = demoStates.find(s => s.id === currentDemoStateId);

                if (target) {
                    gsap.to(cameraModel.position, {
                        x: target.position.x,
                        duration: 1.2,
                        ease: "power3.out",
                        overwrite: "auto"
                    });

                    gsap.to(cameraModel.rotation, {
                        y: target.rotationY,
                        duration: 1.2,
                        overwrite: "auto"
                    });
                }
            }
        }
    }


    // ============================================
    // MOUSE INTERACTION — USE SHARED gMouse
    // ============================================
    const mouseTarget  = { x: 0, y: 0 };
    const mouseCurrent = { x: 0, y: 0 };
    const MOUSE_LERP   = 0.06;
    const MOUSE_RANGE  = 0.3;

    // No new mousemove listener — read from window.gMouse


    // ============================================
    // ANIMATION LOOP — WITH VISIBILITY PAUSE
    // ============================================
    const idleTime = { value: 0 };
    let threePaused = false;

    document.addEventListener('visibilitychange', () => {
        threePaused = document.hidden;
        if (!threePaused) { clock.getDelta(); animate(); }
    });

    window.addEventListener('intro:complete', () => {
        scrollEnabled = true;
    });

    function applyBaseAnimation(model, time, delta, rotSpeed) {
        model.rotation.y += rotSpeed * (delta * 60);
        model.position.y = Math.sin(time) * 0.08;
    }

    const animate = () => {
        if (threePaused) return;
        requestAnimationFrame(animate);

        const delta = clock.getDelta();
        idleTime.value += delta;

        if (aiuMixer) aiuMixer.update(delta);
        if (cameraMixer) cameraMixer.update(delta);

        if (aiuModel && aiuGroup.visible) {
            applyBaseAnimation(aiuModel, idleTime.value, delta, 0.002);
        }

        if (cameraModel && cameraModel.visible) {
            applyBaseAnimation(cameraModel, idleTime.value, delta, 0.007);
        }

        // Read from shared gMouse (no duplicate listener)
        const gm = window.gMouse || { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        mouseTarget.x = (gm.x / window.innerWidth) * 2 - 1;
        mouseTarget.y = (gm.y / window.innerHeight) * 2 - 1;

        mouseCurrent.x += (mouseTarget.x - mouseCurrent.x) * MOUSE_LERP;
        mouseCurrent.y += (mouseTarget.y - mouseCurrent.y) * MOUSE_LERP;

        if (aiuParallaxGroup && aiuGroup.visible) {
            aiuParallaxGroup.rotation.x += (mouseCurrent.y * MOUSE_RANGE - aiuParallaxGroup.rotation.x) * 0.1;
            aiuParallaxGroup.rotation.y += (mouseCurrent.x * MOUSE_RANGE - aiuParallaxGroup.rotation.y) * 0.1;
        }

        if (cameraParallaxGroup && cameraGroup.visible) {
            cameraParallaxGroup.rotation.x += (mouseCurrent.y * MOUSE_RANGE - cameraParallaxGroup.rotation.x) * 0.1;
            cameraParallaxGroup.rotation.y += (mouseCurrent.x * MOUSE_RANGE - cameraParallaxGroup.rotation.y) * 0.1;
        }

        const lightTime = idleTime.value * 0.5;
        bluePoint.position.x  = -4 + Math.sin(lightTime) * 0.5;
        bluePoint.position.y  = 3 + Math.cos(lightTime * 0.8) * 0.3;
        purplePoint.position.x = 4 + Math.cos(lightTime * 0.7) * 0.5;
        purplePoint.position.y = -2 + Math.sin(lightTime * 0.9) * 0.3;

        if (switched) {
            warmRim.position.x = -3 + Math.sin(lightTime * 0.6) * 1.0;
            warmRim.position.y = 2 + Math.cos(lightTime * 0.4) * 0.5;
        }

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

        let scaleFactor = 1;
        if (window.innerWidth < 768) {
            scaleFactor = 0.6;
        } else if (window.innerWidth < 1024) {
            scaleFactor = 0.8;
        }

        aiuGroup.scale.set(scaleFactor, scaleFactor, scaleFactor);
        cameraGroup.scale.set(scaleFactor, scaleFactor, scaleFactor);
    };

    window.addEventListener('resize', handleResize);
    handleResize();

})();
