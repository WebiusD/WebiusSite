import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// renderer setup
 const renderer = new THREE.WebGLRenderer({ antialias: true });
 renderer.outputColorSpace = THREE.SRGBColorSpace;
 renderer.setSize(window.innerWidth, window.innerHeight);
 renderer.setClearColor(0x000000);
 renderer.setPixelRatio(window.devicePixelRation);

 // add support for shadows:
 renderer.shadowMap.enabled = true;
 renderer.shadowMap.type = THREE.PCFSoftShadowMap;

 document.body.appendChild(renderer.domElement);

 const scene = new THREE.Scene()
 const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
 camera.position.set(50, 50, 200);
 camera.lookAt(0, 0, 0);

 // Orbital contols setup:
 const controls = new OrbitControls(camera, renderer.domElement);
 controls.enableDamping = true;
 controls.enablePan = false;
 controls.minDistance = 80;
 controls.maxDistance = 130;
 controls.minPolarAngle = 0.3;
 controls.maxPolarAngle = 1.5;
 controls.autoRotate = false;
 controls.target = new THREE.Vector3(0, 1, 0);
 controls.update();

 const planeGeo = new THREE.PlaneGeometry(350, 350, 350, 350);
 planeGeo.rotateX(-Math.PI / 2);
 const planeMaterial = new THREE.MeshStandardMaterial({
    color: 0x555555,
    side: THREE.DoubleSide,
 });
 const planeMesh = new THREE.Mesh(planeGeo, planeMaterial);
 planeMesh.castShadow = false;
 planeMesh.receiveShadow = true;
 scene.add(planeMesh);

 // Add a light to the scene
 const light = new THREE.SpotLight(0xFFFFFF, 8, 0, Math.PI / 2, 0.2, 0.5);
 light.position.set(-30, 100, 0);
 light.castShadow = true;
 light.shadow.mapSize.width = 2048; // Adjust shadow map size
 light.shadow.mapSize.height = 2048; // Adjust shadow map size
 scene.add(light);

 //Create a helper for the shadow camera (optional)
const helper = new THREE.CameraHelper( light.shadow.camera );
scene.add( helper );

 // ambient light:
// var ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.5); // Color: White, Intensity: 0.5
// scene.add(ambientLight);

 // Load GLTF model
 var loader = new GLTFLoader();
 loader.load('/static/voltaire/Voltaire.glb', function(gltf) {
    var model = gltf.scene;

    // enable shadowing within the model itself:
    model.traverse((child) => {
        if(child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;

            var childName = child.name;
            console.log(`Child name: ${childName}`);
            // Change material color of the mesh
            child.material.color.set(0xff0000);

        }
    });

    model.rotateX(-Math.PI / 2);
    model.position.set(0, 20, 0);

    // Calculate the center of the model's bounding box
    var boundingBox = new THREE.Box3().setFromObject(model);
    var center = boundingBox.getCenter(new THREE.Vector3());

    // Calculate the distance from the camera to the center of the model
    var distance = boundingBox.getSize(new THREE.Vector3()).length();

    // Set the camera position to be behind the model
    // camera.position.copy(center).add(new THREE.Vector3(50, 50, distance));

    // Set the camera's look-at target to the center of the model
    // camera.lookAt(center);

    // Add the model to the scene
    scene.add(model);
 }, undefined, function(error) {
     console.error(error);
 });

 // Animation loop
 function animate() {
     requestAnimationFrame(animate);
     controls.update();
     renderer.render(scene, camera);
 }
 animate();