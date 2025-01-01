import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Scene, camera and renderer setup
 var scene = new THREE.Scene();
 var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
 var renderer = new THREE.WebGLRenderer();
 renderer.setSize(window.innerWidth, window.innerHeight);
 document.body.appendChild(renderer.domElement);

 // Add a light to the scene
 var light = new THREE.PointLight(0xFFFFFF);
 light.position.set(50, 50, 1000);
 scene.add(light);

 // Load GLTF model
 var loader = new GLTFLoader();
 loader.load('/static/voltaire/Voltaire.gltf', function(gltf) {
     console.log(`Trying to load model`);
     scene.add(gltf.scene);
 }, undefined, function(error) {
     console.error(error);
 });

 // Camera position
 camera.position.z = 400;

 // Animation loop
 function animate() {
     requestAnimationFrame(animate);
     renderer.render(scene, camera);
 }
 animate();