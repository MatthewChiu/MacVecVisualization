import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/* ----------------------------------------------------------------------
   Data loading
------------------------------------------------------------------------ */

const DATA_URL = "data/data.json";
const VECTORS_URL = "data/vectors.bin";
const PCA_URL = "data/pca3.bin";
const TSNE_URL = "data/tsne2.bin";
const DIM = 32;
const SPATIAL_SCALE = 62;

let N = 0;
let pc = [];        // array of arrays, pitch-class-count vectors (length 12 each)
let sizeArr = [];   // macroharmony "size" = number of nonzero pitch classes
let totalArr = [];  // sum of the pc vector (note count incl. doublings)
let rootArr = [];   // weight at index 0 (the tonic)
let countArr = [];  // corpus frequency
let vectors = null; // Float32Array, N*32, the raw embedding
let vecNorms = null;// Float32Array, N, precomputed L2 norms
let pcaPos = null;  // Float32Array, N*3
let tsnePos = null; // Float32Array, N*2

async function loadAll() {
  const [meta, vecBuf, pcaBuf, tsneBuf] = await Promise.all([
    fetch(DATA_URL).then(r => r.json()),
    fetch(VECTORS_URL).then(r => r.arrayBuffer()),
    fetch(PCA_URL).then(r => r.arrayBuffer()),
    fetch(TSNE_URL).then(r => r.arrayBuffer()),
  ]);
  pc = meta.pc;
  sizeArr = meta.size;
  totalArr = meta.total;
  rootArr = meta.root;
  countArr = meta.count;
  N = pc.length;
  vectors = new Float32Array(vecBuf);
  pcaPos = new Float32Array(pcaBuf);
  tsnePos = new Float32Array(tsneBuf);

  vecNorms = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const off = i * DIM;
    for (let d = 0; d < DIM; d++) { const v = vectors[off + d]; s += v * v; }
    vecNorms[i] = Math.sqrt(s);
  }
}

/* ----------------------------------------------------------------------
   Color scales
------------------------------------------------------------------------ */

const BRASS = new THREE.Color(0xd4a24e);
const TEAL = new THREE.Color(0x5fb8ae);
const VIOLET = new THREE.Color(0x9a86c9);
const DIM_COLOR = new THREE.Color(0x33323f);

function sizeColor(sz) {
  // categorical-ish ramp across the observed size range 2..12
  const t = THREE.MathUtils.clamp((sz - 2) / 10, 0, 1);
  const c = new THREE.Color();
  if (t < 0.5) c.lerpColors(TEAL, BRASS, t / 0.5);
  else c.lerpColors(BRASS, VIOLET, (t - 0.5) / 0.5);
  return c;
}

/* ----------------------------------------------------------------------
   Filter state
------------------------------------------------------------------------ */

const filterState = {
  sizeSet: null,     // null = all, else Set of ints
  minCount: 0,
  maxCount: Infinity,
  diatonicOnly: false,
  query: "",
};

function matchesFilter(i) {
  if (filterState.sizeSet && !filterState.sizeSet.has(sizeArr[i])) return false;
  if (countArr[i] < filterState.minCount) return false;
  if (countArr[i] > filterState.maxCount) return false;
  if (filterState.diatonicOnly && !isDiatonicScale[i]) return false;
  if (filterState.query) {
    const label = pc[i].join(",");
    if (!label.includes(filterState.query)) return false;
  }
  return true;
}

/* ----------------------------------------------------------------------
   Diatonic scale detection

   Tonic-ranking collapses every transposition of a given mode onto one
   vector (C major, G major, D major... all reduce to the same Ionian
   pattern), but the seven diatonic modes themselves are NOT the same
   pattern relative to their own tonic — each starts the W-W-H-W-W-W-H
   sequence from a different scale degree. These are the seven resulting
   sets of active scale-degree positions (0 = tonic).
------------------------------------------------------------------------ */

const DIATONIC_MODE_SETS = [
  [0, 2, 4, 5, 7, 9, 11],  // Ionian
  [0, 2, 3, 5, 7, 9, 10],  // Dorian
  [0, 1, 3, 5, 7, 8, 10],  // Phrygian
  [0, 2, 4, 6, 7, 9, 11],  // Lydian
  [0, 2, 4, 5, 7, 9, 10],  // Mixolydian
  [0, 2, 3, 5, 7, 8, 10],  // Aeolian
  [0, 1, 3, 5, 6, 8, 10],  // Locrian
].map((arr) => new Set(arr));

function activePositions(vec) {
  const s = new Set();
  for (let k = 0; k < 12; k++) if (vec[k] > 0) s.add(k);
  return s;
}
function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

let isDiatonicScale = null; // Uint8Array, built once data is loaded

function computeDiatonicFlags() {
  isDiatonicScale = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const active = activePositions(pc[i]);
    isDiatonicScale[i] = DIATONIC_MODE_SETS.some((s) => setsEqual(active, s)) ? 1 : 0;
  }
}



/* ----------------------------------------------------------------------
   Three.js scene
------------------------------------------------------------------------ */

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0b10);
scene.fog = new THREE.FogExp2(0x0a0b10, 0.0055);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
camera.position.set(0, 0, 170);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.8;
controls.minDistance = 15;
controls.maxDistance = 500;

let geometry, material, points;
let positionAttr, colorAttr, alphaAttr;

function buildPointCloud() {
  geometry = new THREE.BufferGeometry();
  positionAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  colorAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  alphaAttr = new THREE.BufferAttribute(new Float32Array(N), 1);
  geometry.setAttribute("position", positionAttr);
  geometry.setAttribute("aColor", colorAttr);
  geometry.setAttribute("aAlpha", alphaAttr);

  material = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 5.5 } },
    vertexShader: `
      attribute vec3 aColor;
      attribute float aAlpha;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uSize;
      void main() {
        vColor = aColor;
        vAlpha = aAlpha;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uSize * (280.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = length(c);
        if (d > 0.5) discard;
        float edge = smoothstep(0.5, 0.35, d);
        gl_FragColor = vec4(vColor, vAlpha * edge);
      }
    `,
    transparent: true,
    depthWrite: false,
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);
}

function applyLayout(layout) {
  const src = layout === "pca" ? pcaPos : tsnePos;
  const stride = layout === "pca" ? 3 : 2;
  const arr = positionAttr.array;
  for (let i = 0; i < N; i++) {
    arr[i * 3 + 0] = src[i * stride + 0] * SPATIAL_SCALE;
    arr[i * 3 + 1] = src[i * stride + 1] * SPATIAL_SCALE;
    arr[i * 3 + 2] = stride === 3 ? src[i * stride + 2] * SPATIAL_SCALE : 0;
  }
  positionAttr.needsUpdate = true;
}

const colorMode = "size";
const PATH_COLOR = new THREE.Color(0xffffff);
function applyColors() {
  const arr = colorAttr.array;
  const aArr = alphaAttr.array;
  for (let i = 0; i < N; i++) {
    let c = sizeColor(sizeArr[i]);

    const match = matchesFilter(i);
    if (!match) {
      c = DIM_COLOR;
      aArr[i] = 0.045;
    } else {
      aArr[i] = 0.82;
    }
    if (pathSet.has(i)) {
      c = PATH_COLOR;
      aArr[i] = 1.0;
    }
    arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b;
  }
  colorAttr.needsUpdate = true;
  alphaAttr.needsUpdate = true;
}

/* highlight selection / neighbors with a temporary emphasis pass */
let highlightIdxs = new Set();
function applyHighlight() {
  if (highlightIdxs.size === 0) return;
  const arr = colorAttr.array;
  const aArr = alphaAttr.array;
  highlightIdxs.forEach((i) => {
    aArr[i] = 1.0;
  });
  alphaAttr.needsUpdate = true;
}

function resize() {
  const wrap = document.getElementById("canvas-wrap");
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

/* ----------------------------------------------------------------------
   Picking
------------------------------------------------------------------------ */

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 1.6;
const mouseNDC = new THREE.Vector2();

function pickAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  mouseNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);
  raycaster.params.Points.threshold = 1.6 * (camera.position.length() / 170);
  const hits = raycaster.intersectObject(points);
  if (hits.length === 0) return -1;
  // only ever pick a point that currently passes the filter — points hidden
  // by the sidebar filters should be inert to both hover and click
  hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
  for (const h of hits.slice(0, 8)) {
    if (matchesFilter(h.index)) return h.index;
  }
  return -1;
}

const tip = document.getElementById("hover-tip");
let hoverTimer = null;
canvas.addEventListener("mousemove", (e) => {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    const idx = pickAt(e.clientX, e.clientY);
    if (idx === -1) { tip.classList.add("hidden"); return; }
    const rect = canvas.getBoundingClientRect();
    tip.style.left = (e.clientX - rect.left) + "px";
    tip.style.top = (e.clientY - rect.top) + "px";
    tip.innerHTML = `[${pc[idx].join(",")}]<span class="tip-count">×${countArr[idx]}</span>`;
    tip.classList.remove("hidden");
  }, 30);
});
canvas.addEventListener("mouseleave", () => tip.classList.add("hidden"));

let dragDistance = 0;
canvas.addEventListener("pointerdown", () => { dragDistance = 0; });
canvas.addEventListener("pointermove", (e) => {
  if (e.buttons) dragDistance += Math.abs(e.movementX) + Math.abs(e.movementY);
});
canvas.addEventListener("click", (e) => {
  if (dragDistance > 4) return; // was a drag/orbit, not a click
  const idx = pickAt(e.clientX, e.clientY);
  if (idx !== -1) visitPoint(idx);
});

/* ----------------------------------------------------------------------
   Nearest neighbors
------------------------------------------------------------------------ */

function nearestNeighbors(idx, k = 10) {
  const off = idx * DIM;
  const results = [];
  const normI = vecNorms[idx];
  for (let i = 0; i < N; i++) {
    if (i === idx) continue;
    let dot = 0;
    const off2 = i * DIM;
    for (let d = 0; d < DIM; d++) dot += vectors[off + d] * vectors[off2 + d];
    const sim = dot / (normI * vecNorms[i] + 1e-9);
    results.push([i, sim]);
  }
  results.sort((a, b) => b[1] - a[1]);
  return results.slice(0, k);
}

/* ----------------------------------------------------------------------
   Detail panel
------------------------------------------------------------------------ */

const detailEmpty = document.getElementById("detail-empty");
const detailContent = document.getElementById("detail-content");
const pcWheel = document.getElementById("pc-wheel");

const PC_NAMES = ["Tonic","♭2/♯1","2","♭3/♯2","3","4","♯4/♭5","5","♭6/♯5","6","♭7/♯6","7"];

function drawWheel(vec) {
  const cx = 100, cy = 100, rOuter = 82, rInner = 20;
  const maxV = Math.max(1, ...vec);
  let svg = `<g>`;
  svg += `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="none" stroke="var(--border)" stroke-width="1"/>`;
  for (let k = 0; k < 12; k++) {
    const a0 = (Math.PI * 2 * k) / 12 - Math.PI / 2 - (Math.PI / 12);
    const a1 = (Math.PI * 2 * k) / 12 - Math.PI / 2 + (Math.PI / 12);
    const v = vec[k];
    const r = rInner + (rOuter - rInner) * (v / maxV);
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const xo0 = cx + rInner * Math.cos(a0), yo0 = cy + rInner * Math.sin(a0);
    const xo1 = cx + rInner * Math.cos(a1), yo1 = cy + rInner * Math.sin(a1);
    const color = k === 0 ? "var(--brass)" : (v > 0 ? "var(--teal)" : "var(--border-soft)");
    svg += `<path d="M${xo0},${yo0} L${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1} L${xo1},${yo1} Z" fill="${color}" opacity="${v > 0 ? 0.85 : 0.3}"/>`;
    // label
    const la = (Math.PI * 2 * k) / 12 - Math.PI / 2;
    const lx = cx + (rOuter + 14) * Math.cos(la), ly = cy + (rOuter + 14) * Math.sin(la);
    svg += `<text x="${lx}" y="${ly}" font-size="9" fill="var(--text-faint)" font-family="var(--font-mono)" text-anchor="middle" dominant-baseline="middle">${k}</text>`;
  }
  svg += `</g>`;
  pcWheel.innerHTML = svg;
}

let selectedIdx = -1;

function selectPoint(idx) {
  selectedIdx = idx;
  detailEmpty.classList.add("hidden");
  detailContent.classList.remove("hidden");

  drawWheel(pc[idx]);
  document.getElementById("meta-vector").textContent = `[${pc[idx].join(",")}]`;
  document.getElementById("meta-size").textContent = `${sizeArr[idx]} active pitch classes`;
  document.getElementById("meta-count").textContent = `${countArr[idx].toLocaleString()} occurrences`;

  const neighbors = nearestNeighbors(idx, 10);
  const list = document.getElementById("neighbors-list");
  list.innerHTML = "";
  neighbors.forEach(([ni, sim]) => {
    const li = document.createElement("li");
    li.className = "neighbor-item";
    const barWidth = Math.max(2, sim * 40);
    li.innerHTML = `<span class="neighbor-bar" style="width:${barWidth}px"></span>
                     <span class="neighbor-vec">[${pc[ni].join(",")}]</span>
                     <span class="neighbor-sim">${sim.toFixed(3)}</span>`;
    li.addEventListener("click", () => { visitPoint(ni); });
    list.appendChild(li);
  });

  highlightIdxs = new Set([idx, ...neighbors.map((n) => n[0])]);
  applyColors();
  applyHighlight();
  focusCameraOn(idx);
}

document.getElementById("detail-close").addEventListener("click", () => {
  selectedIdx = -1;
  highlightIdxs = new Set();
  detailEmpty.classList.remove("hidden");
  detailContent.classList.add("hidden");
  applyColors();
});

function focusCameraOn(idx) {
  const x = positionAttr.array[idx * 3], y = positionAttr.array[idx * 3 + 1], z = positionAttr.array[idx * 3 + 2];
  const target = new THREE.Vector3(x, y, z);
  const startTarget = controls.target.clone();
  const startTime = performance.now();
  function step() {
    const t = Math.min(1, (performance.now() - startTime) / 500);
    const e = 1 - Math.pow(1 - t, 3);
    controls.target.lerpVectors(startTarget, target, e);
    if (t < 1) requestAnimationFrame(step);
  }
  step();
}

/* ----------------------------------------------------------------------
   Macroharmonic path

   When path mode is on, every click (canvas or neighbor-list) both selects
   the point as usual AND appends it to an ordered path, drawn as a line
   through the current projection. Works in either PCA or t-SNE layout;
   the line is rebuilt from current positions whenever the layout changes.
------------------------------------------------------------------------ */

let pathMode = false;
let pathIndices = [];
let pathSet = new Set();
let pathLine = null;

const pathLineMaterial = new THREE.LineBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.85,
  depthTest: false,
});

function updatePathLine() {
  if (pathLine) {
    scene.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine = null;
  }
  if (pathIndices.length < 2) return;
  const positions = new Float32Array(pathIndices.length * 3);
  pathIndices.forEach((idx, k) => {
    positions[k * 3 + 0] = positionAttr.array[idx * 3 + 0];
    positions[k * 3 + 1] = positionAttr.array[idx * 3 + 1];
    positions[k * 3 + 2] = positionAttr.array[idx * 3 + 2];
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pathLine = new THREE.Line(geo, pathLineMaterial);
  pathLine.renderOrder = 1;
  scene.add(pathLine);
}

function updatePathList() {
  const listEl = document.getElementById("path-list");
  const countEl = document.getElementById("path-count");
  listEl.innerHTML = "";
  pathIndices.forEach((idx, k) => {
    const li = document.createElement("li");
    li.className = "path-item";
    li.innerHTML = `<span class="path-step">${k + 1}</span><span class="path-vec">[${pc[idx].join(",")}]</span>`;
    li.addEventListener("click", () => { selectPoint(idx); focusCameraOn(idx); });
    listEl.appendChild(li);
  });
  const segments = Math.max(0, pathIndices.length - 1);
  countEl.textContent = pathIndices.length === 0
    ? "no nodes yet"
    : `${pathIndices.length} node${pathIndices.length === 1 ? "" : "s"} · ${segments} segment${segments === 1 ? "" : "s"}`;
}

function addToPath(idx) {
  pathIndices.push(idx);
  pathSet.add(idx);
  updatePathLine();
  updatePathList();
  applyColors();
  if (highlightIdxs.size) applyHighlight();
}

function clearPath() {
  pathIndices = [];
  pathSet = new Set();
  updatePathLine();
  updatePathList();
  applyColors();
  if (highlightIdxs.size) applyHighlight();
}

function visitPoint(idx) {
  if (pathMode) addToPath(idx);
  selectPoint(idx);
}

const pathToggleBtn = document.getElementById("path-toggle");
pathToggleBtn.addEventListener("click", () => {
  pathMode = !pathMode;
  pathToggleBtn.classList.toggle("active", pathMode);
  pathToggleBtn.textContent = pathMode ? "Building path — click nodes to add" : "Click to build a path";
});
document.getElementById("path-clear").addEventListener("click", clearPath);

/* ----------------------------------------------------------------------
   UI: layout toggle
------------------------------------------------------------------------ */

let currentLayout = "pca";
document.querySelectorAll("#layout-toggle .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#layout-toggle .seg-btn").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    currentLayout = btn.dataset.layout;
    applyLayout(currentLayout);
    updatePathLine();
    controls.enableRotate = currentLayout === "pca";
    if (currentLayout === "tsne") {
      camera.position.set(controls.target.x, controls.target.y, 140);
    }
  });
});

document.getElementById("reset-view").addEventListener("click", () => {
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, currentLayout === "tsne" ? 140 : 170);
  controls.update();
});

/* ----------------------------------------------------------------------
   UI: sidebar filters
------------------------------------------------------------------------ */

const sizeFilterEl = document.getElementById("size-filter");
function buildSizeChips() {
  const distinct = Array.from(new Set(sizeArr)).sort((a, b) => a - b);
  distinct.forEach((sz) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = sz;
    chip.dataset.size = sz;
    chip.addEventListener("click", () => {
      if (!filterState.sizeSet) filterState.sizeSet = new Set();
      if (filterState.sizeSet.has(sz)) filterState.sizeSet.delete(sz);
      else filterState.sizeSet.add(sz);
      if (filterState.sizeSet.size === 0) filterState.sizeSet = null;
      syncSizeChips();
      refresh();
    });
    sizeFilterEl.appendChild(chip);
  });
}
function syncSizeChips() {
  sizeFilterEl.querySelectorAll(".chip").forEach((chip) => {
    const sz = Number(chip.dataset.size);
    chip.classList.toggle("active", !!(filterState.sizeSet && filterState.sizeSet.has(sz)));
  });
}
document.getElementById("size-clear").addEventListener("click", () => {
  filterState.sizeSet = null;
  syncSizeChips();
  refresh();
});

const searchInput = document.getElementById("search-input");
searchInput.addEventListener("input", () => {
  filterState.query = searchInput.value.trim();
  refresh();
});

const countSlider = document.getElementById("count-slider");
const countValue = document.getElementById("count-value");
let maxCountObserved = 1;
countSlider.addEventListener("input", () => {
  const t = Number(countSlider.value) / 1000; // 0..1
  // exponential mapping for a log-skewed distribution
  const min = Math.round(Math.exp(t * Math.log(maxCountObserved + 1)) - 1);
  filterState.minCount = min;
  countValue.textContent = min.toLocaleString();
  refresh();
});

function refresh() {
  applyColors();
  if (selectedIdx !== -1) applyHighlight();
  updateStats();
}

function updateStats() {
  let shown = 0;
  for (let i = 0; i < N; i++) if (matchesFilter(i)) shown++;
  document.getElementById("stat-shown").textContent = shown.toLocaleString();
  document.getElementById("stat-total").textContent = N.toLocaleString();
}

/* ----------------------------------------------------------------------
   Collections (curated preset views)
------------------------------------------------------------------------ */

function setPreset({ sizeSet, minCount, maxCount, diatonicOnly, query }) {
  filterState.sizeSet = sizeSet ?? null;
  filterState.minCount = minCount ?? 0;
  filterState.maxCount = maxCount ?? Infinity;
  filterState.diatonicOnly = diatonicOnly ?? false;
  filterState.query = query ?? "";
  searchInput.value = filterState.query;
  countSlider.value = 0;
  countValue.textContent = filterState.minCount.toLocaleString();
  syncSizeChips();
  refresh();
}

function buildCollections() {
  const sortedCounts = [...countArr].sort((a, b) => b - a);
  const top300Threshold = sortedCounts[Math.min(299, sortedCounts.length - 1)];

  const collections = [
    { label: "All macroharmonies", fn: () => setPreset({}) },
    { label: "Most common", fn: () => setPreset({ minCount: top300Threshold }) },
    { label: "Diatonic scales", fn: () => setPreset({ diatonicOnly: true }) },
  ];

  const wrap = document.getElementById("collections");
  collections.forEach((c) => {
    const chip = document.createElement("button");
    chip.className = "chip collection";
    chip.textContent = c.label;
    chip.addEventListener("click", c.fn);
    wrap.appendChild(chip);
  });
}

/* ----------------------------------------------------------------------
   Boot
------------------------------------------------------------------------ */

async function boot() {
  await loadAll();
  computeDiatonicFlags();
  buildSizeChips();
  document.getElementById("subtitle").textContent =
    `${N.toLocaleString()} macroharmony vectors, embedded & projected — click around the space`;

  maxCountObserved = 1;
  for (let i = 0; i < countArr.length; i++) if (countArr[i] > maxCountObserved) maxCountObserved = countArr[i];

  buildPointCloud();
  applyLayout(currentLayout);
  applyColors();
  buildCollections();
  updateStats();
  resize();
  animate();

  document.getElementById("loading").style.opacity = "0";
  setTimeout(() => document.getElementById("loading").classList.add("hidden"), 400);
}

boot();
