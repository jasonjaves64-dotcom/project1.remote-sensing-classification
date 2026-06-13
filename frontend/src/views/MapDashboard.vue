<template>
  <div class="dashboard">
    <header class="toolbar">
      <div class="brand">
        <span class="logo">FCN</span>
        <div>
          <h1>FusionCropNet <span class="ver">V6</span></h1>
          <p class="subtitle">Remote Sensing Crop Classification</p>
        </div>
      </div>

      <div class="controls">
        <select v-model="store.modelType" class="model-select">
          <option value="v5">V5 (Standard)</option>
          <option value="v5edl">V5EDL</option>
          <option value="v5pro">V5Pro (Flagship)</option>
          <option value="v6" selected>V6 (Next-Gen)</option>
          <option value="tsvit">TSViT (Baseline)</option>
        </select>

        <button class="btn btn-primary" :disabled="store.isLoading" @click="runAnalysis">
          <span v-if="store.isLoading" class="spinner"></span>
          {{ store.isLoading ? 'Running...' : 'Run Classification' }}
        </button>
      </div>

      <div class="meta">
        <span class="meta-item">168 tests</span>
        <span class="meta-item dot">49M params</span>
      </div>
    </header>

    <div class="main">
      <div class="map-area">
        <MapView @aoi-ready="onAOIReady" />
      </div>

      <aside class="panel">
        <!-- Data Upload -->
        <section class="panel-section upload-section">
          <h3>📂 DATA UPLOAD</h3>
          <div v-if="store.uploadedFiles.length > 0" class="uploaded-list">
            <div v-for="f in store.uploadedFiles" :key="f.name" class="uploaded-file">
              <span class="file-icon">{{ fileIcon(f.name) }}</span>
              <span class="file-name">{{ f.name }}</span>
              <span class="file-size">{{ formatSize(f.size) }}</span>
              <button class="file-remove" @click="removeFile(f.name)">&times;</button>
            </div>
            <div class="upload-actions">
              <button class="btn btn-sm btn-outline" @click="clearFiles">Clear All</button>
              <span class="file-count">{{ store.uploadedFiles.length }} file(s)</span>
            </div>
          </div>
          <div v-else class="state empty">No data uploaded. Upload .tif, .npy, or .zip files.</div>
          <label class="upload-btn">
            <input type="file" multiple accept=".tif,.tiff,.npy,.npz,.zip" @change="onFileSelect" style="display:none" />
            + Add Files
          </label>
        </section>

        <!-- Model Output -->
        <section class="panel-section">
          <h3>MODEL OUTPUT</h3>
          <div v-if="store.error" class="state error">{{ store.error }}</div>
          <div v-else-if="store.isLoading" class="state loading">
            <span class="spinner"></span> Running {{ store.modelType.toUpperCase() }}...
          </div>
          <div v-else-if="!store.predictionResult" class="state empty">
            Draw an AOI then click "Run Classification".
          </div>
          <div v-else class="results">
            <div class="metric-row">
              <span>Dominant Crop</span>
              <span class="val">{{ store.predictionResult.dominant || '—' }}</span>
            </div>
            <div class="metric-row">
              <span>Confidence</span>
              <span class="val">{{ store.predictionResult.confidence || '—' }}%</span>
            </div>
            <div class="metric-row">
              <span>Inference Time</span>
              <span class="val">{{ store.predictionResult.time || '—' }}s</span>
            </div>
            <div class="metric-row" v-if="store.predictionResult.aux?.lai">
              <span>LAI</span>
              <span class="val">{{ store.predictionResult.aux.lai }}</span>
            </div>
            <div class="metric-row" v-if="store.predictionResult.aux?.growth_stage !== undefined">
              <span>Growth Stage</span>
              <span class="val">{{ store.predictionResult.aux.growth_stage }}</span>
            </div>
            <div class="metric-row" v-if="store.predictionResult.aux?.boundary_coverage">
              <span>Boundary %</span>
              <span class="val">{{ store.predictionResult.aux.boundary_coverage }}%</span>
            </div>
          </div>
        </section>

        <!-- Distribution -->
        <section class="panel-section">
          <h3>CLASS DISTRIBUTION</h3>
          <div v-if="store.predictionResult?.distribution" class="bar-chart">
            <div v-for="(pct, cls) in store.predictionResult.distribution" :key="cls" class="bar-row">
              <span class="bar-label">{{ cls }}</span>
              <div class="bar-track">
                <div class="bar-fill" :style="{ width: Math.max(pct, 1) + '%', background: cropColor(cls) }"></div>
              </div>
              <span class="bar-val">{{ pct }}%</span>
            </div>
          </div>
          <div v-else class="state empty">Run classification to see crop distribution.</div>
        </section>

        <!-- V6 Aux -->
        <section class="panel-section" v-if="store.modelType === 'v6'">
          <h3>V6 AUXILIARY OUTPUTS</h3>
          <div v-if="store.predictionResult?.aux" class="aux-grid">
            <div class="aux-item">
              <div class="aux-name">LAI</div>
              <div class="aux-val">{{ store.predictionResult.aux.lai || '—' }}</div>
            </div>
            <div class="aux-item">
              <div class="aux-name">Growth Stage</div>
              <div class="aux-val">{{ store.predictionResult.aux.growth_stage ?? '—' }}</div>
            </div>
            <div class="aux-item">
              <div class="aux-name">Boundary</div>
              <div class="aux-val">{{ store.predictionResult.aux.boundary_coverage || '—' }}%</div>
            </div>
          </div>
          <div v-else class="state empty">V6 multi-task outputs appear here.</div>
        </section>

        <!-- Geometric Invariants -->
        <section class="panel-section" v-if="store.modelType === 'v6' && store.geoInvariants">
          <h3>📐 GEOMETRIC INVARIANTS</h3>
          <div class="aux-grid">
            <div class="aux-item">
              <div class="aux-name">K (Gaussian)</div>
              <div class="aux-val">{{ fmtNum(store.geoInvariants.K_mean) }}</div>
            </div>
            <div class="aux-item">
              <div class="aux-name">H (Mean)</div>
              <div class="aux-val">{{ fmtNum(store.geoInvariants.H_mean) }}</div>
            </div>
            <div class="aux-item">
              <div class="aux-name">κ₁ (Max)</div>
              <div class="aux-val">{{ fmtNum(store.geoInvariants.k1_mean) }}</div>
            </div>
            <div class="aux-item">
              <div class="aux-name">κ₂ (Min)</div>
              <div class="aux-val">{{ fmtNum(store.geoInvariants.k2_mean) }}</div>
            </div>
            <div class="aux-item" style="grid-column:1/-1">
              <div class="aux-name">τ_g (Torsion)</div>
              <div class="aux-val">{{ fmtNum(store.geoInvariants.tau_g_mean) }}</div>
            </div>
          </div>
        </section>

        <!-- Conflict Analysis -->
        <section class="panel-section" v-if="store.modelType === 'v6' && store.conflictAnalysis">
          <h3>🧬 CONFLICT ANALYSIS</h3>
          <div class="metric-row">
            <span>Type</span>
            <span class="val" :class="conflictClass">{{ store.conflictAnalysis.conflict_type || '—' }}</span>
          </div>
          <div class="metric-row">
            <span>κ (Conflict)</span>
            <span class="val">{{ fmtNum(store.conflictAnalysis.kappa) }}</span>
          </div>
          <div class="metric-row">
            <span>H¹ Norm</span>
            <span class="val">{{ fmtNum(store.conflictAnalysis.h1_norm) }}</span>
          </div>
          <div class="bar-chart" style="margin-top:8px">
            <div class="bar-row">
              <span class="bar-label">Noise</span>
              <div class="bar-track"><div class="bar-fill safe" :style="{width: (store.conflictAnalysis.noise_ratio*100).toFixed(1)+'%'}"></div></div>
              <span class="bar-val">{{ (store.conflictAnalysis.noise_ratio*100).toFixed(0) }}%</span>
            </div>
            <div class="bar-row">
              <span class="bar-label">Structural</span>
              <div class="bar-track"><div class="bar-fill warn" :style="{width: (store.conflictAnalysis.structural_ratio*100).toFixed(1)+'%'}"></div></div>
              <span class="bar-val">{{ (store.conflictAnalysis.structural_ratio*100).toFixed(0) }}%</span>
            </div>
            <div class="bar-row">
              <span class="bar-label">HighOrder</span>
              <div class="bar-track"><div class="bar-fill danger" :style="{width: (store.conflictAnalysis.high_order_ratio*100).toFixed(1)+'%'}"></div></div>
              <span class="bar-val">{{ (store.conflictAnalysis.high_order_ratio*100).toFixed(0) }}%</span>
            </div>
          </div>
        </section>

        <!-- TTA Safety -->
        <section class="panel-section" v-if="store.modelType === 'v6' && store.ttaStatus">
          <h3>🛡 TTA SAFETY</h3>
          <div class="metric-row">
            <span>Level</span>
            <span class="val" :class="ttaLevelClass">{{ ttaLevelLabel }}</span>
          </div>
          <div class="metric-row">
            <span>∇ Alignment</span>
            <span class="val">{{ fmtNum(store.ttaStatus.gradient_alignment) }}</span>
          </div>
          <div class="metric-row">
            <span>C_coh</span>
            <span class="val">{{ fmtNum(store.ttaStatus.cohomology_conflict) }}</span>
          </div>
          <div class="metric-row" v-if="store.ttaStatus.total_interventions">
            <span>Interventions</span>
            <span class="val">L1:{{ store.ttaStatus.total_interventions.L1 }} L2:{{ store.ttaStatus.total_interventions.L2 }} L3:{{ store.ttaStatus.total_interventions.L3 }}</span>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useAnalysisStore } from '../stores/analysis'
import MapView from '../components/MapView.vue'

const store = useAnalysisStore()

const CROP_PALETTE = {
  wheat: '#F4A460', corn: '#FFD700', rice: '#7CFC00',
  soybean: '#228B22', cotton: '#FFE4B5', vegetable: '#9370DB', other: '#808080'
}
function cropColor(name) { return CROP_PALETTE[name] || '#666' }

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase()
  if (ext === 'tif' || ext === 'tiff') return '🖼'
  if (ext === 'npy' || ext === 'npz') return '📊'
  if (ext === 'zip') return '📦'
  return '📄'
}
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB'
  return (bytes/1024/1024).toFixed(1) + ' MB'
}
function fmtNum(v) {
  if (v === undefined || v === null) return '—'
  if (Math.abs(v) < 0.0001) return v.toExponential(2)
  return Number(v).toFixed(4)
}

const conflictClass = computed(() => ({
  'Noise': 'safe', 'Structural': 'warn', 'HighOrder': 'danger'
}[store.conflictAnalysis?.conflict_type] || ''))

const ttaLevelLabel = computed(() => {
  const level = store.ttaStatus?.intervention_level || 0
  return ['🟢 Normal', '🟡 Light (halve LR)', '🟠 Paused', '🔴 Rollback'][level] || '—'
})
const ttaLevelClass = computed(() => ['safe', 'warn', 'warn', 'danger'][store.ttaStatus?.intervention_level || 0])

function onFileSelect(e) { store.addFiles(e.target.files); e.target.value = '' }
function removeFile(name) { store.removeFile(name) }
function clearFiles() { store.clearFiles() }

function runAnalysis() {
  store.runInference({ aoi: store.selectedAOI })
}

function onAOIReady() {
  // Auto-run inference when AOI is drawn
  if (store.selectedAOI) runAnalysis()
}
</script>

<style scoped>
/* ── Layout ── */
.dashboard { display:flex; flex-direction:column; height:100vh }
.toolbar {
  display:flex; align-items:center; gap:24px; padding:10px 20px;
  background:#14161b; border-bottom:1px solid #2a2d35; flex-shrink:0
}
.main { display:flex; flex:1; min-height:0; overflow:hidden }
.map-area { flex:1; min-width:0 }

/* ── Toolbar ── */
.brand { display:flex; align-items:center; gap:12px }
.logo { display:flex; align-items:center; justify-content:center; width:36px; height:36px; background:#e63946; color:#fff; font-weight:900; font-size:12px; border-radius:2px; flex-shrink:0 }
.brand h1 { font-size:17px; font-weight:700; color:#f0f0f0; line-height:1.1 }
.ver { color:#e63946; font-size:12px }
.subtitle { font-size:10px; color:#777 }
.controls { display:flex; gap:10px; margin-left:auto }
.model-select { background:#1e2128; color:#ccc; border:1px solid #333; padding:7px 10px; border-radius:2px; font-size:12px; cursor:pointer }
.btn { padding:7px 18px; border:none; border-radius:2px; font-size:12px; font-weight:600; cursor:pointer; display:flex; align-items:center; gap:8px }
.btn-primary { background:#e63946; color:#fff }
.btn-primary:disabled { opacity:0.5; cursor:not-allowed }
.meta { display:flex; gap:16px; margin-left:auto }
.meta-item { font-size:10px; color:#666; font-weight:600 }
.dot::before { content:'●'; margin-right:5px; color:#e63946; font-size:7px }

/* ── Panel ── */
.panel { width:320px; background:#14161b; border-left:1px solid #2a2d35; overflow-y:auto; flex-shrink:0 }
.panel-section { padding:18px 20px; border-bottom:1px solid #2a2d35 }
.panel-section h3 { font-size:10px; font-weight:700; color:#666; letter-spacing:0.1em; margin-bottom:14px }
.state { font-size:12px; line-height:1.6; padding:12px; border-radius:2px }
.upload-section { background:rgba(230,57,70,0.03) }
.uploaded-list { display:flex; flex-direction:column; gap:6px; margin-bottom:10px }
.uploaded-file { display:flex; align-items:center; gap:8px; padding:6px 8px; background:#1e2128; border-radius:2px; font-size:11px }
.file-icon { flex-shrink:0 }
.file-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#ccc }
.file-size { color:#777; font-size:10px; flex-shrink:0 }
.file-remove { background:none; border:none; color:#e63946; cursor:pointer; font-size:14px; padding:0 4px }
.upload-actions { display:flex; align-items:center; justify-content:space-between }
.file-count { font-size:10px; color:#555 }
.upload-btn { display:block; width:100%; padding:8px; text-align:center; background:#1e2128; color:#999; border:1px dashed #333; border-radius:2px; font-size:11px; cursor:pointer; transition:all 0.15s }
.upload-btn:hover { border-color:#e63946; color:#e63946 }
.btn-sm { padding:4px 10px; font-size:10px }
.btn-outline { background:transparent; color:#777; border:1px solid #333; border-radius:2px; cursor:pointer }
.btn-outline:hover { border-color:#e63946; color:#e63946 }
.state.empty { color:#555 }
.state.error { color:#e63946; background:rgba(230,57,70,0.08); border:1px solid rgba(230,57,70,0.2) }
.state.loading { color:#999; display:flex; align-items:center; gap:10px }

/* ── Results ── */
.metric-row { display:flex; justify-content:space-between; padding:7px 0; font-size:12px; border-bottom:1px solid #1e2128 }
.metric-row .val { font-weight:700; color:#e63946 }

/* ── Bar Chart ── */
.bar-chart { padding-top:4px }
.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:8px }
.bar-label { font-size:11px; color:#aaa; width:72px; text-align:right; flex-shrink:0 }
.bar-track { flex:1; height:14px; background:#1e2128; border-radius:2px; overflow:hidden }
.bar-fill { height:100%; border-radius:2px; transition:width 0.3s; min-width:2px }
.bar-val { font-size:10px; color:#888; width:40px; flex-shrink:0; text-align:right }

/* ── Aux Grid ── */
.aux-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px }
.aux-item { background:#1e2128; padding:10px; border-radius:2px; text-align:center }
.aux-name { font-size:9px; color:#777; margin-bottom:4px }
.aux-val { font-size:13px; font-weight:700; color:#f0f0f0 }

/* ── Spinner ── */
.spinner { width:14px; height:14px; border:2px solid rgba(255,255,255,0.2); border-top-color:#fff; border-radius:50%; animation:spin 0.6s linear infinite; display:inline-block }
@keyframes spin { to { transform:rotate(360deg) } }

/* ── Conflict/TTA Colors ── */
.val.safe { color:#4ade80 }
.val.warn { color:#f59e0b }
.val.danger { color:#ef4444 }
.bar-fill.safe { background:#4ade80 }
.bar-fill.warn { background:#f59e0b }
.bar-fill.danger { background:#ef4444 }

@media (max-width:900px) {
  .panel { width:280px }
  .brand h1 { font-size:14px }
}
</style>
