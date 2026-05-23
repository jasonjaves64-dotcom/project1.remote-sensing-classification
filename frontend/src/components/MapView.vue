<template>
  <div class="map-wrap">
    <div ref="mapDiv" class="map-div"></div>

    <div class="draw-toolbar">
      <button class="tool-btn" :class="{ active: drawMode === 'rectangle' }" @click="toggleDraw('rectangle')">
        &#9633; Rect
      </button>
      <button class="tool-btn" :class="{ active: drawMode === 'polygon' }" @click="toggleDraw('polygon')">
        &#9651; Polygon
      </button>
      <button class="tool-btn danger" v-if="aoiLayer" @click="clearAOI">&times;</button>
    </div>

    <div class="map-status">
      {{ center[0].toFixed(2) }}&deg;N {{ center[1].toFixed(2) }}&deg;E &middot; Z{{ zoom }}
      <span v-if="aoiLayer" style="color:#4ecb71">AOI</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useAnalysisStore } from '../stores/analysis'

const emit = defineEmits(['aoi-ready'])
const store = useAnalysisStore()
const mapDiv = ref(null)
const drawMode = ref(null)
const aoiLayer = ref(null)
const center = ref([39.92, 116.40])
const zoom = ref(5)

let map = null, drawnItems = null, resultGroup = null

const CROP_FILL = {
  wheat: '#F4A460', corn: '#FFD700', rice: '#7CFC00',
  soybean: '#228B22', cotton: '#FFE4B5', vegetable: '#9370DB', other: '#808080'
}

function initMap() {
  const L = window.L
  map = L.map(mapDiv.value).setView([39.92, 116.40], 5)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OSM', maxZoom: 19
  }).addTo(map)

  drawnItems = new L.FeatureGroup()
  map.addLayer(drawnItems)
  resultGroup = L.featureGroup()
  map.addLayer(resultGroup)

  // Draw control
  const drawCtrl = new L.Control.Draw({
    position: 'topright',
    draw: {
      polygon: { allowIntersection: false, showArea: true, shapeOptions: { color: '#e63946' } },
      rectangle: { shapeOptions: { color: '#e63946' } },
      circle: false, circlemarker: false, marker: false, polyline: false
    },
    edit: { featureGroup: drawnItems }
  })
  map.addControl(drawCtrl)

  map.on('moveend', () => { const c = map.getCenter(); center.value = [c.lat, c.lng]; zoom.value = map.getZoom() })

  map.on(L.Draw.Event.CREATED, (e) => {
    drawnItems.clearLayers()
    drawnItems.addLayer(e.layer)
    aoiLayer.value = e.layer
    store.selectedAOI = { type: e.layer.toGeoJSON().geometry.type, coordinates: e.layer.toGeoJSON().geometry.coordinates }
    emit('aoi-ready')
  })
}

function toggleDraw(type) {
  const L = window.L
  if (drawMode.value === type) { drawMode.value = null; return }
  drawMode.value = type
  new L.Draw[type === 'rectangle' ? 'Rectangle' : 'Polygon'](map, {
    shapeOptions: { color: '#e63946', weight: 2, fillOpacity: 0.1 }
  }).enable()
}

function clearAOI() {
  drawnItems?.clearLayers(); resultGroup?.clearLayers()
  aoiLayer.value = null; store.selectedAOI = null; store.predictionResult = null
}

watch(() => store.predictionResult, (result) => {
  resultGroup?.clearLayers()
  if (!result?.geojson?.features) return
  window.L.geoJSON(result.geojson, {
    style: (f) => {
      const crop = f.properties?.crop || 'other'
      const c = CROP_FILL[crop] || '#808080'
      return { fillColor: c, color: c, weight: 1, fillOpacity: 0.5 }
    },
    onEachFeature: (f, layer) => {
      layer.bindPopup(`<b>${f.properties.crop}</b><br>Confidence: ${f.properties.confidence || '—'}%`)
    }
  }).addTo(resultGroup)
}, { deep: true })

onMounted(() => initMap())
onUnmounted(() => { map?.remove() })
</script>

<style scoped>
.map-wrap { width:100%; height:100%; position:relative; min-height:300px }
.map-div { width:100%; height:100% }
.draw-toolbar { position:absolute; top:10px; right:10px; display:flex; gap:6px; z-index:1000 }
.tool-btn { display:inline-flex; align-items:center; gap:4px; padding:6px 11px; background:#14161b; color:#bbb; border:1px solid #333; border-radius:2px; font-size:11px; cursor:pointer }
.tool-btn:hover { background:#1e2128; border-color:#555 }
.tool-btn.active { border-color:#e63946; color:#e63946 }
.tool-btn.danger { color:#e63946; border-color:#e63946 }
.map-status { position:absolute; bottom:8px; left:8px; font-size:10px; color:#777; display:flex; gap:10px; background:rgba(20,22,27,0.85); padding:4px 10px; border-radius:2px; z-index:1000 }
</style>
