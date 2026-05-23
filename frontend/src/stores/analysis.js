import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAnalysisStore = defineStore('analysis', () => {
  const modelType = ref('v6')
  const isLoading = ref(false)
  const predictionResult = ref(null)
  const selectedAOI = ref(null)
  const classificationLayer = ref(null)
  const error = ref(null)
  const uploadedFiles = ref([])
  const dataReady = ref(false)

  function addFiles(files) {
    for (const f of files) {
      if (!uploadedFiles.value.find(x => x.name === f.name)) {
        uploadedFiles.value.push({ name: f.name, size: f.size, file: f })
      }
    }
    dataReady.value = uploadedFiles.value.length > 0
  }

  function removeFile(name) {
    uploadedFiles.value = uploadedFiles.value.filter(f => f.name !== name)
    dataReady.value = uploadedFiles.value.length > 0
  }

  function clearFiles() {
    uploadedFiles.value = []
    dataReady.value = false
  }

  async function runInference(payload) {
    isLoading.value = true
    error.value = null
    try {
      // If files are uploaded, use FormData upload endpoint
      if (uploadedFiles.value.length > 0) {
        const formData = new FormData()
        for (const f of uploadedFiles.value) {
          formData.append('files', f.file, f.name)
        }
        const res = await fetch(`/api/predict/${modelType.value}/upload`, {
          method: 'POST',
          body: formData
        })
        if (!res.ok) {
          const msg = await res.text()
          throw new Error(`${res.status}: ${msg || res.statusText}`)
        }
        const data = await res.json()
        predictionResult.value = data
        return data
      }

      // Fallback: synthetic data inference via JSON
      const res = await fetch(`/api/predict/${modelType.value}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(`${res.status}: ${msg || res.statusText}`)
      }
      const data = await res.json()
      predictionResult.value = data
      return data
    } catch (e) {
      error.value = e.message
      predictionResult.value = null
    } finally {
      isLoading.value = false
    }
  }

  return { modelType, isLoading, predictionResult, selectedAOI, classificationLayer,
           error, uploadedFiles, dataReady, addFiles, removeFile, clearFiles, runInference }
})
