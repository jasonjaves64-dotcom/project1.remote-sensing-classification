import { createRouter, createWebHistory } from 'vue-router'
import MapDashboard from '../views/MapDashboard.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'dashboard', component: MapDashboard }
  ]
})
