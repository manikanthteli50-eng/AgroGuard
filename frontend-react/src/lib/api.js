import axios from 'axios'
import { getToken } from './storage'

/**
 * Backend API client. All data (users, detection history) is persisted in
 * MongoDB through the Flask API. A relative '/api' path is proxied to
 * http://localhost:5000 by Vite; we fall back to absolute URLs otherwise.
 */

const BASES = import.meta.env.VITE_API_URL
  ? [import.meta.env.VITE_API_URL]
  : ['', 'http://localhost:5000', 'http://127.0.0.1:5000']

function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

/** Try each base URL until one responds; throws the last error on failure. */
async function call(method, path, { data, formData, headers } = {}) {
  let lastError = null
  for (const base of BASES) {
    try {
      const res = await axios({
        method,
        url: `${base}${path}`,
        data: formData || data,
        headers: formData
          ? authHeaders({ 'Content-Type': 'multipart/form-data', ...headers })
          : authHeaders({ 'Content-Type': 'application/json', ...headers }),
        timeout: 15000,
      })
      return res.data
    } catch (error) {
      lastError = error
      // A real HTTP response means we reached the server; don't retry other bases.
      if (error.response) break
    }
  }
  throw lastError || new Error('Backend not responding')
}

/** Extract a friendly message from an axios error. */
export function errorMessage(error, fallback = 'Something went wrong') {
  if (error?.response?.data?.error) return error.response.data.error
  if (error?.code === 'ERR_NETWORK') return 'Cannot reach the backend. Is it running on port 5000?'
  return error?.message || fallback
}

/* --------------------------------- Auth --------------------------------- */

export const register = (name, email, password) =>
  call('post', '/api/register', { data: { name, email, password } })

export const login = (email, password) =>
  call('post', '/api/login', { data: { email, password } })

export const fetchMe = () => call('get', '/api/me')

export const updateProfile = (name) =>
  call('patch', '/api/profile', { data: { name } })

/* ------------------------------ Detections ------------------------------ */

export const fetchDetections = () => call('get', '/api/detections')

export const createDetection = (record) =>
  call('post', '/api/detections', { data: record })

export const clearDetections = () => call('delete', '/api/detections')

/* ------------------------------ Prediction ------------------------------ */

export async function predictDisease(file, city) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('city', city || 'New Delhi')
  return call('post', '/api/predict', { formData })
}

export async function checkHealth() {
  try {
    return await call('get', '/api/health')
  } catch {
    return { status: 'unreachable', model_loaded: false, classes: 0, database: 'unreachable' }
  }
}
