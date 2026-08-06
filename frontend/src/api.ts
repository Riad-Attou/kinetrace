import type { Health, Job, MotionResult } from './types'

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      message = payload.detail ?? message
    } catch {
      // Keep the status-based fallback for non-JSON responses.
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export async function getHealth(): Promise<Health> {
  return responseJson<Health>(await fetch('/api/health'))
}

export async function createJob(file: File): Promise<Job> {
  const body = new FormData()
  body.append('video', file)
  return responseJson<Job>(await fetch('/api/jobs', { method: 'POST', body }))
}

export async function getJob(jobId: string): Promise<Job> {
  return responseJson<Job>(await fetch(`/api/jobs/${jobId}`))
}

export async function getMotion(jobId: string): Promise<MotionResult> {
  return responseJson<MotionResult>(await fetch(`/api/jobs/${jobId}/result`))
}
