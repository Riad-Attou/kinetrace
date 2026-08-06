import type { MotionFrame } from './types'

export function formatTime(milliseconds: number): string {
  const totalSeconds = Math.max(milliseconds, 0) / 1000
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  const fraction = Math.floor((totalSeconds % 1) * 10)
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${fraction}`
}

export function nearestFrame(frames: MotionFrame[], milliseconds: number): MotionFrame | null {
  if (!frames.length) return null
  let low = 0
  let high = frames.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (frames[middle].timestampMs < milliseconds) low = middle + 1
    else high = middle
  }
  if (low > 0) {
    const before = frames[low - 1]
    const after = frames[low]
    return milliseconds - before.timestampMs <= after.timestampMs - milliseconds ? before : after
  }
  return frames[low]
}
