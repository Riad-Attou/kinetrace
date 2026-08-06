import { describe, expect, it } from 'vitest'

import { formatTime, nearestFrame } from './utils'

describe('timeline helpers', () => {
  it('formats a compact studio timestamp', () => {
    expect(formatTime(65_780)).toBe('1:05.7')
  })

  it('selects the nearest motion frame', () => {
    const frames = [0, 33, 67].map((timestampMs) => ({
      timestampMs,
      body: [],
      hands: { left: [], right: [] },
    }))
    expect(nearestFrame(frames, 48)?.timestampMs).toBe(33)
    expect(nearestFrame(frames, 60)?.timestampMs).toBe(67)
  })
})
