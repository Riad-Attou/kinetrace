import { Pause, Play } from 'lucide-react'
import type { CSSProperties } from 'react'

import { formatTime } from '../utils'

type TimelineProps = {
  currentMs: number
  durationMs: number
  playing: boolean
  onToggle: () => void
  onSeek: (milliseconds: number) => void
}

export function Timeline({ currentMs, durationMs, playing, onToggle, onSeek }: TimelineProps) {
  const progress = durationMs > 0 ? Math.min((currentMs / durationMs) * 100, 100) : 0
  return (
    <div className="timeline-panel">
      <button className="play-button" type="button" onClick={onToggle} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
      </button>
      <span className="timecode">{formatTime(currentMs)}</span>
      <div className="range-shell" style={{ '--progress': `${progress}%` } as CSSProperties}>
        <input
          type="range"
          min={0}
          max={Math.max(durationMs, 1)}
          step={1}
          value={Math.min(currentMs, durationMs)}
          onChange={(event) => onSeek(Number(event.target.value))}
          aria-label="Animation timeline"
        />
      </div>
      <span className="timecode muted">{formatTime(durationMs)}</span>
    </div>
  )
}
