import { useEffect, type RefObject } from 'react'

import type { Landmark, MotionFrame } from '../types'

type SourceViewerProps = {
  source: string
  frame: MotionFrame | null
  bodyConnections: [number, number][]
  handConnections: [number, number][]
  videoRef: RefObject<HTMLVideoElement | null>
  onPlayState: (playing: boolean) => void
}

const COLORS = {
  body: '#d8ff59',
  left: '#57d3ff',
  right: '#ff6e9f',
}

export function SourceViewer({
  source,
  frame,
  bodyConnections,
  handConnections,
  videoRef,
  onPlayState,
}: SourceViewerProps) {
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const canvas = video.parentElement?.querySelector('canvas')
    if (!(canvas instanceof HTMLCanvasElement)) return
    const context = canvas.getContext('2d')
    if (!context) return

    const width = video.videoWidth || 1280
    const height = video.videoHeight || 720
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
    context.clearRect(0, 0, width, height)
    if (!frame) return

    drawGroup(context, frame.body, bodyConnections, COLORS.body, width, height)
    drawGroup(context, frame.hands.left, handConnections, COLORS.left, width, height)
    drawGroup(context, frame.hands.right, handConnections, COLORS.right, width, height)
  }, [bodyConnections, frame, handConnections, videoRef])

  return (
    <div className="viewer-shell source-viewer">
      <div className="viewer-label">
        <span className="live-dot" /> Source + overlay
      </div>
      <video
        ref={videoRef}
        src={source}
        preload="metadata"
        playsInline
        onPlay={() => onPlayState(true)}
        onPause={() => onPlayState(false)}
        onEnded={() => onPlayState(false)}
      />
      <canvas aria-hidden="true" />
      <div className="overlay-legend">
        <span><i className="legend-body" />Body</span>
        <span><i className="legend-left" />Left hand</span>
        <span><i className="legend-right" />Right hand</span>
      </div>
    </div>
  )
}

function drawGroup(
  context: CanvasRenderingContext2D,
  landmarks: Landmark[],
  connections: [number, number][],
  color: string,
  width: number,
  height: number,
) {
  const points = new Map(landmarks.map((landmark) => [landmark.index, landmark]))
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = Math.max(width / 360, 2)
  context.strokeStyle = color

  for (const [startIndex, endIndex] of connections) {
    const start = points.get(startIndex)
    const end = points.get(endIndex)
    if (!start || !end) continue
    context.globalAlpha = Math.max(Math.min(start.confidence, end.confidence), 0.18)
    context.setLineDash(start.inferred || end.inferred ? [8, 6] : [])
    context.beginPath()
    context.moveTo(start.x * width, start.y * height)
    context.lineTo(end.x * width, end.y * height)
    context.stroke()
  }

  context.setLineDash([])
  for (const point of landmarks) {
    context.globalAlpha = Math.max(point.confidence, 0.24)
    context.fillStyle = point.inferred ? '#ffffff' : color
    context.beginPath()
    context.arc(point.x * width, point.y * height, Math.max(width / 300, 3), 0, Math.PI * 2)
    context.fill()
  }
  context.globalAlpha = 1
}
