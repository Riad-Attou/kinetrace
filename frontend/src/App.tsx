import {
  Activity,
  AlertTriangle,
  Check,
  ChevronRight,
  Download,
  FileJson,
  Fingerprint,
  FolderLock,
  Gauge,
  MonitorUp,
  RefreshCw,
  ScanLine,
  Sparkles,
  Upload,
  Video,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { createJob, getHealth, getJob, getMotion } from './api'
import { SkeletonViewport } from './components/SkeletonViewport'
import { SourceViewer } from './components/SourceViewer'
import { Timeline } from './components/Timeline'
import type { Health, Job, MotionResult } from './types'
import { nearestFrame } from './utils'

const ACCEPTED_VIDEO = '.mp4,.mov,.webm,.mkv,.avi,.m4v,video/*'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [motion, setMotion] = useState<MotionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [currentMs, setCurrentMs] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [looping, setLooping] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'processing')) return
    let busy = false
    const timer = window.setInterval(async () => {
      if (busy) return
      busy = true
      try {
        const updated = await getJob(job.id)
        setJob(updated)
        if (updated.status === 'complete') {
          setMotion(await getMotion(updated.id))
        } else if (updated.status === 'failed') {
          setError(updated.error ?? 'Processing failed')
        }
      } catch (pollError) {
        setError(messageOf(pollError))
      } finally {
        busy = false
      }
    }, 700)
    return () => window.clearInterval(timer)
  }, [job])

  useEffect(() => {
    if (!playing) return
    let animationFrame = 0
    const update = () => {
      const video = videoRef.current
      if (video) setCurrentMs(video.currentTime * 1000)
      animationFrame = requestAnimationFrame(update)
    }
    update()
    return () => cancelAnimationFrame(animationFrame)
  }, [playing])

  const currentFrame = useMemo(
    () => (motion ? nearestFrame(motion.frames, currentMs) : null),
    [currentMs, motion],
  )
  const modelsReady = Boolean(health?.models.pose && health?.models.hands)

  const selectFile = (candidate: File | null) => {
    if (!candidate) return
    setFile(candidate)
    setError(null)
  }

  const startAnalysis = async () => {
    if (!file) return
    setSubmitting(true)
    setError(null)
    try {
      setJob(await createJob(file))
    } catch (uploadError) {
      setError(messageOf(uploadError))
    } finally {
      setSubmitting(false)
    }
  }

  const reset = () => {
    videoRef.current?.pause()
    setFile(null)
    setJob(null)
    setMotion(null)
    setError(null)
    setCurrentMs(0)
    setPlaying(false)
    setLooping(false)
  }

  const togglePlayback = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play()
    else video.pause()
  }

  const seek = (milliseconds: number) => {
    const video = videoRef.current
    if (video) video.currentTime = milliseconds / 1000
    setCurrentMs(milliseconds)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={reset} aria-label="KineTrace home">
          <span className="brand-symbol"><ScanLine size={20} /></span>
          <span>Kine<span>Trace</span></span>
        </button>
        <div className="topbar-status">
          <span className="privacy-badge"><FolderLock size={14} /> Local only</span>
          <span className={`system-light ${health ? 'online' : ''}`} />
          <span className="system-copy">{health ? `Engine ${health.version}` : 'Engine offline'}</span>
        </div>
      </header>

      <main>
        {!job && (
          <Landing
            file={file}
            dragging={dragging}
            engineChecked={health !== null}
            modelsReady={modelsReady}
            submitting={submitting}
            error={error}
            onFile={selectFile}
            onDragging={setDragging}
            onStart={startAnalysis}
          />
        )}

        {job && !motion && (
          <Processing job={job} error={error} onReset={reset} />
        )}

        {job && motion && (
          <Studio
            job={job}
            motion={motion}
            currentFrame={currentFrame}
            currentMs={currentMs}
            playing={playing}
            looping={looping}
            videoRef={videoRef}
            onPlayState={setPlaying}
            onToggle={togglePlayback}
            onToggleLoop={() => setLooping((value) => !value)}
            onSeek={seek}
            onReset={reset}
          />
        )}
      </main>
    </div>
  )
}

type LandingProps = {
  file: File | null
  dragging: boolean
  engineChecked: boolean
  modelsReady: boolean
  submitting: boolean
  error: string | null
  onFile: (file: File | null) => void
  onDragging: (dragging: boolean) => void
  onStart: () => void
}

function Landing({
  file,
  dragging,
  engineChecked,
  modelsReady,
  submitting,
  error,
  onFile,
  onDragging,
  onStart,
}: LandingProps) {
  return (
    <div className="landing">
      <section className="hero-copy">
        <div className="eyebrow"><Sparkles size={14} /> Monocular motion capture</div>
        <h1>Turn movement<br />into <em>motion data.</em></h1>
        <p>
          Drop in a video. KineTrace reconstructs a body-and-finger skeleton you can inspect,
          refine, and take into Blender—entirely on your machine.
        </p>
        <div className="hero-proof">
          <span><Check size={15} /> One actor</span>
          <span><Check size={15} /> 33 body landmarks</span>
          <span><Check size={15} /> 21 joints per hand</span>
        </div>
      </section>

      <section className="upload-column">
        <label
          className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragEnter={(event) => { event.preventDefault(); onDragging(true) }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => onDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            onDragging(false)
            onFile(event.dataTransfer.files.item(0))
          }}
        >
          <input type="file" accept={ACCEPTED_VIDEO} onChange={(event) => onFile(event.target.files?.[0] ?? null)} />
          <span className="upload-icon">{file ? <Video size={27} /> : <MonitorUp size={27} />}</span>
          {file ? (
            <>
              <strong>{file.name}</strong>
              <small>{formatBytes(file.size)} · Ready for local analysis</small>
              <span className="replace-copy">Choose another video</span>
            </>
          ) : (
            <>
              <strong>Drop your movement video here</strong>
              <small>MP4, MOV, WebM, MKV or AVI · up to 1 GB</small>
              <span className="browse-button"><Upload size={15} /> Browse video</span>
            </>
          )}
          <div className="corner corner-one" /><div className="corner corner-two" />
          <div className="corner corner-three" /><div className="corner corner-four" />
        </label>

        {engineChecked && !modelsReady && (
          <div className="notice warning"><AlertTriangle size={16} /> Models are not ready. Run <code>make models</code>.</div>
        )}
        {error && <div className="notice error"><AlertTriangle size={16} /> {error}</div>}
        <button
          className="primary-action"
          type="button"
          disabled={!file || !modelsReady || submitting}
          onClick={onStart}
        >
          {submitting ? <RefreshCw className="spin" size={18} /> : <Activity size={18} />}
          {submitting ? 'Uploading locally…' : 'Reconstruct motion'}
          {!submitting && <ChevronRight size={18} />}
        </button>
        <p className="local-note"><FolderLock size={14} /> No cloud upload. Processing and files stay local.</p>
      </section>

      <section className="pipeline-strip" aria-label="KineTrace processing pipeline">
        <PipelineStep number="01" icon={<Video />} label="Video frames" />
        <PipelineStep number="02" icon={<Fingerprint />} label="Pose + hands" />
        <PipelineStep number="03" icon={<ScanLine />} label="3D skeleton" />
        <PipelineStep number="04" icon={<Download />} label="JSON + BVH" last />
      </section>
    </div>
  )
}

function PipelineStep({ number, icon, label, last = false }: { number: string; icon: React.ReactNode; label: string; last?: boolean }) {
  return (
    <div className="pipeline-step">
      <span>{number}</span>{icon}<strong>{label}</strong>{!last && <ChevronRight className="pipeline-arrow" />}
    </div>
  )
}

function Processing({ job, error, onReset }: { job: Job; error: string | null; onReset: () => void }) {
  const failed = job.status === 'failed' || Boolean(error)
  return (
    <div className="processing-page">
      <div className={`processing-orb ${failed ? 'failed' : ''}`}>
        {failed ? <AlertTriangle size={36} /> : <ScanLine size={36} />}
      </div>
      <div className="eyebrow">{failed ? 'Analysis interrupted' : 'Local reconstruction'}</div>
      <h2>{failed ? 'Something needs attention' : job.stage}</h2>
      <p>{failed ? (error ?? job.error) : job.filename}</p>
      {!failed && (
        <div className="progress-card">
          <div className="progress-track"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
          <div><span>{Math.round(job.progress * 100)}%</span><span>Body + hands · offline quality</span></div>
        </div>
      )}
      {failed && <button className="secondary-action" type="button" onClick={onReset}>Choose another video</button>}
    </div>
  )
}

type StudioProps = {
  job: Job
  motion: MotionResult
  currentFrame: ReturnType<typeof nearestFrame>
  currentMs: number
  playing: boolean
  looping: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  onPlayState: (playing: boolean) => void
  onToggle: () => void
  onToggleLoop: () => void
  onSeek: (milliseconds: number) => void
  onReset: () => void
}

function Studio({
  job,
  motion,
  currentFrame,
  currentMs,
  playing,
  looping,
  videoRef,
  onPlayState,
  onToggle,
  onToggleLoop,
  onSeek,
  onReset,
}: StudioProps) {
  const quality = motion.quality
  const optimization = motion.metadata.optimization
  return (
    <div className="studio-page">
      <div className="studio-heading">
        <div>
          <button className="back-link" type="button" onClick={onReset}>New capture</button>
          <h2>{motion.metadata.sourceFilename}</h2>
          <p>{motion.metadata.frameCount} frames · {motion.metadata.fps.toFixed(1)} FPS · schema {motion.schemaVersion}</p>
        </div>
        <div className="export-actions">
          <a className="secondary-action" href={job.downloads.json} download><FileJson size={16} /> JSON</a>
          <a className="primary-action compact" href={job.downloads.bvh} download><Download size={16} /> Export BVH</a>
        </div>
      </div>

      <div className="quality-row">
        <QualityCard icon={<Activity />} label="Body coverage" value={quality.poseCoverage} />
        <QualityCard icon={<Fingerprint />} label="Left hand tracked" value={quality.leftHandCoverage} />
        <QualityCard icon={<Fingerprint />} label="Right hand tracked" value={quality.rightHandCoverage} />
        <QualityCard icon={<Gauge />} label="Body confidence" value={quality.averageBodyConfidence} />
      </div>

      <div className="viewer-grid">
        <SourceViewer
          source={job.downloads.source}
          frame={currentFrame}
          bodyConnections={motion.skeleton.bodyConnections}
          handConnections={motion.skeleton.handConnections}
          videoRef={videoRef}
          looping={looping}
          onPlayState={onPlayState}
        />
        <SkeletonViewport
          frame={currentFrame}
          bodyConnections={motion.skeleton.bodyConnections}
          handConnections={motion.skeleton.handConnections}
        />
      </div>
      <Timeline
        currentMs={currentMs}
        durationMs={motion.metadata.durationMs}
        playing={playing}
        looping={looping}
        onToggle={onToggle}
        onToggleLoop={onToggleLoop}
        onSeek={onSeek}
      />
      <div className="studio-footnote">
        <span><i className="confidence-good" /> Reliable</span>
        <span><i className="confidence-held" /> Low confidence / temporarily held</span>
        {optimization && (
          <span
            className="optimization-note"
            title={`${optimization.calibration}; stabilized ${optimization.stabilizedContacts.join(', ')}`}
          >
            Auto-optimized · fixed bones · {optimization.stabilizedContacts.length} contacts
          </span>
        )}
        <p>Hand percentages are directly detected frames · single-camera depth is constrained</p>
      </div>
    </div>
  )
}

function QualityCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  const percentage = Math.round(value * 100)
  return (
    <div className="quality-card">
      <span>{icon}</span>
      <div><small>{label}</small><strong>{percentage}%</strong></div>
      <i><b style={{ width: `${percentage}%` }} /></i>
    </div>
  )
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Unexpected error'
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default App
