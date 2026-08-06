export type WorldPoint = {
  x: number
  y: number
  z: number
}

export type Landmark = {
  index: number
  name: string
  x: number
  y: number
  z: number
  world: WorldPoint
  confidence: number
  inferred: boolean
}

export type MotionFrame = {
  timestampMs: number
  body: Landmark[]
  hands: {
    left: Landmark[]
    right: Landmark[]
  }
}

export type MotionResult = {
  schemaVersion: string
  metadata: {
    sourceFilename: string
    width: number
    height: number
    fps: number
    frameCount: number
    durationMs: number
    coordinateSpace: string
    handAssignment: string
    optimization?: {
      method: string
      calibration: string
      fixedBoneLengths: boolean
      rigidHead: boolean
      stabilizedContacts: string[]
      boneVariationBefore: number
      boneVariationAfter: number
      contactDriftBeforeMeters: number
      contactDriftAfterMeters: number
    }
  }
  skeleton: {
    bodyLandmarks: string[]
    handLandmarks: string[]
    bodyConnections: [number, number][]
    handConnections: [number, number][]
  }
  quality: {
    poseCoverage: number
    leftHandCoverage: number
    rightHandCoverage: number
    leftHandUsableCoverage: number
    rightHandUsableCoverage: number
    averageBodyConfidence: number
  }
  frames: MotionFrame[]
}

export type Job = {
  id: string
  filename: string
  status: 'queued' | 'processing' | 'complete' | 'failed'
  progress: number
  stage: string
  error: string | null
  created_at: string
  downloads: {
    source: string
    json?: string
    bvh?: string
  }
}

export type Health = {
  status: string
  version: string
  models: {
    pose: boolean
    hands: boolean
  }
}
