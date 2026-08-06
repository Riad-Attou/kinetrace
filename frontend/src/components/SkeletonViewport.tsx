import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import type { Landmark, MotionFrame } from '../types'
import { createAvatarLayer, updateAvatarLayer, type AvatarLayer } from './AvatarLayer'

type SkeletonViewportProps = {
  frame: MotionFrame | null
  bodyConnections: [number, number][]
  handConnections: [number, number][]
}

type Layer = {
  points: THREE.Points
  lines: THREE.LineSegments
}

type SceneState = {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  layers: {
    body: Layer
    left: Layer
    right: Layer
  }
  avatar: AvatarLayer
  animationFrame: number
}

type ViewMode = 'avatar' | 'skeleton' | 'both'
type CameraView = 'front' | 'side' | 'top'

export function SkeletonViewport({
  frame,
  bodyConnections,
  handConnections,
}: SkeletonViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<SceneState | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('avatar')
  const [cameraView, setCameraView] = useState<CameraView | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.08
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#101215')
    scene.fog = new THREE.Fog('#101215', 3.5, 7)
    const camera = new THREE.PerspectiveCamera(
      38,
      container.clientWidth / Math.max(container.clientHeight, 1),
      0.01,
      50,
    )
    camera.position.set(0.4, 0.15, 3.25)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, -0.18, 0)
    controls.minDistance = 1.4
    controls.maxDistance = 7
    const clearCameraPreset = () => setCameraView(null)
    controls.addEventListener('start', clearCameraPreset)

    const grid = new THREE.GridHelper(4, 16, '#30353c', '#20242a')
    grid.position.y = -1.03
    scene.add(grid)

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(1.7, 64),
      new THREE.MeshBasicMaterial({ color: '#15191e', transparent: true, opacity: 0.72 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -1.035
    scene.add(floor)

    const ambient = new THREE.HemisphereLight('#edf2e3', '#171b20', 1.8)
    const keyLight = new THREE.DirectionalLight('#fff9e8', 2.4)
    keyLight.position.set(2.4, 3.4, 3.2)
    const rimLight = new THREE.DirectionalLight('#b7d8ff', 1.35)
    rimLight.position.set(-2.2, 1.4, -2.5)
    scene.add(ambient, keyLight, rimLight)

    const layers = {
      body: createLayer(scene, '#d8ff59', 0.024),
      left: createLayer(scene, '#57d3ff', 0.018),
      right: createLayer(scene, '#ff6e9f', 0.018),
    }
    const avatar = createAvatarLayer(scene, handConnections)

    const state: SceneState = {
      renderer,
      scene,
      camera,
      controls,
      layers,
      avatar,
      animationFrame: 0,
    }
    stateRef.current = state

    const render = () => {
      controls.update()
      renderer.render(scene, camera)
      state.animationFrame = requestAnimationFrame(render)
    }
    render()

    const resize = new ResizeObserver(() => {
      const width = container.clientWidth
      const height = Math.max(container.clientHeight, 1)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    })
    resize.observe(container)

    return () => {
      resize.disconnect()
      cancelAnimationFrame(state.animationFrame)
      controls.removeEventListener('start', clearCameraPreset)
      controls.dispose()
      renderer.dispose()
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.LineSegments) {
          object.geometry.dispose()
          if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose())
          else object.material.dispose()
        }
      })
      renderer.domElement.remove()
      stateRef.current = null
    }
  }, [])

  useEffect(() => {
    const state = stateRef.current
    if (!state) return
    updateLayer(state.layers.body, frame?.body ?? [], bodyConnections)
    updateLayer(state.layers.left, frame?.hands.left ?? [], handConnections)
    updateLayer(state.layers.right, frame?.hands.right ?? [], handConnections)
    updateAvatarLayer(state.avatar, frame, handConnections)
  }, [bodyConnections, frame, handConnections])

  useEffect(() => {
    const state = stateRef.current
    if (!state) return
    const showSkeleton = viewMode === 'skeleton' || viewMode === 'both'
    setLayerVisible(state.layers.body, showSkeleton)
    setLayerVisible(state.layers.left, showSkeleton)
    setLayerVisible(state.layers.right, showSkeleton)
    state.avatar.group.visible = viewMode === 'avatar' || viewMode === 'both'
  }, [viewMode])

  const selectCameraView = (mode: CameraView) => {
    const state = stateRef.current
    if (!state || !frame) return
    const body = new Map(
      frame.body.map((landmark) => [landmark.index, new THREE.Vector3(...scenePosition(landmark))]),
    )
    const leftShoulder = body.get(11)
    const rightShoulder = body.get(12)
    const leftHip = body.get(23)
    const rightHip = body.get(24)
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return

    const shoulderCenter = leftShoulder.clone().add(rightShoulder).multiplyScalar(0.5)
    const hipCenter = leftHip.clone().add(rightHip).multiplyScalar(0.5)
    const target = shoulderCenter.clone().add(hipCenter).multiplyScalar(0.5)
    const up = new THREE.Vector3(0, 1, 0)
    const lateral = rightShoulder.clone().sub(leftShoulder)
    lateral.y = 0
    lateral.normalize()
    const forward = lateral.clone().cross(up).normalize()
    const direction = mode === 'front' ? forward : mode === 'side' ? lateral : up
    const distance = THREE.MathUtils.clamp(
      state.camera.position.distanceTo(state.controls.target),
      state.controls.minDistance,
      state.controls.maxDistance,
    )

    state.camera.up.copy(mode === 'top' ? forward : up)
    state.camera.position.copy(target).addScaledVector(direction, distance)
    state.controls.target.copy(target)
    state.camera.lookAt(target)
    state.controls.update()
    setCameraView(mode)
  }

  return (
    <div className="viewer-shell skeleton-viewer" ref={containerRef}>
      <div className="viewer-label">
        <span className="cube-mark" /> 3D reconstruction
      </div>
      <div className="viewer-mode-toggle" role="group" aria-label="3D preview mode">
        {(['avatar', 'skeleton', 'both'] as const).map((mode) => (
          <button
            className={viewMode === mode ? 'active' : ''}
            type="button"
            aria-pressed={viewMode === mode}
            onClick={() => setViewMode(mode)}
            key={mode}
          >
            {mode}
          </button>
        ))}
      </div>
      <div className="viewer-camera-toggle" role="group" aria-label="Camera view">
        {(['front', 'side', 'top'] as const).map((mode) => (
          <button
            className={cameraView === mode ? 'active' : ''}
            type="button"
            aria-pressed={cameraView === mode}
            onClick={() => selectCameraView(mode)}
            key={mode}
          >
            {mode}
          </button>
        ))}
      </div>
      <div className="viewport-hint">Drag to orbit · Scroll to zoom</div>
    </div>
  )
}

function setLayerVisible(layer: Layer, visible: boolean) {
  layer.points.visible = visible
  layer.lines.visible = visible
}

function createLayer(scene: THREE.Scene, color: string, pointSize: number): Layer {
  const points = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ color, size: pointSize, sizeAttenuation: true }),
  )
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.88 }),
  )
  scene.add(lines, points)
  return { points, lines }
}

function updateLayer(layer: Layer, landmarks: Landmark[], connections: [number, number][]) {
  const points = new Map(landmarks.map((landmark) => [landmark.index, landmark]))
  const pointPositions = landmarks.flatMap(scenePosition)
  const linePositions: number[] = []
  for (const [startIndex, endIndex] of connections) {
    const start = points.get(startIndex)
    const end = points.get(endIndex)
    if (!start || !end) continue
    linePositions.push(...scenePosition(start), ...scenePosition(end))
  }
  replacePosition(layer.points.geometry, pointPositions)
  replacePosition(layer.lines.geometry, linePositions)
}

function replacePosition(geometry: THREE.BufferGeometry, values: number[]) {
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(values, 3))
  geometry.computeBoundingSphere()
}

function scenePosition(landmark: Landmark): [number, number, number] {
  return [landmark.world.x, -landmark.world.y, -landmark.world.z]
}
