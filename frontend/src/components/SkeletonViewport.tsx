import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import type { Landmark, MotionFrame } from '../types'

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
  animationFrame: number
}

export function SkeletonViewport({
  frame,
  bodyConnections,
  handConnections,
}: SkeletonViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<SceneState | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
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

    const layers = {
      body: createLayer(scene, '#d8ff59', 0.024),
      left: createLayer(scene, '#57d3ff', 0.018),
      right: createLayer(scene, '#ff6e9f', 0.018),
    }

    const state: SceneState = {
      renderer,
      scene,
      camera,
      controls,
      layers,
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
  }, [bodyConnections, frame, handConnections])

  return (
    <div className="viewer-shell skeleton-viewer" ref={containerRef}>
      <div className="viewer-label">
        <span className="cube-mark" /> 3D reconstruction
      </div>
      <div className="viewport-hint">Drag to orbit · Scroll to zoom</div>
    </div>
  )
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
