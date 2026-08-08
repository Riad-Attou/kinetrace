import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

import type { EncodedMesh, Landmark, MotionFrame } from '../types'
import { createAvatarLayer, updateAvatarLayer, type AvatarLayer } from './AvatarLayer'

type SkeletonViewportProps = {
  frame: MotionFrame | null
  bodyConnections: [number, number][]
  handConnections: [number, number][]
  mesh?: EncodedMesh
  meshFrameIndex: number
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
  reconstruction: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial>
  ground: THREE.Group
  animationFrame: number
}

type ViewMode = 'model' | 'skeleton' | 'both'
type CameraView = 'front' | 'side' | 'top'

type DecodedMesh = Omit<EncodedMesh, 'vertices' | 'faces'> & {
  vertices: Int16Array
  faces: Uint16Array
  bounds: THREE.Box3
  groundY: number
}

export function SkeletonViewport({
  frame,
  bodyConnections,
  handConnections,
  mesh,
  meshFrameIndex,
}: SkeletonViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<SceneState | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('model')
  const [cameraView, setCameraView] = useState<CameraView | null>(null)
  const decodedMesh = useMemo(() => decodeMesh(mesh), [mesh])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.12
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#101215')
    scene.fog = new THREE.Fog('#101215', 3.5, 7)
    const environment = new RoomEnvironment()
    const environmentGenerator = new THREE.PMREMGenerator(renderer)
    const environmentMap = environmentGenerator.fromScene(environment, 0.04).texture
    scene.environment = environmentMap
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

    const ground = new THREE.Group()
    ground.position.y = -1.03
    const grid = new THREE.GridHelper(4, 16, '#30353c', '#20242a')
    ground.add(grid)

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(1.7, 64),
      new THREE.MeshStandardMaterial({
        color: '#15191e',
        roughness: 0.92,
        metalness: 0,
        transparent: true,
        opacity: 0.82,
      }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -0.005
    floor.receiveShadow = true
    ground.add(floor)
    scene.add(ground)

    const ambient = new THREE.HemisphereLight('#edf2e3', '#171b20', 1.35)
    const keyLight = new THREE.DirectionalLight('#fff4e5', 3.15)
    keyLight.position.set(2.4, 3.4, 3.2)
    keyLight.castShadow = true
    keyLight.shadow.mapSize.set(2048, 2048)
    keyLight.shadow.camera.near = 0.1
    keyLight.shadow.camera.far = 12
    keyLight.shadow.camera.left = -3
    keyLight.shadow.camera.right = 3
    keyLight.shadow.camera.top = 3
    keyLight.shadow.camera.bottom = -3
    keyLight.shadow.bias = -0.00015
    keyLight.shadow.normalBias = 0.012
    const fillLight = new THREE.DirectionalLight('#ffd8c5', 0.75)
    fillLight.position.set(-2.6, 1.6, 2.4)
    const rimLight = new THREE.DirectionalLight('#a9ceff', 1.7)
    rimLight.position.set(-2.2, 1.4, -2.5)
    scene.add(ambient, keyLight, fillLight, rimLight)

    const layers = {
      body: createLayer(scene, '#d8ff59', 0.024),
      left: createLayer(scene, '#57d3ff', 0.018),
      right: createLayer(scene, '#ff6e9f', 0.018),
    }
    const avatar = createAvatarLayer(scene, handConnections)
    const reconstruction = createReconstructionMesh(scene)

    const state: SceneState = {
      renderer,
      scene,
      camera,
      controls,
      layers,
      avatar,
      reconstruction,
      ground,
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
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.LineSegments) {
          object.geometry.dispose()
          if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose())
          else object.material.dispose()
        }
      })
      environmentMap.dispose()
      environmentGenerator.dispose()
      environment.dispose()
      renderer.dispose()
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
    updateReconstructionMesh(state.reconstruction, decodedMesh, meshFrameIndex)
  }, [bodyConnections, decodedMesh, frame, handConnections, meshFrameIndex])

  useEffect(() => {
    const state = stateRef.current
    if (!state) return
    state.reconstruction.geometry.setIndex(
      decodedMesh ? new THREE.BufferAttribute(decodedMesh.faces, 1) : null,
    )
    if (decodedMesh && state.reconstruction.geometry.hasAttribute('position')) {
      state.reconstruction.geometry.computeVertexNormals()
    }
    state.ground.position.y = decodedMesh ? decodedMesh.groundY : -1.03
    if (decodedMesh) {
      const target = decodedMesh.bounds.getCenter(new THREE.Vector3())
      const cameraOffset = target.clone().sub(state.controls.target)
      state.controls.target.copy(target)
      state.camera.position.add(cameraOffset)
      state.controls.update()
    }
  }, [decodedMesh])

  useEffect(() => {
    const state = stateRef.current
    if (!state) return
    const showModel = viewMode === 'model' || viewMode === 'both'
    const showSkeleton = viewMode === 'skeleton' || viewMode === 'both'
    const hasReconstruction = Boolean(decodedMesh)
    setLayerVisible(state.layers.body, showSkeleton)
    setLayerVisible(state.layers.left, showSkeleton)
    setLayerVisible(state.layers.right, showSkeleton)
    state.reconstruction.visible = showModel && hasReconstruction
    state.avatar.group.visible = showModel && !hasReconstruction
  }, [decodedMesh, viewMode])

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
    const forward = decodedMesh
      ? up.clone().cross(lateral).normalize()
      : lateral.clone().cross(up).normalize()
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
        <span className="cube-mark" /> {decodedMesh ? 'SOMA reconstruction' : '3D reconstruction'}
      </div>
      <div className="viewer-mode-toggle" role="group" aria-label="3D preview mode">
        {(['model', 'skeleton', 'both'] as const).map((mode) => (
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

function createReconstructionMesh(scene: THREE.Scene) {
  const mesh = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshPhysicalMaterial({
      color: '#c9a58e',
      roughness: 0.58,
      metalness: 0,
      sheen: 0.22,
      sheenColor: new THREE.Color('#f0c4ae'),
      sheenRoughness: 0.78,
      side: THREE.DoubleSide,
    }),
  )
  mesh.visible = false
  mesh.frustumCulled = false
  mesh.castShadow = true
  mesh.receiveShadow = true
  scene.add(mesh)
  return mesh
}

function updateReconstructionMesh(
  mesh: THREE.Mesh,
  decoded: DecodedMesh | null,
  frameIndex: number,
) {
  if (!decoded) {
    mesh.geometry.deleteAttribute('position')
    return
  }
  const safeFrame = Math.min(Math.max(frameIndex, 0), decoded.frameCount - 1)
  const frameOffset = safeFrame * decoded.vertexCount * 3
  const currentPosition = mesh.geometry.getAttribute('position')
  const positions = currentPosition instanceof THREE.BufferAttribute
    && currentPosition.array instanceof Float32Array
    && currentPosition.array.length === decoded.vertexCount * 3
    ? currentPosition.array
    : new Float32Array(decoded.vertexCount * 3)
  for (let index = 0; index < positions.length; index += 3) {
    positions[index] = decoded.offset[0] + decoded.vertices[frameOffset + index] * decoded.scale[0]
    positions[index + 1] = -(decoded.offset[1] + decoded.vertices[frameOffset + index + 1] * decoded.scale[1])
    positions[index + 2] = -(decoded.offset[2] + decoded.vertices[frameOffset + index + 2] * decoded.scale[2])
  }
  if (positions === currentPosition?.array) currentPosition.needsUpdate = true
  else mesh.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  mesh.geometry.computeVertexNormals()
  mesh.geometry.computeBoundingSphere()
}

function decodeMesh(mesh: EncodedMesh | undefined): DecodedMesh | null {
  if (!mesh || mesh.encoding !== 'int16-le-base64') return null
  const vertexBytes = decodeBase64(mesh.vertices)
  const faceBytes = decodeBase64(mesh.faces)
  const expectedVertices = mesh.frameCount * mesh.vertexCount * 3
  const expectedFaces = mesh.faceCount * 3
  if (vertexBytes.byteLength !== expectedVertices * 2 || faceBytes.byteLength !== expectedFaces * 2) {
    return null
  }
  const vertexView = new DataView(vertexBytes.buffer, vertexBytes.byteOffset, vertexBytes.byteLength)
  const faceView = new DataView(faceBytes.buffer, faceBytes.byteOffset, faceBytes.byteLength)
  const vertices = new Int16Array(expectedVertices)
  const faces = new Uint16Array(expectedFaces)
  const minimum = new THREE.Vector3(Infinity, Infinity, Infinity)
  const maximum = new THREE.Vector3(-Infinity, -Infinity, -Infinity)
  for (let index = 0; index < expectedVertices; index += 1) {
    vertices[index] = vertexView.getInt16(index * 2, true)
    if (index % 3 === 2) {
      const vertexIndex = index - 2
      const x = mesh.offset[0] + vertices[vertexIndex] * mesh.scale[0]
      const y = -(mesh.offset[1] + vertices[vertexIndex + 1] * mesh.scale[1])
      const z = -(mesh.offset[2] + vertices[vertexIndex + 2] * mesh.scale[2])
      minimum.set(Math.min(minimum.x, x), Math.min(minimum.y, y), Math.min(minimum.z, z))
      maximum.set(Math.max(maximum.x, x), Math.max(maximum.y, y), Math.max(maximum.z, z))
    }
  }
  for (let index = 0; index < expectedFaces; index += 1) {
    faces[index] = faceView.getUint16(index * 2, true)
  }
  return {
    ...mesh,
    vertices,
    faces,
    bounds: new THREE.Box3(minimum, maximum),
    groundY: minimum.y,
  }
}

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
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
