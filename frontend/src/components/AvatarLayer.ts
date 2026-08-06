import * as THREE from 'three'

import type { Landmark, MotionFrame } from '../types'

type MaterialName = 'skin' | 'shirt' | 'shorts' | 'joint'

type BodyBone = {
  key: string
  start: number
  end: number
  radiusScale: number
  material: MaterialName
}

const BODY_BONES: BodyBone[] = [
  { key: 'leftUpperArm', start: 11, end: 13, radiusScale: 0.095, material: 'shirt' },
  { key: 'leftForearm', start: 13, end: 15, radiusScale: 0.075, material: 'skin' },
  { key: 'rightUpperArm', start: 12, end: 14, radiusScale: 0.095, material: 'shirt' },
  { key: 'rightForearm', start: 14, end: 16, radiusScale: 0.075, material: 'skin' },
  { key: 'leftThigh', start: 23, end: 25, radiusScale: 0.13, material: 'shorts' },
  { key: 'leftShin', start: 25, end: 27, radiusScale: 0.095, material: 'joint' },
  { key: 'rightThigh', start: 24, end: 26, radiusScale: 0.13, material: 'shorts' },
  { key: 'rightShin', start: 26, end: 28, radiusScale: 0.095, material: 'joint' },
]

const BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

export type AvatarLayer = {
  group: THREE.Group
  torso: THREE.Mesh
  pelvis: THREE.Mesh
  neck: THREE.Mesh
  head: THREE.Mesh
  bodyBones: Map<string, THREE.Mesh>
  bodyJoints: Map<number, THREE.Mesh>
  feet: { left: THREE.Mesh; right: THREE.Mesh }
  palms: { left: THREE.Mesh; right: THREE.Mesh }
  handBones: {
    left: Map<string, THREE.Mesh>
    right: Map<string, THREE.Mesh>
  }
}

export function createAvatarLayer(
  scene: THREE.Scene,
  handConnections: [number, number][],
): AvatarLayer {
  const group = new THREE.Group()
  group.name = 'procedural-avatar'
  scene.add(group)

  const materials: Record<MaterialName, THREE.MeshStandardMaterial> = {
    skin: new THREE.MeshStandardMaterial({
      color: '#c9aa91',
      roughness: 0.78,
      metalness: 0.02,
    }),
    shirt: new THREE.MeshStandardMaterial({
      color: '#b8cc72',
      roughness: 0.66,
      metalness: 0.04,
    }),
    shorts: new THREE.MeshStandardMaterial({
      color: '#343b43',
      roughness: 0.72,
      metalness: 0.08,
    }),
    joint: new THREE.MeshStandardMaterial({
      color: '#68727b',
      roughness: 0.62,
      metalness: 0.16,
    }),
  }

  const limbGeometry = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false)
  const jointGeometry = new THREE.SphereGeometry(1, 16, 12)
  const torsoGeometry = new THREE.CylinderGeometry(0.5, 0.32, 1, 12, 1, false)

  const torso = mesh(torsoGeometry, materials.shirt, group)
  const pelvis = mesh(limbGeometry, materials.shorts, group)
  const neck = mesh(limbGeometry, materials.skin, group)
  const head = mesh(jointGeometry, materials.skin, group)

  const bodyBones = new Map(
    BODY_BONES.map((bone) => [
      bone.key,
      mesh(limbGeometry, materials[bone.material], group),
    ]),
  )
  const bodyJoints = new Map(
    BODY_JOINTS.map((index) => [index, mesh(jointGeometry, materials.joint, group)]),
  )
  const feet = {
    left: mesh(limbGeometry, materials.shorts, group),
    right: mesh(limbGeometry, materials.shorts, group),
  }
  const palms = {
    left: mesh(jointGeometry, materials.skin, group),
    right: mesh(jointGeometry, materials.skin, group),
  }
  const handBones = {
    left: new Map(
      handConnections.map(([start, end]) => [
        `${start}-${end}`,
        mesh(limbGeometry, materials.skin, group),
      ]),
    ),
    right: new Map(
      handConnections.map(([start, end]) => [
        `${start}-${end}`,
        mesh(limbGeometry, materials.skin, group),
      ]),
    ),
  }

  group.visible = false
  return {
    group,
    torso,
    pelvis,
    neck,
    head,
    bodyBones,
    bodyJoints,
    feet,
    palms,
    handBones,
  }
}

export function updateAvatarLayer(
  avatar: AvatarLayer,
  frame: MotionFrame | null,
  handConnections: [number, number][],
) {
  if (!frame?.body.length) {
    hideAvatarParts(avatar)
    return
  }

  const body = pointMap(frame.body)
  const shoulderWidth = distance(body.get(11), body.get(12)) || 0.38
  const hipWidth = distance(body.get(23), body.get(24)) || shoulderWidth * 0.68
  const bodyScale = Math.max(shoulderWidth, 0.28)
  const shoulderCenter = midpoint(body.get(11), body.get(12))
  const hipCenter = midpoint(body.get(23), body.get(24))

  updateTaperedTorso(
    avatar.torso,
    shoulderCenter,
    hipCenter,
    shoulderWidth,
    Math.max(shoulderWidth * 0.42, hipWidth * 0.56),
  )
  updateBone(avatar.pelvis, body.get(23), body.get(24), bodyScale * 0.13)

  for (const bone of BODY_BONES) {
    const part = avatar.bodyBones.get(bone.key)
    if (part) updateBone(part, body.get(bone.start), body.get(bone.end), bodyScale * bone.radiusScale)
  }

  for (const [index, joint] of avatar.bodyJoints) {
    updateJoint(joint, body.get(index), bodyScale * jointRadius(index))
  }

  updateFoot(avatar.feet.left, body, 27, 29, 31, bodyScale)
  updateFoot(avatar.feet.right, body, 28, 30, 32, bodyScale)

  const facePoints = frame.body
    .filter((landmark) => landmark.index <= 10)
    .map(toVector)
  const headCenter = average(facePoints)
  updateBone(avatar.neck, shoulderCenter, headCenter, bodyScale * 0.072)
  updateHead(avatar.head, headCenter, shoulderWidth)

  updateHand(
    avatar.palms.left,
    avatar.handBones.left,
    frame.hands.left,
    handConnections,
    bodyScale,
  )
  updateHand(
    avatar.palms.right,
    avatar.handBones.right,
    frame.hands.right,
    handConnections,
    bodyScale,
  )
}

function mesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  group: THREE.Group,
) {
  const value = new THREE.Mesh(geometry, material)
  value.visible = false
  group.add(value)
  return value
}

function pointMap(landmarks: Landmark[]) {
  return new Map(landmarks.map((landmark) => [landmark.index, toVector(landmark)]))
}

function toVector(landmark: Landmark) {
  return new THREE.Vector3(landmark.world.x, -landmark.world.y, -landmark.world.z)
}

function midpoint(start?: THREE.Vector3, end?: THREE.Vector3) {
  if (!start || !end) return undefined
  return start.clone().add(end).multiplyScalar(0.5)
}

function average(points: THREE.Vector3[]) {
  if (!points.length) return undefined
  return points.reduce((sum, point) => sum.add(point), new THREE.Vector3()).multiplyScalar(1 / points.length)
}

function distance(start?: THREE.Vector3, end?: THREE.Vector3) {
  return start && end ? start.distanceTo(end) : 0
}

function updateBone(
  part: THREE.Mesh,
  start: THREE.Vector3 | undefined,
  end: THREE.Vector3 | undefined,
  radius: number,
) {
  if (!start || !end) {
    part.visible = false
    return
  }
  const direction = end.clone().sub(start)
  const length = direction.length()
  if (length < 1e-5) {
    part.visible = false
    return
  }
  part.visible = true
  part.position.copy(start).add(end).multiplyScalar(0.5)
  part.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.multiplyScalar(1 / length),
  )
  part.scale.set(radius, length, radius)
}

function updateTaperedTorso(
  torso: THREE.Mesh,
  shoulderCenter: THREE.Vector3 | undefined,
  hipCenter: THREE.Vector3 | undefined,
  width: number,
  depth: number,
) {
  if (!shoulderCenter || !hipCenter) {
    torso.visible = false
    return
  }
  const direction = shoulderCenter.clone().sub(hipCenter)
  const length = direction.length()
  if (length < 1e-5) {
    torso.visible = false
    return
  }
  torso.visible = true
  torso.position.copy(shoulderCenter).add(hipCenter).multiplyScalar(0.5)
  torso.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.multiplyScalar(1 / length),
  )
  torso.scale.set(width, length, depth)
}

function updateJoint(part: THREE.Mesh, point: THREE.Vector3 | undefined, radius: number) {
  if (!point) {
    part.visible = false
    return
  }
  part.visible = true
  part.position.copy(point)
  part.quaternion.identity()
  part.scale.setScalar(radius)
}

function updateHead(part: THREE.Mesh, center: THREE.Vector3 | undefined, shoulderWidth: number) {
  if (!center) {
    part.visible = false
    return
  }
  part.visible = true
  part.position.copy(center)
  part.quaternion.identity()
  part.scale.set(shoulderWidth * 0.24, shoulderWidth * 0.31, shoulderWidth * 0.25)
}

function updateFoot(
  foot: THREE.Mesh,
  body: Map<number, THREE.Vector3>,
  ankleIndex: number,
  heelIndex: number,
  toeIndex: number,
  bodyScale: number,
) {
  const heel = body.get(heelIndex) ?? body.get(ankleIndex)
  const toe = body.get(toeIndex)
  updateBone(foot, heel, toe, bodyScale * 0.105)
}

function updateHand(
  palm: THREE.Mesh,
  bones: Map<string, THREE.Mesh>,
  landmarks: Landmark[],
  connections: [number, number][],
  bodyScale: number,
) {
  const points = pointMap(landmarks)
  const palmPoints = [0, 5, 9, 13, 17]
    .map((index) => points.get(index))
    .filter((point): point is THREE.Vector3 => Boolean(point))
  const palmCenter = average(palmPoints)
  updateJoint(palm, palmCenter, bodyScale * 0.07)
  if (palm.visible) palm.scale.set(bodyScale * 0.08, bodyScale * 0.035, bodyScale * 0.095)

  for (const [start, end] of connections) {
    const bone = bones.get(`${start}-${end}`)
    if (bone) updateBone(bone, points.get(start), points.get(end), bodyScale * 0.013)
  }
}

function jointRadius(index: number) {
  if (index === 15 || index === 16 || index === 27 || index === 28) return 0.075
  if (index === 13 || index === 14) return 0.09
  if (index === 25 || index === 26) return 0.11
  return 0.105
}

function hideAvatarParts(avatar: AvatarLayer) {
  avatar.group.traverse((object) => {
    if (object instanceof THREE.Mesh) object.visible = false
  })
}
