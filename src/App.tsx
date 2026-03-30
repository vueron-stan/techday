import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls, PivotControls } from '@react-three/drei';
import * as THREE from 'three/webgpu';
import {
  float,
  mix,
  perspectiveDepthToViewZ,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  viewZToOrthographicDepth,
} from 'three/tsl';
import { GLTFLoader, HDRLoader } from 'three/examples/jsm/Addons.js';
import {
  GpuFrustumRayHitPass,
  LAYER_DEPTH_MAP_EXTRA,
  LAYER_DEPTH_OCCLUDER,
  LAYER_DEPTH_SIM_INVISIBLE,
  LAYER_FRUSTUM_GUIDE,
} from './GpuFrustumRayHitPass';
import {
  INITIAL_SCENE,
  PRESETS,
  type CarDriveConfig,
  type CyanHitMode,
  type SceneSnapshot,
  type Vec3,
} from './scenePresets';

const HDR_URL =
  'https://viewer.vueron.com/public/model/c8a3cfa6-151f-4d33-bd4c-60795efa271d.hdr';

/** Vite `base` — dev `/`, production build `/techday/` */
const PUBLIC_BASE = import.meta.env.BASE_URL;

/** GLB 루트 스케일 — near와 무관하게 고정 */
const CAR_UNIFORM_SCALE = 4.6;
/** `carDrive` 없을 때 Z 슬라이더 (월드 Z, 앞쪽이 보통 음수) */
const CAR_Z_SLIDER_MIN = -30;
const CAR_Z_SLIDER_MAX = 60;
const CAR_Z_SLIDER_STEP = 0.05;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/** 프리셋/코드에 붙여넣기용 `[x, y, z]` 문자열 */
function vec3TupleClipboardText(v: Vec3): string {
  return `[${v[0]}, ${v[1]}, ${v[2]}]`;
}

/** `INITIAL_SCENE` / `PRESETS` 에 붙여넣기 — `SceneSnapshot` 필드 순서는 scenePresets.ts 와 동일 */
function tsNumberLiteral(n: number): string {
  if (
    Number.isFinite(n) &&
    Math.round(n) === n &&
    Math.abs(n) <= Number.MAX_SAFE_INTEGER
  ) {
    return String(Math.round(n));
  }
  return String(n);
}

function formatSceneSnapshotForPresetPaste(s: SceneSnapshot): string {
  const cam = vec3TupleClipboardText(s.cameraPosition);
  const tgt = vec3TupleClipboardText(s.orbitTarget);
  const lines = [
    '{',
    `  near: ${tsNumberLiteral(s.near)},`,
    `  azimuthSpanDeg: ${tsNumberLiteral(s.azimuthSpanDeg)},`,
    `  polarSpanDeg: ${tsNumberLiteral(s.polarSpanDeg)},`,
    `  azimuthDivisions: ${tsNumberLiteral(s.azimuthDivisions)},`,
    `  polarDivisions: ${tsNumberLiteral(s.polarDivisions)},`,
    '',
    `  sphereOpacity: ${tsNumberLiteral(s.sphereOpacity)},`,
    `  lineOpacity: ${tsNumberLiteral(s.lineOpacity)},`,
    `  planeOpacity: ${tsNumberLiteral(s.planeOpacity)},`,
    `  drawDepthmap: ${s.drawDepthmap},`,
    '',
    `  sphereHitSize: ${tsNumberLiteral(s.sphereHitSize)},`,
    `  sphereHitOpacity: ${tsNumberLiteral(s.sphereHitOpacity)},`,
    `  nearPointSize: ${tsNumberLiteral(s.nearPointSize)},`,
    `  projectMarkersOnNearPlaneOnly: ${s.projectMarkersOnNearPlaneOnly},`,
    `  cyanHitMode: '${s.cyanHitMode}',`,
    '',
    `  hitNoiseLevel: ${tsNumberLiteral(s.hitNoiseLevel)},`,
    `  lidarPyramidHeight: ${tsNumberLiteral(s.lidarPyramidHeight)},`,
    `  lidarPyramidOpacity: ${tsNumberLiteral(s.lidarPyramidOpacity)},`,
    `  lidarSampleRateHz: ${tsNumberLiteral(s.lidarSampleRateHz)},`,
    `  lidarMaxRange: ${tsNumberLiteral(s.lidarMaxRange)},`,
    '',
    `  carOpacity: ${tsNumberLiteral(s.carOpacity)},`,
    `  streetOpacity: ${tsNumberLiteral(s.streetOpacity)},`,
    `  carPosition: ${vec3TupleClipboardText(s.carPosition)},`,
    '',
    `  backgroundIntensity: ${tsNumberLiteral(s.backgroundIntensity)},`,
    `  cameraPosition: ${cam},`,
    `  orbitTarget: ${tgt},`,
  ];
  if (s.carDrive) {
    const { start, end, durationMs } = s.carDrive;
    lines.push(
      '  carDrive: {',
      `    start: ${vec3TupleClipboardText(start)},`,
      `    end: ${vec3TupleClipboardText(end)},`,
      `    durationMs: ${tsNumberLiteral(durationMs)},`,
      '  },',
    );
  }
  lines.push('}');
  return lines.join('\n');
}

function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** GPU 패스 재생성을 묶어 슬라이더/프리셋 연속 변경 시 점이 비었다가 채워지는 깜빡임 완화 */
function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

interface FrustumRaySample {
  dir: THREE.Vector3;
  /** 카메라 전방(z<0)이고 프러스텀 near 직사각형 안이면 near 평면과의 교점 */
  analyticalNear: THREE.Vector3 | null;
}

/** z=−near 에서 수직 범위: y = near·tan(φ)/cos(θ) 이므로 반높이는 near·tan(polar/2)/cos(azimuth/2) */
function nearPlaneHalfHeight(
  near: number,
  azimuthHalfRad: number,
  polarHalfRad: number,
): number {
  const c = Math.max(1e-6, Math.cos(azimuthHalfRad));
  return (near * Math.tan(polarHalfRad)) / c;
}

/** 격자 광선 샘플 (near 직사각형 클립: 가로는 azimuth, 세로는 구면 좌표식 극한 높이) */
function sampleFrustumRays(
  near: number,
  azimuthSpanDeg: number,
  polarSpanDeg: number,
  azimuthDivisions: number,
  polarDivisions: number,
): FrustumRaySample[] {
  const halfX = THREE.MathUtils.degToRad(azimuthSpanDeg / 2);
  const halfY = THREE.MathUtils.degToRad(polarSpanDeg / 2);
  const planeW = 2 * near * Math.tan(halfX);
  const planeH = 2 * nearPlaneHalfHeight(near, halfX, halfY);
  const clipX = (planeW / 2) * 1.01;
  const clipY = (planeH / 2) * 1.01;

  const thetaMin = -THREE.MathUtils.degToRad(azimuthSpanDeg) / 2;
  const thetaMax = THREE.MathUtils.degToRad(azimuthSpanDeg) / 2;
  const phiMin = -THREE.MathUtils.degToRad(polarSpanDeg) / 2;
  const phiMax = THREE.MathUtils.degToRad(polarSpanDeg) / 2;

  const hDiv = Math.max(1, Math.round(azimuthDivisions));
  const vDiv = Math.max(1, Math.round(polarDivisions));
  const out: FrustumRaySample[] = [];

  for (let i = 0; i <= hDiv; i++) {
    for (let j = 0; j <= vDiv; j++) {
      const theta = thetaMin + (i / hDiv) * (thetaMax - thetaMin);
      const phi = phiMin + (j / vDiv) * (phiMax - phiMin);
      const dirX = Math.sin(theta) * Math.cos(phi);
      const dirY = Math.sin(phi);
      const dirZ = -Math.cos(theta) * Math.cos(phi);
      const dir = new THREE.Vector3(dirX, dirY, dirZ).normalize();

      let analyticalNear: THREE.Vector3 | null = null;
      if (dirZ < 0) {
        const t = -near / dirZ;
        const p = dir.clone().multiplyScalar(t);
        if (Math.abs(p.x) <= clipX && Math.abs(p.y) <= clipY) analyticalNear = p;
      }
      out.push({ dir, analyticalNear });
    }
  }
  return out;
}

/**
 * 꼭짓점 원점, 전방 −Z. `sampleFrustumRays` 와 동일 (θ,φ) 모서리 4광선과 평면 z=−height 교선으로 밑면.
 */
function buildLidarPyramidGeometry(
  height: number,
  azimuthSpanDeg: number,
  polarSpanDeg: number,
): THREE.BufferGeometry {
  const h = Math.max(1e-4, height);
  const thetaMin = -THREE.MathUtils.degToRad(azimuthSpanDeg) / 2;
  const thetaMax = THREE.MathUtils.degToRad(azimuthSpanDeg) / 2;
  const phiMin = -THREE.MathUtils.degToRad(polarSpanDeg) / 2;
  const phiMax = THREE.MathUtils.degToRad(polarSpanDeg) / 2;

  const corner = (theta: number, phi: number): [number, number, number] => {
    const dirX = Math.sin(theta) * Math.cos(phi);
    const dirY = Math.sin(phi);
    const dirZ = -Math.cos(theta) * Math.cos(phi);
    const dz = dirZ < -1e-8 ? dirZ : -1e-8;
    const t = -h / dz;
    return [t * dirX, t * dirY, t * dirZ];
  };

  const c1 = corner(thetaMin, phiMin);
  const c2 = corner(thetaMax, phiMin);
  const c3 = corner(thetaMax, phiMax);
  const c4 = corner(thetaMin, phiMax);

  const positions = new Float32Array([
    0,
    0,
    0,
    ...c1,
    ...c2,
    ...c3,
    ...c4,
  ]);
  const indices = [
    0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 3, 2, 1, 4, 3,
  ];
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

interface FrustumData {
  lines: THREE.Vector3[];
  /** near 평면 해석 교점 (레이캐스트 폴백·표시용) */
  nearPlanePoints: THREE.Vector3[];
  /** 광선 끝 — 외접 구면 위 점 */
  sphereHitPoints: THREE.Vector3[];
  sphereRadius: number;
  near: number;
  planeWidth: number;
  planeHeight: number;
  samples: FrustumRaySample[];
}

/** 광선–외접 구면 교점 하이라이트 색 */
const SPHERE_HIT_COLOR = '#ffff00';

/**
 * WebGPU 는 `Points` 프리티브 크기를 1px 로만 그릴 수 있어 `PointsMaterial.size`가 무시됨.
 * 인스턴스된 단위구(반지름 1)에 `scale=s`를 주어 슬라이더 크기가 반영되게 함.
 */
function InstancedSphereMarkers({
  id,
  positions,
  size,
  color,
  opacity,
  depthTest = true,
}: {
  id: string;
  positions: Float32Array;
  size: number;
  color: THREE.ColorRepresentation;
  opacity: number;
  /** false 면 뒤에 있는 가이드/자동차 깊이와 무관하게 항상 위에 그림 */
  depthTest?: boolean;
}) {
  const meshRef = useRef<THREE.InstancedMesh | null>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const count = positions.length / 3;

  const geometry = useMemo(() => new THREE.SphereGeometry(1, 10, 10), []);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || count === 0) return;
    for (let i = 0; i < count; i++) {
      dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      dummy.scale.setScalar(size);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [positions, size, count, dummy]);

  if (count === 0) return null;

  return (
    <instancedMesh
      key={`${id}-${count}`}
      ref={meshRef}
      args={[geometry, null as unknown as THREE.Material, count]}
    >
      <meshStandardMaterial
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
        depthTest={depthTest}
      />
    </instancedMesh>
  );
}

function setOrbitControlsEnabled(controls: unknown, enabled: boolean) {
  const c = controls as { enabled?: boolean } | null | undefined;
  if (c && typeof c.enabled === 'boolean') c.enabled = enabled;
}

/** `SceneSnapshot` 필드 + 런타임 `carRaycastTarget`(GLB 마운트) */
interface FrustumVisualizerProps {
  sphereOpacity: number;
  lineOpacity: number;
  planeOpacity: number;
  near: number;
  azimuthSpanDeg: number;
  polarSpanDeg: number;
  azimuthDivisions: number;
  polarDivisions: number;
  sphereHitSize: number;
  sphereHitOpacity: number;
  nearPointSize: number;
  /** GLB 루트 — 레이캐스트 대상 (없으면 near/구면 폴백) */
  carRaycastTarget: THREE.Object3D | null;
  /** true: 노란 투영점을 near 평면 식으로만 둠. false + 차 있으면 `Raycaster`로 표면 교차 */
  projectMarkersOnNearPlaneOnly: boolean;
  /** 시안: 구면만 / GPU 깊이(차·바닥) 시뮬 */
  cyanHitMode: CyanHitMode;
  /** GPU 깊이 히트 위치 노이즈 (world 단위, 0–0.3) */
  hitNoiseLevel: number;
  /** 라이다 원점 표시: 전방 −Z로 선 사각 피라미드 높이(꼭짓점~밑면) */
  lidarPyramidHeight: number;
  /** 라이다 원점 피라미드 메시 투명도 */
  lidarPyramidOpacity: number;
  /** GPU 깊이/포인트 `execute` 갱신 주파수 (Hz) */
  lidarSampleRateHz: number;
  /** 라이다 시뮬 최대 거리(far) */
  lidarMaxRange: number;
  /** 차량 주행(carDrive) 중 와이어 외접 구가 차 메시와 깊이 정렬로 깜빡일 때 뒤로 밀리지 않게 renderOrder 상승 */
  carDriveActive: boolean;
  /** 빨간 near 평면에 센서 시점 깊이 텍스처 */
  drawDepthmap: boolean;
}

function MainCameraEnableFrustumGuideLayer() {
  const camera = useThree((s) => s.camera);
  useLayoutEffect(() => {
    camera.layers.enable(LAYER_FRUSTUM_GUIDE);
  }, [camera]);
  return null;
}

const FrustumVisualizer: React.FC<FrustumVisualizerProps> = ({
  sphereOpacity,
  lineOpacity,
  planeOpacity,
  near,
  azimuthSpanDeg,
  polarSpanDeg,
  azimuthDivisions,
  polarDivisions,
  sphereHitSize,
  sphereHitOpacity,
  nearPointSize,
  carRaycastTarget,
  projectMarkersOnNearPlaneOnly,
  cyanHitMode,
  hitNoiseLevel,
  lidarPyramidHeight,
  lidarPyramidOpacity,
  lidarSampleRateHz,
  lidarMaxRange,
  carDriveActive,
  drawDepthmap,
}) => {
  const { gl, scene, controls } = useThree();
  const guideRootRef = useRef<THREE.Group>(null);
  const depthSolidSphereRef = useRef<THREE.Mesh>(null);
  const planeDepthRTRef = useRef<THREE.RenderTarget | null>(null);
  const sensorDepthCamRef = useRef(
    new THREE.PerspectiveCamera(50, 1, 0.05, 1000),
  );
  const sensorDepthMatUniformsRef = useRef<{
    farU: ReturnType<typeof uniform>;
    opacityU: ReturnType<typeof uniform>;
  } | null>(null);
  const [sensorPlaneDepthMat, setSensorPlaneDepthMat] =
    useState<THREE.MeshBasicNodeMaterial | null>(null);
  const gpuPassForFrameRef = useRef<GpuFrustumRayHitPass | null>(null);
  const [gpuPass, setGpuPass] = useState<GpuFrustumRayHitPass | null>(null);
  const lastGpuSampleTimeRef = useRef<number>(-Infinity);
  /** 라이다 센서(프러스텀 가이드 + GPU 레이 원점) — 피라미드 클릭 후 PivotControls 로 편집 */
  const [sensorMatrix, setSensorMatrix] = useState(() => new THREE.Matrix4());
  const [lidarPivotSelected, setLidarPivotSelected] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setLidarPivotSelected(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    setOrbitControlsEnabled(controls, !lidarPivotSelected);
  }, [controls, lidarPivotSelected]);

  useEffect(() => {
    return () => setOrbitControlsEnabled(controls, true);
  }, [controls]);

  useLayoutEffect(() => {
    lastGpuSampleTimeRef.current = -Infinity;
  }, [lidarSampleRateHz]);

  const lidarPyramidGeometry = useMemo(
    () => buildLidarPyramidGeometry(lidarPyramidHeight, azimuthSpanDeg, polarSpanDeg),
    [lidarPyramidHeight, azimuthSpanDeg, polarSpanDeg],
  );
  useEffect(
    () => () => {
      lidarPyramidGeometry.dispose();
    },
    [lidarPyramidGeometry],
  );

  useLayoutEffect(() => {
    if (!drawDepthmap) {
      setSensorPlaneDepthMat((prev) => {
        prev?.dispose();
        return null;
      });
      planeDepthRTRef.current?.dispose();
      planeDepthRTRef.current = null;
      sensorDepthMatUniformsRef.current = null;
      return;
    }

    const rt = new THREE.RenderTarget(512, 512);
    const depthTex = new THREE.DepthTexture(512, 512);
    depthTex.minFilter = THREE.NearestFilter;
    depthTex.magFilter = THREE.NearestFilter;
    rt.depthTexture = depthTex;
    rt.depthBuffer = true;
    planeDepthRTRef.current = rt;

    const nearU = uniform(0.05);
    const farU = uniform(lidarMaxRange);
    const opacityU = uniform(planeOpacity);
    sensorDepthMatUniformsRef.current = { farU, opacityU };

    /** Y-flip: depth RT ↔ 평면 UV */
    const uvFlip = vec2(uv().x, float(1).sub(uv().y));
    /** 샘플은 깊이 버퍼 0–1; reversed Z 등은 renderer 설정에 맞춰 perspectiveDepthToViewZ 가 처리 */
    const d = texture(depthTex, uvFlip).clamp(1e-6, 1);
    const viewZ = perspectiveDepthToViewZ(d, nearU, farU);
    /** 카메라 near–far 사이 선형 깊이를 0–1 로 정규화(뷰 공간 Z 기준) */
    const linDepth = viewZToOrthographicDepth(viewZ, nearU, farU).clamp(0, 1);
    const depthGray = vec3(linDepth);
    const red = vec3(1.0, 0.27, 0.27);
    const mat = new THREE.MeshBasicNodeMaterial();
    /** 거의 순수 그레이스케일 깊이 + 약한 빨간 틴트 */
    mat.colorNode = mix(depthGray, red, float(0.1));
    mat.transparent = true;
    mat.opacityNode = opacityU;
    mat.side = THREE.DoubleSide;
    mat.depthWrite = false;
    mat.depthTest = true;

    setSensorPlaneDepthMat(mat);
    return () => {
      mat.dispose();
      rt.dispose();
      planeDepthRTRef.current = null;
      sensorDepthMatUniformsRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- lidarMaxRange·planeOpacity → uniform, 다음 effect
  }, [drawDepthmap]);

  useLayoutEffect(() => {
    const u = sensorDepthMatUniformsRef.current;
    if (!u) return;
    u.farU.value = lidarMaxRange;
    u.opacityU.value = planeOpacity;
  }, [lidarMaxRange, planeOpacity, sensorPlaneDepthMat]);

  /** 차·바닥 opacity 0 이라도 전용 레이어로 깊이 패스에 남음 */
  const depthSceneActive = true;
  const cyanDepthSim = cyanHitMode === 'depthSim';
  const runGpuHits =
    depthSceneActive &&
    (!projectMarkersOnNearPlaneOnly || cyanDepthSim);

  const GPU_PASS_RECREATE_DEBOUNCE_MS = 90;
  const debouncedAzimuthSpanDeg = useDebouncedValue(
    azimuthSpanDeg,
    GPU_PASS_RECREATE_DEBOUNCE_MS,
  );
  const debouncedPolarSpanDeg = useDebouncedValue(
    polarSpanDeg,
    GPU_PASS_RECREATE_DEBOUNCE_MS,
  );
  const debouncedAzimuthDivisions = useDebouncedValue(
    azimuthDivisions,
    GPU_PASS_RECREATE_DEBOUNCE_MS,
  );
  const debouncedPolarDivisions = useDebouncedValue(
    polarDivisions,
    GPU_PASS_RECREATE_DEBOUNCE_MS,
  );

  const { lines, nearPlanePoints, sphereHitPoints, sphereRadius, planeWidth, planeHeight, samples } =
    useMemo<FrustumData>(() => {
      const far = Math.max(1, lidarMaxRange);
      const halfX = THREE.MathUtils.degToRad(azimuthSpanDeg / 2);
      const halfY = THREE.MathUtils.degToRad(polarSpanDeg / 2);
      const farX = far * Math.tan(halfX);
      const farY = nearPlaneHalfHeight(far, halfX, halfY);
      const radius = Math.sqrt(farX * farX + farY * farY + far * far);

      const planeW = 2 * near * Math.tan(halfX);
      const planeH = 2 * nearPlaneHalfHeight(near, halfX, halfY);

      const raySamples = sampleFrustumRays(
        near,
        azimuthSpanDeg,
        polarSpanDeg,
        azimuthDivisions,
        polarDivisions,
      );

      const linePoints: THREE.Vector3[] = [];
      const nearPts: THREE.Vector3[] = [];
      const spherePts: THREE.Vector3[] = [];

      for (const s of raySamples) {
        const endPt = s.dir.clone().multiplyScalar(radius);
        linePoints.push(new THREE.Vector3(0, 0, 0), endPt);
        spherePts.push(endPt);
        if (s.analyticalNear) nearPts.push(s.analyticalNear);
      }

      return {
        lines: linePoints,
        nearPlanePoints: nearPts,
        sphereHitPoints: spherePts,
        sphereRadius: radius,
        near,
        planeWidth: planeW,
        planeHeight: planeH,
        samples: raySamples,
      };
    }, [near, azimuthSpanDeg, polarSpanDeg, azimuthDivisions, polarDivisions, lidarMaxRange]);

  useLayoutEffect(() => {
    const m = depthSolidSphereRef.current;
    if (!m) return;
    m.layers.set(LAYER_DEPTH_MAP_EXTRA);
    m.traverse((o) => {
      if (o instanceof THREE.Mesh) o.layers.set(LAYER_DEPTH_MAP_EXTRA);
    });
  }, [drawDepthmap, sphereRadius]);

  const projectionMarkers = useMemo(() => {
    const useAnalytical = projectMarkersOnNearPlaneOnly || !carRaycastTarget;
    if (useAnalytical) {
      return new Float32Array(nearPlanePoints.flatMap((v) => [v.x, v.y, v.z]));
    }
    if (runGpuHits) {
      return new Float32Array(0);
    }

    const origin = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    sensorMatrix.decompose(origin, quat, scl);
    const inv = new THREE.Matrix4().copy(sensorMatrix).invert();
    const worldDir = new THREE.Vector3();

    carRaycastTarget.updateMatrixWorld(true);
    const raycaster = new THREE.Raycaster();
    const hits: THREE.Vector3[] = [];

    for (const s of samples) {
      if (s.dir.z >= 0) continue;
      worldDir.copy(s.dir).applyQuaternion(quat).normalize();
      raycaster.set(origin, worldDir);
      const res = raycaster.intersectObject(carRaycastTarget, true);
      if (res.length > 0) hits.push(res[0].point.clone().applyMatrix4(inv));
    }

    return new Float32Array(hits.flatMap((v) => [v.x, v.y, v.z]));
  }, [
    projectMarkersOnNearPlaneOnly,
    carRaycastTarget,
    samples,
    nearPlanePoints,
    runGpuHits,
    sensorMatrix,
  ]);

  const sphereHitDisplayPositions = useMemo(() => {
    if (cyanHitMode === 'sphereOnly' || !carRaycastTarget) {
      return new Float32Array(sphereHitPoints.flatMap((v) => [v.x, v.y, v.z]));
    }
    if (runGpuHits) {
      return new Float32Array(0);
    }

    const origin = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    sensorMatrix.decompose(origin, quat, scl);
    const inv = new THREE.Matrix4().copy(sensorMatrix).invert();
    const worldDir = new THREE.Vector3();

    carRaycastTarget.updateMatrixWorld(true);
    const raycaster = new THREE.Raycaster();
    const pts: THREE.Vector3[] = [];

    for (const s of samples) {
      worldDir.copy(s.dir).applyQuaternion(quat).normalize();
      raycaster.set(origin, worldDir);
      const res = raycaster.intersectObject(carRaycastTarget, true);
      if (res.length > 0) pts.push(res[0].point.clone().applyMatrix4(inv));
      else pts.push(s.dir.clone().multiplyScalar(sphereRadius));
    }

    return new Float32Array(pts.flatMap((v) => [v.x, v.y, v.z]));
  }, [
    cyanHitMode,
    carRaycastTarget,
    samples,
    sphereHitPoints,
    sphereRadius,
    runGpuHits,
    sensorMatrix,
  ]);

  /* GpuFrustumRayHitPass 생성/해제를 React 트리(<primitive>)와 맞추려면 effect 내 setState가 필요함 */
  useLayoutEffect(() => {
    if (!runGpuHits) {
      setGpuPass(null);
      gpuPassForFrameRef.current = null;
      return;
    }
    const cfg = {
      azimuthSpanDeg: debouncedAzimuthSpanDeg,
      polarSpanDeg: debouncedPolarSpanDeg,
      azimuthDivisions: debouncedAzimuthDivisions,
      polarDivisions: debouncedPolarDivisions,
      depthNear: 0.05,
      depthTextureSize: 512,
    };
    const p = new GpuFrustumRayHitPass(cfg);
    setGpuPass(p);
    gpuPassForFrameRef.current = p;
    lastGpuSampleTimeRef.current = -Infinity;
    return () => {
      p.dispose();
      gpuPassForFrameRef.current = null;
    };
  }, [
    runGpuHits,
    debouncedAzimuthSpanDeg,
    debouncedPolarSpanDeg,
    debouncedAzimuthDivisions,
    debouncedPolarDivisions,
  ]);

  useLayoutEffect(() => {
    if (!gpuPass) return;
    gpuPassForFrameRef.current = gpuPass;
    gpuPass.setMaxRange(sphereRadius);
    gpuPass.setMarkerScales(sphereHitSize, nearPointSize);
    gpuPass.setCyanUsesHitOnly(cyanDepthSim);
    gpuPass.setHitNoiseLevel(hitNoiseLevel);
    gpuPass.setMeshVisibility(
      cyanDepthSim,
      !projectMarkersOnNearPlaneOnly && nearPointSize > 0,
    );
    gpuPass.setMarkerOpacities(
      sphereHitOpacity,
      nearPointSize > 0 ? 1 : 0,
    );
  }, [
    gpuPass,
    sphereRadius,
    sphereHitSize,
    nearPointSize,
    sphereHitOpacity,
    cyanDepthSim,
    projectMarkersOnNearPlaneOnly,
    hitNoiseLevel,
  ]);

  useFrame((state) => {
    const pass = gpuPassForFrameRef.current;
    if (!pass) return;
    const renderer = gl as unknown as THREE.WebGPURenderer;
    if (!renderer.compute) return;
    const t = state.clock.elapsedTime;
    const hz = THREE.MathUtils.clamp(lidarSampleRateHz, 0.1, 20);
    const interval = 1 / hz;
    if (t - lastGpuSampleTimeRef.current < interval) return;
    lastGpuSampleTimeRef.current = t;
    pass.execute(renderer, scene, sensorMatrix);
  }, -1);

  useFrame(() => {
    if (!drawDepthmap || !planeDepthRTRef.current || planeOpacity <= 0.001) return;
    const renderer = gl as unknown as THREE.WebGPURenderer;
    const rt = planeDepthRTRef.current;
    const cam = sensorDepthCamRef.current;

    const vFovRad = 2 * Math.atan((planeHeight / 2) / Math.max(1e-6, near));
    cam.fov = THREE.MathUtils.radToDeg(vFovRad);
    cam.aspect = planeWidth / Math.max(1e-6, planeHeight);
    cam.near = 0.05;
    cam.far = Math.max(cam.near + 0.1, lidarMaxRange);
    cam.updateProjectionMatrix();

    const origin = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    sensorMatrix.decompose(origin, quat, scl);
    cam.position.copy(origin);
    cam.quaternion.copy(quat);
    cam.updateMatrixWorld(true);

    cam.layers.disableAll();
    cam.layers.enable(LAYER_DEPTH_OCCLUDER);
    cam.layers.enable(LAYER_DEPTH_SIM_INVISIBLE);
    cam.layers.enable(LAYER_DEPTH_MAP_EXTRA);

    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(rt);
    renderer.clear(true, true, true);
    renderer.render(scene, cam);
    renderer.setRenderTarget(prev);
  }, -2);

  const linePositions = new Float32Array(lines.flatMap((v) => [v.x, v.y, v.z]));

  const showCpuCyan =
    !runGpuHits || cyanHitMode === 'sphereOnly' || sphereHitOpacity <= 0.001;
  const showCpuYellow =
    nearPointSize > 0 && (!runGpuHits || projectMarkersOnNearPlaneOnly);
  /** 차 표면에 찍힌 CPU 점이 depth buffer에 밀리지 않게 */
  const cpuCyanDrawOverCar =
    showCpuCyan && cyanHitMode === 'depthSim' && carRaycastTarget != null;
  const cpuYellowDrawOverCar =
    showCpuYellow && !projectMarkersOnNearPlaneOnly && carRaycastTarget != null;

  useLayoutEffect(() => {
    const g = guideRootRef.current;
    if (!g) return;
    g.traverse((obj) => {
      if (
        obj instanceof THREE.Mesh ||
        obj instanceof THREE.LineSegments ||
        obj instanceof THREE.Line
      ) {
        obj.layers.set(LAYER_FRUSTUM_GUIDE);
      }
    });
  }, [
    lines,
    near,
    sphereRadius,
    planeWidth,
    planeHeight,
    sphereOpacity,
    lineOpacity,
    planeOpacity,
    showCpuCyan,
    showCpuYellow,
    sphereHitDisplayPositions,
    projectionMarkers,
    lidarPyramidHeight,
    azimuthSpanDeg,
    polarSpanDeg,
    lidarPyramidGeometry,
  ]);

  return (
    <>
      <PivotControls
        matrix={sensorMatrix}
        onDrag={(mL) => {
          setSensorMatrix(mL.clone());
        }}
        visible={lidarPivotSelected}
        enabled={lidarPivotSelected}
        disableScaling
        depthTest={false}
        scale={1.25}
        lineWidth={3}
      >
        <group ref={guideRootRef}>
          <mesh renderOrder={carDriveActive ? 120 : 0}>
            <sphereGeometry args={[sphereRadius, 32, 32]} />
            <meshStandardMaterial
              color="#444444"
              wireframe
              transparent
              opacity={sphereOpacity}
              depthWrite={false}
              depthTest
            />
          </mesh>

          <lineSegments>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[linePositions, 3]} />
            </bufferGeometry>
            <lineBasicMaterial
              color="#8888ff"
              transparent
              opacity={lineOpacity}
              depthWrite={false}
            />
          </lineSegments>

          <mesh position={[0, 0, -near]}>
            <planeGeometry args={[planeWidth, planeHeight]} />
            {sensorPlaneDepthMat && drawDepthmap ? (
              <primitive object={sensorPlaneDepthMat} attach="material" />
            ) : (
              <meshStandardMaterial
                color="#ff4444"
                transparent
                opacity={planeOpacity}
                side={THREE.DoubleSide}
                depthWrite={false}
                depthTest
              />
            )}
          </mesh>

          {showCpuCyan ? (
            <InstancedSphereMarkers
              id="sphere-hit"
              positions={sphereHitDisplayPositions}
              size={sphereHitSize}
              color={SPHERE_HIT_COLOR}
              opacity={sphereHitOpacity}
              depthTest={!cpuCyanDrawOverCar}
            />
          ) : null}

          {showCpuYellow ? (
            <InstancedSphereMarkers
              id="projection-hit"
              positions={projectionMarkers}
              size={nearPointSize}
              color="#ffff00"
              opacity={1}
              depthTest={!cpuYellowDrawOverCar}
            />
          ) : null}

          <mesh
            geometry={lidarPyramidGeometry}
            onPointerDown={(e) => {
              e.stopPropagation();
              // 같은 클릭에서 Orbit이 먼저 잡지 않도록 즉시 끔 (useEffect는 한 틱 늦을 수 있음)
              setOrbitControlsEnabled(controls, false);
              setLidarPivotSelected(true);
            }}
          >
            <meshStandardMaterial
              color={lidarPivotSelected ? '#ff4444' : '#cc2222'}
              transparent
              opacity={lidarPyramidOpacity}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>
        {drawDepthmap ? (
          <mesh ref={depthSolidSphereRef} renderOrder={carDriveActive ? 120 : 0}>
            <sphereGeometry args={[sphereRadius, 32, 32]} />
            <meshBasicMaterial color="#6a6a6a" />
          </mesh>
        ) : null}
      </PivotControls>

      {gpuPass ? (
        <>
          <primitive object={gpuPass.cyanMesh} />
          <primitive object={gpuPass.yellowMesh} />
        </>
      ) : null}
    </>
  );
};

function CameraAndControlsSync({
  position,
  target,
}: {
  position: Vec3;
  target: Vec3;
}) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);

  useLayoutEffect(() => {
    camera.position.set(position[0], position[1], position[2]);
    const oc = controls as unknown as { target?: THREE.Vector3; update?: () => void } | null;
    if (oc?.target && oc.update) {
      oc.target.set(target[0], target[1], target[2]);
      oc.update();
    }
  }, [camera, controls, position, target]);
  return null;
}

function Background({ intensity }: { intensity: number }) {
  const scene = useThree((s) => s.scene);
  const intensityRef = useRef(intensity);
  intensityRef.current = intensity;

  useLayoutEffect(() => {
    // Three.js Scene.backgroundIntensity 공식 API
    // eslint-disable-next-line react-hooks/immutability -- three.js 가변 Scene
    scene.backgroundIntensity = intensity;
  }, [scene, intensity]);

  useEffect(() => {
    let alive = true;
    const loader = new HDRLoader();
    loader.load(HDR_URL, (texture) => {
      if (!alive) {
        texture.dispose();
        return;
      }
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      scene.background = texture;
      scene.backgroundRotation = new THREE.Euler(0, Math.PI, 0);
      scene.backgroundIntensity = intensityRef.current;
      scene.environment = texture;
      scene.environmentIntensity = intensityRef.current;
    });
    return () => {
      alive = false;
    };
  }, [scene]);

  return null;
}

function ManagedOrbitControls({ onRelease }: { onRelease: (pos: Vec3, tgt: Vec3) => void }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);
  return (
    <OrbitControls
      makeDefault
      onEnd={() => {
        const oc = controls as unknown as { target: THREE.Vector3 } | null;
        if (!oc?.target) return;
        onRelease(
          [camera.position.x, camera.position.y, camera.position.z],
          [oc.target.x, oc.target.y, oc.target.z],
        );
      }}
    />
  );
}

/** 패널에서 버튼으로 스냅샷할 때 사용 — R3F 카메라·OrbitControls.target 을 읽음 */
function ViewportPoseGetterBridge({
  getterRef,
}: {
  getterRef: React.MutableRefObject<(() => { camera: Vec3; target: Vec3 } | null) | null>;
}) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);
  useLayoutEffect(() => {
    getterRef.current = () => {
      const oc = controls as unknown as { target?: THREE.Vector3 } | null;
      if (!oc?.target) return null;
      return {
        camera: [camera.position.x, camera.position.y, camera.position.z],
        target: [oc.target.x, oc.target.y, oc.target.z],
      };
    };
    return () => {
      getterRef.current = null;
    };
  }, [camera, controls, getterRef]);
  return null;
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (n: number) => void;
}) {
  return (
    <label
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        gap: '8px 12px',
        alignItems: 'center',
        marginBottom: 10,
        fontSize: 13,
      }}
    >
      <span style={{ color: '#ccc' }}>
        {label}{' '}
        <span style={{ color: '#8cf', fontVariantNumeric: 'tabular-nums' }}>
          {value.toFixed(step >= 1 ? 0 : step >= 0.1 ? 1 : 2)}
          {unit}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: 120 }}
      />
    </label>
  );
}

function PanelGroup({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        padding: '12px 14px 14px',
        marginBottom: 6,
        borderRadius: 8,
        background: 'rgba(255,255,255,0.035)',
        boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.06)',
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: 'rgba(150, 200, 255, 0.8)',
          letterSpacing: '0.03em',
          marginBottom: 10,
        }}
      >
        {title}
      </div>
      {children}
    </section>
  );
}

const PRESET_DURATION_MS = 1800;

/**
 * GLB 루트: opacity>0 이면 레이어 0에서 일반 렌더.
 * opacity≤0 이면 메인 뷰에서는 보이지 않지만 `LAYER_DEPTH_SIM_INVISIBLE` 로 LiDAR 깊이 패스에만 기여(포인트 시뮬 유지).
 */
function applyGltfRootSurfaceState(
  root: THREE.Object3D,
  opacity: number,
  depthWrite: boolean,
) {
  root.visible = true;
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    if (opacity <= 0) {
      child.layers.set(LAYER_DEPTH_SIM_INVISIBLE);
    } else {
      child.layers.set(LAYER_DEPTH_OCCLUDER);
    }
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    for (const mat of mats) {
      if (!mat) continue;
      const m = mat as THREE.Material & { colorWrite?: boolean };
      if (opacity <= 0) {
        m.transparent = false;
        m.opacity = 1;
        m.depthWrite = depthWrite;
        m.depthTest = true;
        m.colorWrite = false;
        m.needsUpdate = true;
      } else {
        m.transparent = opacity < 1;
        m.opacity = opacity;
        m.depthWrite = depthWrite;
        m.depthTest = true;
        m.colorWrite = true;
        m.needsUpdate = true;
      }
    }
  });
}

function Car({
  worldPosition,
  opacity,
  depthWrite,
  onSceneMount,
}: {
  worldPosition: Vec3;
  opacity: number;
  depthWrite: boolean;
  onSceneMount?: (root: THREE.Object3D | null) => void;
}) {
  const gltf = useLoader(GLTFLoader, `${PUBLIC_BASE}car3.glb`);

  useLayoutEffect(() => {
    applyGltfRootSurfaceState(gltf.scene, opacity, depthWrite);
  }, [gltf, opacity, depthWrite]);

  useLayoutEffect(() => {
    onSceneMount?.(gltf.scene);
    return () => onSceneMount?.(null);
  }, [gltf, onSceneMount]);

  return (
    <primitive
      object={gltf.scene}
      scale={[CAR_UNIFORM_SCALE, CAR_UNIFORM_SCALE, CAR_UNIFORM_SCALE]}
      position={[worldPosition[0], worldPosition[1], worldPosition[2]]}
      rotation={[0, 0, 0]}
    />
  );
}

const STREET_SEGMENT_OFFSETS: readonly (readonly [number, number, number])[] = [
  [0, -2, 0],
  [0, -2, -30],
  [0, -2, -60],
  [0, -2, -90],
];

function Street({
  opacity,
  depthWrite,
}: {
  opacity: number;
  depthWrite: boolean;
}) {
  const gltf = useLoader(GLTFLoader, `${PUBLIC_BASE}street.glb`);
  const segmentRoots = useMemo(() => {
    const roots: THREE.Object3D[] = [gltf.scene];
    for (let i = 0; i < STREET_SEGMENT_OFFSETS.length - 1; i++) {
      roots.push(gltf.scene.clone(true));
    }
    return roots;
  }, [gltf]);

  useLayoutEffect(() => {
    for (const root of segmentRoots) {
      applyGltfRootSurfaceState(root, opacity, depthWrite);
    }
  }, [segmentRoots, opacity, depthWrite]);

  return (
    <group>
      {segmentRoots.map((root, i) => (
        <primitive
          key={i}
          object={root}
          scale={[0.02, 0.02, 0.02]}
          position={STREET_SEGMENT_OFFSETS[i] as [number, number, number]}
          rotation={[0, 0, 0]}
        />
      ))}
    </group>
  );
}

export default function App() {
  const [sphereOpacity, setSphereOpacity] = useState(INITIAL_SCENE.sphereOpacity);
  const [lineOpacity, setLineOpacity] = useState(INITIAL_SCENE.lineOpacity);
  const [planeOpacity, setPlaneOpacity] = useState(INITIAL_SCENE.planeOpacity);
  const [drawDepthmap, setDrawDepthmap] = useState(INITIAL_SCENE.drawDepthmap);
  const [near, setNear] = useState(INITIAL_SCENE.near);
  const [azimuthSpanDeg, setAzimuthSpanDeg] = useState(INITIAL_SCENE.azimuthSpanDeg);
  const [polarSpanDeg, setPolarSpanDeg] = useState(INITIAL_SCENE.polarSpanDeg);
  const [azimuthDivisions, setAzimuthDivisions] = useState(INITIAL_SCENE.azimuthDivisions);
  const [polarDivisions, setPolarDivisions] = useState(INITIAL_SCENE.polarDivisions);
  const [sphereHitSize, setSphereHitSize] = useState(INITIAL_SCENE.sphereHitSize);
  const [sphereHitOpacity, setSphereHitOpacity] = useState(INITIAL_SCENE.sphereHitOpacity);
  const [nearPointSize, setNearPointSize] = useState(INITIAL_SCENE.nearPointSize);
  const [backgroundIntensity, setBackgroundIntensity] = useState(
    INITIAL_SCENE.backgroundIntensity,
  );
  const [cameraPosition, setCameraPosition] = useState<Vec3>(INITIAL_SCENE.cameraPosition);
  const [orbitTarget, setOrbitTarget] = useState<Vec3>(INITIAL_SCENE.orbitTarget);
  /** 체크 시: 노란 투영점을 near 평면 식으로만 계산 (레이캐스트 안 함). */
  const [projectMarkersOnNearPlaneOnly, setProjectMarkersOnNearPlaneOnly] = useState(
    INITIAL_SCENE.projectMarkersOnNearPlaneOnly,
  );
  const [cyanHitMode, setCyanHitMode] = useState<CyanHitMode>(INITIAL_SCENE.cyanHitMode);
  const [carRaycastTarget, setCarRaycastTarget] = useState<THREE.Object3D | null>(null);
  const [carOpacity, setCarOpacity] = useState(INITIAL_SCENE.carOpacity);
  const [streetOpacity, setStreetOpacity] = useState(INITIAL_SCENE.streetOpacity);
  const [carX, setCarX] = useState(INITIAL_SCENE.carPosition[0]);
  const [carY, setCarY] = useState(INITIAL_SCENE.carPosition[1]);
  const [carZ, setCarZ] = useState(INITIAL_SCENE.carPosition[2]);
  const animFrameRef = useRef<number | null>(null);
  const carDriveRafRef = useRef<number | null>(null);
  /** `carDrive` 애니 중이면 `applySnapshot`이 차 위치를 덮어쓰지 않음 */
  const carDriveActiveRef = useRef(false);
  /** `null`이면 `carPosition` 슬라이더 사용. `carDrive` 중에는 보간 위치 */
  const [carWorldPosition, setCarWorldPosition] = useState<Vec3 | null>(null);
  const [hitNoiseLevel, setHitNoiseLevel] = useState(INITIAL_SCENE.hitNoiseLevel);
  const [lidarPyramidHeight, setLidarPyramidHeight] = useState(
    INITIAL_SCENE.lidarPyramidHeight,
  );
  const [lidarPyramidOpacity, setLidarPyramidOpacity] = useState(
    INITIAL_SCENE.lidarPyramidOpacity,
  );
  const [lidarSampleRateHz, setLidarSampleRateHz] = useState(
    INITIAL_SCENE.lidarSampleRateHz,
  );
  const [lidarMaxRange, setLidarMaxRange] = useState(INITIAL_SCENE.lidarMaxRange);
  /** Camera projection 옵션(슬라이더) 패널 — 기본 접힘 */
  const [projectionOptionsOpen, setProjectionOptionsOpen] = useState(false);
  /** 우하단 프리셋 번호 — 마지막으로 누른 스텝(초기 1번) */
  const [activePresetIndex, setActivePresetIndex] = useState(0);

  const viewportPoseGetterRef = useRef<(() => { camera: Vec3; target: Vec3 } | null) | null>(
    null,
  );
  const [panelCameraReadout, setPanelCameraReadout] = useState<Vec3>(() => [
    ...INITIAL_SCENE.cameraPosition,
  ]);
  const [panelTargetReadout, setPanelTargetReadout] = useState<Vec3>(() => [
    ...INITIAL_SCENE.orbitTarget,
  ]);

  const refreshViewportPoseReadout = useCallback(() => {
    const snap = viewportPoseGetterRef.current?.();
    if (!snap) return;
    setPanelCameraReadout([...snap.camera]);
    setPanelTargetReadout([...snap.target]);
  }, []);

  const copyCurrentScenePresetSnippet = useCallback(() => {
    const pose = viewportPoseGetterRef.current?.();
    const cam = pose?.camera ?? ([...cameraPosition] as Vec3);
    const tgt = pose?.target ?? ([...orbitTarget] as Vec3);
    const s: SceneSnapshot = {
      near,
      azimuthSpanDeg,
      polarSpanDeg,
      azimuthDivisions,
      polarDivisions,
      sphereOpacity,
      lineOpacity,
      planeOpacity,
      drawDepthmap,
      sphereHitSize,
      sphereHitOpacity,
      nearPointSize,
      projectMarkersOnNearPlaneOnly,
      cyanHitMode,
      hitNoiseLevel,
      lidarPyramidHeight,
      lidarPyramidOpacity,
      lidarSampleRateHz,
      lidarMaxRange,
      carOpacity,
      streetOpacity,
      carPosition:
        carWorldPosition ??
        ([carX, carY, carZ] as [number, number, number]),
      backgroundIntensity,
      cameraPosition: [...cam],
      orbitTarget: [...tgt],
    };
    setCameraPosition([...cam]);
    setOrbitTarget([...tgt]);
    setPanelCameraReadout([...cam]);
    setPanelTargetReadout([...tgt]);
    void navigator.clipboard.writeText(formatSceneSnapshotForPresetPaste(s));
  }, [
    near,
    azimuthSpanDeg,
    polarSpanDeg,
    azimuthDivisions,
    polarDivisions,
    sphereOpacity,
    lineOpacity,
    planeOpacity,
    drawDepthmap,
    sphereHitSize,
    sphereHitOpacity,
    nearPointSize,
    projectMarkersOnNearPlaneOnly,
    cyanHitMode,
    hitNoiseLevel,
    lidarPyramidHeight,
    lidarPyramidOpacity,
    lidarSampleRateHz,
    lidarMaxRange,
    carOpacity,
    streetOpacity,
    carX,
    carY,
    carZ,
    carWorldPosition,
    backgroundIntensity,
    cameraPosition,
    orbitTarget,
  ]);

  const onCarSceneMount = useCallback((root: THREE.Object3D | null) => {
    setCarRaycastTarget(root);
  }, []);

  useEffect(() => {
    return () => {
      if (animFrameRef.current != null) cancelAnimationFrame(animFrameRef.current);
      if (carDriveRafRef.current != null) cancelAnimationFrame(carDriveRafRef.current);
    };
  }, []);

  const snapshot = (): SceneSnapshot => ({
    sphereOpacity,
    lineOpacity,
    planeOpacity,
    drawDepthmap,
    near,
    azimuthSpanDeg,
    polarSpanDeg,
    azimuthDivisions,
    polarDivisions,
    sphereHitSize,
    sphereHitOpacity,
    nearPointSize,
    backgroundIntensity,
    cameraPosition,
    orbitTarget,
    carOpacity,
    streetOpacity,
    carPosition: carWorldPosition ?? ([carX, carY, carZ] as Vec3),
    hitNoiseLevel,
    lidarPyramidHeight,
    lidarPyramidOpacity,
    lidarSampleRateHz,
    lidarMaxRange,
    projectMarkersOnNearPlaneOnly,
    cyanHitMode,
  });

  const applySnapshot = (s: SceneSnapshot) => {
    setSphereOpacity(s.sphereOpacity);
    setLineOpacity(s.lineOpacity);
    setPlaneOpacity(s.planeOpacity);
    setDrawDepthmap(s.drawDepthmap);
    setNear(s.near);
    setAzimuthSpanDeg(s.azimuthSpanDeg);
    setPolarSpanDeg(s.polarSpanDeg);
    setAzimuthDivisions(s.azimuthDivisions);
    setPolarDivisions(s.polarDivisions);
    setSphereHitSize(s.sphereHitSize);
    setSphereHitOpacity(s.sphereHitOpacity);
    setNearPointSize(s.nearPointSize);
    setBackgroundIntensity(s.backgroundIntensity);
    setCameraPosition([...s.cameraPosition]);
    setOrbitTarget([...s.orbitTarget]);
    setCarOpacity(s.carOpacity);
    setStreetOpacity(s.streetOpacity);
    if (!carDriveActiveRef.current) {
      setCarX(s.carPosition[0]);
      setCarY(s.carPosition[1]);
      setCarZ(s.carPosition[2]);
    }
    setHitNoiseLevel(s.hitNoiseLevel);
    setLidarPyramidHeight(s.lidarPyramidHeight);
    setLidarPyramidOpacity(s.lidarPyramidOpacity);
    setLidarSampleRateHz(s.lidarSampleRateHz);
    setLidarMaxRange(s.lidarMaxRange);
    setProjectMarkersOnNearPlaneOnly(s.projectMarkersOnNearPlaneOnly);
    setCyanHitMode(s.cyanHitMode);
  };

  const cancelCarDrive = useCallback(() => {
    if (carDriveRafRef.current != null) {
      cancelAnimationFrame(carDriveRafRef.current);
      carDriveRafRef.current = null;
    }
    carDriveActiveRef.current = false;
    setCarWorldPosition(null);
  }, []);

  const startCarDrive = useCallback(
    (cfg: CarDriveConfig) => {
      cancelCarDrive();
      carDriveActiveRef.current = true;
      const [sx, sy, sz] = cfg.start;
      const [ex, ey, ez] = cfg.end;
      const dur = Math.max(1, cfg.durationMs);
      const t0 = performance.now();
      setCarWorldPosition([sx, sy, sz]);
      setCarX(sx);
      setCarY(sy);
      setCarZ(sz);
      const driveTick = (now: number) => {
        const u = Math.min(1, (now - t0) / dur);
        const k = easeInOutCubic(u);
        const x = lerp(sx, ex, k);
        const y = lerp(sy, ey, k);
        const z = lerp(sz, ez, k);
        setCarWorldPosition([x, y, z]);
        setCarX(x);
        setCarY(y);
        setCarZ(z);
        if (u < 1) carDriveRafRef.current = requestAnimationFrame(driveTick);
        else {
          carDriveRafRef.current = null;
          carDriveActiveRef.current = false;
          setCarWorldPosition(null);
          setCarX(ex);
          setCarY(ey);
          setCarZ(ez);
        }
      };
      carDriveRafRef.current = requestAnimationFrame(driveTick);
    },
    [cancelCarDrive],
  );

  const runPreset = (presetIndex: number) => {
    const goal = PRESETS[presetIndex];
    if (!goal) return;
    setActivePresetIndex(presetIndex);
    if (animFrameRef.current != null) cancelAnimationFrame(animFrameRef.current);
    const from = snapshot();
    if (goal.carDrive) {
      startCarDrive(goal.carDrive);
    } else {
      cancelCarDrive();
    }
    let tStart = -1;

    const tick = (now: number) => {
      if (tStart < 0) tStart = now;
      const rawT = Math.min(1, (now - tStart) / PRESET_DURATION_MS);
      const k = easeInOutCubic(rawT);
      applySnapshot({
        sphereOpacity: lerp(from.sphereOpacity, goal.sphereOpacity, k),
        lineOpacity: lerp(from.lineOpacity, goal.lineOpacity, k),
        planeOpacity: lerp(from.planeOpacity, goal.planeOpacity, k),
        near: lerp(from.near, goal.near, k),
        azimuthSpanDeg: lerp(from.azimuthSpanDeg, goal.azimuthSpanDeg, k),
        polarSpanDeg: lerp(from.polarSpanDeg, goal.polarSpanDeg, k),
        azimuthDivisions: Math.round(lerp(from.azimuthDivisions, goal.azimuthDivisions, k)),
        polarDivisions: Math.round(lerp(from.polarDivisions, goal.polarDivisions, k)),
        sphereHitSize: lerp(from.sphereHitSize, goal.sphereHitSize, k),
        sphereHitOpacity: lerp(from.sphereHitOpacity, goal.sphereHitOpacity, k),
        nearPointSize: lerp(from.nearPointSize, goal.nearPointSize, k),
        backgroundIntensity: lerp(from.backgroundIntensity, goal.backgroundIntensity, k),
        cameraPosition: lerpVec3(from.cameraPosition, goal.cameraPosition, k),
        orbitTarget: lerpVec3(from.orbitTarget, goal.orbitTarget, k),
        carOpacity: lerp(from.carOpacity, goal.carOpacity, k),
        streetOpacity: lerp(from.streetOpacity, goal.streetOpacity, k),
        hitNoiseLevel: lerp(from.hitNoiseLevel, goal.hitNoiseLevel, k),
        lidarPyramidHeight: lerp(from.lidarPyramidHeight, goal.lidarPyramidHeight, k),
        lidarPyramidOpacity: lerp(from.lidarPyramidOpacity, goal.lidarPyramidOpacity, k),
        lidarSampleRateHz: lerp(from.lidarSampleRateHz, goal.lidarSampleRateHz, k),
        lidarMaxRange: lerp(from.lidarMaxRange, goal.lidarMaxRange, k),
        projectMarkersOnNearPlaneOnly: goal.projectMarkersOnNearPlaneOnly,
        cyanHitMode: goal.cyanHitMode,
        drawDepthmap: goal.drawDepthmap,
        carPosition: goal.carDrive
          ? from.carPosition
          : lerpVec3(from.carPosition, goal.carPosition, k),
      });
      if (rawT < 1) animFrameRef.current = requestAnimationFrame(tick);
      else animFrameRef.current = null;
    };
    animFrameRef.current = requestAnimationFrame(tick);
  };

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        minHeight: '100svh',
        flex: 1,
        background: '#111',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <Canvas
        style={{ width: '100%', height: '100%', display: 'block' }}
        camera={{ position: [...cameraPosition] as [number, number, number], fov: 60 }}
        gl={async (props) => {
          const renderer = new THREE.WebGPURenderer(
            props as ConstructorParameters<typeof THREE.WebGPURenderer>[0],
          );
          await renderer.init();
          return renderer;
        }}
      >
        <Background intensity={backgroundIntensity} />
        {/* <ambientLight intensity={1} /> */}
        <CameraAndControlsSync position={cameraPosition} target={orbitTarget} />
        <ViewportPoseGetterBridge getterRef={viewportPoseGetterRef} />
        <MainCameraEnableFrustumGuideLayer />
        <Car
          worldPosition={carWorldPosition ?? ([carX, carY, carZ] as Vec3)}
          opacity={carOpacity}
          depthWrite={true}
          onSceneMount={onCarSceneMount}
        />
        <Street opacity={streetOpacity} depthWrite />
        <FrustumVisualizer
          sphereOpacity={sphereOpacity}
          lineOpacity={lineOpacity}
          planeOpacity={planeOpacity}
          near={near}
          azimuthSpanDeg={azimuthSpanDeg}
          polarSpanDeg={polarSpanDeg}
          azimuthDivisions={azimuthDivisions}
          polarDivisions={polarDivisions}
          sphereHitSize={sphereHitSize}
          sphereHitOpacity={sphereHitOpacity}
          nearPointSize={nearPointSize}
          carRaycastTarget={carRaycastTarget}
          projectMarkersOnNearPlaneOnly={projectMarkersOnNearPlaneOnly}
          cyanHitMode={cyanHitMode}
          hitNoiseLevel={hitNoiseLevel}
          lidarPyramidHeight={lidarPyramidHeight}
          lidarPyramidOpacity={lidarPyramidOpacity}
          lidarSampleRateHz={lidarSampleRateHz}
          lidarMaxRange={lidarMaxRange}
          carDriveActive={carWorldPosition !== null}
          drawDepthmap={drawDepthmap}
        />
        <ManagedOrbitControls
          onRelease={(pos, tgt) => {
            setCameraPosition(pos);
            setOrbitTarget(tgt);
          }}
        />

      </Canvas>

      <div
        style={{
          position: 'absolute',
          top: 8,
          left: 8,
          ...(projectionOptionsOpen ? { bottom: 8 } : {}),
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          color: 'white',
          fontFamily: 'system-ui, sans-serif',
          background: 'rgba(0,0,0,0.55)',
          borderRadius: 8,
          maxWidth: 340,
          backdropFilter: 'blur(6px)',
          zIndex: 10,
          overflow: 'hidden',
        }}
      >
        <button
          type="button"
          id="projection-panel-heading"
          aria-expanded={projectionOptionsOpen}
          aria-controls="projection-options-panel"
          onClick={() => setProjectionOptionsOpen((o) => !o)}
          style={{
            margin: projectionOptionsOpen ? '0 0 12px' : 0,
            fontSize: 15,
            fontWeight: 600,
            backdropFilter: 'blur(6px)',
            padding: '8px 18px',
            flexShrink: 0,
            border: 'none',
            background: 'transparent',
            color: 'inherit',
            fontFamily: 'inherit',
            textAlign: 'left',
            cursor: 'pointer',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            borderRadius: 8,
          }}
        >
          <span
            aria-hidden
            style={{
              display: 'inline-block',
              width: '1em',
              color: '#9df',
              fontSize: 12,
              lineHeight: 1,
            }}
          >
            {projectionOptionsOpen ? '▼' : '▶'}
          </span>
          Camera projection (WebGPU)
        </button>
        {projectionOptionsOpen ? (
          <div
            id="projection-options-panel"
            role="region"
            aria-labelledby="projection-panel-heading"
            style={{
              padding: '8px 4px',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              flex: 1,
              minHeight: 0,
              overflowY: 'auto',
              WebkitOverflowScrolling: 'touch',
            }}
          >
            <div style={{ padding: '0 10px 4px', flexShrink: 0 }}>
              <button
                type="button"
                title="뷰포트 카메라·오빗 타깃으로 동기화한 뒤 SceneSnapshot 리터럴을 클립보드에 복사"
                onClick={copyCurrentScenePresetSnippet}
                style={{
                  width: '100%',
                  fontSize: 13,
                  fontWeight: 600,
                  padding: '8px 12px',
                  borderRadius: 8,
                  border: '1px solid rgba(120, 200, 255, 0.35)',
                  background: 'rgba(80, 140, 220, 0.22)',
                  color: '#dff4ff',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                현재값 복사
              </button>
            </div>
            {/*
              패널 순서(의도 유지): 차량·바닥 → 카메라·배경 → 프러스텀·격자 → 가이드(뷰 평면 투명도 슬라이더는 주석 유지) → 교차점·시안 → 라이다 → 카메라·오빗 타겟(읽기).
              FrustumVisualizer: near 빨간 뷰 평면 mesh 주석은 필요 시 여기와 슬라이더를 함께 복구.
            */}
            <PanelGroup title="차량 · 바닥">
              <SliderRow
                label="자동차 투명도 (0 = 완전 숨김)"
                value={carOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setCarOpacity}
              />
              <SliderRow
                label="차량 월드 X"
                value={carX}
                min={-12}
                max={12}
                step={CAR_Z_SLIDER_STEP}
                unit=""
                onChange={(x) => {
                  cancelCarDrive();
                  setCarX(x);
                }}
              />
              <SliderRow
                label="차량 월드 Y"
                value={carY}
                min={-12}
                max={12}
                step={CAR_Z_SLIDER_STEP}
                unit=""
                onChange={(y) => {
                  cancelCarDrive();
                  setCarY(y);
                }}
              />
              <SliderRow
                label="차량 월드 Z (전후, carDrive 없을 때)"
                value={carZ}
                min={CAR_Z_SLIDER_MIN}
                max={CAR_Z_SLIDER_MAX}
                step={CAR_Z_SLIDER_STEP}
                unit=""
                onChange={(z) => {
                  cancelCarDrive();
                  setCarZ(z);
                }}
              />
              <SliderRow
                label="바닥(street) 투명도 (0 = 레이·깊이 통과)"
                value={streetOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setStreetOpacity}
              />
            </PanelGroup>
            <PanelGroup title="카메라 · 배경">
              <SliderRow
                label="HDR 배경 강도 (backgroundIntensity)"
                value={backgroundIntensity}
                min={0}
                max={3}
                step={0.05}
                unit=""
                onChange={setBackgroundIntensity}
              />
            </PanelGroup>
            <PanelGroup title="프러스텀 · 격자">
              <SliderRow
                label="뷰 평면 ↔ 원점 거리 (near)"
                value={near}
                min={0.4}
                max={12}
                step={0.1}
                unit=""
                onChange={setNear}
              />
              <SliderRow
                label="가로 — 방위각(azimuth, θ) 범위"
                value={azimuthSpanDeg}
                min={15}
                max={180}
                step={1}
                unit="°"
                onChange={setAzimuthSpanDeg}
              />
              <SliderRow
                label="세로 — 극각(polar, φ) 범위"
                value={polarSpanDeg}
                min={15}
                max={180}
                step={1}
                unit="°"
                onChange={setPolarSpanDeg}
              />
              <SliderRow
                label="가로 격자 구간 수 (θ)"
                value={azimuthDivisions}
                min={2}
                max={256}
                step={1}
                unit="개"
                onChange={setAzimuthDivisions}
              />
              <SliderRow
                label="세로 격자 구간 수 (φ)"
                value={polarDivisions}
                min={2}
                max={128}
                step={1}
                unit="개"
                onChange={setPolarDivisions}
              />
            </PanelGroup>
            <PanelGroup title="가이드 (구·선·뷰 평면)">
              <SliderRow
                label="광선(선분) 투명도"
                value={lineOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setLineOpacity}
              />
              <SliderRow
                label="구(와이어프레임) 투명도"
                value={sphereOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setSphereOpacity}
              />
              {/* 뷰 평면 mesh 비활성 시 슬라이더도 함께 주석 — 복구 시 FrustumVisualizer near plane 주석 해제 */}
              <SliderRow
                label="뷰 평면(near 면) 투명도"
                value={planeOpacity}
                min={0}
                max={1}
                step={0.01}
                unit=""
                onChange={setPlaneOpacity}
              />
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                  marginTop: 8,
                  fontSize: 13,
                  color: '#ccc',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={drawDepthmap}
                  onChange={() => setDrawDepthmap((v) => !v)}
                  style={{ marginTop: 3 }}
                />
                <span>
                  빨간 평면에 센서 시점 깊이 텍스처
                  <span style={{ color: '#888', fontSize: 11, display: 'block', marginTop: 4 }}>
                    구·차·바닥을 라이다 각도로 렌더한 깊이를 평면에 표시합니다.
                  </span>
                </span>
              </label>
            </PanelGroup>
            <PanelGroup title="교차점 · 시안">
              <SliderRow
                label="구면 교점(시안) 크기"
                value={sphereHitSize}
                min={0.02}
                max={0.5}
                step={0.01}
                unit=""
                onChange={setSphereHitSize}
              />
              <SliderRow
                label="구면 교점 투명도"
                value={sphereHitOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setSphereHitOpacity}
              />
              <fieldset
                style={{
                  border: 'none',
                  padding: 0,
                  margin: '0 0 4px',
                }}
              >
                <legend
                  style={{
                    fontSize: 13,
                    color: '#ccc',
                    marginBottom: 8,
                    padding: 0,
                  }}
                >
                  시안(구면 교점)
                </legend>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                    marginBottom: 8,
                    fontSize: 13,
                    color: '#ccc',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="radio"
                    name="cyanHitMode"
                    checked={cyanHitMode === 'sphereOnly'}
                    onChange={() => setCyanHitMode('sphereOnly')}
                    style={{ marginTop: 3 }}
                  />
                  <span>
                    <strong style={{ color: '#9df' }}>구면에만</strong> 투영
                    <span style={{ color: '#888', fontSize: 11, display: 'block', marginTop: 4 }}>
                      외접 구면과의 해석적 교차(차·바닥과 무관).
                    </span>
                  </span>
                </label>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                    fontSize: 13,
                    color: '#ccc',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="radio"
                    name="cyanHitMode"
                    checked={cyanHitMode === 'depthSim'}
                    onChange={() => setCyanHitMode('depthSim')}
                    style={{ marginTop: 3 }}
                  />
                  <span>
                    <strong style={{ color: '#fd9' }}>차량, 바닥</strong>에 시뮬레이션
                    <span style={{ color: '#888', fontSize: 11, display: 'block', marginTop: 4 }}>
                      GPU 깊이 맵으로 메시 표면에 포인트(차·street 가시 시).
                    </span>
                  </span>
                </label>
              </fieldset>
            </PanelGroup>
            <PanelGroup title="라이다 시뮬">
              <SliderRow
                label="라이다 위치 피라미드 (전방 길이)"
                value={lidarPyramidHeight}
                min={0.08}
                max={2.5}
                step={0.02}
                unit=""
                onChange={setLidarPyramidHeight}
              />
              <SliderRow
                label="라이다 위치 피라미드 투명도"
                value={lidarPyramidOpacity}
                min={0}
                max={1}
                step={0.02}
                unit=""
                onChange={setLidarPyramidOpacity}
              />
              <SliderRow
                label="GPU 히트 위치 노이즈"
                value={hitNoiseLevel}
                min={0}
                max={0.3}
                step={0.01}
                unit=""
                onChange={setHitNoiseLevel}
              />
              <SliderRow
                label="라이다/깊이 포인트 갱신 주파수"
                value={lidarSampleRateHz}
                min={0.1}
                max={20}
                step={0.1}
                unit=" Hz"
                onChange={setLidarSampleRateHz}
              />
              <SliderRow
                label="라이다 최대 거리 (far / range)"
                value={lidarMaxRange}
                min={5}
                max={100}
                step={1}
                unit=""
                onChange={setLidarMaxRange}
              />
            </PanelGroup>

            <PanelGroup title="카메라 · 오빗 타겟 (읽기)">
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  marginBottom: 8,
                }}
              >
                <button
                  type="button"
                  onClick={refreshViewportPoseReadout}
                  style={{
                    fontSize: 12,
                    padding: '4px 10px',
                    borderRadius: 6,
                    border: '1px solid rgba(255,255,255,0.2)',
                    background: 'rgba(255,255,255,0.08)',
                    color: '#e8f4ff',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  업데이트
                </button>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  marginBottom: 6,
                }}
              >
                <span style={{ fontSize: 11, color: '#9ab' }}>카메라 위치 (world)</span>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard.writeText(vec3TupleClipboardText(panelCameraReadout));
                  }}
                  style={{
                    fontSize: 11,
                    padding: '3px 8px',
                    borderRadius: 6,
                    border: '1px solid rgba(255,255,255,0.2)',
                    background: 'rgba(255,255,255,0.06)',
                    color: '#cfe8ff',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    flexShrink: 0,
                  }}
                >
                  복사
                </button>
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '14px 1fr',
                  gap: '4px 6px',
                  marginBottom: 10,
                  alignItems: 'center',
                }}
              >
                {(['X', 'Y', 'Z'] as const).map((axis, i) => (
                  <React.Fragment key={`cam-${axis}`}>
                    <span style={{ color: '#8ac', fontVariantNumeric: 'tabular-nums' }}>{axis}</span>
                    <input
                      readOnly
                      value={panelCameraReadout[i].toFixed(4)}
                      aria-label={`카메라 ${axis}`}
                      style={{
                        width: '100%',
                        boxSizing: 'border-box',
                        fontSize: 12,
                        padding: '4px 8px',
                        borderRadius: 4,
                        border: '1px solid rgba(255,255,255,0.12)',
                        background: 'rgba(0,0,0,0.35)',
                        color: '#e8f0ff',
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    />
                  </React.Fragment>
                ))}
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  marginBottom: 6,
                }}
              >
                <span style={{ fontSize: 11, color: '#9ab' }}>컨트롤 타겟 (orbit)</span>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard.writeText(vec3TupleClipboardText(panelTargetReadout));
                  }}
                  style={{
                    fontSize: 11,
                    padding: '3px 8px',
                    borderRadius: 6,
                    border: '1px solid rgba(255,255,255,0.2)',
                    background: 'rgba(255,255,255,0.06)',
                    color: '#cfe8ff',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    flexShrink: 0,
                  }}
                >
                  복사
                </button>
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '14px 1fr',
                  gap: '4px 6px',
                  alignItems: 'center',
                }}
              >
                {(['X', 'Y', 'Z'] as const).map((axis, i) => (
                  <React.Fragment key={`tgt-${axis}`}>
                    <span style={{ color: '#8ac', fontVariantNumeric: 'tabular-nums' }}>{axis}</span>
                    <input
                      readOnly
                      value={panelTargetReadout[i].toFixed(4)}
                      aria-label={`오빗 타겟 ${axis}`}
                      style={{
                        width: '100%',
                        boxSizing: 'border-box',
                        fontSize: 12,
                        padding: '4px 8px',
                        borderRadius: 4,
                        border: '1px solid rgba(255,255,255,0.12)',
                        background: 'rgba(0,0,0,0.35)',
                        color: '#e8f0ff',
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    />
                  </React.Fragment>
                ))}
              </div>
            </PanelGroup>

          </div>
        ) : null}

      </div>

      <div
        role="toolbar"
        aria-label="장면 프리셋"
        style={{
          position: 'absolute',
          right: 20,
          bottom: 20,
          display: 'flex',
          flexDirection: 'row',
          gap: 10,
          zIndex: 20,
        }}
      >
        {PRESETS.map((_, i) => {
          const selected = i === activePresetIndex;
          return (
            <button
              key={i}
              type="button"
              aria-pressed={selected}
              title={`발표 스텝 ${i + 1}/${PRESETS.length}`}
              onClick={() => runPreset(i)}
              style={{
                width: 44,
                height: 44,
                borderRadius: '50%',
                border: selected
                  ? '2px solid rgba(120, 210, 255, 0.95)'
                  : '2px solid rgba(255,255,255,0.35)',
                background: selected
                  ? 'rgba(55, 95, 140, 0.92)'
                  : 'rgba(30,30,35,0.85)',
                color: selected ? '#eaf8ff' : '#fff',
                fontSize: 17,
                fontWeight: selected ? 700 : 600,
                cursor: 'pointer',
                boxShadow: selected
                  ? '0 0 0 2px rgba(100, 190, 255, 0.35), 0 4px 16px rgba(0,0,0,0.45)'
                  : '0 4px 14px rgba(0,0,0,0.35)',
                backdropFilter: 'blur(8px)',
              }}
            >
              {i + 1}
            </button>
          );
        })}
      </div>
    </div>
  );
}
