import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three/webgpu';
import { GLTFLoader, HDRLoader } from 'three/examples/jsm/Addons.js';
import {
  GpuFrustumRayHitPass,
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

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
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

/** 격자 광선 샘플 (시야 클립은 near 직사각형과 동일) */
function sampleFrustumRays(
  near: number,
  azimuthSpanDeg: number,
  polarSpanDeg: number,
  azimuthDivisions: number,
  polarDivisions: number,
  frustumFovXDeg: number,
  frustumFovYDeg: number,
): FrustumRaySample[] {
  const halfX = THREE.MathUtils.degToRad(frustumFovXDeg / 2);
  const halfY = THREE.MathUtils.degToRad(frustumFovYDeg / 2);
  const planeW = 2 * near * Math.tan(halfX);
  const planeH = 2 * near * Math.tan(halfY);
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
const SPHERE_HIT_COLOR = '#00ffc8';

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

const frustumFovY = 45;
const frustumFovX = THREE.MathUtils.radToDeg(
  2 * Math.atan((16 / 9) * Math.tan(THREE.MathUtils.degToRad(frustumFovY / 2))),
);

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
  /** 바닥(street) 투명도 — 0이면 깊이/레이 통과 */
  streetOpacity: number;
  /** GPU 깊이 히트 위치 노이즈 (world 단위, 0–0.3) */
  hitNoiseLevel: number;
  /** 라이다 원점 표시: 전방 −Z로 선 사각 피라미드 높이(꼭짓점~밑면) */
  lidarPyramidHeight: number;
  /** GPU 깊이/포인트 `execute` 갱신 주파수 (Hz) */
  lidarSampleRateHz: number;
  /** 라이다 시뮬 최대 거리(far) */
  lidarMaxRange: number;
  /** 차량 주행 중 등 — Hz 스로틀 없이 매 프레임 깊이·포인트 갱신 */
  forceLidarEveryFrame: boolean;
}

const SENSOR_MATRIX = new THREE.Matrix4();

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
  streetOpacity,
  hitNoiseLevel,
  lidarPyramidHeight,
  lidarSampleRateHz,
  lidarMaxRange,
  forceLidarEveryFrame,
}) => {
  const { gl, scene } = useThree();
  const guideRootRef = useRef<THREE.Group>(null);
  const gpuPassForFrameRef = useRef<GpuFrustumRayHitPass | null>(null);
  const [gpuPass, setGpuPass] = useState<GpuFrustumRayHitPass | null>(null);
  const lastGpuSampleTimeRef = useRef<number>(-Infinity);

  useLayoutEffect(() => {
    lastGpuSampleTimeRef.current = -Infinity;
  }, [lidarSampleRateHz]);

  useLayoutEffect(() => {
    if (forceLidarEveryFrame) lastGpuSampleTimeRef.current = -Infinity;
  }, [forceLidarEveryFrame]);

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

  const depthSceneActive =
    streetOpacity > 0 || carRaycastTarget != null;
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
      const halfX = THREE.MathUtils.degToRad(frustumFovX / 2);
      const halfY = THREE.MathUtils.degToRad(frustumFovY / 2);
      const farX = far * Math.tan(halfX);
      const farY = far * Math.tan(halfY);
      const radius = Math.sqrt(farX * farX + farY * farY + far * far);

      const planeW = 2 * near * Math.tan(halfX);
      const planeH = 2 * near * Math.tan(halfY);

      const raySamples = sampleFrustumRays(
        near,
        azimuthSpanDeg,
        polarSpanDeg,
        azimuthDivisions,
        polarDivisions,
        frustumFovX,
        frustumFovY,
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

  const projectionMarkers = useMemo(() => {
    const useAnalytical = projectMarkersOnNearPlaneOnly || !carRaycastTarget;
    if (useAnalytical) {
      return new Float32Array(nearPlanePoints.flatMap((v) => [v.x, v.y, v.z]));
    }
    if (runGpuHits) {
      return new Float32Array(0);
    }

    carRaycastTarget.updateMatrixWorld(true);
    const raycaster = new THREE.Raycaster();
    const origin = new THREE.Vector3(0, 0, 0);
    const hits: THREE.Vector3[] = [];

    for (const s of samples) {
      if (s.dir.z >= 0) continue;
      raycaster.set(origin, s.dir);
      const res = raycaster.intersectObject(carRaycastTarget, true);
      if (res.length > 0) hits.push(res[0].point.clone());
    }

    return new Float32Array(hits.flatMap((v) => [v.x, v.y, v.z]));
  }, [
    projectMarkersOnNearPlaneOnly,
    carRaycastTarget,
    samples,
    nearPlanePoints,
    runGpuHits,
  ]);

  const sphereHitDisplayPositions = useMemo(() => {
    if (cyanHitMode === 'sphereOnly' || !carRaycastTarget) {
      return new Float32Array(sphereHitPoints.flatMap((v) => [v.x, v.y, v.z]));
    }
    if (runGpuHits) {
      return new Float32Array(0);
    }

    carRaycastTarget.updateMatrixWorld(true);
    const raycaster = new THREE.Raycaster();
    const origin = new THREE.Vector3(0, 0, 0);
    const pts: THREE.Vector3[] = [];

    for (const s of samples) {
      raycaster.set(origin, s.dir);
      const res = raycaster.intersectObject(carRaycastTarget, true);
      if (res.length > 0) pts.push(res[0].point.clone());
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
  ]);

  /* GpuFrustumRayHitPass 생성/해제를 React 트리(<primitive>)와 맞추려면 effect 내 setState가 필요함 */
  /* eslint-disable react-hooks/set-state-in-effect */
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
  /* eslint-enable react-hooks/set-state-in-effect */

  useLayoutEffect(() => {
    if (!gpuPass) return;
    gpuPassForFrameRef.current = gpuPass;
    gpuPass.setMaxRange(sphereRadius);
    gpuPass.setMarkerScales(sphereHitSize, nearPointSize);
    gpuPass.setMarkerOpacities(sphereHitOpacity, 1);
    gpuPass.setCyanUsesHitOnly(cyanDepthSim);
    gpuPass.setHitNoiseLevel(hitNoiseLevel);
    gpuPass.setMeshVisibility(
      cyanDepthSim,
      !projectMarkersOnNearPlaneOnly,
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
    if (!forceLidarEveryFrame) {
      const hz = THREE.MathUtils.clamp(lidarSampleRateHz, 0.1, 20);
      const interval = 1 / hz;
      if (t - lastGpuSampleTimeRef.current < interval) return;
    }
    lastGpuSampleTimeRef.current = t;
    pass.execute(renderer, scene, SENSOR_MATRIX);
  }, -1);

  const linePositions = new Float32Array(lines.flatMap((v) => [v.x, v.y, v.z]));

  const showCpuCyan =
    !runGpuHits || cyanHitMode === 'sphereOnly' || sphereHitOpacity <= 0.001;
  const showCpuYellow = !runGpuHits || projectMarkersOnNearPlaneOnly;

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
      <group ref={guideRootRef}>
        <mesh>
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
          <meshStandardMaterial
            color="#ff4444"
            transparent
            opacity={planeOpacity}
            side={THREE.DoubleSide}
            depthWrite={false}
            depthTest
          />
        </mesh>

        {showCpuCyan ? (
          <InstancedSphereMarkers
            id="sphere-hit"
            positions={sphereHitDisplayPositions}
            size={sphereHitSize}
            color={SPHERE_HIT_COLOR}
            opacity={sphereHitOpacity}
          />
        ) : null}

        {showCpuYellow ? (
          <InstancedSphereMarkers
            id="projection-hit"
            positions={projectionMarkers}
            size={nearPointSize}
            color="#ffff00"
            opacity={1}
          />
        ) : null}

        <mesh geometry={lidarPyramidGeometry}>
          <meshStandardMaterial
            color="#cc2222"
            transparent
            opacity={0.3}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      </group>

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

const PRESET_DURATION_MS = 1800;

/** GLB 루트: opacity 0 이면 완전 숨김(레이·깊이 통과), 아니면 표면·깊이 버퍼에 기여 */
function applyGltfRootSurfaceState(
  root: THREE.Object3D,
  opacity: number,
  depthWrite: boolean,
) {
  if (opacity <= 0) {
    root.visible = false;
    return;
  }

  root.visible = true;
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    for (const mat of mats) {
      if (!mat) continue;
      const m = mat as THREE.Material;
      m.transparent = opacity < 1;
      m.opacity = opacity;
      m.depthWrite = depthWrite;
      m.depthTest = true;
      m.needsUpdate = true;
    }
  });
}

function Car({
  distance,
  worldPosition,
  opacity,
  depthWrite,
  onSceneMount,
}: {
  distance: number;
  /** 없으면 기본 `[-3, -2.1, -distance * 0.7]` */
  worldPosition?: Vec3;
  opacity: number;
  depthWrite: boolean;
  onSceneMount?: (root: THREE.Object3D | null) => void;
}) {
  const gltf = useLoader(GLTFLoader, '/car3.glb');
  const multiplier = distance*0.8;
  const pos = worldPosition ?? ([-3, -2.1, -distance * 0.7] as Vec3);

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
      scale={[multiplier, multiplier, multiplier]}
      position={[pos[0], pos[1], pos[2]]}
      rotation={[0, 0, 0]}
    />
  );
}

function Street({
  opacity,
  depthWrite,
}: {
  opacity: number;
  depthWrite: boolean;
}) {
  const gltf = useLoader(GLTFLoader, '/street.glb');

  useLayoutEffect(() => {
    applyGltfRootSurfaceState(gltf.scene, opacity, depthWrite);
  }, [gltf, opacity, depthWrite]);

  return (
    <group>
      <primitive
        object={gltf.scene}
        scale={[0.02, 0.02, 0.02]}
        position={[0, -2, 0]}
        rotation={[0, 0, 0]}
      />
      <primitive
        object={gltf.scene.clone(true)}
        scale={[0.02, 0.02, 0.02]}
        position={[0, -2, -30]}
        rotation={[0, 0, 0]}
      />
      <primitive
        object={gltf.scene.clone(true)}
        scale={[0.02, 0.02, 0.02]}
        position={[0, -2, -60]}
        rotation={[0, 0, 0]}
      />
      <primitive
        object={gltf.scene.clone(true)}
        scale={[0.02, 0.02, 0.02]}
        position={[0, -2, -90]}
        rotation={[0, 0, 0]}
      />
    </group>

  );
}

export default function App() {
  const [sphereOpacity, setSphereOpacity] = useState(INITIAL_SCENE.sphereOpacity);
  const [lineOpacity, setLineOpacity] = useState(INITIAL_SCENE.lineOpacity);
  const [planeOpacity, setPlaneOpacity] = useState(INITIAL_SCENE.planeOpacity);
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
  const [hitNoiseLevel, setHitNoiseLevel] = useState(INITIAL_SCENE.hitNoiseLevel);
  const [lidarPyramidHeight, setLidarPyramidHeight] = useState(
    INITIAL_SCENE.lidarPyramidHeight,
  );
  const [lidarSampleRateHz, setLidarSampleRateHz] = useState(
    INITIAL_SCENE.lidarSampleRateHz,
  );
  const [lidarMaxRange, setLidarMaxRange] = useState(INITIAL_SCENE.lidarMaxRange);
  /** Camera projection 옵션(슬라이더) 패널 — 기본 접힘 */
  const [projectionOptionsOpen, setProjectionOptionsOpen] = useState(false);

  const onCarSceneMount = useCallback((root: THREE.Object3D | null) => {
    setCarRaycastTarget(root);
  }, []);

  const animFrameRef = useRef<number | null>(null);
  const carDriveRafRef = useRef<number | null>(null);
  /** `null`이면 Car z는 `carZSlider`. `carDrive` 중에는 이 값으로 보간 */
  const [carWorldPosition, setCarWorldPosition] = useState<Vec3 | null>(null);
  const [carZSlider, setCarZSlider] = useState(() => -INITIAL_SCENE.near * 0.7);

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
    hitNoiseLevel,
    lidarPyramidHeight,
    lidarSampleRateHz,
    lidarMaxRange,
    projectMarkersOnNearPlaneOnly,
    cyanHitMode,
  });

  const applySnapshot = (s: SceneSnapshot) => {
    setSphereOpacity(s.sphereOpacity);
    setLineOpacity(s.lineOpacity);
    setPlaneOpacity(s.planeOpacity);
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
    setHitNoiseLevel(s.hitNoiseLevel);
    setLidarPyramidHeight(s.lidarPyramidHeight);
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
    setCarWorldPosition(null);
  }, []);

  const startCarDrive = useCallback(
    (cfg: CarDriveConfig) => {
      cancelCarDrive();
      const [sx, sy, sz] = cfg.start;
      const [ex, ey, ez] = cfg.end;
      const dur = Math.max(1, cfg.durationMs);
      const t0 = performance.now();
      setCarWorldPosition([sx, sy, sz]);
      setCarZSlider(sz);
      const driveTick = (now: number) => {
        const u = Math.min(1, (now - t0) / dur);
        const k = easeInOutCubic(u);
        const x = lerp(sx, ex, k);
        const y = lerp(sy, ey, k);
        const z = lerp(sz, ez, k);
        setCarWorldPosition([x, y, z]);
        setCarZSlider(z);
        if (u < 1) carDriveRafRef.current = requestAnimationFrame(driveTick);
        else {
          carDriveRafRef.current = null;
          setCarWorldPosition(null);
          setCarZSlider(ez);
        }
      };
      carDriveRafRef.current = requestAnimationFrame(driveTick);
    },
    [cancelCarDrive],
  );

  const runPreset = (presetIndex: number) => {
    const goal = PRESETS[presetIndex];
    if (!goal) return;
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
        lidarSampleRateHz: lerp(from.lidarSampleRateHz, goal.lidarSampleRateHz, k),
        lidarMaxRange: lerp(from.lidarMaxRange, goal.lidarMaxRange, k),
        projectMarkersOnNearPlaneOnly: goal.projectMarkersOnNearPlaneOnly,
        cyanHitMode: goal.cyanHitMode,
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
        <MainCameraEnableFrustumGuideLayer />
        <Car
          distance={near}
          worldPosition={carWorldPosition ?? ([-3, -2.1, carZSlider] as Vec3)}
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
          carRaycastTarget={carOpacity > 0 ? carRaycastTarget : null}
          projectMarkersOnNearPlaneOnly={projectMarkersOnNearPlaneOnly}
          cyanHitMode={cyanHitMode}
          streetOpacity={streetOpacity}
          hitNoiseLevel={hitNoiseLevel}
          lidarPyramidHeight={lidarPyramidHeight}
          lidarSampleRateHz={lidarSampleRateHz}
          lidarMaxRange={lidarMaxRange}
          forceLidarEveryFrame={carWorldPosition !== null}
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
              label="구(와이어프레임) 투명도"
              value={sphereOpacity}
              min={0}
              max={1}
              step={0.02}
              unit=""
              onChange={setSphereOpacity}
            />
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
            <SliderRow
              label="투영 점(노랑) 크기 — 레이캐스트 또는 near"
              value={nearPointSize}
              min={0.02}
              max={0.5}
              step={0.01}
              unit=""
              onChange={setNearPointSize}
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
            <SliderRow
              label="뷰 평면(near 면) 투명도"
              value={planeOpacity}
              min={0}
              max={1}
              step={0.02}
              unit=""
              onChange={setPlaneOpacity}
            />
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
              label="차량 월드 Z (전후)"
              value={carZSlider}
              min={-90}
              max={10}
              step={0.5}
              unit=""
              onChange={(z) => {
                cancelCarDrive();
                setCarZSlider(z);
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
            <label
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                marginBottom: 12,
                fontSize: 13,
                color: '#ccc',
                cursor: 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={projectMarkersOnNearPlaneOnly}
                onChange={(e) => setProjectMarkersOnNearPlaneOnly(e.target.checked)}
                style={{ marginTop: 3 }}
              />
              <span>
                노란 투영점을 <strong style={{ color: '#9df' }}>near 평면(수식)</strong>만 사용
                <span style={{ color: '#888', fontSize: 11, display: 'block', marginTop: 4 }}>
                  끄면 원점에서 광선마다 <strong style={{ color: '#fd9' }}>Raycaster → car.glb</strong> 표면
                  교차(미적중 시 near 평면 규칙으로 폴백).
                </span>
              </span>
            </label>
            <fieldset
              style={{
                border: 'none',
                padding: 0,
                margin: '0 0 12px',
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
            <SliderRow
              label="HDR 배경 강도 (backgroundIntensity)"
              value={backgroundIntensity}
              min={0}
              max={3}
              step={0.05}
              unit=""
              onChange={setBackgroundIntensity}
            />
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
              max={80}
              step={1}
              unit="개"
              onChange={setAzimuthDivisions}
            />
            <SliderRow
              label="세로 격자 구간 수 (φ)"
              value={polarDivisions}
              min={2}
              max={80}
              step={1}
              unit="개"
              onChange={setPolarDivisions}
            />
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
        {PRESETS.map((_, i) => (
          <button
            key={i}
            type="button"
            title={`발표 스텝 ${i + 1}/${PRESETS.length}`}
            onClick={() => runPreset(i)}
            style={{
              width: 44,
              height: 44,
              borderRadius: '50%',
              border: '2px solid rgba(255,255,255,0.35)',
              background: 'rgba(30,30,35,0.85)',
              color: '#fff',
              fontSize: 17,
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
              backdropFilter: 'blur(8px)',
            }}
          >
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  );
}
