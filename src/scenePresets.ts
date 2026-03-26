export type Vec3 = readonly [number, number, number];

/** 시안 포인트: 외접 구면만 vs GPU 깊이(차·바닥) 시뮬 */
export type CyanHitMode = 'sphereOnly' | 'depthSim';

/** 프리셋 진입 시 차량 월드 위치 주행(없으면 App 기본 `[-3,-2.1,-near·0.7]`) */
export interface CarDriveConfig {
  start: Vec3;
  end: Vec3;
  durationMs: number;
}

/**
 * 프리셋·초기값·슬라이더가 공유하는 스냅샷 — `PRESETS` 항목을 자유롭게 수정하세요.
 * (`App`의 `FrustumVisualizer`에 넘기는 값과 동일. `carRaycastTarget`만 GLB에서 런타임 주입)
 */
export interface SceneSnapshot {
  sphereOpacity: number;
  lineOpacity: number;
  planeOpacity: number;
  near: number;
  azimuthSpanDeg: number;
  polarSpanDeg: number;
  azimuthDivisions: number;
  polarDivisions: number;
  /** 광선과 외접 구면 교점(점 스프라이트) */
  sphereHitSize: number;
  sphereHitOpacity: number;
  /** near 평면 교차점(노란 점) */
  nearPointSize: number;
  backgroundIntensity: number;
  cameraPosition: Vec3;
  orbitTarget: Vec3;
  /** 0이면 차량 숨김 — 레이캐스트 타깃도 끔 */
  carOpacity: number;
  /** 0이면 바닥(street) 깊이·레이 비기여 — `FrustumVisualizer`의 `streetOpacity`와 동일 */
  streetOpacity: number;
  /** GPU 깊이 히트 위치 노이즈 (world, 0–0.3 권장) */
  hitNoiseLevel: number;
  /** 라이다 원점 피라미드 전방 길이 */
  lidarPyramidHeight: number;
  /** GPU 깊이 패스·포인트 갱신 주파수 (Hz) */
  lidarSampleRateHz: number;
  /** 라이다/프러스텀 최대 거리(far, 월드) — 광선 끝·외접 구 반경·GPU maxRange */
  lidarMaxRange: number;
  /** true면 노란 점을 near 평면 해석만 사용 */
  projectMarkersOnNearPlaneOnly: boolean;
  /** 시안: 구면 해석만 / GPU 깊이로 차·바닥 시뮬 */
  cyanHitMode: CyanHitMode;
  /** 프리셋 버튼으로 진입할 때만 사용 — 슬라이더 스냅샷에는 보통 없음 */
  carDrive?: CarDriveConfig;
}

/** 발표 스텝 1: 하늘(HDR) + 센서 원점(빨간 점), 프러스텀 거의 끔 */
export const INITIAL_SCENE: SceneSnapshot = {
  sphereOpacity: 0.0,
  lineOpacity: 0.1,
  planeOpacity: 0,
  near: 7,
  azimuthSpanDeg: 120,
  polarSpanDeg: 30,
  azimuthDivisions: 80,
  polarDivisions: 32,
  sphereHitSize: 0.05,
  sphereHitOpacity: 0,
  nearPointSize: 0.004,
  backgroundIntensity: 1.15,
  cameraPosition: [7, 5, 9],
  orbitTarget: [0, 0, 0],
  carOpacity: 0,
  streetOpacity: 1,
  hitNoiseLevel: 0.03,
  lidarPyramidHeight: 1,
  lidarSampleRateHz: 10,
  lidarMaxRange: 70,
  projectMarkersOnNearPlaneOnly: false,
  cyanHitMode: 'sphereOnly',
};

/**
 * 우하단 1…8 버튼 — LiDAR 시뮬 발표 시나리오 (ease 보간 목표).
 * 5: 차는 보이나 시안은 구면만, 6: 시안 depthSim(차 표면).
 */
export const PRESETS: SceneSnapshot[] = [
  // 1 — 하늘 + 원점 (INITIAL과 동일 의도; 카메라만 살짝 다를 수 있음)
  {
    ...INITIAL_SCENE,
    cameraPosition: [7.2, 5.2, 9.2],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1.2,
  },
  // 2 — 희소 레이 + 차량 주행 + GPU 깊이 시뮬 (`carDrive`: 시작/끝 위치·시간)
  {
    ...INITIAL_SCENE,
    // lineOpacity: 0.42,
    // sphereOpacity: 0.02,
    planeOpacity: 0.02,
    sphereHitOpacity: 0.45,
    nearPointSize: 0.004,
    near: 6,
    cameraPosition: [5.5, 3.8, 7.5],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1.05,
    carOpacity: 1,
    streetOpacity: 1,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
    carDrive: {
      start: [-3, -2.1, -(6 * 0.7) - 45],
      end: [-3, -2.1, -5],
      durationMs: 8000,
    },
  },
  // 3 — 촘촘한 격자·넓은 FOV
  {
    ...INITIAL_SCENE,
    // lineOpacity: 0.42,
    // sphereOpacity: 0.02,
    planeOpacity: 0.02,
    sphereHitOpacity: 0.45,
    nearPointSize: 0.004,
    near: 6,
    cameraPosition: [5.5, 3.8, 7.5],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1.05,
    carOpacity: 0.1,
    streetOpacity: 1,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
    carDrive: {
      start: [-3, -2.1, -(6 * 0.7) - 45],
      end: [-3, -2.1, -5],
      durationMs: 8000,
    },
  },
  // 4 — 거리 구면 + 시안(구면), 차 없음
  {
    ...INITIAL_SCENE,
    sphereOpacity: 0.36,
    lineOpacity: 0.4,
    planeOpacity: 0.08,
    sphereHitOpacity: 0.9,
    nearPointSize: 0.1,
    near: 6,
    azimuthSpanDeg: 72,
    polarSpanDeg: 52,
    azimuthDivisions: 18,
    polarDivisions: 11,
    cameraPosition: [5.2, 3.2, 7.2],
    orbitTarget: [0, -0.3, 0],
    backgroundIntensity: 0.95,
    carOpacity: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'sphereOnly',
  },
  // 5 — 차량 등장, 구면 히트 유지(관통)
  {
    ...INITIAL_SCENE,
    sphereOpacity: 0.32,
    lineOpacity: 0.38,
    planeOpacity: 0.1,
    sphereHitOpacity: 0.88,
    nearPointSize: 0.12,
    near: 6,
    azimuthSpanDeg: 72,
    polarSpanDeg: 52,
    azimuthDivisions: 18,
    polarDivisions: 11,
    cameraPosition: [6.5, 2.8, 6.8],
    orbitTarget: [0, -0.9, -3.2],
    backgroundIntensity: 0.9,
    carOpacity: 1,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'sphereOnly',
  },
  // 6 — 메시 히트
  {
    ...INITIAL_SCENE,
    sphereOpacity: 0.22,
    lineOpacity: 0.34,
    planeOpacity: 0.12,
    sphereHitOpacity: 0.92,
    nearPointSize: 0.14,
    near: 5.5,
    azimuthSpanDeg: 72,
    polarSpanDeg: 52,
    azimuthDivisions: 18,
    polarDivisions: 11,
    cameraPosition: [6.2, 2.4, 6.2],
    orbitTarget: [0, -0.85, -3.1],
    backgroundIntensity: 0.88,
    carOpacity: 1,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
  },
  // 7 — near 평면 강조 + 3D 히트(노랑 레이캐스트)
  {
    ...INITIAL_SCENE,
    sphereOpacity: 0.12,
    lineOpacity: 0.28,
    planeOpacity: 0.7,
    sphereHitOpacity: 0.85,
    nearPointSize: 0.16,
    near: 4.2,
    azimuthSpanDeg: 68,
    polarSpanDeg: 48,
    azimuthDivisions: 16,
    polarDivisions: 10,
    cameraPosition: [-3.8, 2.6, 7.4],
    orbitTarget: [0, -0.5, -2.6],
    backgroundIntensity: 0.85,
    carOpacity: 1,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
  },
  // 8 — 평면 위 샘플만 (range / depth)
  {
    ...INITIAL_SCENE,
    sphereOpacity: 0.08,
    lineOpacity: 0.22,
    planeOpacity: 0.78,
    sphereHitOpacity: 0.32,
    nearPointSize: 0.18,
    near: 4,
    azimuthSpanDeg: 68,
    polarSpanDeg: 48,
    azimuthDivisions: 16,
    polarDivisions: 10,
    cameraPosition: [-4.2, 3, 7.8],
    orbitTarget: [0, -0.2, -3.6],
    backgroundIntensity: 0.82,
    carOpacity: 1,
    projectMarkersOnNearPlaneOnly: true,
    cyanHitMode: 'depthSim',
  },
];
