export type Vec3 = readonly [number, number, number];

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
  /** true면 노란 점을 near 평면 해석만 사용 */
  projectMarkersOnNearPlaneOnly: boolean;
  /** true면 시안 점을 차량 메시 레이캐스트에 둠 */
  sphereHitProjectOnCar: boolean;
}

/** 발표 스텝 1: 하늘(HDR) + 센서 원점(빨간 점), 프러스텀 거의 끔 */
export const INITIAL_SCENE: SceneSnapshot = {
  sphereOpacity: 0.02,
  lineOpacity: 0.02,
  planeOpacity: 0,
  near: 7,
  azimuthSpanDeg: 50,
  polarSpanDeg: 40,
  azimuthDivisions: 12,
  polarDivisions: 8,
  sphereHitSize: 0.05,
  sphereHitOpacity: 0,
  nearPointSize: 0.004,
  backgroundIntensity: 1.15,
  cameraPosition: [7, 5, 9],
  orbitTarget: [0, 0, 0],
  carOpacity: 0,
  streetOpacity: 1,
  hitNoiseLevel: 0.03,
  lidarPyramidHeight: 2,
  lidarSampleRateHz: 10,
  projectMarkersOnNearPlaneOnly: false,
  sphereHitProjectOnCar: false,
};

/**
 * 우하단 1…8 버튼 — LiDAR 시뮬 발표 시나리오 (ease 보간 목표).
 * 5: 차는 보이나 시안은 구면(의도적 관통), 6: 메시 히트 on.
 */
export const PRESETS: SceneSnapshot[] = [
  // 1 — 하늘 + 원점 (INITIAL과 동일 의도; 카메라만 살짝 다를 수 있음)
  {
    ...INITIAL_SCENE,
    cameraPosition: [7.2, 5.2, 9.2],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1.2,
  },
  // 2 — 희소 레이만
  {
    ...INITIAL_SCENE,
    lineOpacity: 0.42,
    sphereOpacity: 0.02,
    planeOpacity: 0.02,
    sphereHitOpacity: 0.02,
    nearPointSize: 0.004,
    near: 6,
    azimuthSpanDeg: 32,
    polarSpanDeg: 26,
    azimuthDivisions: 5,
    polarDivisions: 4,
    cameraPosition: [5.5, 3.8, 7.5],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1.05,
    carOpacity: 0,
    projectMarkersOnNearPlaneOnly: false,
    sphereHitProjectOnCar: false,
  },
  // 3 — 촘촘한 격자·넓은 FOV
  {
    ...INITIAL_SCENE,
    lineOpacity: 0.48,
    sphereOpacity: 0.04,
    planeOpacity: 0.04,
    sphereHitOpacity: 0.04,
    nearPointSize: 0.006,
    near: 6.5,
    azimuthSpanDeg: 95,
    polarSpanDeg: 72,
    azimuthDivisions: 28,
    polarDivisions: 16,
    cameraPosition: [6.2, 4.5, 8.8],
    orbitTarget: [0, 0, 0],
    backgroundIntensity: 1,
    carOpacity: 0,
    projectMarkersOnNearPlaneOnly: false,
    sphereHitProjectOnCar: false,
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
    sphereHitProjectOnCar: false,
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
    sphereHitProjectOnCar: false,
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
    sphereHitProjectOnCar: true,
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
    sphereHitProjectOnCar: true,
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
    sphereHitProjectOnCar: true,
  },
];
