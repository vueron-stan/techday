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
 *
 * 필드는 왼쪽 패널·문서 가독을 위해 카테고리 순으로 배열했습니다.
 */
export interface SceneSnapshot {
  /* ─── 프러스텀 · 격자 ─── */
  near: number;
  azimuthSpanDeg: number;
  polarSpanDeg: number;
  azimuthDivisions: number;
  polarDivisions: number;

  /* ─── 가이드 (구·선·뷰 평면) ─── */
  sphereOpacity: number;
  lineOpacity: number;
  planeOpacity: number;

  /* ─── 교차점 · 투영점 ─── */
  /** 광선과 외접 구면 교점(점 스프라이트) */
  sphereHitSize: number;
  sphereHitOpacity: number;
  /** near 평면 교차점(노란 점) — 0이면 비표시 */
  nearPointSize: number;
  /** true면 노란 점을 near 평면 해석만 사용 */
  projectMarkersOnNearPlaneOnly: boolean;
  /** 시안: 구면 해석만 / GPU 깊이로 차·바닥 시뮬 */
  cyanHitMode: CyanHitMode;

  /* ─── 라이다 시뮬 ─── */
  /** GPU 깊이 히트 위치 노이즈 (world, 0–0.3 권장) */
  hitNoiseLevel: number;
  /** 라이다 원점 피라미드 전방 길이 */
  lidarPyramidHeight: number;
  /** 라이다 원점 피라미드 메시 투명도 (0–1) */
  lidarPyramidOpacity: number;
  /** GPU 깊이 패스·포인트 갱신 주파수 (Hz) — 차량 주행 중에도 동일 간격으로만 갱신 */
  lidarSampleRateHz: number;
  /** 라이다/프러스텀 최대 거리(far, 월드) — 광선 끝·외접 구 반경·GPU maxRange */
  lidarMaxRange: number;

  /* ─── 차량 · 바닥 ─── */
  /** 0이면 차량을 화면에만 숨김 — LiDAR 깊이·레이캐스트 타깃은 유지 */
  carOpacity: number;
  /** 0이면 바닥을 화면에만 숨김 — LiDAR 깊이 패스에는 여전히 메시가 있어 포인트 시뮬 유지 */
  streetOpacity: number;

  /* ─── 카메라 · 환경 ─── */
  backgroundIntensity: number;
  cameraPosition: Vec3;
  orbitTarget: Vec3;

  /** 프리셋 버튼으로 진입할 때만 사용 — 슬라이더 스냅샷에는 보통 없음 */
  carDrive?: CarDriveConfig;
}

/** 발표 스텝 1: 하늘(HDR) + 센서 원점(빨간 점), 프러스텀 거의 끔 */
export const INITIAL_SCENE: SceneSnapshot = {
  near: 7,
  azimuthSpanDeg: 120,
  polarSpanDeg: 30,
  azimuthDivisions: 80,
  polarDivisions: 32,

  sphereOpacity: 0.0,
  lineOpacity: 0,
  planeOpacity: 0,

  sphereHitSize: 0.05,
  sphereHitOpacity: 0,
  nearPointSize: 0,
  projectMarkersOnNearPlaneOnly: false,
  cyanHitMode: 'sphereOnly',

  hitNoiseLevel: 0.03,
  lidarPyramidHeight: 1,
  lidarPyramidOpacity: 0,
  lidarSampleRateHz: 10,
  lidarMaxRange: 70,

  carOpacity: 0,
  streetOpacity: 0,

  backgroundIntensity: 1.15,
  cameraPosition: [7.475317110307151, -0.7281688354023441, 8.925257285566499],
    orbitTarget: [0.49166223312671203, 1.900876456763511, -1.459187110171474],
};

/**
 * 우하단 1…8 버튼 — LiDAR 시뮬 발표 시나리오 (ease 보간 목표).
 * 5: 차는 보이나 시안은 구면만, 6: 시안 depthSim(차 표면).
 */
export const PRESETS: SceneSnapshot[] = [
  // 1 — 하늘 + 원점 (INITIAL과 동일 의도; 카메라만 살짝 다를 수 있음)
  {
    ...INITIAL_SCENE,
    cameraPosition: [7.475317110307151, -0.7281688354023441, 8.925257285566499],
    orbitTarget: [0.49166223312671203, 1.900876456763511, -1.459187110171474],

  },
  // 2 — 하늘 + 원점 (INITIAL과 동일 의도; 카메라만 살짝 다를 수 있음)
  {
    ...INITIAL_SCENE,
    cameraPosition: [7.475317110307151, -0.7281688354023441, 8.925257285566499],
    orbitTarget: [0.49166223312671203, 1.900876456763511, -1.459187110171474],
    streetOpacity:1,
    lineOpacity:0,
    lidarPyramidOpacity:0,
  },
  // 3 — 희소 레이 + 차량 주행 + GPU 깊이 시뮬 (`carDrive`: 시작/끝 위치·시간)
  {
    ...INITIAL_SCENE,
    // lineOpacity: 0.42,
    // sphereOpacity: 0.02,
    planeOpacity: 0.02,
    sphereHitOpacity: 0.45,
    nearPointSize: 0,
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
      end: [-3, -2.1, -10],
      durationMs: 8000,
    },
  },
  // 4 — 촘촘한 격자·넓은 FOV
  {
    ...INITIAL_SCENE,
    // lineOpacity: 0.42,
    // sphereOpacity: 0.02,
    planeOpacity: 0.02,
    sphereHitOpacity: 0.45,
    nearPointSize: 0,
    lineOpacity:0,
    near: 6,
    cameraPosition: [15.5, 3.8, 3],
    orbitTarget: [0, 0, -5],
    backgroundIntensity: 0.2,
    carOpacity: 0.02,
    streetOpacity: 0.02,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
    carDrive: {
      start: [-3, -2.1, -(6 * 0.7) - 45],
      end: [-3, -2.1, -10],
      durationMs: 8000,
    },
  },
  // 5 — 차량 등장, 구면 히트 유지(관통)
  {
    near: 6,
    azimuthSpanDeg: 45,
    polarSpanDeg: 45,
    azimuthDivisions: 4,
    polarDivisions: 4,
  
    sphereOpacity: 1,
    lineOpacity: 1,
    planeOpacity: 0,
  
    sphereHitSize: 0.5,
    sphereHitOpacity: 0.9,
    nearPointSize: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'sphereOnly',
  
    hitNoiseLevel: 0.03,
    lidarPyramidHeight: 2.5,
    lidarPyramidOpacity: 0.44,
    lidarSampleRateHz: 10,
    lidarMaxRange: 70,
  
    carOpacity: 0,
    streetOpacity: 0,
  
    backgroundIntensity: 0,
    cameraPosition: [23.504981566786327, 20.624815957369886, 35.18566275850949],
    orbitTarget: [9.15143119596536, 12.005749769497985, -13.912439184324123],
  },
  // 6 — 거리 구면 + 시안(구면), 차 없음
  {
    near: 6,
    azimuthSpanDeg: 120,
    polarSpanDeg: 45,
    azimuthDivisions: 18,
    polarDivisions: 11,
  
    sphereOpacity: 1,
    lineOpacity: 0.94,
    planeOpacity: 0,
  
    sphereHitSize: 0.5,
    sphereHitOpacity: 0.9,
    nearPointSize: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'sphereOnly',
  
    hitNoiseLevel: 0.03,
    lidarPyramidHeight: 1.5,
    lidarPyramidOpacity: 0.3,
    lidarSampleRateHz: 10,
    lidarMaxRange: 70,
  
    carOpacity: 0,
    streetOpacity: 0,
  
    backgroundIntensity: 0.05,
    cameraPosition: [35.3241618322496, 38.40126360945008, 64.23891503904804],
    orbitTarget: [11.60515558744173, 12.321567765229588, -5.555514980666803],
  },
  // 7 — 메시 히트
  {
    near: 6,
    azimuthSpanDeg: 120,
    polarSpanDeg: 45,
    azimuthDivisions: 256,
    polarDivisions: 64,
  
    sphereOpacity: 0.30000000000000004,
    lineOpacity: 0.94,
    planeOpacity: 0,
  
    sphereHitSize: 0.2,
    sphereHitOpacity: 0.9,
    nearPointSize: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'sphereOnly',
  
    hitNoiseLevel: 0.03,
    lidarPyramidHeight: 1.5,
    lidarPyramidOpacity: 0.3,
    lidarSampleRateHz: 10,
    lidarMaxRange: 70,
  
    carOpacity: 0,
    streetOpacity: 0,
  
    backgroundIntensity: 0.05,
    cameraPosition: [46.51370163784948, 49.298951967688836, 87.8722732335296],
    orbitTarget: [11.60515558744173, 12.321567765229588, -5.555514980666803],
  },
  // 7 — near 평면 강조 + 3D 히트(노랑 레이캐스트)
  {
    ...INITIAL_SCENE,
    near: 6,
    azimuthSpanDeg: 120,
    polarSpanDeg: 45,
    azimuthDivisions: 256,
    polarDivisions: 64,
  
    sphereOpacity: 0.30000000000000004,
    lineOpacity: 0,
    planeOpacity: 0,
  
    sphereHitSize: 0.02,
    sphereHitOpacity: 0.3,
    nearPointSize: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
  
    hitNoiseLevel: 0,
    lidarPyramidHeight: 1.5,
    lidarPyramidOpacity: 0.3,
    lidarSampleRateHz: 10,
    lidarMaxRange: 70,
    cameraPosition: [-3.8, 2.6, 7.4],
    orbitTarget: [0, -0.5, -2.6],
    carDrive:{
      start: [-3, -2.1, -10],
      end: [-3, -2.1, -10],
      durationMs: 1,
    },
    backgroundIntensity: 0.85,
    carOpacity: 1,
    streetOpacity:1,
  },
  // 8 — 평면 위 샘플만 (range / depth)
  {
    ...INITIAL_SCENE,
    near: 6,
    azimuthSpanDeg: 120,
    polarSpanDeg: 45,
    azimuthDivisions: 256,
    polarDivisions: 64,
  
    sphereOpacity: 0.30000000000000004,
    lineOpacity: 0,
    planeOpacity: 0,
  
    sphereHitSize: 0.02,
    sphereHitOpacity: 0.3,
    nearPointSize: 0,
    projectMarkersOnNearPlaneOnly: false,
    cyanHitMode: 'depthSim',
  
    hitNoiseLevel: 0.03,
    lidarPyramidHeight: 1.5,
    lidarPyramidOpacity: 0.3,
    lidarSampleRateHz: 10,
    lidarMaxRange: 70,
    cameraPosition: [-3.8, 2.6, 12.4],
    orbitTarget: [0, -0.5, -2.6],
    carDrive:{
      start: [-3, -2.1, -10],
      end: [-3, -2.1, -10],
      durationMs: 1,
    },
    backgroundIntensity: 0.85,
    carOpacity: 1,
    streetOpacity:1,
  },
];
