/**
 * 프러스텀 격자 레이 → 깊이 RT + TSL compute 히트 ([GpuLidarSimulator.ts] 패턴).
 * CPU Raycaster 대신 WebGPU compute로 마커 위치 갱신.
 */
import {
  Fn,
  If,
  instanceIndex,
  positionLocal,
  storage,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import * as THREE from 'three/webgpu';

const DEG2RAD = Math.PI / 180;

/** 차·지면 등 깊이에 쓸 오클루더 (기본 레이어 0) */
export const LAYER_DEPTH_OCCLUDER = 0;
/** 프러스텀 가이드(구·선·평면·원점) — 깊이 패스에서 제외 */
export const LAYER_FRUSTUM_GUIDE = 1;

interface Partition {
  start: number;
  count: number;
  /** 파티션 중심 방위각 (라디안). 월드 전방은 −Z, 레이도 동일 — GpuLidar(+Z) 와 달리 180° 보정 없음 */
  centerAzimuthRad: number;
  hFovDeg: number;
}

interface CameraPass {
  camera: THREE.PerspectiveCamera;
  depthTarget: THREE.RenderTarget;
  depthTexture: THREE.DepthTexture;
  computeNode: object;
  viewMatrixUniform: ReturnType<typeof uniform>;
  projectionMatrixUniform: ReturnType<typeof uniform>;
  localRotation: THREE.Quaternion;
}

function buildDirectionsAndPartitions(
  azimuthSpanDeg: number,
  polarSpanDeg: number,
  azimuthDivisions: number,
  polarDivisions: number,
): { directions: Float32Array; partitions: Partition[]; numPoints: number } {
  const hDiv = Math.max(1, Math.round(azimuthDivisions));
  const vDiv = Math.max(1, Math.round(polarDivisions));
  const numJ = vDiv + 1;
  const numIRows = hDiv + 1;
  const numPoints = numIRows * numJ;

  const thetaMin = THREE.MathUtils.degToRad(-azimuthSpanDeg / 2);
  const thetaMax = THREE.MathUtils.degToRad(azimuthSpanDeg / 2);
  const phiMin = THREE.MathUtils.degToRad(-polarSpanDeg / 2);
  const phiMax = THREE.MathUtils.degToRad(polarSpanDeg / 2);

  const dirs = new Float32Array(numPoints * 3);
  let w = 0;
  for (let i = 0; i <= hDiv; i++) {
    for (let j = 0; j <= vDiv; j++) {
      const theta = thetaMin + (i / hDiv) * (thetaMax - thetaMin);
      const phi = phiMin + (j / vDiv) * (phiMax - phiMin);
      dirs[w++] = Math.sin(theta) * Math.cos(phi);
      dirs[w++] = Math.sin(phi);
      dirs[w++] = -Math.cos(theta) * Math.cos(phi);
    }
  }

  const numCams = Math.max(1, Math.ceil(azimuthSpanDeg / 90));
  const partitions: Partition[] = [];

  for (let c = 0; c < numCams; c++) {
    const iStart = Math.round((c / numCams) * (hDiv + 1));
    const iEnd = Math.round(((c + 1) / numCams) * (hDiv + 1));
    if (iEnd <= iStart) continue;

    const start = iStart * numJ;
    const count = (iEnd - iStart) * numJ;

    const thetaL = thetaMin + (iStart / hDiv) * (thetaMax - thetaMin);
    const thetaR = thetaMin + ((iEnd - 1) / hDiv) * (thetaMax - thetaMin);
    const thetaC = (thetaL + thetaR) / 2;
    const hFovDeg =
      THREE.MathUtils.radToDeg(Math.abs(thetaR - thetaL)) + 3;

    partitions.push({
      start,
      count,
      centerAzimuthRad: thetaC,
      hFovDeg,
    });
  }

  return { directions: dirs, partitions, numPoints };
}

export interface GpuFrustumRayHitPassConfig {
  azimuthSpanDeg: number;
  polarSpanDeg: number;
  azimuthDivisions: number;
  polarDivisions: number;
  depthNear: number;
  depthTextureSize: number;
}

export class GpuFrustumRayHitPass {
  private passes: CameraPass[] = [];
  private directionStorage = null as unknown as {
    element: (i: unknown) => any;
  };
  private positionStorage = null as unknown as {
    element: (i: unknown) => any;
  };
  private visibilityStorage = null as unknown as {
    element: (i: unknown) => any;
  };
  private numPoints = 0;

  private rotationMatrixUniform = uniform(new THREE.Matrix4());
  private originUniform = uniform(new THREE.Vector3());
  private nearUniform = uniform(0.05);
  private farUniform = uniform(50);
  private maxRangeUniform = uniform(50);
  private sphereRadiusUniform = uniform(1);
  private cyanScaleUniform = uniform(0.14);
  private yellowScaleUniform = uniform(0.15);
  private cyanOpacityUniform = uniform(1);
  private yellowOpacityUniform = uniform(1);

  readonly cyanMesh: THREE.InstancedMesh;
  readonly yellowMesh: THREE.InstancedMesh;
  private cyanMaterial: THREE.MeshBasicNodeMaterial;
  private yellowMaterial: THREE.MeshBasicNodeMaterial;

  constructor(initialConfig: GpuFrustumRayHitPassConfig) {
    const { directions, partitions, numPoints } = buildDirectionsAndPartitions(
      initialConfig.azimuthSpanDeg,
      initialConfig.polarSpanDeg,
      initialConfig.azimuthDivisions,
      initialConfig.polarDivisions,
    );
    this.numPoints = numPoints;

    const dirAttr = new THREE.InstancedBufferAttribute(directions, 3);
    const posAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(numPoints * 3),
      3,
    );
    const visAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(numPoints),
      1,
    );

    this.directionStorage = storage(dirAttr as never, 'vec3', numPoints) as unknown as typeof this.directionStorage;
    this.positionStorage = storage(posAttr as never, 'vec3', numPoints) as unknown as typeof this.positionStorage;
    this.visibilityStorage = storage(visAttr as never, 'float', numPoints) as unknown as typeof this.visibilityStorage;

    const vFovDeg = polarSpanDegToCameraVFov(initialConfig.polarSpanDeg);
    const depthNear = initialConfig.depthNear;
    const texSize = initialConfig.depthTextureSize;

    for (const p of partitions) {
      if (p.count === 0) continue;

      const safeHFov = p.hFovDeg + 0.5;
      const aspect = Math.max(
        0.1,
        Math.tan((safeHFov / 2) * DEG2RAD) / Math.tan((vFovDeg / 2) * DEG2RAD),
      );

      const camera = new THREE.PerspectiveCamera(
        vFovDeg,
        aspect,
        depthNear,
        1000,
      );
      camera.layers.disableAll();
      camera.layers.enable(LAYER_DEPTH_OCCLUDER);

      /** Ry(θ)·(0,0,−1)=(−sin θ,0,−cos θ) 이므로 (sin θ,0,−cos θ) 정렬에 Y축 −θ */
      const localRotation = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        -p.centerAzimuthRad,
      );

      const depthTexture = new THREE.DepthTexture(texSize, texSize);
      depthTexture.minFilter = THREE.NearestFilter;
      depthTexture.magFilter = THREE.NearestFilter;
      depthTexture.generateMipmaps = false;

      const depthTarget = new THREE.RenderTarget(texSize, texSize);
      depthTarget.depthTexture = depthTexture;
      depthTarget.depthBuffer = true;

      const viewMatrixUniform = uniform(new THREE.Matrix4());
      const projectionMatrixUniform = uniform(new THREE.Matrix4());
      const startIdx = Math.floor(p.start);

      const directionStorage = this.directionStorage;
      const positionStorage = this.positionStorage;
      const visibilityStorage = this.visibilityStorage;
      const originUniform = this.originUniform;
      const nearUniform = this.nearUniform;
      const farUniform = this.farUniform;
      const maxRangeUniform = this.maxRangeUniform;
      const rotationMatrixUniform = this.rotationMatrixUniform;

      const computeNode = Fn(() => {
        const i = instanceIndex.add(startIdx);
        const dir = directionStorage.element(i);
        const worldDir = rotationMatrixUniform
          .mul(vec4(dir, 0.0))
          .xyz.normalize();

        If(worldDir.z.greaterThanEqual(0.0), () => {
          visibilityStorage.element(i).assign(0.0);
        }).Else(() => {
          const origin = originUniform;
          const worldPosFar = origin.add(worldDir.mul(maxRangeUniform));

          const clipPos = projectionMatrixUniform
            .mul(viewMatrixUniform)
            .mul(vec4(worldPosFar, 1.0));
          const ndc = clipPos.xyz.div(clipPos.w);
          const uv = ndc.xy.mul(0.5).add(0.5);
          const uvFlipped = vec2(uv.x, uv.y.mul(-1.0).add(1.0));

          const depth = texture(depthTexture, uvFlipped).x;
          const outside = uvFlipped.x
            .lessThan(0.0)
            .or(uvFlipped.x.greaterThan(1.0))
            .or(uvFlipped.y.lessThan(0.0))
            .or(uvFlipped.y.greaterThan(1.0))
            .or(depth.greaterThan(0.9999));

          const far = farUniform;
          const near = nearUniform;
          const viewZ = near.mul(far).div(far.sub(near).mul(depth).sub(far));
          const dirView = viewMatrixUniform.mul(vec4(worldDir, 0.0)).xyz;
          const t = viewZ.div(dirView.z);
          const hitPos = origin.add(worldDir.mul(t));

          If(
            outside.or(t.lessThan(0.0)).or(t.greaterThan(maxRangeUniform)),
            () => {
              visibilityStorage.element(i).assign(0.0);
            },
          ).Else(() => {
            visibilityStorage.element(i).assign(1.0);
            positionStorage.element(i).assign(hitPos);
          });
        });
      })().compute(p.count);

      this.passes.push({
        camera,
        depthTarget,
        depthTexture,
        computeNode,
        viewMatrixUniform,
        projectionMatrixUniform,
        localRotation,
      });
    }

    const sphereGeom = new THREE.SphereGeometry(1, 10, 10);

    this.cyanMaterial = new THREE.MeshBasicNodeMaterial();
    this.cyanMaterial.transparent = true;
    this.cyanMaterial.depthWrite = false;
    const cyanVis = this.visibilityStorage.element(instanceIndex) as any;
    const cyanDir = this.directionStorage.element(instanceIndex) as any;
    const cyanHit = this.positionStorage.element(instanceIndex) as any;
    const cyanFallback = cyanDir.mul(this.sphereRadiusUniform);
    const cyanPos = cyanVis.greaterThan(0.5).select(cyanHit, cyanFallback);
    this.cyanMaterial.colorNode = vec3(0, 1, 0.78);
    this.cyanMaterial.opacityNode = this.cyanOpacityUniform;
    this.cyanMaterial.positionNode = cyanPos.add(
      positionLocal.mul(this.cyanScaleUniform),
    );

    this.yellowMaterial = new THREE.MeshBasicNodeMaterial();
    this.yellowMaterial.transparent = true;
    this.yellowMaterial.depthWrite = false;
    const yVis = this.visibilityStorage.element(instanceIndex) as any;
    const yHit = this.positionStorage.element(instanceIndex) as any;
    this.yellowMaterial.colorNode = vec3(1, 1, 0);
    this.yellowMaterial.opacityNode = this.yellowOpacityUniform.mul(yVis);
    this.yellowMaterial.positionNode = yHit.add(
      positionLocal.mul(this.yellowScaleUniform),
    );

    this.cyanMesh = new THREE.InstancedMesh(
      sphereGeom,
      this.cyanMaterial,
      numPoints,
    );
    this.cyanMesh.frustumCulled = false;
    this.cyanMesh.renderOrder = 1000;
    this.cyanMesh.layers.set(LAYER_FRUSTUM_GUIDE);

    this.yellowMesh = new THREE.InstancedMesh(
      sphereGeom.clone(),
      this.yellowMaterial,
      numPoints,
    );
    this.yellowMesh.frustumCulled = false;
    this.yellowMesh.renderOrder = 1001;
    this.yellowMesh.layers.set(LAYER_FRUSTUM_GUIDE);
  }

  setSphereRadius(r: number): void {
    this.sphereRadiusUniform.value = r;
  }

  setMaxRange(r: number): void {
    this.maxRangeUniform.value = r;
    this.farUniform.value = r;
    for (const p of this.passes) {
      p.camera.far = r;
      p.camera.updateProjectionMatrix();
    }
  }

  setMarkerScales(cyan: number, yellow: number): void {
    this.cyanScaleUniform.value = Math.max(0.01, cyan);
    this.yellowScaleUniform.value = Math.max(0.01, yellow);
  }

  setMarkerOpacities(cyan: number, yellow: number): void {
    this.cyanOpacityUniform.value = cyan;
    this.yellowOpacityUniform.value = yellow;
  }

  setMeshVisibility(cyanVisible: boolean, yellowVisible: boolean): void {
    this.cyanMesh.visible = cyanVisible;
    this.yellowMesh.visible = yellowVisible;
  }

  setCyanUsesHitOnly(useHitOnly: boolean): void {
    const cyanDir = this.directionStorage.element(instanceIndex) as any;
    const cyanHit = this.positionStorage.element(instanceIndex) as any;
    const cyanVis = this.visibilityStorage.element(instanceIndex) as any;
    const cyanFallback = cyanDir.mul(this.sphereRadiusUniform);
    const cyanPos = useHitOnly
      ? cyanHit
      : cyanVis.greaterThan(0.5).select(cyanHit, cyanFallback);
    this.cyanMaterial.positionNode = cyanPos.add(
      positionLocal.mul(this.cyanScaleUniform),
    );
    this.cyanMaterial.opacityNode = useHitOnly
      ? this.cyanOpacityUniform.mul(cyanVis)
      : this.cyanOpacityUniform;
    this.cyanMaterial.needsUpdate = true;
  }

  execute(
    renderer: THREE.WebGPURenderer,
    scene: THREE.Scene,
    sensorMatrixWorld: THREE.Matrix4,
  ): void {
    if (!renderer.compute || this.numPoints === 0) return;

    const sensorQuat = new THREE.Quaternion();
    const sensorScale = new THREE.Vector3();
    sensorMatrixWorld.decompose(
      this.originUniform.value,
      sensorQuat,
      sensorScale,
    );
    (this.rotationMatrixUniform.value as THREE.Matrix4).makeRotationFromQuaternion(
      sensorQuat,
    );

    const baseRot = sensorQuat.clone();

    const prevTarget = renderer.getRenderTarget();

    for (const pass of this.passes) {
      const camRot = baseRot.clone().multiply(pass.localRotation);
      pass.camera.position.set(
        this.originUniform.value.x,
        this.originUniform.value.y,
        this.originUniform.value.z,
      );
      pass.camera.quaternion.copy(camRot);
      pass.camera.updateMatrixWorld(true);

      (pass.viewMatrixUniform.value as THREE.Matrix4).copy(
        pass.camera.matrixWorldInverse,
      );
      (pass.projectionMatrixUniform.value as THREE.Matrix4).copy(
        pass.camera.projectionMatrix,
      );

      renderer.setRenderTarget(pass.depthTarget);
      renderer.render(scene, pass.camera);
      renderer.compute(pass.computeNode as never);
    }

    renderer.setRenderTarget(prevTarget);
  }

  dispose(): void {
    this.cyanMesh.geometry.dispose();
    this.yellowMesh.geometry.dispose();
    this.cyanMaterial.dispose();
    this.yellowMaterial.dispose();
    for (const p of this.passes) {
      p.depthTarget.dispose();
      p.depthTexture.dispose();
    }
    this.passes = [];
  }
}

function polarSpanDegToCameraVFov(polarSpanDeg: number): number {
  return Math.max(45, polarSpanDeg + 24);
}
