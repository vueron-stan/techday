import { Layer, resolveLidarPointColorForShader } from '$simulation/constant';
import type { LidarRenderOptions, LidarSpec } from '$simulation/type';
import {
  Fn,
  If,
  fract,
  instanceIndex,
  positionLocal,
  sin,
  storage,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import * as THREE from 'three/webgpu';

const DEG2RAD = Math.PI / 180;

interface Partition {
  start: number;
  count: number;
  yaw: number;
  hFov: number;
}

interface CameraPass {
  camera: THREE.PerspectiveCamera;
  depthTarget: THREE.RenderTarget;
  depthTexture: THREE.DepthTexture;
  computeNode: any;
  viewMatrixUniform: ReturnType<typeof uniform>;
  projectionMatrixUniform: ReturnType<typeof uniform>;
  localRotation: THREE.Quaternion;
}

const computeDirections = (
  spec: LidarSpec,
  scale: number,
): { directions: Float32Array; partitions: Partition[] } => {
  const { hFovDeg, hResolution, vFovMinDeg, vFovMaxDeg, channels } = spec;

  // 카메라 갯수 = 90도로 나눈 올림값
  const numCameras = Math.ceil(hFovDeg / 90);
  const hStepsTotal = Math.max(numCameras, Math.round(hResolution * scale));
  const totalPoints = hStepsTotal * channels;
  const dirs = new Float32Array(totalPoints * 3);
  const partitions: Partition[] = [];

  let currentIdx = 0;
  const hStepDeg = hFovDeg / hStepsTotal;
  const vStepDeg =
    channels > 1 ? (vFovMaxDeg - vFovMinDeg) / (channels - 1) : 0;

  for (let c = 0; c < numCameras; c++) {
    const startHStep = Math.round((c / numCameras) * hStepsTotal);
    const endHStep = Math.round(((c + 1) / numCameras) * hStepsTotal);
    const stepsInCam = endHStep - startHStep;

    const count = stepsInCam * channels;
    const start = currentIdx / 3;

    const startAngle = startHStep * hStepDeg - hFovDeg / 2;
    const endAngle = endHStep * hStepDeg - hFovDeg / 2;
    const cameraYaw = (startAngle + endAngle) / 2;
    const cameraHFov = endAngle - startAngle;

    for (let h = startHStep; h < endHStep; h++) {
      const hAngle = (h * hStepDeg - hFovDeg / 2) * DEG2RAD;
      const cosH = Math.cos(hAngle);
      const sinH = Math.sin(hAngle);

      for (let v = 0; v < channels; v++) {
        const vAngle = (vFovMinDeg + v * vStepDeg) * DEG2RAD;
        const cosV = Math.cos(vAngle);
        const sinV = Math.sin(vAngle);

        dirs[currentIdx++] = cosV * sinH; // X
        dirs[currentIdx++] = sinV; // Y
        dirs[currentIdx++] = cosV * cosH; // Z
      }
    }

    partitions.push({ start, count, yaw: cameraYaw, hFov: cameraHFov });
  }

  return { directions: dirs, partitions };
};

export class GpuLidarSimulator {
  private renderOptions: LidarRenderOptions;
  private pointsMesh: THREE.InstancedMesh;
  private material: THREE.MeshBasicNodeMaterial;
  private directionStorage: ReturnType<typeof storage>;
  private positionStorage: ReturnType<typeof storage>;
  private visibilityStorage: ReturnType<typeof storage>;

  private passes: CameraPass[] = [];
  private rotationMatrix = new THREE.Matrix4();

  private rotationMatrixUniform = uniform(new THREE.Matrix4());
  private originUniform = uniform(new THREE.Vector3());
  private nearUniform = uniform(0.1);
  private farUniform: ReturnType<typeof uniform>;
  private maxRangeUniform: ReturnType<typeof uniform>;
  private noiseUniform: ReturnType<typeof uniform>;
  private timeUniform: ReturnType<typeof uniform>;
  private colorUniform = uniform(new THREE.Color(0xffffff));
  private opacityUniform = uniform(1);
  private pointScaleUniform = uniform(1);
  private pointScale = 1;

  constructor(spec: LidarSpec, renderOptions: LidarRenderOptions) {
    this.renderOptions = { ...renderOptions };
    this.farUniform = uniform(spec.maxRange);
    this.maxRangeUniform = uniform(spec.maxRange);
    this.noiseUniform = uniform(renderOptions.noiseLevel);
    this.timeUniform = uniform(0);

    const { directions, partitions } = computeDirections(
      spec,
      renderOptions.resolutionScale,
    );
    const numPoints = directions.length / 3;

    const dirAttr = new THREE.InstancedBufferAttribute(directions, 3);
    const posAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(numPoints * 3),
      3,
    );
    const visAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(numPoints),
      1,
    );

    this.directionStorage = storage(dirAttr as any, 'vec3', numPoints);
    this.positionStorage = storage(posAttr as any, 'vec3', numPoints);
    this.visibilityStorage = storage(visAttr as any, 'float', numPoints);

    this.colorUniform.value = new THREE.Color(
      resolveLidarPointColorForShader(renderOptions.pointColor),
    );
    this.opacityUniform.value = renderOptions.opacity;
    this.pointScale = Math.max(0.1, renderOptions.pointSize);
    this.pointScaleUniform.value = this.pointScale;

    this.material = new THREE.MeshBasicNodeMaterial();
    this.material.transparent = true;
    this.material.depthWrite = false;

    const visibility = this.visibilityStorage.element(instanceIndex);
    this.material.colorNode = visibility
      .greaterThan(0.5)
      .select(this.colorUniform, vec3(0, 0, 0));
    this.material.opacityNode = this.opacityUniform.mul(visibility);
    this.material.positionNode = this.positionStorage
      .element(instanceIndex)
      .add(positionLocal.mul(this.pointScaleUniform));

    const geometry = new THREE.SphereGeometry(0.04, 6, 6);
    this.pointsMesh = new THREE.InstancedMesh(
      geometry,
      this.material,
      numPoints,
    );
    this.pointsMesh.frustumCulled = false;
    this.pointsMesh.renderOrder = 999;
    this.pointsMesh.scale.setScalar(1);
    this.pointsMesh.visible = renderOptions.enabled;

    const maxVAbs = Math.max(
      Math.abs(spec.vFovMinDeg),
      Math.abs(spec.vFovMaxDeg),
    );
    const cameraVFov = Math.max(1, maxVAbs * 2);

    for (const p of partitions) {
      if (p.count === 0) continue;

      const safeHFov = p.hFov + 0.2;
      const aspect = Math.max(
        0.1,
        Math.tan((safeHFov / 2) * DEG2RAD) /
          Math.tan((cameraVFov / 2) * DEG2RAD),
      );

      const camera = new THREE.PerspectiveCamera(
        cameraVFov,
        aspect,
        this.nearUniform.value,
        spec.maxRange,
      );
      // 임포트 모델 메시(LIDAR_MODEL_LAYER)와 지면만 깊이 패스에 포함합니다(바운딩박스·헬퍼는 레이어 0).
      camera.layers.disableAll();
      camera.layers.enable(Layer.LIDAR_MODEL_LAYER);
      camera.layers.enable(Layer.GROUND_LAYER);

      // [FIX] 카메라는 -Z 방향을 바라보므로 레이어의 방향과 맞추기 위해 180도를 추가 회전합니다.
      const localRotation = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        (p.yaw + 180) * DEG2RAD,
      );

      // const width = Math.max(4, Math.round(p.count / spec.channels));
      // const height = Math.max(1, spec.channels);
      const width = Math.max(1024, Math.round(p.count / spec.channels));
      const height = Math.max(1024, spec.channels);

      const depthTexture = new THREE.DepthTexture(width, height);
      depthTexture.minFilter = THREE.NearestFilter;
      depthTexture.magFilter = THREE.NearestFilter;
      depthTexture.generateMipmaps = false;

      const depthTarget = new THREE.RenderTarget(width, height);
      depthTarget.depthTexture = depthTexture;
      depthTarget.depthBuffer = true;

      const viewMatrixUniform = uniform(new THREE.Matrix4());
      const projectionMatrixUniform = uniform(new THREE.Matrix4());

      // [FIX] JS Number와 Shader Node간의 타입 문제를 피하기 위해 확실한 정수화 적용
      const startIdx = Math.floor(p.start);

      const computeNode = Fn(() => {
        const i = instanceIndex.add(startIdx);
        const dir = this.directionStorage.element(i);
        const worldDir = this.rotationMatrixUniform
          .mul(vec4(dir, 0.0))
          .xyz.normalize();
        const origin = this.originUniform;
        const worldPosFar = origin.add(worldDir.mul(this.maxRangeUniform));

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
          .or(depth.greaterThan(0.9999)); // [FIX] 깊이 정밀도 여유값 상향

        const far = this.farUniform;
        const near = this.nearUniform;
        const viewZ = near.mul(far).div(far.sub(near).mul(depth).sub(far));
        const dirView = viewMatrixUniform.mul(vec4(worldDir, 0.0)).xyz;
        const t = viewZ.div(dirView.z);
        const hitPos = origin.add(worldDir.mul(t));

        const seed = i.toFloat().add(this.timeUniform.mul(10.0));
        const randX = fract(sin(seed.mul(12.9898)).mul(43758.5453));
        const randY = fract(sin(seed.mul(78.233)).mul(12345.6789));
        const randZ = fract(sin(seed.mul(39.346)).mul(23421.631));
        const noiseVec = vec3(randX, randY, randZ)
          .sub(0.5)
          .mul(this.noiseUniform.mul(2.0));

        If(
          outside.or(t.lessThan(0.0)).or(t.greaterThan(this.maxRangeUniform)),
          () => {
            this.visibilityStorage.element(i).assign(0.0);
          },
        ).Else(() => {
          this.visibilityStorage.element(i).assign(1.0);
          this.positionStorage.element(i).assign(hitPos.add(noiseVec));
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
  }

  getPointsMesh(): THREE.InstancedMesh {
    return this.pointsMesh;
  }

  updateRenderOptions(options: LidarRenderOptions): boolean {
    if (options.resolutionScale !== this.renderOptions.resolutionScale) {
      return true;
    }
    this.renderOptions = { ...options };
    this.colorUniform.value = new THREE.Color(
      resolveLidarPointColorForShader(options.pointColor),
    );
    this.opacityUniform.value = options.opacity;
    this.pointsMesh.visible = options.enabled;
    this.pointScale = Math.max(0.1, options.pointSize);
    this.pointScaleUniform.value = this.pointScale;
    this.noiseUniform.value = options.noiseLevel;
    this.material.needsUpdate = true;
    return false;
  }

  simulate(
    origin: {
      x: number;
      y: number;
      z: number;
    },
    rotationQuat: THREE.Quaternion,
    scene: THREE.Scene,
    renderer: THREE.WebGPURenderer,
    exclude: Set<THREE.Object3D>,
  ): void {
    if (!renderer.compute) return;
    if (!this.renderOptions.enabled) return;

    this.originUniform.value.set(origin.x, origin.y, origin.z);
    this.noiseUniform.value = this.renderOptions.noiseLevel;
    this.timeUniform.value = performance.now() * 0.001;

    const baseRot = rotationQuat.clone();
    this.rotationMatrix.makeRotationFromQuaternion(baseRot);
    this.rotationMatrixUniform.value.copy(this.rotationMatrix);

    const hidden: Array<{ obj: THREE.Object3D; visible: boolean }> = [];
    for (const obj of exclude) {
      hidden.push({ obj, visible: obj.visible });
      obj.visible = false;
    }

    const prevTarget = renderer.getRenderTarget();

    for (const pass of this.passes) {
      const camRot = baseRot.clone().multiply(pass.localRotation);

      pass.camera.position.set(origin.x, origin.y, origin.z);
      pass.camera.quaternion.copy(camRot);
      pass.camera.updateMatrixWorld(true);

      (pass.viewMatrixUniform.value as any).copy(
        pass.camera.matrixWorldInverse,
      );
      (pass.projectionMatrixUniform.value as any).copy(
        pass.camera.projectionMatrix,
      );

      renderer.setRenderTarget(pass.depthTarget);
      renderer.render(scene, pass.camera);

      renderer.compute(pass.computeNode);
    }

    renderer.setRenderTarget(prevTarget);

    for (const { obj, visible } of hidden) {
      obj.visible = visible;
    }
  }

  dispose(): void {
    this.pointsMesh.geometry.dispose();
    this.material.dispose();
    for (const pass of this.passes) {
      pass.depthTarget.dispose();
      pass.depthTexture.dispose();
    }
  }
}
