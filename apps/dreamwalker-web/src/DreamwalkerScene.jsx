import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { Application, Entity } from '@playcanvas/react';
import { Camera, GSplat, Script } from '@playcanvas/react/components';
import { useApp, useAppEvent, useSplat } from '@playcanvas/react/hooks';
import { CameraControls } from 'playcanvas/scripts/esm/camera-controls.mjs';
import {
  BLEND_NONE,
  Color,
  PIXELFORMAT_R32F,
  PIXELFORMAT_RGBA8,
  RenderTarget,
  SHADER_PREPASS,
  Texture,
  Vec3
} from 'playcanvas';

const WalkRuntime = lazy(() => import('./WalkRuntime.jsx'));
const defaultRobotFrameStreamFps = 10;

function dynamicMapDiagnosticsEnabled() {
  if (typeof window === 'undefined') {
    return false;
  }

  const value = new URLSearchParams(window.location.search)
    .get('dynamicMapDiagnostics')
    ?.trim()
    .toLowerCase();

  return value === '1' || value === 'true';
}

function updateDynamicMapDiagnostics(update) {
  if (!dynamicMapDiagnosticsEnabled()) {
    return;
  }

  const current = window.__dreamwalkerDynamicMapDiagnostics ?? {
    activeSplatUrls: [],
    events: [],
    splatRefs: {}
  };
  const normalizedCurrent = {
    ...current,
    activeSplatUrls: Array.isArray(current.activeSplatUrls)
      ? current.activeSplatUrls
      : [],
    events: Array.isArray(current.events) ? current.events : [],
    splatRefs: current.splatRefs && typeof current.splatRefs === 'object'
      ? current.splatRefs
      : {}
  };
  const next = update(normalizedCurrent);

  window.__dreamwalkerDynamicMapDiagnostics = {
    ...current,
    ...next,
    events: next.events.slice(-80),
    updatedAt: Date.now()
  };
}

function recordSplatLifecycleEvent(type, splatUrl) {
  updateDynamicMapDiagnostics((current) => {
    const splatRefs = { ...current.splatRefs };
    const currentCount = Number(splatRefs[splatUrl] ?? 0);

    if (type === 'mount') {
      splatRefs[splatUrl] = currentCount + 1;
    } else if (type === 'unmount') {
      const nextCount = Math.max(0, currentCount - 1);
      if (nextCount > 0) {
        splatRefs[splatUrl] = nextCount;
      } else {
        delete splatRefs[splatUrl];
      }
    }

    return {
      ...current,
      activeSplatUrls: Object.keys(splatRefs),
      events: [
        ...current.events,
        {
          at: Date.now(),
          type,
          url: splatUrl
        }
      ],
      splatRefs
    };
  });
}

function SplatContent({ splatUrl, entityRef, onStatusChange }) {
  const { asset } = useSplat(splatUrl);

  useEffect(() => {
    recordSplatLifecycleEvent('mount', splatUrl);

    return () => {
      recordSplatLifecycleEvent('unmount', splatUrl);
    };
  }, [splatUrl]);

  useEffect(() => {
    onStatusChange?.({
      status: asset ? 'ready' : 'loading',
      url: splatUrl
    });

    if (asset) {
      recordSplatLifecycleEvent('asset-ready', splatUrl);
    }
  }, [asset, onStatusChange, splatUrl]);

  if (!asset) {
    return null;
  }

  return (
    <Entity
      ref={entityRef}
      position={[0, 0, 0]}
      rotation={[0, 0, 180]}>
      <GSplat asset={asset} />
    </Entity>
  );
}

function parseRobotFrameStreamConfigFromSearch() {
  if (typeof window === 'undefined') {
    return {
      enabled: false,
      fps: defaultRobotFrameStreamFps
    };
  }

  const searchParams = new URLSearchParams(window.location.search);
  const streamParam = searchParams.get('robotFrameStream')?.trim().toLowerCase() ?? '';
  const fpsParam = Number.parseFloat(searchParams.get('robotFrameFps') ?? '');

  return {
    enabled: streamParam === '1' || streamParam === 'true',
    fps: Number.isFinite(fpsParam) && fpsParam > 0 ? fpsParam : defaultRobotFrameStreamFps
  };
}

function parseRobotDepthStreamConfigFromSearch() {
  if (typeof window === 'undefined') {
    return {
      enabled: false,
      fps: defaultRobotFrameStreamFps
    };
  }

  const searchParams = new URLSearchParams(window.location.search);
  const streamParam = searchParams.get('robotDepthStream')?.trim().toLowerCase() ?? '';
  const fpsParam = Number.parseFloat(
    searchParams.get('robotDepthFps') ?? searchParams.get('robotFrameFps') ?? ''
  );

  return {
    enabled: streamParam === '1' || streamParam === 'true',
    fps: Number.isFinite(fpsParam) && fpsParam > 0 ? fpsParam : defaultRobotFrameStreamFps
  };
}

function captureCanvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    if (!canvas || typeof canvas.toBlob !== 'function') {
      reject(new Error('Canvas does not support blob capture'));
      return;
    }

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Canvas capture produced an empty blob'));
          return;
        }

        resolve(blob);
      },
      'image/jpeg',
      0.85
    );
  });
}

function createCaptureMetadata(cameraEntity, cameraComponent, width, height) {
  const position = cameraEntity.getPosition();
  const rotation = cameraEntity.getRotation();

  return {
    timestamp: new Date().toISOString(),
    width,
    height,
    fov: cameraComponent.fov,
    pose: {
      position: [position.x, position.y, position.z],
      orientation: [rotation.x, rotation.y, rotation.z, rotation.w]
    }
  };
}

function buildPackedFloatClearColor(value) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setFloat32(0, value, false);
  return new Color(bytes[0] / 255, bytes[1] / 255, bytes[2] / 255, bytes[3] / 255);
}

function decodePackedDepthBuffer(rawBuffer) {
  const depthValues = new Float32Array(rawBuffer.length / 4);
  const view = new DataView(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);

  for (let index = 0; index < depthValues.length; index += 1) {
    depthValues[index] = view.getFloat32(index * 4, false);
  }

  return depthValues;
}

function flipDepthBufferY(depthValues, width, height) {
  const flipped = new Float32Array(depthValues.length);

  for (let row = 0; row < height; row += 1) {
    const srcStart = (height - row - 1) * width;
    const dstStart = row * width;
    flipped.set(depthValues.subarray(srcStart, srcStart + width), dstStart);
  }

  return flipped;
}

function createFallbackDepthBuffer(width, height, depthValue) {
  const fallback = new Float32Array(width * height);
  fallback.fill(depthValue);
  return fallback;
}

function destroyDepthCaptureResources(resources) {
  if (!resources) {
    return;
  }

  if (Array.isArray(resources.viewBindGroups)) {
    for (const bindGroup of resources.viewBindGroups) {
      bindGroup?.defaultUniformBuffer?.destroy?.();
      bindGroup?.destroy?.();
    }
  }

  resources.renderTarget?.destroy?.();
  resources.texture?.destroy?.();
}

function findGsplatMeshInstance(cameraComponent) {
  const app = cameraComponent?.system?.app;
  const scene = app?.scene;
  const camera = cameraComponent?.camera;
  const layerIds = Array.isArray(cameraComponent?.layers) ? cameraComponent.layers : [];

  if (!scene || !camera || !layerIds.length) {
    return null;
  }

  for (const layerId of layerIds) {
    const layer = scene.layers.getLayerById(layerId);
    if (!layer) {
      continue;
    }

    const unifiedMeshInstance =
      app?.renderer?.gsplatDirector?.camerasMap
        ?.get(camera)
        ?.layersMap?.get(layer)
        ?.gsplatManager?.renderer?.meshInstance ?? null;

    if (unifiedMeshInstance) {
      return unifiedMeshInstance;
    }

    const culledInstances = layer.getCulledInstances(camera);
    const visibleInstances = [
      ...(culledInstances?.opaque ?? []),
      ...(culledInstances?.transparent ?? [])
    ];
    const gsplatMeshInstance = visibleInstances.find((meshInstance) => meshInstance?.gsplatInstance);

    if (gsplatMeshInstance) {
      return gsplatMeshInstance;
    }
  }

  return null;
}

function ensureDepthCaptureResources(app, resourcesRef, width, height) {
  const existing = resourcesRef.current;
  const format = app.graphicsDevice.textureFloatRenderable ? PIXELFORMAT_R32F : PIXELFORMAT_RGBA8;

  if (
    existing &&
    existing.width === width &&
    existing.height === height &&
    existing.format === format
  ) {
    return existing;
  }

  destroyDepthCaptureResources(existing);

  const texture = new Texture(app.graphicsDevice, {
    name: 'DreamwalkerDepthCaptureTexture',
    width,
    height,
    format,
    mipmaps: false
  });
  const renderTarget = new RenderTarget({
    name: 'DreamwalkerDepthCaptureTarget',
    colorBuffer: texture,
    depth: true,
    samples: 1
  });
  const next = {
    width,
    height,
    format,
    texture,
    renderTarget,
    viewBindGroups: []
  };

  resourcesRef.current = next;
  return next;
}

function readDepthPixels(app, resources, width, height) {
  const device = app.graphicsDevice;
  const gl = device.gl;

  if (!gl) {
    return device.readTextureAsync(resources.texture, 0, 0, width, height, {
      immediate: true,
      renderTarget: resources.renderTarget
    });
  }

  const previousRenderTarget = device.renderTarget ?? null;
  const readBuffer =
    resources.format === PIXELFORMAT_R32F
      ? new Float32Array(width * height)
      : new Uint8Array(width * height * 4);
  const textureImpl = resources.texture.impl;

  device.setRenderTarget(resources.renderTarget);
  device.updateBegin();

  try {
    gl.readPixels(
      0,
      0,
      width,
      height,
      textureImpl?._glFormat ?? gl.RGBA,
      textureImpl?._glPixelType ?? gl.UNSIGNED_BYTE,
      readBuffer
    );
  } finally {
    device.updateEnd();
    device.setRenderTarget(previousRenderTarget);
  }

  return readBuffer;
}

async function captureDepthBuffer(app, cameraComponent, meshInstance, resourcesRef, width, height) {
  const resources = ensureDepthCaptureResources(app, resourcesRef, width, height);
  const sceneCamera = cameraComponent.camera;
  const material = meshInstance.material;
  const previousBlendType = material.blendType;
  const previousDepthWrite = material.depthWrite;
  const previousClearColor = new Color(
    sceneCamera.clearColor.r,
    sceneCamera.clearColor.g,
    sceneCamera.clearColor.b,
    sceneCamera.clearColor.a
  );
  const previousClearColorBuffer = sceneCamera.clearColorBuffer;
  const previousClearDepth = sceneCamera.clearDepth;
  const previousClearDepthBuffer = sceneCamera.clearDepthBuffer;
  const farClip = cameraComponent.farClip;

  try {
    material.blendType = BLEND_NONE;
    material.depthWrite = true;
    sceneCamera.clearColor =
      resources.format === PIXELFORMAT_R32F
        ? new Color(farClip, 0, 0, 0)
        : buildPackedFloatClearColor(farClip);
    sceneCamera.clearColorBuffer = true;
    sceneCamera.clearDepth = 1;
    sceneCamera.clearDepthBuffer = true;

    app.renderer.renderForwardLayer(
      sceneCamera,
      resources.renderTarget,
      null,
      false,
      SHADER_PREPASS,
      resources.viewBindGroups,
      {
        meshInstances: [meshInstance],
        clearColor: true,
        clearDepth: true
      }
    );

    const rawBuffer = await readDepthPixels(app, resources, width, height);
    const linearDepth =
      resources.format === PIXELFORMAT_R32F
        ? rawBuffer
        : decodePackedDepthBuffer(rawBuffer);

    return flipDepthBufferY(linearDepth, width, height);
  } finally {
    material.blendType = previousBlendType;
    material.depthWrite = previousDepthWrite;
    sceneCamera.clearColor = previousClearColor;
    sceneCamera.clearColorBuffer = previousClearColorBuffer;
    sceneCamera.clearDepth = previousClearDepth;
    sceneCamera.clearDepthBuffer = previousClearDepthBuffer;
  }
}

function ProjectionBridge({ cameraEntityRef, points, onProjected }) {
  const worldPositionsRef = useRef(
    points.map((point) => ({
      id: point.id,
      position: new Vec3(point.position[0], point.position[1], point.position[2])
    }))
  );
  const lastEmitTimeRef = useRef(0);
  const lastSignatureRef = useRef('');

  useEffect(() => {
    worldPositionsRef.current = points.map((point) => ({
      id: point.id,
      position: new Vec3(point.position[0], point.position[1], point.position[2])
    }));
  }, [points]);

  useAppEvent('postrender', () => {
    const now = performance.now();
    if (now - lastEmitTimeRef.current < 50) {
      return;
    }

    const cameraEntity = cameraEntityRef.current;
    const cameraComponent = cameraEntity?.camera;

    if (!cameraComponent) {
      return;
    }

    const width = cameraComponent.system.app.graphicsDevice.width;
    const height = cameraComponent.system.app.graphicsDevice.height;
    const nextProjected = [];

    for (let index = 0; index < points.length; index += 1) {
      const point = points[index];
      const worldPosition = worldPositionsRef.current[index]?.position;

      if (!worldPosition) {
        continue;
      }

      const screenPosition = cameraComponent.worldToScreen(worldPosition);
      const normalizedX = screenPosition.x / width;
      const normalizedY = 1 - screenPosition.y / height;
      const isVisible =
        screenPosition.z > 0 &&
        normalizedX >= -0.15 &&
        normalizedX <= 1.15 &&
        normalizedY >= -0.15 &&
        normalizedY <= 1.15;

      if (!isVisible) {
        continue;
      }

      nextProjected.push({
        id: point.id,
        xPercent: Number((normalizedX * 100).toFixed(2)),
        yPercent: Number((normalizedY * 100).toFixed(2)),
        zDepth: Number(screenPosition.z.toFixed(3))
      });
    }

    const signature = JSON.stringify(nextProjected);
    if (signature === lastSignatureRef.current) {
      return;
    }

    lastEmitTimeRef.current = now;
    lastSignatureRef.current = signature;
    onProjected(nextProjected);
  });

  return null;
}

function FrameStreamBridge({
  cameraEntityRef,
  splatEntityRef,
  enabled,
  fps,
  onFrame,
  onDepthFrame
}) {
  const app = useApp();
  const depthStreamConfig = parseRobotDepthStreamConfigFromSearch();
  const lastCaptureTimeRef = useRef(0);
  const captureInFlightRef = useRef(false);
  const depthCaptureResourcesRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      lastCaptureTimeRef.current = 0;
      captureInFlightRef.current = false;
    }
  }, [enabled, fps]);

  useEffect(
    () => () => {
      destroyDepthCaptureResources(depthCaptureResourcesRef.current);
      depthCaptureResourcesRef.current = null;
    },
    []
  );

  useAppEvent('postrender', () => {
    const frameCaptureEnabled = enabled && typeof onFrame === 'function';
    const depthCaptureEnabled =
      depthStreamConfig.enabled && typeof onDepthFrame === 'function';

    if ((!frameCaptureEnabled && !depthCaptureEnabled) || captureInFlightRef.current) {
      return;
    }

    const now = performance.now();
    const activeFps = [
      frameCaptureEnabled ? fps : null,
      depthCaptureEnabled ? depthStreamConfig.fps : null
    ].filter((value) => Number.isFinite(value) && value > 0);
    const captureIntervalMs = 1000 / Math.min(...activeFps);

    if (now - lastCaptureTimeRef.current < captureIntervalMs) {
      return;
    }

    const cameraEntity = cameraEntityRef.current;
    const cameraComponent = cameraEntity?.camera;
    const canvas = cameraComponent?.system?.app?.graphicsDevice?.canvas;

    if (!cameraComponent || !canvas) {
      return;
    }

    const depthMeshInstance =
      splatEntityRef.current?.gsplat?.instance?.meshInstance ??
      findGsplatMeshInstance(cameraComponent);
    const canRenderDepthFromGsplat =
      Boolean(app && depthMeshInstance && depthMeshInstance.instancingCount > 0);

    lastCaptureTimeRef.current = now;
    captureInFlightRef.current = true;
    const captureMetadata = createCaptureMetadata(
      cameraEntity,
      cameraComponent,
      canvas.width,
      canvas.height
    );
    const captureTasks = [];

    if (frameCaptureEnabled) {
      captureTasks.push(
        captureCanvasBlob(canvas).then((blob) =>
          Promise.resolve(onFrame(blob, { ...captureMetadata }))
        )
      );
    }

    if (depthCaptureEnabled) {
      captureTasks.push(
        (
          canRenderDepthFromGsplat
            ? captureDepthBuffer(
                app,
                cameraComponent,
                depthMeshInstance,
                depthCaptureResourcesRef,
                canvas.width,
                canvas.height
              )
            : Promise.resolve(
                createFallbackDepthBuffer(
                  canvas.width,
                  canvas.height,
                  cameraComponent.farClip
                )
              )
        ).then((depthBuffer) => {
          return Promise.resolve(
            onDepthFrame(depthBuffer, {
              ...captureMetadata,
              nearClip: cameraComponent.nearClip,
              farClip: cameraComponent.farClip
            })
          );
        })
      );
    }

    Promise.allSettled(captureTasks)
      .then((results) => {
        for (const result of results) {
          if (result.status === 'rejected') {
            console.error('DreamWalker sensor capture failed', result.reason);
          }
        }
      })
      .finally(() => {
        captureInFlightRef.current = false;
      });
  });

  return null;
}

function rotateLocalOffset(localOffset, yawDegrees) {
  const radians = (yawDegrees * Math.PI) / 180;
  const sin = Math.sin(radians);
  const cos = Math.cos(radians);
  const [x, y, z] = localOffset;

  return new Vec3(
    x * cos - z * sin,
    y,
    -x * sin - z * cos
  );
}

function CameraPresetBridge({
  orbitCameraEntityRef,
  playerEntityRef,
  playerControllerRef,
  walkCameraEntityRef,
  cameraMode,
  currentPreset,
  cameraLocalHeight,
  roboticsCameraEnabled,
  onPresetApplied
}) {
  const pendingSignatureRef = useRef(`${cameraMode}:${currentPreset.id}`);

  useEffect(() => {
    pendingSignatureRef.current = `${cameraMode}:${currentPreset.id}`;
  }, [cameraMode, currentPreset.id]);

  useAppEvent('update', () => {
    if (roboticsCameraEnabled && cameraMode !== 'walk') {
      pendingSignatureRef.current = '';
      return;
    }

    const signature = `${cameraMode}:${currentPreset.id}`;
    if (pendingSignatureRef.current !== signature) {
      return;
    }

    const position = new Vec3(
      currentPreset.position[0],
      currentPreset.position[1],
      currentPreset.position[2]
    );

    if (cameraMode === 'walk') {
      const playerEntity = playerEntityRef.current;
      const walkCameraEntity = walkCameraEntityRef.current;

      if (!playerEntity || !walkCameraEntity) {
        return;
      }

      const walkOrigin = new Vec3(position.x, position.y - cameraLocalHeight, position.z);
      if (playerEntity.rigidbody?.teleport) {
        playerEntity.rigidbody.teleport(walkOrigin);
        playerEntity.rigidbody.linearVelocity = new Vec3(0, 0, 0);
        playerEntity.rigidbody.angularVelocity = new Vec3(0, 0, 0);
      } else {
        playerEntity.setPosition(walkOrigin);
      }

      walkCameraEntity.setLocalPosition(0, cameraLocalHeight, 0);

      if (currentPreset.focusPoint) {
        walkCameraEntity.lookAt(
          currentPreset.focusPoint[0],
          currentPreset.focusPoint[1],
          currentPreset.focusPoint[2]
        );
      } else {
        walkCameraEntity.setEulerAngles(
          currentPreset.rotation[0],
          currentPreset.rotation[1],
          currentPreset.rotation[2]
        );
      }

      const walkAngles = walkCameraEntity.getLocalEulerAngles().clone();
      if (!playerControllerRef.current?._angles) {
        return;
      }

      playerControllerRef.current._angles.copy(walkAngles);
      walkCameraEntity.setLocalEulerAngles(walkAngles);

      pendingSignatureRef.current = '';
      onPresetApplied(currentPreset.label);
      return;
    }

    const orbitCameraEntity = orbitCameraEntityRef.current;
    if (!orbitCameraEntity) {
      return;
    }

    const controller = orbitCameraEntity.script?.cameraControls;

    if (controller && currentPreset.focusPoint) {
      controller.reset(
        new Vec3(
          currentPreset.focusPoint[0],
          currentPreset.focusPoint[1],
          currentPreset.focusPoint[2]
        ),
        position
      );
    } else {
      orbitCameraEntity.setPosition(position);
      orbitCameraEntity.setEulerAngles(
        currentPreset.rotation[0],
        currentPreset.rotation[1],
        currentPreset.rotation[2]
      );
    }

    pendingSignatureRef.current = '';
    onPresetApplied(currentPreset.label);
  });

  return null;
}

function RobotCameraBridge({
  orbitCameraEntityRef,
  enabled,
  robotPose,
  selectedCamera
}) {
  useAppEvent('update', () => {
    if (!enabled || !selectedCamera || !robotPose) {
      return;
    }

    const orbitCameraEntity = orbitCameraEntityRef.current;
    if (!orbitCameraEntity) {
      return;
    }

    const positionOffset = rotateLocalOffset(
      selectedCamera.localOffset ?? [0, 1.2, 0],
      robotPose.yawDegrees
    );
    const lookAtOffset = rotateLocalOffset(
      selectedCamera.lookAtOffset ?? [0, 0.8, 1],
      robotPose.yawDegrees
    );

    orbitCameraEntity.setPosition(
      robotPose.position[0] + positionOffset.x,
      robotPose.position[1] + positionOffset.y,
      robotPose.position[2] + positionOffset.z
    );
    orbitCameraEntity.lookAt(
      robotPose.position[0] + lookAtOffset.x,
      robotPose.position[1] + lookAtOffset.y,
      robotPose.position[2] + lookAtOffset.z
    );
  });

  return null;
}

function MapPositionBridge({
  cameraEntityRef,
  enabled,
  onPositionChange,
  robotPose,
  roboticsCameraEnabled
}) {
  const lastEmitTimeRef = useRef(0);
  const lastSignatureRef = useRef('');

  useAppEvent('update', () => {
    if (!enabled || typeof onPositionChange !== 'function') {
      return;
    }

    const now = performance.now();
    if (now - lastEmitTimeRef.current < 250) {
      return;
    }

    const source = roboticsCameraEnabled && robotPose ? 'robot' : 'camera';
    const position =
      source === 'robot'
        ? {
            x: Number(robotPose.position?.[0] ?? 0),
            y: Number(robotPose.position?.[1] ?? 0),
            z: Number(robotPose.position?.[2] ?? 0)
          }
        : (() => {
            const cameraPosition = cameraEntityRef.current?.getPosition?.();
            return cameraPosition
              ? {
                  x: Number(cameraPosition.x),
                  y: Number(cameraPosition.y),
                  z: Number(cameraPosition.z)
                }
              : null;
          })();

    if (!position || ![position.x, position.y, position.z].every(Number.isFinite)) {
      return;
    }

    const signature = [
      source,
      position.x.toFixed(2),
      position.y.toFixed(2),
      position.z.toFixed(2)
    ].join(':');
    if (signature === lastSignatureRef.current) {
      return;
    }

    lastEmitTimeRef.current = now;
    lastSignatureRef.current = signature;
    onPositionChange({
      source,
      position,
      timestamp: Date.now()
    });
  });

  return null;
}

export default function DreamwalkerScene({
  worldConfig,
  cameraMode,
  currentPreset,
  hotspots,
  loopItems,
  robotPoints = [],
  robotRoutePoints = [],
  benchmarkRoutePoints = [],
  semanticZonePoints = [],
  semanticZoneSurfacePoints = [],
  roboticsCamera,
  onColliderStatusChange,
  onPresetApplied,
  onHotspotsProjected,
  onLoopItemsProjected,
  onRobotPointsProjected,
  onRobotRoutePointsProjected,
  onBenchmarkRoutePointsProjected,
  onSemanticZonePointsProjected,
  onSemanticZoneSurfacePointsProjected,
  onFrame,
  onDepthFrame,
  onMapPositionChange,
  onSplatStatusChange,
  dynamicMapPositionTrackingEnabled = true,
  splatUrl
}) {
  const frameStreamConfig = parseRobotFrameStreamConfigFromSearch();
  const isWalkMode = cameraMode === 'walk';
  const isRoboticsCameraEnabled = Boolean(roboticsCamera?.enabled && roboticsCamera?.selectedCamera);
  const hasColliderMesh = Boolean(worldConfig.colliderMeshUrl);
  const orbitCameraEntityRef = useRef(null);
  const playerEntityRef = useRef(null);
  const playerControllerRef = useRef(null);
  const splatEntityRef = useRef(null);
  const walkCameraEntityRef = useRef(null);
  const [colliderStatus, setColliderStatus] = useState(() =>
    hasColliderMesh ? { mode: 'idle', error: null } : { mode: 'proxy', error: null }
  );
  const [walkCameraEntity, setWalkCameraEntity] = useState(null);
  const activeCameraEntityRef = isWalkMode ? walkCameraEntityRef : orbitCameraEntityRef;
  const shouldUseProxyColliders =
    isWalkMode &&
    (!hasColliderMesh || colliderStatus.mode === 'loading' || colliderStatus.mode === 'error');

  useEffect(() => {
    const nextStatus = hasColliderMesh
      ? isWalkMode
        ? { mode: 'loading', error: null }
        : { mode: 'idle', error: null }
      : { mode: 'proxy', error: null };

    setColliderStatus(nextStatus);
    onColliderStatusChange?.(nextStatus);
  }, [hasColliderMesh, isWalkMode, onColliderStatusChange]);

  const handleColliderStatusChange = useCallback(
    (nextStatus) => {
      setColliderStatus(nextStatus);
      onColliderStatusChange?.(nextStatus);
    },
    [onColliderStatusChange]
  );

  const handleWalkCameraRef = useCallback((entity) => {
    walkCameraEntityRef.current = entity;
    setWalkCameraEntity(entity);
  }, []);
  const handleSplatStatusChange = useCallback(
    (nextStatus) => {
      onSplatStatusChange?.({
        ...nextStatus,
        fragmentId: worldConfig.fragmentId
      });
    },
    [onSplatStatusChange, worldConfig.fragmentId]
  );

  return (
    <Application
      className="dreamwalker-canvas"
      usePhysics={isWalkMode}
      graphicsDeviceOptions={{
        alpha: false,
        antialias: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true
      }}>
      <Entity
        name="OrbitCamera"
        ref={orbitCameraEntityRef}
        position={currentPreset.position}
        rotation={currentPreset.rotation}>
        <Camera enabled={!isWalkMode} />
        <Script
          script={CameraControls}
          enabled={!isWalkMode && !isRoboticsCameraEnabled}
          enableFly
          enableOrbit
          enablePan
          moveSpeed={10}
          moveFastSpeed={20}
          moveSlowSpeed={5}
          rotateSpeed={0.2}
        />
      </Entity>

      {isWalkMode ? (
        <Suspense fallback={null}>
          <WalkRuntime
            currentPreset={currentPreset}
            playerEntityRef={playerEntityRef}
            playerControllerRef={playerControllerRef}
            handleWalkCameraRef={handleWalkCameraRef}
            walkCameraEntity={walkCameraEntity}
            walkController={worldConfig.walkController}
            walkProxyColliders={worldConfig.walkProxyColliders}
            colliderMeshUrl={worldConfig.colliderMeshUrl}
            showColliderDebug={worldConfig.showColliderDebug}
            shouldUseProxyColliders={shouldUseProxyColliders}
            onStatusChange={handleColliderStatusChange}
          />
        </Suspense>
      ) : null}
      {splatUrl ? (
        <SplatContent
          key={splatUrl}
          splatUrl={splatUrl}
          entityRef={splatEntityRef}
          onStatusChange={handleSplatStatusChange}
        />
      ) : null}
      <CameraPresetBridge
        orbitCameraEntityRef={orbitCameraEntityRef}
        playerEntityRef={playerEntityRef}
        playerControllerRef={playerControllerRef}
        walkCameraEntityRef={walkCameraEntityRef}
        cameraMode={cameraMode}
        currentPreset={currentPreset}
        cameraLocalHeight={worldConfig.walkController.cameraLocalHeight}
        roboticsCameraEnabled={isRoboticsCameraEnabled}
        onPresetApplied={onPresetApplied}
      />
      <RobotCameraBridge
        orbitCameraEntityRef={orbitCameraEntityRef}
        enabled={isRoboticsCameraEnabled}
        robotPose={roboticsCamera?.robotPose}
        selectedCamera={roboticsCamera?.selectedCamera}
      />
      <MapPositionBridge
        cameraEntityRef={activeCameraEntityRef}
        enabled={Boolean(dynamicMapPositionTrackingEnabled)}
        onPositionChange={onMapPositionChange}
        robotPose={roboticsCamera?.robotPose}
        roboticsCameraEnabled={isRoboticsCameraEnabled}
      />
      <FrameStreamBridge
        cameraEntityRef={activeCameraEntityRef}
        splatEntityRef={splatEntityRef}
        enabled={frameStreamConfig.enabled}
        fps={frameStreamConfig.fps}
        onFrame={onFrame}
        onDepthFrame={onDepthFrame}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={hotspots}
        onProjected={onHotspotsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={loopItems}
        onProjected={onLoopItemsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={robotPoints}
        onProjected={onRobotPointsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={robotRoutePoints}
        onProjected={onRobotRoutePointsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={benchmarkRoutePoints}
        onProjected={onBenchmarkRoutePointsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={semanticZonePoints}
        onProjected={onSemanticZonePointsProjected}
      />
      <ProjectionBridge
        cameraEntityRef={activeCameraEntityRef}
        points={semanticZoneSurfacePoints}
        onProjected={onSemanticZoneSurfacePointsProjected}
      />
    </Application>
  );
}
