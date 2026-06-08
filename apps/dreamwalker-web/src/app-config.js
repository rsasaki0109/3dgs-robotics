import { resolvePublicUrl } from './public-url.js';

const defaultRoboticsCameras = [
  {
    id: 'front',
    label: 'Front Camera',
    localOffset: [0, 1.18, 0.14],
    lookAtOffset: [0, 0.96, 2.4]
  },
  {
    id: 'chase',
    label: 'Chase Camera',
    localOffset: [0, 1.85, -3.2],
    lookAtOffset: [0, 0.92, 2.2]
  },
  {
    id: 'top',
    label: 'Top View',
    localOffset: [0, 6.4, 0.2],
    lookAtOffset: [0, 0, 0.8]
  }
];

const residencyFragment = {
  fragmentId: 'residency',
  fragmentLabel: 'Residency',
  homePresetId: 'foyer',
  overlayBranding: {
    id: 'residency-branding',
    label: 'Residency Broadcast',
    accent: '#f4ca72',
    highlight: '#85e3e1',
    glow: 'rgba(244, 202, 114, 0.28)',
    badge: 'Residency Broadcast',
    strapline: 'late-night return / window talk / home fragment'
  },
  assetBundle: {
    label: 'Residency Marble',
    expectedSplatUrl: '/splats/residency-main.sog',
    expectedColliderMeshUrl: '/colliders/residency-main-collider.glb',
    worldNote:
      'Residency fragment 用の World Labs Marble export を差し込む場所。' +
      '実ファイル未配置の間は demo splat へフォールバックする。'
  },
  walkProxyColliders: [
    {
      id: 'dreamwalker-floor-proxy',
      position: [0, -0.5, 6],
      halfExtents: [10, 0.5, 18]
    }
  ],
  robotics: {
    spawnPosition: [0, 0, 5.8],
    spawnYawDegrees: 180,
    semanticZoneMapUrl: '/manifests/robotics-residency.zones.json'
  },
  cameraPresets: [
    {
      id: 'foyer',
      label: 'Foyer',
      position: [0, 1.5, 6],
      rotation: [0, 180, 0],
      focusPoint: [0, 1.5, 0],
      description: '最初の印象を作る導入構図'
    },
    {
      id: 'window',
      label: 'Window',
      position: [4, 2.2, 5],
      rotation: [-8, 140, 0],
      focusPoint: [0.2, 1.4, 1.5],
      description: '配信の雑談に向く静かな斜め構図'
    },
    {
      id: 'gate',
      label: 'Gate',
      position: [0, 1.8, 10],
      rotation: [0, 180, 0],
      focusPoint: [0, 1.6, 6.5],
      description: 'DreamGate や shard の告知カット向け'
    }
  ],
  streamScenes: [
    {
      id: 'residency-intro',
      label: 'Intro',
      title: 'Residency Intro',
      topic: '配信開始 / ただいまの一言',
      presetId: 'foyer',
      overlayMemo: {
        title: 'Opening Beats',
        items: ['ただいまの一言', '今日の配信テーマ', '最初に歩く場所'],
        footer: 'home fragment start'
      },
      overlayBrandingOverrides: {
        badge: 'Residency Intro',
        strapline: 'standby / homecoming / opening line'
      }
    },
    {
      id: 'window-talk',
      label: 'Window Talk',
      title: 'Window Talk',
      topic: '雑談 / 近況 / 深夜の空気',
      presetId: 'window',
      overlayMemo: {
        title: 'Window Memo',
        items: ['最近の進捗を 1 本', '夢っぽい話題を 1 本', '写真モード告知を 1 回'],
        footer: 'slow talk block'
      },
      overlayBrandingOverrides: {
        badge: 'Window Talk',
        strapline: 'soft night talk / residency window / low-tempo stream',
        highlight: '#b7f3ef'
      }
    },
    {
      id: 'gate-recap',
      label: 'Gate Recap',
      title: 'Gate Recap',
      topic: '章の締め / 次 fragment への振り',
      presetId: 'gate',
      overlayMemo: {
        title: 'Gate Recap Notes',
        items: ['今日拾った shard', '次 fragment の導線', '締めの一言'],
        footer: 'chapter handoff'
      },
      overlayBrandingOverrides: {
        badge: 'Gate Recap',
        strapline: 'chapter handoff / distortion collected / next fragment',
        accent: '#85e3e1'
      }
    }
  ],
  shards: [
    {
      id: 'shard-01',
      kind: 'distortion-shard',
      label: 'Shard 01',
      title: 'Distortion Shard 01',
      position: [-1.75, 1.1, 4.4],
      accentColor: '#9deaf4',
      body: 'Dynamic Map Viewer の最初の欠片。導入トークや最初の感想を置く場所。'
    },
    {
      id: 'shard-02',
      kind: 'distortion-shard',
      label: 'Shard 02',
      title: 'Distortion Shard 02',
      position: [0.1, 1.15, 6.4],
      accentColor: '#9fd9ff',
      body: '中央の shard。視線誘導と「この世界を歩いている」実感を作る。'
    },
    {
      id: 'shard-03',
      kind: 'distortion-shard',
      label: 'Shard 03',
      title: 'Distortion Shard 03',
      position: [1.8, 1.2, 8.1],
      accentColor: '#a6f1da',
      body: '最後の欠片。回収すると DreamGate が配信終盤の話題として使える。'
    }
  ],
  hotspots: [
    {
      id: 'home-note',
      kind: 'echo-note',
      label: 'Residency Note',
      title: 'Echo Note: ここに住んでいる設定',
      position: [0, 1.75, 2.6],
      accentColor: '#f4ca72',
      body:
        'Dynamic Map Viewer では、この世界は「訪れる場所」ではなく、' +
        '配信者が帰ってくる生活圏として扱う。\n\n' +
        '毎回の導入、待機、雑談、告知、深夜の散歩まで同じ空間の延長で語れるようにする。'
    },
    {
      id: 'photo-spot',
      kind: 'photo-spot',
      label: 'Photo Spot',
      title: 'Photo Spot: プロフ画像向けの定点',
      position: [3.5, 2.1, 4.2],
      accentColor: '#85e3e1',
      presetId: 'window',
      body:
        'ここはプロフィール画像と告知画像を量産するための固定構図ポイント。\n\n' +
        'Photo Mode に入って 4:5 か 1:1 を選び、フィルタを Soft Dream にすると使いやすい。'
    },
    {
      id: 'stream-topic',
      kind: 'stream-topic',
      label: 'Live Topic',
      title: 'Echo Note: 今日の配信テーマ',
      position: [0.1, 2.1, 9.6],
      accentColor: '#a7d4ff',
      body:
        'DreamGate 付近は「新しい記憶」「次の世界」「最近見た夢」みたいな話題に繋げやすい。\n\n' +
        '今後は chat command で gate pulse や shard glow を起こせるようにする。'
    }
  ],
  gate: {
    id: 'dream-gate',
    kind: 'dream-gate',
    position: [0, 1.95, 11.2],
    presetId: 'gate',
    targetFragmentId: 'echo-chamber',
    targetFragmentLabel: 'Echo Chamber',
    lockedAccentColor: '#f4ca72',
    openAccentColor: '#85e3e1',
    lockedTitle: 'Dream Gate: まだ閉ざされている',
    openTitle: 'Dream Gate: Echo Chamber へ接続',
    lockedBody:
      '歪みの欠片を集めると gate が目を覚ます。\n\n' +
      'Dynamic Map Viewer では、この gate は次の fragment を切り替える browser portal として扱う。',
    openBody:
      'DreamGate は Echo Chamber に接続された。\n\n' +
      'ここから別の fragment へ遷移して、配信の章立てを切り替えられる。'
  }
};

const echoChamberFragment = {
  fragmentId: 'echo-chamber',
  fragmentLabel: 'Echo Chamber',
  homePresetId: 'threshold',
  overlayBranding: {
    id: 'echo-chamber-branding',
    label: 'Echo Chamber Feed',
    accent: '#86e1ff',
    highlight: '#b8c8ff',
    glow: 'rgba(134, 225, 255, 0.24)',
    badge: 'Echo Chamber Feed',
    strapline: 'residual chorus / chapter shift / aftertalk'
  },
  assetBundle: {
    label: 'Echo Chamber Marble',
    expectedSplatUrl: '/splats/echo-chamber-main.sog',
    expectedColliderMeshUrl: '/colliders/echo-chamber-main-collider.glb',
    worldNote:
      'Echo Chamber fragment 用の Marble export を差し込む場所。' +
      'chapter ごとに別 splat を見せる前提の設定。'
  },
  walkProxyColliders: [
    {
      id: 'echo-floor-proxy',
      position: [0, -0.5, 4.5],
      halfExtents: [8, 0.5, 14]
    }
  ],
  robotics: {
    spawnPosition: [0, 0, 4.8],
    spawnYawDegrees: 180,
    semanticZoneMapUrl: '/manifests/robotics-echo-chamber.zones.json'
  },
  cameraPresets: [
    {
      id: 'threshold',
      label: 'Threshold',
      position: [0, 1.6, 4.8],
      rotation: [0, 180, 0],
      focusPoint: [0, 1.55, 0.8],
      description: 'fragment 遷移直後の導入カット'
    },
    {
      id: 'chorus',
      label: 'Chorus',
      position: [-3, 2.1, 4],
      rotation: [-6, 214, 0],
      focusPoint: [0.2, 1.4, 2.6],
      description: '残響の塊を背景にした横構図'
    },
    {
      id: 'return-gate',
      label: 'Return Gate',
      position: [0, 1.8, 7.4],
      rotation: [0, 180, 0],
      focusPoint: [0, 1.6, 5.4],
      description: 'Residency へ戻る門の告知カット'
    }
  ],
  streamScenes: [
    {
      id: 'echo-threshold-live',
      label: 'Threshold',
      title: 'Echo Threshold',
      topic: 'chapter 切替直後の導入',
      presetId: 'threshold',
      overlayMemo: {
        title: 'Threshold Beats',
        items: ['chapter 切替の一言', '空気の差分', '次に寄るスポット'],
        footer: 'echo fragment start'
      },
      overlayBrandingOverrides: {
        badge: 'Echo Threshold',
        strapline: 'chapter shift / resonance check / threshold open'
      }
    },
    {
      id: 'echo-chorus-live',
      label: 'Chorus',
      title: 'Chorus Break',
      topic: '切り抜き / 告知 / 異常な静けさ',
      presetId: 'chorus',
      overlayMemo: {
        title: 'Chorus Notes',
        items: ['切り抜きにしたい一言', '告知を 1 本', '静かな絵を数秒維持'],
        footer: 'chorus segment'
      },
      overlayBrandingOverrides: {
        badge: 'Chorus Break',
        strapline: 'echo chorus / clipped memory / broadcast shimmer',
        highlight: '#d4dcff'
      }
    },
    {
      id: 'echo-return-live',
      label: 'Return Gate',
      title: 'Return Gate',
      topic: 'Residency 帰還前の締め',
      presetId: 'return-gate',
      overlayMemo: {
        title: 'Return Checklist',
        items: ['Residency に持ち帰る話題', '次回予告', '閉じの挨拶'],
        footer: 'return vector'
      },
      overlayBrandingOverrides: {
        badge: 'Return Gate',
        strapline: 'return vector / residency recall / fragment close',
        accent: '#93ffd4'
      }
    }
  ],
  shards: [
    {
      id: 'echo-shard-01',
      kind: 'distortion-shard',
      label: 'Echo 01',
      title: 'Echo Shard 01',
      position: [-1.4, 1.1, 2.8],
      accentColor: '#b8c8ff',
      body: 'ここでは shard が配信の残響や切り抜きネタとして扱われる。'
    },
    {
      id: 'echo-shard-02',
      kind: 'distortion-shard',
      label: 'Echo 02',
      title: 'Echo Shard 02',
      position: [0.2, 1.2, 4.2],
      accentColor: '#86e1ff',
      body: '中央の echo shard。ここで chapter の空気を切り替える。'
    },
    {
      id: 'echo-shard-03',
      kind: 'distortion-shard',
      label: 'Echo 03',
      title: 'Echo Shard 03',
      position: [1.7, 1.15, 5.7],
      accentColor: '#93ffd4',
      body: '最後の echo shard。Residency へ戻るか次章へ行く前の締めに使う。'
    }
  ],
  hotspots: [
    {
      id: 'echo-note',
      kind: 'echo-note',
      label: 'Echo Note',
      title: 'Echo Note: 残響の部屋',
      position: [0, 1.75, 1.8],
      accentColor: '#f4ca72',
      body:
        'Echo Chamber は雑談の残響、配信後の余韻、切り抜きの反射でできた部屋。\n\n' +
        'Residency より少しだけ抽象度が高く、話題の回収と次章への接続に向いている。'
    },
    {
      id: 'chorus-spot',
      kind: 'photo-spot',
      label: 'Chorus Spot',
      title: 'Photo Spot: Echo Chamber の固定構図',
      position: [-2.8, 2.05, 3.6],
      accentColor: '#85e3e1',
      presetId: 'chorus',
      body:
        '告知画像よりも、配信サムネや切り抜き用の「異常な静けさ」を出したい時の構図。'
    },
    {
      id: 'return-topic',
      kind: 'stream-topic',
      label: 'Return Topic',
      title: 'Echo Note: Residency に持ち帰る話題',
      position: [0, 1.95, 6.2],
      accentColor: '#a7d4ff',
      body:
        'この fragment で拾ったトークの残骸を Residency に持ち帰る。\n\n' +
        '配信の chapter 構成を world の移動として見せるためのノート。'
    }
  ],
  gate: {
    id: 'echo-return-gate',
    kind: 'dream-gate',
    position: [0, 1.95, 8.6],
    presetId: 'return-gate',
    targetFragmentId: 'residency',
    targetFragmentLabel: 'Residency',
    lockedAccentColor: '#f4ca72',
    openAccentColor: '#85e3e1',
    lockedTitle: 'Dream Gate: まだ帰れない',
    openTitle: 'Dream Gate: Residency へ帰還',
    lockedBody:
      'Echo shards を揃えると Residency へ戻る gate が安定する。\n\n' +
      'fragment を行き来して chapter の切り替えを見せる。',
    openBody:
      'Echo Chamber の gate は Residency へ戻る準備ができた。\n\n' +
      '帰還して次の導入、雑談、締めに繋げる。'
  }
};

export const dreamwalkerConfig = {
  appTitle: 'Dynamic Map Viewer',
  subtitle: 'Browser-first 3DGS dynamic-map loading and route playback',
  defaultFragmentId: 'residency',
  assetManifest: {
    defaultUrl: '/manifests/dreamwalker-live.assets.json',
    queryParam: 'assetManifest'
  },
  tileCatalog: {
    defaultUrl: '',
    queryParam: 'tileCatalog',
    tileQueryParam: 'tileId',
    preloadLimitQueryParam: 'tilePreloadLimit',
    residentLimitQueryParam: 'tileResidentLimit'
  },
  largeScaleDemo: {
    label: 'Outdoor Production Grid Large',
    tileCatalogUrl: '/manifests/outdoor-production-grid-large-tile-catalog.json',
    preloadMode: 'metadata',
    tilePreloadLimit: 4,
    tileResidentLimit: 6,
    routeUrl: '/robot-routes/outdoor-production-grid-large-route.json',
    routePlayback: true,
    routePlaybackLoop: true,
    routePlaybackMs: 1200
  },
  studioBundle: {
    defaultUrl: '',
    queryParam: 'studioBundle'
  },
  studioBundleCatalog: {
    defaultUrl: '/studio-bundles/index.json',
    queryParam: 'studioBundleCatalog'
  },
  robotRoute: {
    defaultUrl: '',
    queryParam: 'robotRoute',
    playbackQueryParam: 'robotRoutePlayback',
    playbackIntervalQueryParam: 'robotRoutePlaybackMs',
    playbackLoopQueryParam: 'robotRoutePlaybackLoop'
  },
  robotRouteCatalog: {
    defaultUrl: '/robot-routes/index.json',
    queryParam: 'robotRouteCatalog'
  },
  robotMission: {
    defaultUrl: '',
    queryParam: 'robotMission'
  },
  robotMissionCatalog: {
    defaultUrl: '/robot-missions/index.json',
    queryParam: 'robotMissionCatalog'
  },
  defaultSplatUrl: '',
  demoSplatUrl: 'https://developer.playcanvas.com/assets/toy-cat.sog',
  colliderMeshUrl: '',
  showColliderDebug: false,
  robotics: {
    defaultCameraId: 'front',
    moveStep: 0.8,
    turnStepDegrees: 16,
    waypointDistance: 2.8,
    footprintRadius: 0.55,
    trailPointLimit: 18,
    trailHeight: 0.16,
    zoneAnchorHeight: 0.42,
    routeZoneRadius: 0.85,
    routeZoneCost: 18,
    zoneBoundsPadding: 1,
    gamepadDeadzone: 0.35,
    gamepadRepeatMs: 180,
    gamepadButtonRepeatMs: 240,
    cameras: defaultRoboticsCameras
  },
  interactSettings: {
    maxScreenDistance: 16,
    maxDepth: 6.5
  },
  walkController: {
    cameraLocalHeight: 0.62,
    capsuleRadius: 0.42,
    capsuleHeight: 1.8,
    mass: 90,
    friction: 0.8,
    lookSensitivity: 0.08,
    speedGround: 55,
    speedAir: 7,
    sprintMultiplier: 1.5,
    jumpForce: 600
  },
  photoRatios: [
    { id: 'free', label: 'Free', frameWidth: 1, frameHeight: 1 },
    { id: 'square', label: '1:1', frameWidth: 1, frameHeight: 1 },
    { id: 'portrait', label: '4:5', frameWidth: 4, frameHeight: 5 },
    { id: 'story', label: '9:16', frameWidth: 9, frameHeight: 16 }
  ],
  dreamFilters: [
    {
      id: 'soft-dream',
      label: 'Soft Dream',
      description: '青白い霞を足した静かな夢景色'
    },
    {
      id: 'broken-memory',
      label: 'Broken Memory',
      description: 'コントラスト強めの記憶断片'
    },
    {
      id: 'cold-marble',
      label: 'Cold Marble',
      description: '冷たい石の空気を強調する配信向けトーン'
    }
  ],
  overlayPresets: [
    {
      id: 'lower-third',
      label: 'Lower Third',
      description: '左下に大きめのトークカードを置く標準レイアウト'
    },
    {
      id: 'side-stack',
      label: 'Side Stack',
      description: '右上に細身のカードを置いて avatar 面積を広く残す'
    },
    {
      id: 'headline-ribbon',
      label: 'Headline',
      description: '上部中央に横長 ribbon を出して告知感を強くする'
    }
  ],
  fragments: {
    residency: residencyFragment,
    'echo-chamber': echoChamberFragment
  }
};

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function firstNonEmptyString(...values) {
  for (const value of values) {
    if (hasNonEmptyString(value)) {
      return value.trim();
    }
  }

  return '';
}

export function resolveWorldAssetBundle(worldConfig, assetManifest) {
  const manifestDefaults = assetManifest?.defaults ?? {};
  const manifestFragment = assetManifest?.fragments?.[worldConfig.fragmentId] ?? {};
  const configAsset = worldConfig.assetBundle ?? {};

  const configuredSplatUrl = firstNonEmptyString(
    manifestFragment.splatUrl,
    configAsset.splatUrl,
    worldConfig.defaultSplatUrl,
    manifestDefaults.splatUrl,
    dreamwalkerConfig.defaultSplatUrl
  );
  const demoSplatUrl = firstNonEmptyString(
    manifestFragment.demoSplatUrl,
    configAsset.demoSplatUrl,
    worldConfig.demoSplatUrl,
    manifestDefaults.demoSplatUrl,
    dreamwalkerConfig.demoSplatUrl
  );
  const colliderMeshUrl = firstNonEmptyString(
    manifestFragment.colliderMeshUrl,
    configAsset.colliderMeshUrl,
    worldConfig.colliderMeshUrl,
    manifestDefaults.colliderMeshUrl,
    dreamwalkerConfig.colliderMeshUrl
  );

  return {
    assetLabel: firstNonEmptyString(
      manifestFragment.label,
      configAsset.label,
      `${worldConfig.fragmentLabel} World`
    ),
    worldNote: firstNonEmptyString(
      manifestFragment.worldNote,
      configAsset.worldNote,
      assetManifest?.note
    ),
    manifestLabel: firstNonEmptyString(assetManifest?.label),
    splatUrl: resolvePublicUrl(configuredSplatUrl || demoSplatUrl),
    colliderMeshUrl: resolvePublicUrl(colliderMeshUrl),
    expectedSplatUrl: firstNonEmptyString(
      manifestFragment.expectedSplatUrl,
      configAsset.expectedSplatUrl
    ),
    expectedColliderMeshUrl: firstNonEmptyString(
      manifestFragment.expectedColliderMeshUrl,
      configAsset.expectedColliderMeshUrl
    ),
    splatSource: hasNonEmptyString(manifestFragment.splatUrl)
      ? 'manifest'
      : configuredSplatUrl
        ? 'config'
        : demoSplatUrl
          ? 'demo'
          : 'missing',
    colliderSource: hasNonEmptyString(manifestFragment.colliderMeshUrl)
      ? 'manifest'
      : colliderMeshUrl
        ? 'config'
        : 'proxy',
    usesDemoFallback: !configuredSplatUrl && Boolean(demoSplatUrl),
    hasConfiguredSplat: Boolean(configuredSplatUrl),
    hasColliderMesh: Boolean(colliderMeshUrl)
  };
}

export function resolveDreamwalkerConfig(fragmentId) {
  const fallbackFragment =
    dreamwalkerConfig.fragments[dreamwalkerConfig.defaultFragmentId];
  const fragment =
    dreamwalkerConfig.fragments[fragmentId] ?? fallbackFragment;

  return {
    ...dreamwalkerConfig,
    ...fragment,
    interactSettings: {
      ...dreamwalkerConfig.interactSettings,
      ...fragment.interactSettings
    },
    walkController: {
      ...dreamwalkerConfig.walkController,
      ...fragment.walkController
    },
    robotics: {
      ...dreamwalkerConfig.robotics,
      ...fallbackFragment.robotics,
      ...fragment.robotics,
      cameras:
        fragment.robotics?.cameras ??
        fallbackFragment.robotics?.cameras ??
        dreamwalkerConfig.robotics.cameras
    },
    overlayBranding: {
      ...fallbackFragment.overlayBranding,
      ...fragment.overlayBranding
    },
    assetBundle: {
      ...fallbackFragment.assetBundle,
      ...fragment.assetBundle
    },
    gate: {
      ...fallbackFragment.gate,
      ...fragment.gate
    },
    walkProxyColliders:
      fragment.walkProxyColliders ?? fallbackFragment.walkProxyColliders,
    cameraPresets: fragment.cameraPresets ?? fallbackFragment.cameraPresets,
    photoRatios: fragment.photoRatios ?? dreamwalkerConfig.photoRatios,
    dreamFilters: fragment.dreamFilters ?? dreamwalkerConfig.dreamFilters,
    overlayPresets: fragment.overlayPresets ?? dreamwalkerConfig.overlayPresets,
    streamScenes: fragment.streamScenes ?? fallbackFragment.streamScenes ?? [],
    shards: fragment.shards ?? fallbackFragment.shards,
    hotspots: fragment.hotspots ?? fallbackFragment.hotspots,
    fragmentId: fragment.fragmentId ?? dreamwalkerConfig.defaultFragmentId,
    fragmentLabel: fragment.fragmentLabel ?? fallbackFragment.fragmentLabel
  };
}
