import { dreamwalkerConfig } from './app-config.js';

function normalizeStringList(items) {
  return Array.isArray(items)
    ? items.filter((item) => typeof item === 'string' && item.trim())
    : [];
}

export function resolveOverlayPresetFromPayload(overlayState) {
  return (
    dreamwalkerConfig.overlayPresets.find(
      (preset) => preset.id === overlayState?.overlayPresetId
    ) ?? dreamwalkerConfig.overlayPresets[0]
  );
}

export function normalizeOverlayMemoItems(items) {
  return normalizeStringList(items);
}

export function resolveOverlayBrandingFromPayload(overlayState) {
  const fragmentConfig = dreamwalkerConfig.fragments[overlayState?.fragmentId];
  const fallbackBranding =
    dreamwalkerConfig.fragments[dreamwalkerConfig.defaultFragmentId]?.overlayBranding;

  return {
    ...fallbackBranding,
    ...fragmentConfig?.overlayBranding,
    id: overlayState?.overlayBrandingId ?? fragmentConfig?.overlayBranding?.id ?? fallbackBranding?.id,
    label:
      overlayState?.overlayBrandingLabel ??
      fragmentConfig?.overlayBranding?.label ??
      fallbackBranding?.label,
    badge:
      overlayState?.overlayBrandingBadge ??
      fragmentConfig?.overlayBranding?.badge ??
      fallbackBranding?.badge,
    strapline:
      overlayState?.overlayBrandingStrapline ??
      fragmentConfig?.overlayBranding?.strapline ??
      fallbackBranding?.strapline,
    accent:
      overlayState?.overlayBrandingAccent ??
      fragmentConfig?.overlayBranding?.accent ??
      fallbackBranding?.accent,
    highlight:
      overlayState?.overlayBrandingHighlight ??
      fragmentConfig?.overlayBranding?.highlight ??
      fallbackBranding?.highlight,
    glow:
      overlayState?.overlayBrandingGlow ??
      fragmentConfig?.overlayBranding?.glow ??
      fallbackBranding?.glow
  };
}

export function resolveOverlayBrandingForScene(baseBranding, streamScene) {
  return {
    ...baseBranding,
    ...(streamScene?.overlayBrandingOverrides ?? {})
  };
}

export function resolveOverlayMemoFromPayload(overlayState) {
  const fragmentConfig = dreamwalkerConfig.fragments[overlayState?.fragmentId];
  const streamScene = fragmentConfig?.streamScenes?.find(
    (candidate) => candidate.id === overlayState?.streamSceneId
  );
  const fallbackMemo = streamScene?.overlayMemo;
  const overlayMemoItems = normalizeOverlayMemoItems(overlayState?.overlayMemoItems);
  const fallbackMemoItems = normalizeOverlayMemoItems(fallbackMemo?.items);
  const resolvedMemo = {
    title: overlayState?.overlayMemoTitle ?? fallbackMemo?.title ?? null,
    items: overlayMemoItems.length > 0 ? overlayMemoItems : fallbackMemoItems,
    footer: overlayState?.overlayMemoFooter ?? fallbackMemo?.footer ?? null
  };

  if (
    !resolvedMemo.title &&
    !resolvedMemo.footer &&
    resolvedMemo.items.length === 0
  ) {
    return null;
  }

  return resolvedMemo;
}

function OverlaySceneCard({ overlayState, overlayPreset, overlayBranding, preview = false }) {
  const activeState = overlayState ?? null;
  const resolvedPreset =
    overlayPreset ?? resolveOverlayPresetFromPayload(overlayState);
  const resolvedBranding =
    overlayBranding ?? resolveOverlayBrandingFromPayload(overlayState);
  const cardClassName = `obs-overlay-card glass-panel overlay-card-${resolvedPreset.id}${preview ? ' overlay-card-preview live-scene-card' : ''}`;
  const cardStyle = {
    '--overlay-accent': resolvedBranding?.accent ?? '#f4ca72',
    '--overlay-highlight': resolvedBranding?.highlight ?? '#85e3e1',
    '--overlay-glow': resolvedBranding?.glow ?? 'rgba(244, 202, 114, 0.22)'
  };

  if (!activeState) {
    return (
      <section
        className={cardClassName}
        data-overlay-branding={resolvedBranding?.id ?? 'default-branding'}
        data-overlay-preset={resolvedPreset.id}
        style={cardStyle}>
        <p className="obs-overlay-kicker">Dynamic Map Viewer</p>
        <h1>No Live Scene Published</h1>
        <p className="obs-overlay-topic">
          通常画面で Live Mode を開くと、scene 情報がここへ同期されます。
        </p>
      </section>
    );
  }

  return (
    <section
      className={cardClassName}
      data-overlay-branding={resolvedBranding?.id ?? 'default-branding'}
      data-overlay-preset={resolvedPreset.id}
      style={cardStyle}>
      <div className="obs-overlay-accent-line" />
      <div className="obs-overlay-live-row">
        <span className="obs-live-pill">{resolvedBranding?.badge ?? 'LIVE'}</span>
        <span className="obs-overlay-fragment">{activeState.fragmentLabel ?? 'Dynamic Map Viewer'}</span>
        <span className="obs-overlay-scene-chip">
          {activeState.streamSceneLabel ?? resolvedPreset.label}
        </span>
      </div>
      <p className="obs-overlay-kicker">{activeState.fragmentLabel ?? 'Dynamic Map Viewer'}</p>
      <h1>{activeState.streamSceneTitle ?? activeState.appTitle ?? 'Live Scene'}</h1>
      <p className="obs-overlay-topic">
        {activeState.streamSceneTopic ?? '配信用 scene 情報がここに表示されます。'}
      </p>
      {resolvedBranding?.strapline ? (
        <p className="obs-overlay-strapline">{resolvedBranding.strapline}</p>
      ) : null}
      <div className="obs-overlay-meta">
        <span>{activeState.cameraPresetLabel ?? 'Camera'}</span>
        <span>{activeState.dreamFilterLabel ?? 'Filter'}</span>
        <span>{activeState.overlayPresetLabel ?? resolvedPreset.label}</span>
        <span>{activeState.gateStatus ?? 'gate'}</span>
      </div>
    </section>
  );
}

function OverlayMemoPanel({
  overlayMemo,
  overlayPreset,
  overlayBranding,
  preview = false
}) {
  if (!overlayMemo) {
    return null;
  }

  const resolvedBranding = overlayBranding ?? resolveOverlayBrandingFromPayload(null);
  const resolvedPreset =
    overlayPreset ?? dreamwalkerConfig.overlayPresets[0];
  const panelClassName = `obs-overlay-memo glass-panel overlay-memo-${resolvedPreset.id}${preview ? ' overlay-memo-preview' : ''}`;
  const panelStyle = {
    '--overlay-accent': resolvedBranding?.accent ?? '#f4ca72',
    '--overlay-highlight': resolvedBranding?.highlight ?? '#85e3e1',
    '--overlay-glow': resolvedBranding?.glow ?? 'rgba(244, 202, 114, 0.22)'
  };

  return (
    <aside
      className={panelClassName}
      data-overlay-branding={resolvedBranding?.id ?? 'default-branding'}
      data-overlay-preset={resolvedPreset.id}
      style={panelStyle}>
      <p className="obs-overlay-memo-kicker">Scene Memo</p>
      {overlayMemo.title ? <h2>{overlayMemo.title}</h2> : null}
      {overlayMemo.items.length > 0 ? (
        <ul className="obs-overlay-memo-list">
          {overlayMemo.items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      {overlayMemo.footer ? (
        <p className="obs-overlay-memo-footer">{overlayMemo.footer}</p>
      ) : null}
    </aside>
  );
}

export function OverlayStage({
  overlayState,
  overlayPreset,
  overlayBranding,
  preview = false
}) {
  const overlayMemo = resolveOverlayMemoFromPayload(overlayState);

  return (
    <div
      className={`overlay-scene-stack overlay-stack-${overlayPreset.id}${overlayMemo ? ' has-overlay-memo' : ''}`}
      data-has-overlay-memo={overlayMemo ? 'true' : 'false'}>
      <OverlaySceneCard
        overlayState={overlayState}
        overlayPreset={overlayPreset}
        overlayBranding={overlayBranding}
        preview={preview}
      />
      <OverlayMemoPanel
        overlayMemo={overlayMemo}
        overlayPreset={overlayPreset}
        overlayBranding={overlayBranding}
        preview={preview}
      />
    </div>
  );
}

export function ObsOverlayView({ overlayState }) {
  const activeState = overlayState ?? null;
  const overlayPreset = resolveOverlayPresetFromPayload(activeState);
  const overlayBranding = resolveOverlayBrandingFromPayload(activeState);

  return (
    <div
      className={`obs-overlay-shell overlay-layout-${overlayPreset.id}`}
      data-overlay-branding={overlayBranding?.id ?? 'default-branding'}
      data-overlay-preset={overlayPreset.id}>
      <OverlayStage
        overlayState={activeState}
        overlayPreset={overlayPreset}
        overlayBranding={overlayBranding}
      />
    </div>
  );
}
