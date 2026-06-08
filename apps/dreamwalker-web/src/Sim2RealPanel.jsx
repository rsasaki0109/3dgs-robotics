import { useEffect, useMemo, useRef, useState } from 'react'
import {
  buildSim2realImageBenchmarkRequest,
  buildSim2realPreviewFromMessage,
  buildSim2realRenderRequest,
  parseSim2realMessage,
  sim2realDefaultUrl
} from './sim2real-query.js'
import {
  applyLocalizationEstimateStreamMessage,
  buildLocalizationBenchmarkReport,
  localizationMonitorDefaultUrl,
  normalizeLocalizationEstimate,
  normalizeLocalizationMonitorUrl,
  parseLocalizationEstimateDocument
} from './sim2real-localization.js'
import { importLocalizationReviewBundleDocument } from './sim2real-review-bundle-import.js'

const sim2realCaptureShelfStorageKey = 'dreamwalker-live-sim2real-capture-shelf'
const sim2realLocalizationRunShelfStorageKey = 'dreamwalker-live-sim2real-localization-run-shelf'

function formatConnectionStatus(status) {
  if (status === 'connected') {
    return 'Connected'
  }

  if (status === 'connecting') {
    return 'Connecting'
  }

  if (status === 'error') {
    return 'Error'
  }

  if (status === 'closed') {
    return 'Closed'
  }

  return 'Disabled'
}

function normalizeYawDegrees(value) {
  const normalized = value % 360
  return normalized < 0 ? normalized + 360 : normalized
}

function isPosition(value) {
  return (
    Array.isArray(value) &&
    value.length >= 3 &&
    value.every((item) => Number.isFinite(item))
  )
}

function inferYawDegreesFromRoute(route, index, fallbackYawDegrees) {
  const current = route[index]
  const next = route[index + 1] ?? null
  const previous = route[index - 1] ?? null
  let dx = 0
  let dz = 0

  if (isPosition(next)) {
    dx = next[0] - current[0]
    dz = next[2] - current[2]
  } else if (isPosition(previous)) {
    dx = current[0] - previous[0]
    dz = current[2] - previous[2]
  }

  if (Math.hypot(dx, dz) < 0.0001) {
    return normalizeYawDegrees(fallbackYawDegrees)
  }

  return normalizeYawDegrees((Math.atan2(-dx, -dz) * 180) / Math.PI)
}

function sleep(durationMs) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs)
  })
}

function downloadTextFile(fileName, content, type = 'application/json') {
  const blob = new Blob([content], { type: `${type};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

function buildCaptureFileName(fragmentId, capturedAt) {
  const safeFragmentId =
    typeof fragmentId === 'string' && fragmentId.trim() ? fragmentId.trim() : 'dreamwalker'
  const safeTimestamp = String(capturedAt || new Date().toISOString()).replace(/[:.]/g, '-')
  return `${safeFragmentId}-sim2real-route-capture-${safeTimestamp}.json`
}

function buildLocalizationBenchmarkFileName(fragmentId, createdAt) {
  const safeFragmentId =
    typeof fragmentId === 'string' && fragmentId.trim() ? fragmentId.trim() : 'dreamwalker'
  const safeTimestamp = String(createdAt || new Date().toISOString()).replace(/[:.]/g, '-')
  return `${safeFragmentId}-localization-benchmark-${safeTimestamp}.json`
}

function buildLocalizationRunFileName(fragmentId, savedAt) {
  const safeFragmentId =
    typeof fragmentId === 'string' && fragmentId.trim() ? fragmentId.trim() : 'dreamwalker'
  const safeTimestamp = String(savedAt || new Date().toISOString()).replace(/[:.]/g, '-')
  return `${safeFragmentId}-localization-run-${safeTimestamp}.json`
}

function buildLocalizationRunCompareFileName(fragmentId, createdAt, extension = 'json') {
  const safeFragmentId =
    typeof fragmentId === 'string' && fragmentId.trim() ? fragmentId.trim() : 'dreamwalker'
  const safeTimestamp = String(createdAt || new Date().toISOString()).replace(/[:.]/g, '-')
  return `${safeFragmentId}-localization-run-compare-${safeTimestamp}.${extension}`
}

function buildLocalizationReviewBundleFileName(fragmentId, createdAt) {
  const safeFragmentId =
    typeof fragmentId === 'string' && fragmentId.trim() ? fragmentId.trim() : 'dreamwalker'
  const safeTimestamp = String(createdAt || new Date().toISOString()).replace(/[:.]/g, '-')
  return `${safeFragmentId}-localization-review-bundle-${safeTimestamp}.json`
}

function formatMeters(value) {
  return Number.isFinite(value) ? `${value.toFixed(3)} m` : 'n/a'
}

function formatDegrees(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} deg` : 'n/a'
}

function formatDb(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} dB` : 'n/a'
}

function formatLpips(value) {
  return Number.isFinite(value) ? value.toFixed(3) : 'n/a'
}

function formatSignedMetric(value, formatter, zeroText = null) {
  if (!Number.isFinite(value)) {
    return 'n/a'
  }

  const normalized = Number(value)
  if (zeroText && Math.abs(normalized) <= 0.0005) {
    return zeroText
  }

  return normalized > 0 ? `+${formatter(normalized)}` : formatter(normalized)
}

function classifyMetricDelta(value, tolerance = 0.0005) {
  if (!Number.isFinite(value)) {
    return 'unknown'
  }

  if (Math.abs(Number(value)) <= tolerance) {
    return 'neutral'
  }

  return Number(value) < 0 ? 'better' : 'worse'
}

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(3)} s` : 'n/a'
}

function formatLiveEstimateStatus(status) {
  if (status === 'connected') {
    return 'Connected'
  }

  if (status === 'connecting') {
    return 'Connecting'
  }

  if (status === 'error') {
    return 'Error'
  }

  if (status === 'closed') {
    return 'Closed'
  }

  return 'Disabled'
}

function formatImageBenchmarkBatchMode(mode) {
  if (mode === 'all') {
    return 'refresh-all'
  }

  if (mode === 'missing') {
    return 'missing-only'
  }

  return 'idle'
}

function csvEscapeCell(value) {
  const text = value === null || value === undefined ? '' : String(value)
  if (!/[",\n]/.test(text)) {
    return text
  }

  return `"${text.replaceAll('"', '""')}"`
}

function buildLocalizationRunCompareCsv(rows) {
  const header = [
    'rank',
    'label',
    'sourceType',
    'matchedCount',
    'ateRmseMeters',
    'ateDeltaMeters',
    'yawRmseDegrees',
    'yawDeltaDegrees',
    'psnrMean',
    'ssimMean',
    'lpipsMean',
    'lpipsDelta',
    'worstLpipsValue',
    'worstLpipsFrame',
    'worstLpipsGroundTruthLabel',
    'worstLpipsEstimateLabel',
    'isBaseline',
    'isBestAte',
    'isBestYaw',
    'isBestLpips',
    'isLatest',
    'isActive'
  ]
  const lines = [
    header.join(','),
    ...rows.map((row) =>
      [
        row.rank,
        row.label,
        row.summary.sourceType,
        row.summary.matchedCount,
        row.summary.ateRmseMeters,
        row.ateDeltaMeters,
        row.summary.yawRmseDegrees,
        row.yawDeltaDegrees,
        row.psnrMean,
        row.ssimMean,
        row.lpipsMean,
        row.lpipsDelta,
        row.worstLpipsValue,
        row.worstLpipsFrameIndex !== null ? row.worstLpipsFrameIndex + 1 : null,
        row.worstLpipsGroundTruthLabel,
        row.worstLpipsEstimateLabel,
        row.isBaseline,
        row.isBestAte,
        row.isBestYaw,
        row.isBestLpips,
        row.isLatest,
        row.isActive
      ]
        .map(csvEscapeCell)
        .join(',')
    )
  ]

  return `${lines.join('\n')}\n`
}

function markdownEscapeCell(value) {
  return String(value === null || value === undefined ? '' : value).replaceAll('|', '\\|')
}

function buildLocalizationRunCompareMarkdown(comparison, { fragmentId, fragmentLabel, createdAt } = {}) {
  const titleLabel =
    readNonEmptyString(fragmentLabel) || readNonEmptyString(fragmentId) || 'Dynamic Map Viewer'
  const lines = [
    `# Localization Run Compare`,
    '',
    `- Fragment: ${titleLabel}`,
    `- Created At: ${createdAt || new Date().toISOString()}`,
    `- Baseline: ${comparison.baselineRun?.label || 'n/a'}`,
    `- Runs: ${comparison.rows.length}`,
    `- Image Metrics: ${comparison.runsWithImageBenchmarkCount} attached / ${comparison.missingImageBenchmarkCount} missing`,
    ''
  ]

  if (comparison.bestAteRun || comparison.bestYawRun || comparison.bestLpipsRun) {
    lines.push('## Highlights', '')
    lines.push(`- Best ATE: ${comparison.bestAteRun?.label || 'n/a'} (${formatMeters(comparison.bestAteRun?.summary?.ateRmseMeters)})`)
    lines.push(`- Best Yaw: ${comparison.bestYawRun?.label || 'n/a'} (${formatDegrees(comparison.bestYawRun?.summary?.yawRmseDegrees)})`)
    lines.push(`- Best LPIPS: ${comparison.bestLpipsRun?.label || 'n/a'} (${formatLpips(comparison.bestLpipsRun?.imageBenchmark?.summary?.lpipsMean)})`)
    lines.push(`- Latest: ${comparison.latestRun?.label || 'n/a'}`)
    lines.push('')
  }

  lines.push('## Table', '')
  lines.push(
    '| Rank | Run | Source | Matched | ATE | ΔATE | Yaw | ΔYaw | PSNR | SSIM | LPIPS | ΔLPIPS | Worst LPIPS |'
  )
  lines.push(
    '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |'
  )

  comparison.rows.forEach((row) => {
    const worstLpipsLabel =
      Number.isFinite(row.worstLpipsValue) && row.worstLpipsFrameIndex !== null
        ? `${formatLpips(row.worstLpipsValue)} @ frame ${row.worstLpipsFrameIndex + 1}`
        : 'n/a'
    const runLabel = [
      row.label,
      row.isBaseline ? 'baseline' : '',
      row.isBestAte ? 'best-ate' : '',
      row.isBestYaw ? 'best-yaw' : '',
      row.isBestLpips ? 'best-lpips' : '',
      row.isLatest ? 'latest' : '',
      row.isActive ? 'active' : ''
    ]
      .filter(Boolean)
      .join(' / ')

    lines.push(
      `| ${row.rank} | ${markdownEscapeCell(runLabel)} | ${markdownEscapeCell(row.summary.sourceType)} | ${row.summary.matchedCount} | ${markdownEscapeCell(formatMeters(row.summary.ateRmseMeters))} | ${markdownEscapeCell(formatSignedMetric(row.ateDeltaMeters, formatMeters, '0.000 m'))} | ${markdownEscapeCell(formatDegrees(row.summary.yawRmseDegrees))} | ${markdownEscapeCell(formatSignedMetric(row.yawDeltaDegrees, formatDegrees, '0.00 deg'))} | ${markdownEscapeCell(formatDb(row.psnrMean))} | ${markdownEscapeCell(formatLpips(row.ssimMean))} | ${markdownEscapeCell(formatLpips(row.lpipsMean))} | ${markdownEscapeCell(formatSignedMetric(row.lpipsDelta, formatLpips, '0.000'))} | ${markdownEscapeCell(worstLpipsLabel)} |`
    )
  })

  lines.push('')
  return `${lines.join('\n')}\n`
}

function buildWorstLpipsPreviewArtifact(entry) {
  const highlight = entry?.imageBenchmark?.highlights?.lpips ?? null

  if (
    !highlight ||
    !readNonEmptyString(highlight.groundTruthColorJpegBase64) ||
    !readNonEmptyString(highlight.renderedColorJpegBase64)
  ) {
    return null
  }

  return {
    entryId: readNonEmptyString(entry.id),
    runLabel: readNonEmptyString(entry.label),
    metricName: 'lpips',
    value: normalizeOptionalMetricNumber(highlight.value),
    frameIndex: Number.isFinite(Number(highlight.frameIndex)) ? Number(highlight.frameIndex) : null,
    groundTruthLabel: readNonEmptyString(highlight.groundTruthLabel),
    estimateLabel: readNonEmptyString(highlight.estimateLabel),
    groundTruthColorJpegBase64: readNonEmptyString(highlight.groundTruthColorJpegBase64),
    renderedColorJpegBase64: readNonEmptyString(highlight.renderedColorJpegBase64)
  }
}

function buildRouteCaptureBundle({
  fragmentId,
  fragmentLabel,
  endpoint,
  serverInfo,
  requestSettings,
  routePreviewPoses,
  captures
}) {
  return {
    protocol: 'dreamwalker-sim2real-capture/v1',
    type: 'route-capture-bundle',
    capturedAt: new Date().toISOString(),
    fragmentId: fragmentId || '',
    fragmentLabel: fragmentLabel || '',
    endpoint,
    server: {
      frameId: serverInfo?.frameId || '',
      renderer: serverInfo?.renderer || '',
      rendererReason: serverInfo?.rendererReason || '',
      defaults: serverInfo?.defaults ?? null
    },
    request: {
      width: Number(requestSettings.width),
      height: Number(requestSettings.height),
      fovDegrees: Number(requestSettings.fovDegrees),
      nearClip: Number(requestSettings.nearClip),
      farClip: Number(requestSettings.farClip),
      pointRadius: Number(requestSettings.pointRadius),
      routeDelayMs: Number(requestSettings.routeDelayMs)
    },
    route: routePreviewPoses.map((pose, index) => ({
      index,
      position: [...pose.position],
      yawDegrees: Number(pose.yawDegrees),
      relativeTimeSeconds: normalizeOptionalTimestampSeconds(captures[index]?.relativeTimeSeconds)
    })),
    captures
  }
}

function readNonEmptyString(value) {
  return typeof value === 'string' ? value.trim() : ''
}

function normalizeCapturePose(poseLike) {
  const pose = poseLike && typeof poseLike === 'object' ? poseLike : {}
  const position = Array.isArray(pose.position) ? pose.position.slice(0, 3).map(Number) : null
  const yawDegrees = Number(pose.yawDegrees)

  if (!position || position.length !== 3 || position.some((value) => !Number.isFinite(value))) {
    throw new Error('capture pose.position must be three finite numbers')
  }

  if (!Number.isFinite(yawDegrees)) {
    throw new Error('capture pose.yawDegrees must be finite')
  }

  return {
    position,
    yawDegrees
  }
}

function normalizeOptionalTimestampSeconds(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function normalizeOptionalMetricNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function metricSortValue(value) {
  return Number.isFinite(value) ? Number(value) : Number.POSITIVE_INFINITY
}

function timestampSortValue(value) {
  const parsed = Date.parse(readNonEmptyString(value))
  return Number.isFinite(parsed) ? parsed : 0
}

function metricEquals(left, right, tolerance = 0.0005) {
  if (!Number.isFinite(left) || !Number.isFinite(right)) {
    return false
  }

  return Math.abs(Number(left) - Number(right)) <= tolerance
}

function buildLocalizationRunSummary(reportLike) {
  const report = reportLike && typeof reportLike === 'object' ? reportLike : {}

  return {
    createdAt: readNonEmptyString(report.createdAt) || new Date().toISOString(),
    alignment: readNonEmptyString(report.alignment) || 'auto',
    requestedAlignment:
      readNonEmptyString(report.requestedAlignment) ||
      readNonEmptyString(report.alignment) ||
      'auto',
    groundTruthLabel: readNonEmptyString(report.groundTruth?.label) || 'Ground Truth Capture',
    estimateLabel: readNonEmptyString(report.estimate?.label) || 'Localization Estimate',
    sourceType: readNonEmptyString(report.estimate?.sourceType) || 'poses',
    interpolationMode: readNonEmptyString(report.estimate?.interpolationMode) || 'none',
    matchedCount: Number.isFinite(Number(report.matching?.matchedCount))
      ? Number(report.matching.matchedCount)
      : 0,
    groundTruthPoseCount: Number.isFinite(Number(report.groundTruth?.poseCount))
      ? Number(report.groundTruth.poseCount)
      : 0,
    estimatePoseCount: Number.isFinite(Number(report.estimate?.poseCount))
      ? Number(report.estimate.poseCount)
      : 0,
    ateRmseMeters: normalizeOptionalMetricNumber(report.metrics?.ateRmseMeters),
    yawRmseDegrees: normalizeOptionalMetricNumber(report.metrics?.yaw?.rmse),
    rpeTranslationRmseMeters: normalizeOptionalMetricNumber(
      report.metrics?.rpeTranslationRmseMeters
    ),
    rpeYawRmseDegrees: normalizeOptionalMetricNumber(report.metrics?.rpeYawRmseDegrees),
    timeDeltaMeanSeconds: normalizeOptionalMetricNumber(report.metrics?.timeDelta?.mean),
    timeDeltaMaxSeconds: normalizeOptionalMetricNumber(report.metrics?.timeDelta?.max),
    timeAligned: Boolean(report.matching?.timeAligned),
    interpolatedCount: Number.isFinite(Number(report.matching?.interpolatedCount))
      ? Number(report.matching.interpolatedCount)
      : 0,
    clampedCount: Number.isFinite(Number(report.matching?.clampedCount))
      ? Number(report.matching.clampedCount)
      : 0
  }
}

function buildJpegDataUrl(base64Payload) {
  const normalized = readNonEmptyString(base64Payload)
  return normalized ? `data:image/jpeg;base64,${normalized}` : ''
}

function normalizeLocalizationImageBenchmarkHighlight(highlightLike) {
  const highlight = highlightLike && typeof highlightLike === 'object' ? highlightLike : {}
  const value = normalizeOptionalMetricNumber(highlight.value)
  const frameIndex = Number.isFinite(Number(highlight.frameIndex))
    ? Number(highlight.frameIndex)
    : null

  if (!Number.isFinite(value) || frameIndex === null) {
    return null
  }

  return {
    ordering: readNonEmptyString(highlight.ordering) || 'max',
    frameIndex,
    value,
    groundTruthLabel: readNonEmptyString(highlight.groundTruthLabel),
    estimateLabel: readNonEmptyString(highlight.estimateLabel),
    interpolationKind: readNonEmptyString(highlight.interpolationKind),
    timeDeltaSeconds: normalizeOptionalMetricNumber(highlight.timeDeltaSeconds),
    groundTruthColorJpegBase64: readNonEmptyString(highlight.groundTruthColorJpegBase64),
    renderedColorJpegBase64: readNonEmptyString(highlight.renderedColorJpegBase64)
  }
}

function normalizeLocalizationImageBenchmarkReport(reportLike) {
  const report = reportLike && typeof reportLike === 'object' ? reportLike : {}

  if (report.type !== 'localization-image-benchmark-report') {
    throw new Error('image benchmark report type must be localization-image-benchmark-report')
  }

  const summary =
    report.metrics?.summary && typeof report.metrics.summary === 'object'
      ? report.metrics.summary
      : report.summary && typeof report.summary === 'object'
        ? {
            psnr: { mean: report.summary.psnrMean },
            ssim: { mean: report.summary.ssimMean },
            lpips: { mean: report.summary.lpipsMean }
          }
        : {}
  const highlights =
    report.metrics?.highlights && typeof report.metrics.highlights === 'object'
      ? report.metrics.highlights
      : report.highlights && typeof report.highlights === 'object'
        ? report.highlights
        : {}

  return {
    protocol: readNonEmptyString(report.protocol) || 'dreamwalker-localization-image-benchmark/v1',
    type: 'localization-image-benchmark-report',
    createdAt: readNonEmptyString(report.createdAt) || new Date().toISOString(),
    endpoint: readNonEmptyString(report.endpoint),
    alignment: readNonEmptyString(report.alignment) || 'auto',
    estimateLabel:
      readNonEmptyString(report.estimate?.label) ||
      readNonEmptyString(report.estimateLabel) ||
      'Localization Estimate',
    groundTruthLabel:
      readNonEmptyString(report.groundTruth?.label) ||
      readNonEmptyString(report.groundTruth?.fragmentLabel) ||
      readNonEmptyString(report.groundTruth?.fragmentId) ||
      'Ground Truth Capture',
    matchedCount: Number.isFinite(Number(report.matching?.matchedCount))
      ? Number(report.matching.matchedCount)
      : 0,
    summary: {
      psnrMean: normalizeOptionalMetricNumber(summary.psnr?.mean),
      ssimMean: normalizeOptionalMetricNumber(summary.ssim?.mean),
      lpipsMean: normalizeOptionalMetricNumber(summary.lpips?.mean)
    },
    highlights: {
      psnr: normalizeLocalizationImageBenchmarkHighlight(highlights.psnr),
      ssim: normalizeLocalizationImageBenchmarkHighlight(highlights.ssim),
      lpips: normalizeLocalizationImageBenchmarkHighlight(highlights.lpips)
    }
  }
}

function normalizeLocalizationRunSummary(summaryLike, fallbackSummary = {}) {
  const summary = summaryLike && typeof summaryLike === 'object' ? summaryLike : {}

  return {
    createdAt:
      readNonEmptyString(summary.createdAt) ||
      readNonEmptyString(fallbackSummary.createdAt) ||
      new Date().toISOString(),
    alignment:
      readNonEmptyString(summary.alignment) ||
      readNonEmptyString(fallbackSummary.alignment) ||
      'auto',
    requestedAlignment:
      readNonEmptyString(summary.requestedAlignment) ||
      readNonEmptyString(fallbackSummary.requestedAlignment) ||
      readNonEmptyString(summary.alignment) ||
      readNonEmptyString(fallbackSummary.alignment) ||
      'auto',
    groundTruthLabel:
      readNonEmptyString(summary.groundTruthLabel) ||
      readNonEmptyString(fallbackSummary.groundTruthLabel) ||
      'Ground Truth Capture',
    estimateLabel:
      readNonEmptyString(summary.estimateLabel) ||
      readNonEmptyString(fallbackSummary.estimateLabel) ||
      'Localization Estimate',
    sourceType:
      readNonEmptyString(summary.sourceType) ||
      readNonEmptyString(fallbackSummary.sourceType) ||
      'poses',
    interpolationMode:
      readNonEmptyString(summary.interpolationMode) ||
      readNonEmptyString(fallbackSummary.interpolationMode) ||
      'none',
    matchedCount: Number.isFinite(Number(summary.matchedCount))
      ? Number(summary.matchedCount)
      : Number.isFinite(Number(fallbackSummary.matchedCount))
        ? Number(fallbackSummary.matchedCount)
        : 0,
    groundTruthPoseCount: Number.isFinite(Number(summary.groundTruthPoseCount))
      ? Number(summary.groundTruthPoseCount)
      : Number.isFinite(Number(fallbackSummary.groundTruthPoseCount))
        ? Number(fallbackSummary.groundTruthPoseCount)
        : 0,
    estimatePoseCount: Number.isFinite(Number(summary.estimatePoseCount))
      ? Number(summary.estimatePoseCount)
      : Number.isFinite(Number(fallbackSummary.estimatePoseCount))
        ? Number(fallbackSummary.estimatePoseCount)
        : 0,
    ateRmseMeters: normalizeOptionalMetricNumber(
      summary.ateRmseMeters ?? fallbackSummary.ateRmseMeters
    ),
    yawRmseDegrees: normalizeOptionalMetricNumber(
      summary.yawRmseDegrees ?? fallbackSummary.yawRmseDegrees
    ),
    rpeTranslationRmseMeters: normalizeOptionalMetricNumber(
      summary.rpeTranslationRmseMeters ?? fallbackSummary.rpeTranslationRmseMeters
    ),
    rpeYawRmseDegrees: normalizeOptionalMetricNumber(
      summary.rpeYawRmseDegrees ?? fallbackSummary.rpeYawRmseDegrees
    ),
    timeDeltaMeanSeconds: normalizeOptionalMetricNumber(
      summary.timeDeltaMeanSeconds ?? fallbackSummary.timeDeltaMeanSeconds
    ),
    timeDeltaMaxSeconds: normalizeOptionalMetricNumber(
      summary.timeDeltaMaxSeconds ?? fallbackSummary.timeDeltaMaxSeconds
    ),
    timeAligned:
      typeof summary.timeAligned === 'boolean'
        ? summary.timeAligned
        : Boolean(fallbackSummary.timeAligned),
    interpolatedCount: Number.isFinite(Number(summary.interpolatedCount))
      ? Number(summary.interpolatedCount)
      : Number.isFinite(Number(fallbackSummary.interpolatedCount))
        ? Number(fallbackSummary.interpolatedCount)
        : 0,
    clampedCount: Number.isFinite(Number(summary.clampedCount))
      ? Number(summary.clampedCount)
      : Number.isFinite(Number(fallbackSummary.clampedCount))
        ? Number(fallbackSummary.clampedCount)
        : 0
  }
}

function buildLocalizationRunSnapshot({
  label,
  estimate,
  groundTruthSource,
  report
}) {
  const normalizedEstimate = normalizeLocalizationEstimate(estimate)
  const summary = buildLocalizationRunSummary(report)

  return {
    protocol: 'dreamwalker-localization-run/v1',
    type: 'localization-run-snapshot',
    label:
      readNonEmptyString(label) ||
      `${summary.estimateLabel} / ATE ${formatMeters(summary.ateRmseMeters)}`,
    savedAt: new Date().toISOString(),
    groundTruth: {
      sourceId: readNonEmptyString(groundTruthSource?.id) || 'current-capture',
      label:
        readNonEmptyString(groundTruthSource?.label) ||
        readNonEmptyString(summary.groundTruthLabel) ||
        'Ground Truth Capture',
      bundle:
        readNonEmptyString(groundTruthSource?.id) === 'current-capture'
          ? groundTruthSource?.bundle ?? null
          : null
    },
    estimate: normalizedEstimate,
    benchmark: {
      alignment: readNonEmptyString(report?.alignment) || 'auto',
      requestedAlignment:
        readNonEmptyString(report?.requestedAlignment) ||
        readNonEmptyString(report?.alignment) ||
        'auto',
      reportCreatedAt: readNonEmptyString(report?.createdAt)
    },
    summary,
    report: report && typeof report === 'object' ? report : null
  }
}

function normalizeCaptureResponse(responseLike) {
  const response = responseLike && typeof responseLike === 'object' ? responseLike : {}

  if (response.type !== 'render-result') {
    throw new Error('capture response must be render-result')
  }

  if (!Number.isFinite(Number(response.width)) || !Number.isFinite(Number(response.height))) {
    throw new Error('capture response width/height are required')
  }

  if (!readNonEmptyString(response.colorJpegBase64) || !readNonEmptyString(response.depthBase64)) {
    throw new Error('capture response image payloads are required')
  }

      return {
        ...response,
        width: Number(response.width),
        height: Number(response.height)
      }
}

function normalizeRouteCaptureBundle(bundleLike) {
  const bundle = bundleLike && typeof bundleLike === 'object' ? bundleLike : {}

  if (bundle.type !== 'route-capture-bundle') {
    throw new Error('capture bundle type must be route-capture-bundle')
  }

  const captures = Array.isArray(bundle.captures)
    ? bundle.captures.map((entry, index) => {
        const capture = entry && typeof entry === 'object' ? entry : {}

        return {
          index: Number.isFinite(Number(capture.index)) ? Number(capture.index) : index,
          label: readNonEmptyString(capture.label) || `capture:${index + 1}`,
          capturedAt: readNonEmptyString(capture.capturedAt),
          relativeTimeSeconds: normalizeOptionalTimestampSeconds(capture.relativeTimeSeconds),
          pose: normalizeCapturePose(capture.pose),
          response: normalizeCaptureResponse(capture.response)
        }
      })
    : []

  if (captures.length === 0) {
    throw new Error('capture bundle must include at least one capture')
  }

  return {
    protocol: readNonEmptyString(bundle.protocol) || 'dreamwalker-sim2real-capture/v1',
    type: 'route-capture-bundle',
    capturedAt: readNonEmptyString(bundle.capturedAt) || new Date().toISOString(),
    fragmentId: readNonEmptyString(bundle.fragmentId),
    fragmentLabel: readNonEmptyString(bundle.fragmentLabel),
    endpoint: readNonEmptyString(bundle.endpoint),
    server: bundle.server && typeof bundle.server === 'object' ? bundle.server : {},
    request: bundle.request && typeof bundle.request === 'object' ? bundle.request : {},
    route: Array.isArray(bundle.route) ? bundle.route : [],
    captures
  }
}

function normalizeCaptureShelfEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {}
  const bundle = normalizeRouteCaptureBundle(entry.bundle ?? entry)
  const fallbackLabel = `${bundle.fragmentLabel || bundle.fragmentId || 'Capture'} / ${bundle.captures.length} frames`

  return {
    id: readNonEmptyString(entry.id) || `sim2real-capture-${index}-${bundle.capturedAt}`,
    label: readNonEmptyString(entry.label) || fallbackLabel,
    savedAt: readNonEmptyString(entry.savedAt) || new Date().toISOString(),
    bundle
  }
}

function loadCaptureShelf() {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(sim2realCaptureShelfStorageKey)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed)
      ? parsed.map((entry, index) => normalizeCaptureShelfEntry(entry, index))
      : []
  } catch {
    return []
  }
}

function normalizeLocalizationRunSnapshot(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {}
  const estimate = normalizeLocalizationEstimate(
    entry.estimate ?? entry.estimateInput ?? entry.localizationEstimate
  )
  const groundTruth = entry.groundTruth && typeof entry.groundTruth === 'object' ? entry.groundTruth : {}
  const benchmark = entry.benchmark && typeof entry.benchmark === 'object' ? entry.benchmark : {}
  const groundTruthBundle = groundTruth.bundle ? normalizeRouteCaptureBundle(groundTruth.bundle) : null
  const groundTruthLabel = readNonEmptyString(groundTruth.label)
  const requestedAlignment =
    readNonEmptyString(benchmark.requestedAlignment) ||
    readNonEmptyString(benchmark.alignment) ||
    readNonEmptyString(entry.report?.requestedAlignment) ||
    readNonEmptyString(entry.report?.alignment) ||
    'auto'
  let report = entry.report && typeof entry.report === 'object' ? entry.report : null

  if (!report && groundTruthBundle) {
    try {
      report = buildLocalizationBenchmarkReport({
        alignment: requestedAlignment,
        groundTruthBundle,
        groundTruthLabel,
        estimateInput: estimate,
        estimateLabel: estimate.label
      })
    } catch {
      report = null
    }
  }

  const fallbackSummary = report
    ? buildLocalizationRunSummary(report)
    : {
        createdAt: readNonEmptyString(benchmark.reportCreatedAt),
        alignment: requestedAlignment,
        requestedAlignment,
        groundTruthLabel,
        estimateLabel: estimate.label,
        sourceType: estimate.sourceType,
        estimatePoseCount: estimate.poses.length
      }
  const summary = normalizeLocalizationRunSummary(entry.summary, fallbackSummary)
  const fallbackLabel = `${summary.estimateLabel} / ATE ${formatMeters(summary.ateRmseMeters)}`
  let imageBenchmark = null

  try {
    if (entry.imageBenchmark) {
      imageBenchmark = normalizeLocalizationImageBenchmarkReport(entry.imageBenchmark)
    }
  } catch {
    imageBenchmark = null
  }

  return {
    protocol: readNonEmptyString(entry.protocol) || 'dreamwalker-localization-run/v1',
    type: 'localization-run-snapshot',
    id: readNonEmptyString(entry.id) || `sim2real-localization-run-${index}-${summary.createdAt}`,
    label: readNonEmptyString(entry.label) || fallbackLabel,
    savedAt: readNonEmptyString(entry.savedAt) || summary.createdAt || new Date().toISOString(),
    groundTruth: {
      sourceId: readNonEmptyString(groundTruth.sourceId) || 'current-capture',
      label: groundTruthLabel || summary.groundTruthLabel,
      bundle: groundTruthBundle
    },
    estimate,
    benchmark: {
      alignment:
        readNonEmptyString(benchmark.alignment) ||
        readNonEmptyString(report?.alignment) ||
        summary.alignment,
      requestedAlignment,
      reportCreatedAt:
        readNonEmptyString(benchmark.reportCreatedAt) ||
        readNonEmptyString(report?.createdAt) ||
        summary.createdAt
    },
    summary,
    report,
    imageBenchmark
  }
}

function loadLocalizationRunShelf() {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(sim2realLocalizationRunShelfStorageKey)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed)
      ? parsed.map((entry, index) => normalizeLocalizationRunSnapshot(entry, index))
      : []
  } catch {
    return []
  }
}

export default function Sim2RealPanel({
  config,
  fragmentId,
  fragmentLabel,
  onBenchmarkOverlayChange,
  robotPose,
  robotTrail,
  robotWaypoint,
  onStatusMessage
}) {
  const [connectionState, setConnectionState] = useState(() => ({
    status: config.enabled ? 'connecting' : 'disabled',
    error: null
  }))
  const [serverInfo, setServerInfo] = useState(null)
  const [requestPending, setRequestPending] = useState(false)
  const [reconnectNonce, setReconnectNonce] = useState(0)
  const [replayState, setReplayState] = useState({
    active: false,
    index: 0,
    total: 0,
    mode: null
  })
  const [captureState, setCaptureState] = useState({
    status: 'idle',
    bundle: null,
    error: null
  })
  const [captureShelf, setCaptureShelf] = useState(loadCaptureShelf)
  const [captureShelfLabel, setCaptureShelfLabel] = useState('')
  const [captureImportError, setCaptureImportError] = useState('')
  const [localizationRunShelf, setLocalizationRunShelf] = useState(loadLocalizationRunShelf)
  const [localizationRunShelfLabel, setLocalizationRunShelfLabel] = useState('')
  const [localizationRunShelfError, setLocalizationRunShelfError] = useState('')
  const [benchmarkSourceId, setBenchmarkSourceId] = useState('current-capture')
  const [benchmarkAlignmentMode, setBenchmarkAlignmentMode] = useState('auto')
  const [estimateSourceMode, setEstimateSourceMode] = useState('auto')
  const [importedLocalizationEstimate, setImportedLocalizationEstimate] = useState(null)
  const [liveLocalizationEstimate, setLiveLocalizationEstimate] = useState(null)
  const [localizationImportError, setLocalizationImportError] = useState('')
  const [liveEstimateMonitorState, setLiveEstimateMonitorState] = useState(() => ({
    url: localizationMonitorDefaultUrl,
    enabled: false,
    status: 'disabled',
    error: null,
    messageCount: 0,
    lastMessageAt: '',
    label: ''
  }))
  const [requestSettings, setRequestSettings] = useState({
    width: 640,
    height: 480,
    fovDegrees: 60,
    nearClip: 0.05,
    farClip: 50,
    pointRadius: 1,
    routeDelayMs: 250
  })
  const [previewState, setPreviewState] = useState({
    status: 'idle',
    label: 'Current Pose',
    frameId: '',
    width: 0,
    height: 0,
    colorSrc: '',
    depthSrc: '',
    depthMin: null,
    depthMax: null,
    error: null
  })
  const socketRef = useRef(null)
  const pendingRequestRef = useRef(null)
  const replayAbortRef = useRef(false)
  const serverDefaultsHydratedRef = useRef(false)
  const captureFileInputRef = useRef(null)
  const localizationFileInputRef = useRef(null)
  const imageBenchmarkFileInputRef = useRef(null)
  const reviewBundleFileInputRef = useRef(null)
  const liveEstimateSocketRef = useRef(null)
  const liveLocalizationEstimateRef = useRef(null)
  const localizationRunShelfRef = useRef(localizationRunShelf)
  const [liveEstimateReconnectNonce, setLiveEstimateReconnectNonce] = useState(0)
  const [imageBenchmarkPreview, setImageBenchmarkPreview] = useState(null)
  const [imageBenchmarkRequestEntryId, setImageBenchmarkRequestEntryId] = useState('')
  const [imageBenchmarkBatchState, setImageBenchmarkBatchState] = useState({
    active: false,
    mode: 'missing',
    completed: 0,
    failed: 0,
    total: 0,
    currentLabel: ''
  })
  const [baselineLocalizationRunId, setBaselineLocalizationRunId] = useState('')

  const routeCount = Array.isArray(robotTrail) ? robotTrail.length : 0
  const connectionLabel = formatConnectionStatus(connectionState.status)
  const liveEstimateStatusLabel = formatLiveEstimateStatus(liveEstimateMonitorState.status)
  const captureCount = Array.isArray(captureState.bundle?.captures)
    ? captureState.bundle.captures.length
    : 0
  const routePreviewPoses = useMemo(
    () =>
      (Array.isArray(robotTrail) ? robotTrail : [])
        .filter((position) => isPosition(position))
        .map((position, index, route) => ({
          position,
          yawDegrees: inferYawDegreesFromRoute(route, index, robotPose.yawDegrees)
        })),
    [robotPose.yawDegrees, robotTrail]
  )
  const benchmarkGroundTruthOptions = useMemo(() => {
    const options = []

    if (captureState.bundle) {
      options.push({
        id: 'current-capture',
        label: `Current Capture / ${captureCount} frames`,
        bundle: captureState.bundle
      })
    }

    captureShelf.forEach((entry) => {
      options.push({
        id: `capture-shelf:${entry.id}`,
        label: entry.label,
        bundle: entry.bundle
      })
    })

    return options
  }, [captureCount, captureShelf, captureState.bundle])
  const activeBenchmarkSource =
    benchmarkGroundTruthOptions.find((entry) => entry.id === benchmarkSourceId) ??
    benchmarkGroundTruthOptions[0] ??
    null
  const activeLocalizationEstimate = useMemo(() => {
    if (estimateSourceMode === 'live') {
      return liveLocalizationEstimate
    }

    if (estimateSourceMode === 'imported') {
      return importedLocalizationEstimate
    }

    return liveLocalizationEstimate ?? importedLocalizationEstimate
  }, [estimateSourceMode, importedLocalizationEstimate, liveLocalizationEstimate])
  const localizationBenchmark = useMemo(() => {
    if (!activeBenchmarkSource?.bundle || !activeLocalizationEstimate) {
      return {
        report: null,
        error: ''
      }
    }

    try {
      return {
        report: buildLocalizationBenchmarkReport({
          groundTruthBundle: activeBenchmarkSource.bundle,
          groundTruthLabel: activeBenchmarkSource.label,
          alignment: benchmarkAlignmentMode,
          estimateInput: activeLocalizationEstimate,
          estimateLabel: activeLocalizationEstimate.label
        }),
        error: ''
      }
    } catch (error) {
      return {
        report: null,
        error: error instanceof Error ? error.message : String(error)
      }
    }
  }, [activeBenchmarkSource, activeLocalizationEstimate, benchmarkAlignmentMode])
  const activeLocalizationRunId = useMemo(() => {
    if (!localizationBenchmark.report || !activeLocalizationEstimate) {
      return ''
    }

    const activeAteRmseMeters = localizationBenchmark.report.metrics.ateRmseMeters
    const activeMatchedCount = localizationBenchmark.report.matching.matchedCount
    const activeGroundTruthLabel = localizationBenchmark.report.groundTruth.label

    return (
      localizationRunShelf.find((entry) => {
        return (
          entry.estimate.label === activeLocalizationEstimate.label &&
          entry.summary.groundTruthLabel === activeGroundTruthLabel &&
          entry.summary.matchedCount === activeMatchedCount &&
          metricEquals(entry.summary.ateRmseMeters, activeAteRmseMeters)
        )
      })?.id ?? ''
    )
  }, [activeLocalizationEstimate, localizationBenchmark.report, localizationRunShelf])
  const localizationRunComparison = useMemo(() => {
    if (localizationRunShelf.length === 0) {
      return null
    }

    const runsWithImageBenchmark = localizationRunShelf.filter((entry) => entry.imageBenchmark)
    const missingImageBenchmarkCount = localizationRunShelf.length - runsWithImageBenchmark.length

    const sortedByAte = [...localizationRunShelf].sort((left, right) => {
      return (
        metricSortValue(left.summary.ateRmseMeters) -
          metricSortValue(right.summary.ateRmseMeters) ||
        metricSortValue(left.summary.yawRmseDegrees) -
          metricSortValue(right.summary.yawRmseDegrees) ||
        Number(right.summary.matchedCount || 0) - Number(left.summary.matchedCount || 0) ||
        timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
      )
    })
    const bestAteRun = sortedByAte[0] ?? null
    const bestYawRun =
      [...localizationRunShelf]
        .filter((entry) => Number.isFinite(entry.summary.yawRmseDegrees))
        .sort((left, right) => {
          return (
            metricSortValue(left.summary.yawRmseDegrees) -
              metricSortValue(right.summary.yawRmseDegrees) ||
            metricSortValue(left.summary.ateRmseMeters) -
              metricSortValue(right.summary.ateRmseMeters) ||
            timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
          )
        })[0] ?? null
    const latestRun =
      [...localizationRunShelf].sort(
        (left, right) => timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
      )[0] ?? null
    const worstAteRun =
      [...localizationRunShelf]
        .filter((entry) => Number.isFinite(entry.summary.ateRmseMeters))
        .sort((left, right) => {
          return (
            metricSortValue(right.summary.ateRmseMeters) -
              metricSortValue(left.summary.ateRmseMeters) ||
            timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
          )
        })[0] ?? null
    const ateSpreadMeters =
      bestAteRun && worstAteRun
        ? normalizeOptionalMetricNumber(
            Number(worstAteRun.summary.ateRmseMeters) - Number(bestAteRun.summary.ateRmseMeters)
          )
        : null
    const bestLpipsRun =
      [...localizationRunShelf]
        .filter((entry) => Number.isFinite(entry.imageBenchmark?.summary?.lpipsMean))
        .sort((left, right) => {
          return (
            metricSortValue(left.imageBenchmark?.summary?.lpipsMean) -
              metricSortValue(right.imageBenchmark?.summary?.lpipsMean) ||
            timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
          )
        })[0] ?? null
    const worstLpipsRun =
      [...localizationRunShelf]
        .filter((entry) => Number.isFinite(entry.imageBenchmark?.summary?.lpipsMean))
        .sort((left, right) => {
          return (
            metricSortValue(right.imageBenchmark?.summary?.lpipsMean) -
              metricSortValue(left.imageBenchmark?.summary?.lpipsMean) ||
            timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt)
          )
        })[0] ?? null
    const lpipsSpread =
      bestLpipsRun && worstLpipsRun
        ? normalizeOptionalMetricNumber(
            Number(worstLpipsRun.imageBenchmark?.summary?.lpipsMean) -
              Number(bestLpipsRun.imageBenchmark?.summary?.lpipsMean)
          )
        : null
    const baselineRun =
      localizationRunShelf.find((entry) => entry.id === baselineLocalizationRunId) ??
      bestAteRun ??
      null

    return {
      bestAteRun,
      bestYawRun,
      bestLpipsRun,
      baselineRun,
      latestRun,
      ateSpreadMeters,
      lpipsSpread,
      runsWithImageBenchmarkCount: runsWithImageBenchmark.length,
      missingImageBenchmarkCount,
      rows: sortedByAte.map((entry, index) => ({
        ...entry,
        rank: index + 1,
        isActive: entry.id === activeLocalizationRunId,
        isBaseline: baselineRun ? entry.id === baselineRun.id : false,
        isBestAte: bestAteRun ? entry.id === bestAteRun.id : false,
        isBestYaw: bestYawRun ? entry.id === bestYawRun.id : false,
        isBestLpips: bestLpipsRun ? entry.id === bestLpipsRun.id : false,
        isLatest: latestRun ? entry.id === latestRun.id : false,
        psnrMean: entry.imageBenchmark?.summary?.psnrMean ?? null,
        ssimMean: entry.imageBenchmark?.summary?.ssimMean ?? null,
        lpipsMean: entry.imageBenchmark?.summary?.lpipsMean ?? null,
        worstLpipsValue: entry.imageBenchmark?.highlights?.lpips?.value ?? null,
        worstLpipsFrameIndex: Number.isFinite(entry.imageBenchmark?.highlights?.lpips?.frameIndex)
          ? Number(entry.imageBenchmark.highlights.lpips.frameIndex)
          : null,
        worstLpipsGroundTruthLabel: entry.imageBenchmark?.highlights?.lpips?.groundTruthLabel ?? '',
        worstLpipsEstimateLabel: entry.imageBenchmark?.highlights?.lpips?.estimateLabel ?? '',
        ateDeltaMeters:
          baselineRun && Number.isFinite(entry.summary.ateRmseMeters) && Number.isFinite(baselineRun.summary.ateRmseMeters)
            ? normalizeOptionalMetricNumber(
                Number(entry.summary.ateRmseMeters) - Number(baselineRun.summary.ateRmseMeters)
              )
            : null,
        yawDeltaDegrees:
          baselineRun && Number.isFinite(entry.summary.yawRmseDegrees) && Number.isFinite(baselineRun.summary.yawRmseDegrees)
            ? normalizeOptionalMetricNumber(
                Number(entry.summary.yawRmseDegrees) - Number(baselineRun.summary.yawRmseDegrees)
              )
            : null,
        lpipsDelta:
          baselineRun &&
          Number.isFinite(entry.imageBenchmark?.summary?.lpipsMean) &&
          Number.isFinite(baselineRun.imageBenchmark?.summary?.lpipsMean)
            ? normalizeOptionalMetricNumber(
                Number(entry.imageBenchmark.summary.lpipsMean) -
                  Number(baselineRun.imageBenchmark.summary.lpipsMean)
              )
            : null
      }))
    }
  }, [activeLocalizationRunId, baselineLocalizationRunId, localizationRunShelf])
  const largestBenchmarkTranslationSample = useMemo(() => {
    if (!Array.isArray(localizationBenchmark.report?.samples)) {
      return null
    }

    return localizationBenchmark.report.samples.reduce((largest, sample) => {
      if (!largest || sample.translationErrorMeters > largest.translationErrorMeters) {
        return sample
      }

      return largest
    }, null)
  }, [localizationBenchmark.report])
  const benchmarkOverlayPayload = useMemo(() => {
    if (!localizationBenchmark.report) {
      return null
    }

    return {
      groundTruthLabel: localizationBenchmark.report.groundTruth.label,
      estimateLabel: localizationBenchmark.report.estimate.label,
      matchedCount: localizationBenchmark.report.matching.matchedCount,
      ateRmseMeters: localizationBenchmark.report.metrics.ateRmseMeters,
      yawRmseDegrees: localizationBenchmark.report.metrics.yaw?.rmse ?? null,
      worstSampleIndex: largestBenchmarkTranslationSample?.index ?? null,
      samples: localizationBenchmark.report.samples.map((sample) => ({
        index: sample.index,
        translationErrorMeters: sample.translationErrorMeters,
        yawErrorDegrees: sample.yawErrorDegrees,
        groundTruthPosition: [...sample.groundTruth.position],
        estimatePosition: [...sample.estimate.position]
      }))
    }
  }, [largestBenchmarkTranslationSample?.index, localizationBenchmark.report])

  useEffect(() => {
    onBenchmarkOverlayChange?.(benchmarkOverlayPayload)
  }, [benchmarkOverlayPayload, onBenchmarkOverlayChange])

  useEffect(
    () => () => {
      onBenchmarkOverlayChange?.(null)
    },
    [onBenchmarkOverlayChange]
  )

  function rejectPendingRequest(errorMessage) {
    const pending = pendingRequestRef.current

    if (!pending) {
      return
    }

    window.clearTimeout(pending.timeoutId)
    pendingRequestRef.current = null
    setRequestPending(false)
    setImageBenchmarkRequestEntryId('')
    pending.reject(new Error(errorMessage))
  }

  function reconnect() {
    setReconnectNonce((current) => current + 1)
  }

  useEffect(() => {
    if (benchmarkGroundTruthOptions.length === 0) {
      if (benchmarkSourceId !== 'current-capture') {
        setBenchmarkSourceId('current-capture')
      }
      return
    }

    if (!benchmarkGroundTruthOptions.some((entry) => entry.id === benchmarkSourceId)) {
      setBenchmarkSourceId(benchmarkGroundTruthOptions[0].id)
    }
  }, [benchmarkGroundTruthOptions, benchmarkSourceId])

  function persistCaptureShelf(nextShelf) {
    const normalizedShelf = nextShelf
      .map((entry, index) => normalizeCaptureShelfEntry(entry, index))
      .slice(0, 12)

    setCaptureShelf(normalizedShelf)

    try {
      if (normalizedShelf.length === 0) {
        window.localStorage.removeItem(sim2realCaptureShelfStorageKey)
      } else {
        window.localStorage.setItem(
          sim2realCaptureShelfStorageKey,
          JSON.stringify(normalizedShelf)
        )
      }
      return true
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'sim2real capture shelf を保存できませんでした'
      setCaptureImportError(message)
      onStatusMessage?.(`Sim2Real capture shelf error: ${message}`)
      return false
    }
  }

  function persistLocalizationRunShelf(nextShelf) {
    const normalizedShelf = nextShelf
      .map((entry, index) => normalizeLocalizationRunSnapshot(entry, index))
      .slice(0, 12)

    localizationRunShelfRef.current = normalizedShelf
    setLocalizationRunShelf(normalizedShelf)

    try {
      if (normalizedShelf.length === 0) {
        window.localStorage.removeItem(sim2realLocalizationRunShelfStorageKey)
      } else {
        window.localStorage.setItem(
          sim2realLocalizationRunShelfStorageKey,
          JSON.stringify(normalizedShelf)
        )
      }
      return true
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'localization run shelf を保存できませんでした'
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization run shelf error: ${message}`)
      return false
    }
  }

  function clearCaptureBundle() {
    setCaptureState({
      status: 'idle',
      bundle: null,
      error: null
    })
    onStatusMessage?.('Sim2Real route capture cleared')
  }

  function downloadCaptureBundle() {
    if (!captureState.bundle) {
      return
    }

    downloadTextFile(
      buildCaptureFileName(fragmentId, captureState.bundle.capturedAt),
      JSON.stringify(captureState.bundle, null, 2)
    )
    onStatusMessage?.(`Sim2Real route capture downloaded: ${captureCount} frames`)
  }

  function saveCaptureSnapshot() {
    if (!captureState.bundle) {
      return
    }

    const fallbackLabel = `${fragmentLabel || fragmentId || 'Capture'} / ${captureCount} frames`
    const nextEntry = {
      id: `sim2real-capture-${Date.now()}`,
      label: readNonEmptyString(captureShelfLabel) || fallbackLabel,
      savedAt: new Date().toISOString(),
      bundle: captureState.bundle
    }
    const nextShelf = [nextEntry, ...captureShelf.filter((entry) => entry.label !== nextEntry.label)]

    if (persistCaptureShelf(nextShelf)) {
      setCaptureShelfLabel('')
      setCaptureImportError('')
      onStatusMessage?.(`Sim2Real capture saved to shelf: ${nextEntry.label}`)
    }
  }

  function removeCaptureShelfEntry(entryId) {
    persistCaptureShelf(captureShelf.filter((entry) => entry.id !== entryId))
    onStatusMessage?.('Sim2Real capture removed from shelf')
  }

  function clearCaptureShelf() {
    persistCaptureShelf([])
    onStatusMessage?.('Sim2Real capture shelf cleared')
  }

  function previewCaptureShelfEntry(entry) {
    const latestCapture = entry?.bundle?.captures?.at?.(-1) ?? null

    if (!latestCapture?.response) {
      return
    }

    setPreviewState({
      status: 'ready',
      label: `${entry.label} / Preview`,
      error: null,
      ...buildSim2realPreviewFromMessage(latestCapture.response)
    })
    setCaptureState({
      status: 'ready',
      bundle: entry.bundle,
      error: null
    })
    onStatusMessage?.(`Sim2Real capture preview loaded: ${entry.label}`)
  }

  function downloadCaptureShelfEntry(entry) {
    downloadTextFile(
      buildCaptureFileName(
        entry.bundle.fragmentId || fragmentId,
        entry.bundle.capturedAt || entry.savedAt
      ),
      JSON.stringify(entry.bundle, null, 2)
    )
    onStatusMessage?.(`Sim2Real shelf capture downloaded: ${entry.label}`)
  }

  async function handleCaptureFileImport(event) {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''

    if (!file) {
      return
    }

    try {
      const rawText = await file.text()
      const importedBundle = normalizeRouteCaptureBundle(JSON.parse(rawText))
      const nextEntry = {
        id: `sim2real-capture-import-${Date.now()}`,
        label:
          readNonEmptyString(captureShelfLabel) ||
          `${importedBundle.fragmentLabel || importedBundle.fragmentId || file.name} / ${importedBundle.captures.length} frames`,
        savedAt: new Date().toISOString(),
        bundle: importedBundle
      }

      if (persistCaptureShelf([nextEntry, ...captureShelf])) {
        setCaptureState({
          status: 'ready',
          bundle: importedBundle,
          error: null
        })
        setCaptureImportError('')
        setCaptureShelfLabel('')
        previewCaptureShelfEntry(nextEntry)
        onStatusMessage?.(`Sim2Real capture imported: ${nextEntry.label}`)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setCaptureImportError(message)
      onStatusMessage?.(`Sim2Real capture import error: ${message}`)
    }
  }

  useEffect(() => {
    liveLocalizationEstimateRef.current = liveLocalizationEstimate
  }, [liveLocalizationEstimate])

  useEffect(() => {
    localizationRunShelfRef.current = localizationRunShelf
  }, [localizationRunShelf])

  function clearImportedLocalizationEstimate() {
    setImportedLocalizationEstimate(null)
    setLocalizationImportError('')
    onStatusMessage?.('Imported localization estimate cleared')
  }

  function clearLiveLocalizationEstimate() {
    liveLocalizationEstimateRef.current = null
    setLiveLocalizationEstimate(null)
    setLiveEstimateMonitorState((current) => ({
      ...current,
      messageCount: 0,
      lastMessageAt: '',
      label: ''
    }))
    onStatusMessage?.('Live localization estimate cleared')
  }

  function connectLiveEstimateMonitor() {
    setLiveEstimateMonitorState((current) => ({
      ...current,
      enabled: true,
      status: 'connecting',
      error: null,
      url: normalizeLocalizationMonitorUrl(current.url)
    }))
    setLiveEstimateReconnectNonce((current) => current + 1)
  }

  function disconnectLiveEstimateMonitor() {
    setLiveEstimateMonitorState((current) => ({
      ...current,
      enabled: false,
      status: 'disabled',
      error: null
    }))
    liveEstimateSocketRef.current?.close(1000, 'live monitor disabled')
    onStatusMessage?.('Live localization monitor disconnected')
  }

  function downloadLocalizationBenchmarkReport() {
    if (!localizationBenchmark.report) {
      return
    }

    downloadTextFile(
      buildLocalizationBenchmarkFileName(fragmentId, localizationBenchmark.report.createdAt),
      JSON.stringify(localizationBenchmark.report, null, 2)
    )
    onStatusMessage?.(
      `Localization benchmark report downloaded: ${localizationBenchmark.report.matching.matchedCount} matched poses`
    )
  }

  function saveLocalizationRunSnapshot() {
    if (!localizationBenchmark.report || !activeBenchmarkSource?.bundle || !activeLocalizationEstimate) {
      return
    }

    const nextEntry = {
      id: `sim2real-localization-run-${Date.now()}`,
      ...buildLocalizationRunSnapshot({
        label: localizationRunShelfLabel,
        estimate: activeLocalizationEstimate,
        groundTruthSource: activeBenchmarkSource,
        report: localizationBenchmark.report
      })
    }
    const nextShelf = [
      nextEntry,
      ...localizationRunShelf.filter((entry) => entry.label !== nextEntry.label)
    ]

    if (persistLocalizationRunShelf(nextShelf)) {
      setLocalizationRunShelfLabel('')
      setLocalizationRunShelfError('')
      onStatusMessage?.(`Localization run saved: ${nextEntry.label}`)
    }
  }

  function buildLocalizationImagePreview(entry, metricName = 'lpips') {
    const highlight = entry?.imageBenchmark?.highlights?.[metricName] ?? null

    if (
      !highlight ||
      !readNonEmptyString(highlight.groundTruthColorJpegBase64) ||
      !readNonEmptyString(highlight.renderedColorJpegBase64)
    ) {
      return null
    }

    return {
      entryId: entry.id,
      runLabel: entry.label,
      metricName,
      value: highlight.value,
      frameIndex: highlight.frameIndex,
      groundTruthLabel: highlight.groundTruthLabel || entry.summary.groundTruthLabel,
      estimateLabel: highlight.estimateLabel || entry.summary.estimateLabel,
      groundTruthSrc: buildJpegDataUrl(highlight.groundTruthColorJpegBase64),
      renderedSrc: buildJpegDataUrl(highlight.renderedColorJpegBase64)
    }
  }

  function previewLocalizationImageBenchmark(entry, metricName = 'lpips') {
    const nextPreview = buildLocalizationImagePreview(entry, metricName)

    if (!nextPreview) {
      const message = `image benchmark preview unavailable: ${entry.label}`
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization image preview error: ${message}`)
      return
    }

    setImageBenchmarkPreview(nextPreview)
    setLocalizationRunShelfError('')
    onStatusMessage?.(
      `Localization image preview loaded: ${entry.label} / ${metricName} ${formatLpips(nextPreview.value)}`
    )
  }

  function resolveLocalizationRunEntry(entryLike) {
    const entryId =
      typeof entryLike === 'string'
        ? readNonEmptyString(entryLike)
        : readNonEmptyString(entryLike?.id)
    const currentShelf = localizationRunShelfRef.current

    if (!entryId) {
      return entryLike && typeof entryLike === 'object' ? entryLike : null
    }

    return currentShelf.find((entry) => entry.id === entryId) ?? null
  }

  function attachImageBenchmarkReportToShelf(report, options = {}) {
    const currentShelf = localizationRunShelfRef.current
    const matchingEntries = currentShelf.filter((entry) => {
      const estimateLabelMatches =
        entry.summary.estimateLabel === report.estimateLabel || entry.label === report.estimateLabel
      const matchedCountMatches =
        !report.matchedCount || !entry.summary.matchedCount || entry.summary.matchedCount === report.matchedCount

      return estimateLabelMatches && matchedCountMatches
    })
    const activeEntry =
      activeLocalizationRunId
        ? currentShelf.find((entry) => entry.id === activeLocalizationRunId)
        : null
    const targetEntry =
      (readNonEmptyString(options.entryId)
        ? currentShelf.find((entry) => entry.id === options.entryId) ?? null
        : null) ??
      (activeEntry &&
      (activeEntry.summary.estimateLabel === report.estimateLabel || activeEntry.label === report.estimateLabel)
        ? activeEntry
        : null) ??
      matchingEntries.sort((left, right) => timestampSortValue(right.savedAt) - timestampSortValue(left.savedAt))[0] ??
      null

    if (!targetEntry) {
      throw new Error(
        `matching localization run not found for image benchmark: ${report.estimateLabel}`
      )
    }

    const nextShelf = currentShelf.map((entry) =>
      entry.id === targetEntry.id
        ? {
            ...entry,
            imageBenchmark: report
          }
        : entry
    )

    if (persistLocalizationRunShelf(nextShelf)) {
      const updatedEntry = nextShelf.find((entry) => entry.id === targetEntry.id)
      if (updatedEntry) {
        const preview = buildLocalizationImagePreview(updatedEntry, 'lpips')
        if (preview) {
          setImageBenchmarkPreview(preview)
        }
      }
      setLocalizationRunShelfError('')
      onStatusMessage?.(
        `Localization image benchmark attached: ${targetEntry.label} / LPIPS ${formatLpips(report.summary.lpipsMean)}`
      )
      return updatedEntry ?? null
    }

    return null
  }

  function mergeEntriesById(importedEntries, existingEntries) {
    const importedIds = new Set(importedEntries.map((entry) => entry.id))
    return [...importedEntries, ...existingEntries.filter((entry) => !importedIds.has(entry.id))]
  }

  async function handleLocalizationReviewBundleImport(event) {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''

    if (!file) {
      return
    }

    try {
      const rawText = await file.text()
      const importedBundle = importLocalizationReviewBundleDocument(rawText)
      const importedCaptureEntries = importedBundle.captureShelfEntries
      const importedRunEntries = importedBundle.runShelfEntries
      const nextCaptureShelf = mergeEntriesById(importedCaptureEntries, captureShelf)
      const nextRunShelf = mergeEntriesById(importedRunEntries, localizationRunShelfRef.current)
      const capturesPersisted = persistCaptureShelf(nextCaptureShelf)
      const runsPersisted = persistLocalizationRunShelf(nextRunShelf)

      if (!capturesPersisted || !runsPersisted) {
        return
      }

      if (readNonEmptyString(importedBundle.baselineRunId)) {
        setBaselineLocalizationRunId(importedBundle.baselineRunId)
      }

      const firstImportedCapture = importedCaptureEntries[0] ?? null
      const firstImportedRun = importedRunEntries[0] ?? null

      if (firstImportedCapture?.bundle) {
        previewCaptureShelfEntry(firstImportedCapture)
      } else if (firstImportedRun?.groundTruth?.bundle) {
        previewLocalizationGroundTruthBundle(
          firstImportedRun.groundTruth.bundle,
          `${firstImportedRun.label} / Ground Truth`
        )
      }

      setCaptureImportError('')
      setLocalizationRunShelfError('')
      onStatusMessage?.(
        `Localization review bundle imported: ${importedRunEntries.length} runs / ${importedCaptureEntries.length} captures`
      )
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization review bundle import error: ${message}`)
    }
  }

  async function handleLocalizationImageBenchmarkImport(event) {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''

    if (!file) {
      return
    }

    try {
      const rawText = await file.text()
      const importedReport = normalizeLocalizationImageBenchmarkReport(JSON.parse(rawText))
      attachImageBenchmarkReportToShelf(importedReport)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization image benchmark import error: ${message}`)
    }
  }

  function clearLocalizationImageBenchmark(entryId) {
    const nextShelf = localizationRunShelf.map((entry) =>
      entry.id === entryId
        ? {
            ...entry,
            imageBenchmark: null
          }
        : entry
    )

    if (persistLocalizationRunShelf(nextShelf)) {
      setLocalizationRunShelfError('')
      setImageBenchmarkPreview((current) => (current?.entryId === entryId ? null : current))
      onStatusMessage?.('Localization image benchmark cleared')
    }
  }

  function buildLocalizationRunCompareReport(comparison, createdAt = new Date().toISOString()) {
    if (!comparison) {
      return null
    }

    return {
      protocol: 'dreamwalker-localization-run-compare/v1',
      type: 'localization-run-compare-report',
      createdAt,
      fragmentId: readNonEmptyString(fragmentId),
      fragmentLabel: readNonEmptyString(fragmentLabel),
      baselineRunId: comparison.baselineRun?.id || '',
      baselineLabel: comparison.baselineRun?.label || '',
      baseline: comparison.baselineRun
        ? {
            id: comparison.baselineRun.id,
            label: comparison.baselineRun.label,
            summary: {
              ...comparison.baselineRun.summary
            },
            imageBenchmark: comparison.baselineRun.imageBenchmark
              ? {
                  summary: {
                    ...comparison.baselineRun.imageBenchmark.summary
                  },
                  highlights: {
                    ...comparison.baselineRun.imageBenchmark.highlights
                  }
                }
              : null
          }
        : null,
      summary: {
        runCount: comparison.rows.length,
        runsWithImageBenchmarkCount: comparison.runsWithImageBenchmarkCount,
        missingImageBenchmarkCount: comparison.missingImageBenchmarkCount,
        bestAteRunId: comparison.bestAteRun?.id || '',
        bestYawRunId: comparison.bestYawRun?.id || '',
        bestLpipsRunId: comparison.bestLpipsRun?.id || '',
        latestRunId: comparison.latestRun?.id || '',
        ateSpreadMeters: comparison.ateSpreadMeters,
        lpipsSpread: comparison.lpipsSpread,
        bestAteLabel: comparison.bestAteRun?.label || '',
        bestYawLabel: comparison.bestYawRun?.label || '',
        bestLpipsLabel: comparison.bestLpipsRun?.label || '',
        latestLabel: comparison.latestRun?.label || ''
      },
      highlights: {
        bestAte: comparison.bestAteRun
          ? {
              id: comparison.bestAteRun.id,
              label: comparison.bestAteRun.label,
              ateRmseMeters: comparison.bestAteRun.summary.ateRmseMeters
            }
          : null,
        bestYaw: comparison.bestYawRun
          ? {
              id: comparison.bestYawRun.id,
              label: comparison.bestYawRun.label,
              yawRmseDegrees: comparison.bestYawRun.summary.yawRmseDegrees
            }
          : null,
        bestLpips: comparison.bestLpipsRun
          ? {
              id: comparison.bestLpipsRun.id,
              label: comparison.bestLpipsRun.label,
              lpipsMean: comparison.bestLpipsRun.imageBenchmark?.summary?.lpipsMean ?? null,
              worstFrameIndex: comparison.bestLpipsRun.imageBenchmark?.highlights?.lpips?.frameIndex ?? null,
              worstFrameLabel:
                comparison.bestLpipsRun.imageBenchmark?.highlights?.lpips?.groundTruthLabel ?? ''
            }
          : null
      },
      rows: comparison.rows.map((entry) => ({
        id: entry.id,
        rank: entry.rank,
        label: entry.label,
        flags: {
          isBaseline: entry.isBaseline,
          isBestAte: entry.isBestAte,
          isBestYaw: entry.isBestYaw,
          isBestLpips: entry.isBestLpips,
          isLatest: entry.isLatest,
          isActive: entry.isActive
        },
        summary: {
          ...entry.summary
        },
        imageBenchmark: entry.imageBenchmark
          ? {
              summary: {
                ...entry.imageBenchmark.summary
              },
              highlights: {
                ...entry.imageBenchmark.highlights
              }
            }
          : null,
        imageBenchmarkSummary: {
          psnrMean: entry.psnrMean,
          ssimMean: entry.ssimMean,
          lpipsMean: entry.lpipsMean,
          worstLpipsValue: entry.worstLpipsValue,
          worstLpipsFrameIndex: entry.worstLpipsFrameIndex,
          worstLpipsGroundTruthLabel: entry.worstLpipsGroundTruthLabel,
          worstLpipsEstimateLabel: entry.worstLpipsEstimateLabel
        },
        deltas: {
          ateMeters: entry.ateDeltaMeters,
          yawDegrees: entry.yawDeltaDegrees,
          lpips: entry.lpipsDelta
        }
      }))
    }
  }

  function downloadLocalizationRunCompareJson() {
    if (!localizationRunComparison) {
      return
    }

    const report = buildLocalizationRunCompareReport(localizationRunComparison)

    downloadTextFile(
      buildLocalizationRunCompareFileName(fragmentId, report.createdAt, 'json'),
      JSON.stringify(report, null, 2)
    )
    onStatusMessage?.(`Localization run compare exported: ${report.rows.length} runs`)
  }

  function downloadLocalizationRunCompareCsv() {
    if (!localizationRunComparison) {
      return
    }

    downloadTextFile(
      buildLocalizationRunCompareFileName(fragmentId, new Date().toISOString(), 'csv'),
      buildLocalizationRunCompareCsv(localizationRunComparison.rows),
      'text/csv'
    )
    onStatusMessage?.(`Localization run compare CSV exported: ${localizationRunComparison.rows.length} runs`)
  }

  function downloadLocalizationRunCompareMarkdown() {
    if (!localizationRunComparison) {
      return
    }

    const createdAt = new Date().toISOString()
    downloadTextFile(
      buildLocalizationRunCompareFileName(fragmentId, createdAt, 'md'),
      buildLocalizationRunCompareMarkdown(localizationRunComparison, {
        fragmentId,
        fragmentLabel,
        createdAt
      }),
      'text/markdown'
    )
    onStatusMessage?.(
      `Localization run compare Markdown exported: ${localizationRunComparison.rows.length} runs`
    )
  }

  function downloadLocalizationReviewBundle() {
    if (!localizationRunComparison) {
      return
    }

    const createdAt = new Date().toISOString()
    const compareReport = buildLocalizationRunCompareReport(localizationRunComparison, createdAt)
    const compareCsv = buildLocalizationRunCompareCsv(localizationRunComparison.rows)
    const compareMarkdown = buildLocalizationRunCompareMarkdown(localizationRunComparison, {
      fragmentId,
      fragmentLabel,
      createdAt
    })
    const worstLpipsPreviews = localizationRunComparison.rows
      .map((entry) => buildWorstLpipsPreviewArtifact(entry))
      .filter(Boolean)
    const linkedCaptureMap = new Map()
    const runs = localizationRunComparison.rows.map((entry) => {
      const snapshot = buildPortableLocalizationRunSnapshot(entry)
      const captureSourceId = readNonEmptyString(snapshot.groundTruth?.sourceId) || 'current-capture'

      if (!linkedCaptureMap.has(captureSourceId)) {
        const captureBundle = snapshot.groundTruth?.bundle ?? null
        linkedCaptureMap.set(captureSourceId, {
          sourceId: captureSourceId,
          label: snapshot.groundTruth?.label || '',
          fragmentId: captureBundle?.fragmentId || '',
          fragmentLabel: captureBundle?.fragmentLabel || '',
          captureCount: Array.isArray(captureBundle?.captures) ? captureBundle.captures.length : 0,
          bundle: captureBundle
        })
      }

      return {
        id: entry.id,
        label: entry.label,
        rank: entry.rank,
        reviewArtifacts: {
          worstLpipsPreview: buildWorstLpipsPreviewArtifact(entry)
        },
        snapshot
      }
    })

    const bundle = {
      protocol: 'dreamwalker-localization-review-bundle/v1',
      type: 'localization-review-bundle',
      createdAt,
      fragmentId: readNonEmptyString(fragmentId),
      fragmentLabel: readNonEmptyString(fragmentLabel),
      selection: {
        runIds: localizationRunComparison.rows.map((entry) => entry.id),
        baselineRunId: localizationRunComparison.baselineRun?.id || '',
        baselineLabel: localizationRunComparison.baselineRun?.label || ''
      },
      compareReport,
      artifacts: {
        compareCsv,
        compareMarkdown,
        worstLpipsPreviews
      },
      linkedCaptures: Array.from(linkedCaptureMap.values()),
      runs
    }

    downloadTextFile(
      buildLocalizationReviewBundleFileName(fragmentId, createdAt),
      JSON.stringify(bundle, null, 2)
    )
    onStatusMessage?.(`Localization review bundle exported: ${runs.length} runs`)
  }

  async function runLocalizationImageBenchmark(entryLike, options = {}) {
    const entry = resolveLocalizationRunEntry(entryLike)

    if (!entry) {
      const message = 'saved localization run が見つかりません'
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization image benchmark error: ${message}`)
      return null
    }

    const resolvedGroundTruth = resolveLocalizationRunGroundTruth(entry)

    if (!resolvedGroundTruth?.bundle) {
      const message =
        'saved localization run に必要な ground truth capture が見つかりません。capture shelf entry を残すか、capture bundle を再importしてください'
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization image benchmark error: ${message}`)
      return null
    }

    const matchedCount = Math.max(
      1,
      Number(entry.summary.matchedCount || 0) || resolvedGroundTruth.bundle.captures?.length || 1
    )
    const responseTimeoutSeconds = Math.min(180, Math.max(30, matchedCount * 8))
    const payload = buildSim2realImageBenchmarkRequest({
      groundTruthBundle: resolvedGroundTruth.bundle,
      estimate: entry.estimate,
      alignment: entry.benchmark.requestedAlignment || entry.summary.requestedAlignment || 'auto',
      metrics: ['psnr', 'ssim', 'lpips'],
      responseTimeoutSeconds
    })

    setLocalizationRunShelfError('')
    setImageBenchmarkRequestEntryId(entry.id)

    try {
      const reportMessage = await sendSocketRequest(payload, {
        expectedResponseType: 'localization-image-benchmark-report',
        timeoutMs: responseTimeoutSeconds * 1000 + 5_000
      })
      const normalizedReport = normalizeLocalizationImageBenchmarkReport(reportMessage)
      const updatedEntry = attachImageBenchmarkReportToShelf(normalizedReport, { entryId: entry.id })
      if (!options.suppressSuccessStatus) {
        onStatusMessage?.(
          `Localization image benchmark completed: ${entry.label} / LPIPS ${formatLpips(normalizedReport.summary.lpipsMean)}`
        )
      }
      return {
        report: normalizedReport,
        entry: updatedEntry
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization image benchmark error: ${message}`)
      return null
    } finally {
      setImageBenchmarkRequestEntryId((current) => (current === entry.id ? '' : current))
    }
  }

  async function runLocalizationImageBenchmarkBatch(mode = 'missing') {
    const currentShelf = localizationRunShelfRef.current
    const targetEntries = currentShelf.filter((entry) =>
      mode === 'all' ? true : !entry.imageBenchmark
    )

    if (targetEntries.length === 0) {
      onStatusMessage?.(
        mode === 'all'
          ? 'Localization image benchmark batch skipped: no saved runs'
          : 'Localization image benchmark batch skipped: no runs are missing image metrics'
      )
      return
    }

    setLocalizationRunShelfError('')
    setImageBenchmarkBatchState({
      active: true,
      mode,
      completed: 0,
      failed: 0,
      total: targetEntries.length,
      currentLabel: targetEntries[0].label
    })

    let completed = 0
    let failed = 0

    for (const targetEntry of targetEntries) {
      setImageBenchmarkBatchState((current) => ({
        ...current,
        currentLabel: targetEntry.label,
        completed,
        failed
      }))

      const result = await runLocalizationImageBenchmark(targetEntry.id, {
        suppressSuccessStatus: true
      })

      if (result) {
        completed += 1
      } else {
        failed += 1
      }

      setImageBenchmarkBatchState((current) => ({
        ...current,
        completed,
        failed
      }))
    }

    setImageBenchmarkBatchState((current) => ({
      ...current,
      active: false,
      currentLabel: ''
    }))
    onStatusMessage?.(
      `Localization image benchmark batch completed: ${completed} succeeded / ${failed} failed / mode ${formatImageBenchmarkBatchMode(mode)}`
    )
  }

  function resolveLocalizationRunGroundTruth(entry) {
    const sourceId = readNonEmptyString(entry?.groundTruth?.sourceId)

    if (entry?.groundTruth?.bundle) {
      return {
        sourceId: sourceId || 'current-capture',
        label: entry.groundTruth.label,
        bundle: entry.groundTruth.bundle
      }
    }

    if (sourceId.startsWith('capture-shelf:')) {
      const matchingCaptureEntry = captureShelf.find(
        (captureEntry) => `capture-shelf:${captureEntry.id}` === sourceId
      )

      if (matchingCaptureEntry) {
        return {
          sourceId,
          label: matchingCaptureEntry.label,
          bundle: matchingCaptureEntry.bundle
        }
      }
    }

    return null
  }

  function previewLocalizationGroundTruthBundle(bundle, label) {
    const latestCapture = bundle?.captures?.at?.(-1) ?? null

    if (!latestCapture?.response) {
      return
    }

    setPreviewState({
      status: 'ready',
      label,
      error: null,
      ...buildSim2realPreviewFromMessage(latestCapture.response)
    })
  }

  function loadLocalizationRunSnapshot(entry) {
    const resolvedGroundTruth = resolveLocalizationRunGroundTruth(entry)

    if (!resolvedGroundTruth?.bundle) {
      const message =
        'saved localization run に必要な ground truth capture が見つかりません。capture shelf entry を残すか、capture bundle を再importしてください'
      setLocalizationRunShelfError(message)
      onStatusMessage?.(`Localization run load error: ${message}`)
      return
    }

    if (resolvedGroundTruth.sourceId === 'current-capture') {
      setCaptureState({
        status: 'ready',
        bundle: resolvedGroundTruth.bundle,
        error: null
      })
    }

    previewLocalizationGroundTruthBundle(
      resolvedGroundTruth.bundle,
      `${entry.label} / Ground Truth`
    )
    setImportedLocalizationEstimate(entry.estimate)
    setEstimateSourceMode('imported')
    setBenchmarkAlignmentMode(
      entry.benchmark.requestedAlignment || entry.benchmark.alignment || 'auto'
    )
    setBenchmarkSourceId(resolvedGroundTruth.sourceId || 'current-capture')
    setLocalizationImportError('')
    setLocalizationRunShelfError('')
    onStatusMessage?.(`Localization run loaded: ${entry.label}`)
  }

  function buildPortableLocalizationRunSnapshot(entry) {
    const resolvedGroundTruth = resolveLocalizationRunGroundTruth(entry)

    return {
      ...entry,
      groundTruth: {
        ...entry.groundTruth,
        sourceId: entry.groundTruth.sourceId,
        label: entry.groundTruth.label,
        bundle: resolvedGroundTruth?.bundle ?? entry.groundTruth.bundle ?? null
      }
    }
  }

  function downloadLocalizationRunSnapshot(entry) {
    const portableEntry = buildPortableLocalizationRunSnapshot(entry)
    downloadTextFile(
      buildLocalizationRunFileName(
        portableEntry.groundTruth?.bundle?.fragmentId || fragmentId,
        portableEntry.savedAt
      ),
      JSON.stringify(portableEntry, null, 2)
    )
    onStatusMessage?.(`Localization run downloaded: ${entry.label}`)
  }

  function removeLocalizationRunSnapshot(entryId) {
    persistLocalizationRunShelf(localizationRunShelf.filter((entry) => entry.id !== entryId))
    setImageBenchmarkPreview((current) => (current?.entryId === entryId ? null : current))
    onStatusMessage?.('Localization run removed from shelf')
  }

  function clearLocalizationRunShelf() {
    persistLocalizationRunShelf([])
    setImageBenchmarkPreview(null)
    setImageBenchmarkBatchState({
      active: false,
      mode: 'missing',
      completed: 0,
      failed: 0,
      total: 0,
      currentLabel: ''
    })
    onStatusMessage?.('Localization run shelf cleared')
  }

  async function handleLocalizationEstimateImport(event) {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''

    if (!file) {
      return
    }

    try {
      const rawText = await file.text()
      const normalizedEstimate = parseLocalizationEstimateDocument(rawText, {
        fileName: file.name
      })
      const nextEstimate = {
        ...normalizedEstimate,
        fileName: file.name,
        label: readNonEmptyString(normalizedEstimate.label) || file.name
      }

      setImportedLocalizationEstimate(nextEstimate)
      setLocalizationImportError('')
      setEstimateSourceMode((current) => (current === 'live' ? current : 'imported'))
      onStatusMessage?.(
        `Localization estimate imported: ${nextEstimate.label} / ${nextEstimate.poses.length} poses`
      )
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setLocalizationImportError(message)
      onStatusMessage?.(`Localization estimate import error: ${message}`)
    }
  }

  useEffect(() => {
    if (!liveEstimateMonitorState.enabled) {
      liveEstimateSocketRef.current?.close(1000, 'live monitor disabled')
      liveEstimateSocketRef.current = null
      return
    }

    let disposed = false
    const socket = new WebSocket(normalizeLocalizationMonitorUrl(liveEstimateMonitorState.url))
    liveEstimateSocketRef.current = socket
    setLiveEstimateMonitorState((current) => ({
      ...current,
      status: 'connecting',
      error: null
    }))

    socket.addEventListener('open', () => {
      if (disposed) {
        return
      }

      setLiveEstimateMonitorState((current) => ({
        ...current,
        status: 'connected',
        error: null
      }))
      onStatusMessage?.(`Live localization monitor connected: ${normalizeLocalizationMonitorUrl(liveEstimateMonitorState.url)}`)
    })

    socket.addEventListener('message', (event) => {
      if (disposed) {
        return
      }

      try {
        const result = applyLocalizationEstimateStreamMessage(liveLocalizationEstimateRef.current, event.data, {
          maxPoses: 240,
          defaultLabel: 'Live Localization Estimate'
        })

        liveLocalizationEstimateRef.current = result.estimate
        setLiveLocalizationEstimate(result.estimate)
        setLiveEstimateMonitorState((current) => ({
          ...current,
          status: current.status === 'error' ? 'connected' : current.status,
          error: null,
          messageCount: result.kind === 'clear' ? 0 : current.messageCount + 1,
          lastMessageAt: result.kind === 'clear' ? '' : new Date().toISOString(),
          label: result.estimate?.label || current.label
        }))
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        setLiveEstimateMonitorState((current) => ({
          ...current,
          status: 'error',
          error: message
        }))
        onStatusMessage?.(`Live localization monitor error: ${message}`)
      }
    })

    socket.addEventListener('error', () => {
      if (disposed) {
        return
      }

      setLiveEstimateMonitorState((current) => ({
        ...current,
        status: 'error',
        error: `live localization websocket に接続できません: ${normalizeLocalizationMonitorUrl(liveEstimateMonitorState.url)}`
      }))
    })

    socket.addEventListener('close', () => {
      if (liveEstimateSocketRef.current === socket) {
        liveEstimateSocketRef.current = null
      }

      if (disposed) {
        return
      }

      setLiveEstimateMonitorState((current) => ({
        ...current,
        status: current.enabled ? (current.status === 'error' ? 'error' : 'closed') : 'disabled'
      }))
    })

    return () => {
      disposed = true
      if (liveEstimateSocketRef.current === socket) {
        liveEstimateSocketRef.current = null
      }
      socket.close(1000, 'cleanup')
    }
  }, [liveEstimateMonitorState.enabled, liveEstimateMonitorState.url, liveEstimateReconnectNonce, onStatusMessage])

  async function sendSocketRequest(payload, { expectedResponseType, timeoutMs, onTimeout } = {}) {
    const socket = socketRef.current

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error(`sim2real socket is not connected: ${config.url}`)
    }

    if (pendingRequestRef.current) {
      throw new Error('sim2real render request already in flight')
    }

    setRequestPending(true)

    return new Promise((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        pendingRequestRef.current = null
        setRequestPending(false)
        onTimeout?.()
        const errorMessage = `sim2real ${expectedResponseType || 'request'} timed out`
        reject(new Error(errorMessage))
      }, Math.max(1_000, Number(timeoutMs) || 10_000))

      pendingRequestRef.current = {
        expectedResponseType,
        resolve,
        reject,
        timeoutId
      }

      try {
        socket.send(JSON.stringify(payload))
      } catch (error) {
        window.clearTimeout(timeoutId)
        pendingRequestRef.current = null
        setRequestPending(false)
        reject(error instanceof Error ? error : new Error('sim2real send failed'))
      }
    })
  }

  async function sendRenderRequest(pose, label) {
    const payload = buildSim2realRenderRequest(pose, requestSettings)

    setPreviewState((current) => ({
      ...current,
      status: 'loading',
      label,
      error: null
    }))

    return sendSocketRequest(payload, {
      expectedResponseType: 'render-result',
      timeoutMs: 10_000,
      onTimeout: () => {
        setPreviewState((current) => ({
          ...current,
          status: 'error',
          error: 'sim2real render request timed out'
        }))
      }
    })
  }

  async function renderCurrentPose() {
    try {
      await sendRenderRequest(robotPose, 'Current Pose')
      onStatusMessage?.('Sim2Real: current pose rendered')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setConnectionState((current) => ({
        ...current,
        status: 'error',
        error: message
      }))
      onStatusMessage?.(`Sim2Real error: ${message}`)
    }
  }

  async function renderWaypointPose() {
    if (!robotWaypoint?.position) {
      return
    }

    try {
      await sendRenderRequest(
        {
          position: robotWaypoint.position,
          yawDegrees: robotPose.yawDegrees
        },
        'Waypoint'
      )
      onStatusMessage?.('Sim2Real: waypoint pose rendered')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setConnectionState((current) => ({
        ...current,
        status: 'error',
        error: message
      }))
      onStatusMessage?.(`Sim2Real error: ${message}`)
    }
  }

  async function runRouteSequence(mode) {
    const isCaptureMode = mode === 'capture'

    if (replayState.active) {
      const activeMode = replayState.mode
      replayAbortRef.current = true
      setReplayState((current) => ({
        ...current,
        active: false
      }))
      onStatusMessage?.(
        activeMode === 'capture'
          ? 'Sim2Real route capture stopped'
          : 'Sim2Real route replay stopped'
      )
      return
    }

    if (!routePreviewPoses.length) {
      onStatusMessage?.(
        isCaptureMode
          ? 'Sim2Real route capture: no route nodes'
          : 'Sim2Real route replay: no route nodes'
      )
      return
    }

    replayAbortRef.current = false
    setReplayState({
      active: true,
      index: 0,
      total: routePreviewPoses.length,
      mode
    })

    if (isCaptureMode) {
      setCaptureState({
        status: 'capturing',
        bundle: null,
        error: null
      })
      onStatusMessage?.(`Sim2Real route capture started: ${routePreviewPoses.length} nodes`)
    } else {
      onStatusMessage?.(`Sim2Real route replay started: ${routePreviewPoses.length} nodes`)
    }

    try {
      const capturedFrames = []
      const captureStartedAtMs = Date.now()

      for (let index = 0; index < routePreviewPoses.length; index += 1) {
        if (replayAbortRef.current) {
          break
        }

        setReplayState({
          active: true,
          index: index + 1,
          total: routePreviewPoses.length,
          mode
        })

        const labelPrefix = isCaptureMode ? 'Capture' : 'Route'
        const response = await sendRenderRequest(
          routePreviewPoses[index],
          `${labelPrefix} ${index + 1}/${routePreviewPoses.length}`
        )

        if (isCaptureMode) {
          const capturedAt = new Date().toISOString()
          const relativeTimeSeconds = Number(
            (((Date.now() - captureStartedAtMs) || 0) / 1000).toFixed(3)
          )
          capturedFrames.push({
            index,
            label: `${fragmentLabel || fragmentId || 'route'}:${index + 1}`,
            capturedAt,
            relativeTimeSeconds,
            pose: {
              position: [...routePreviewPoses[index].position],
              yawDegrees: Number(routePreviewPoses[index].yawDegrees)
            },
            response
          })
        }

        if (index < routePreviewPoses.length - 1 && !replayAbortRef.current) {
          await sleep(Math.max(50, Number(requestSettings.routeDelayMs) || 250))
        }
      }

      if (!replayAbortRef.current && isCaptureMode) {
        const bundle = buildRouteCaptureBundle({
          fragmentId,
          fragmentLabel,
          endpoint: config.url,
          serverInfo,
          requestSettings,
          routePreviewPoses,
          captures: capturedFrames
        })
        setCaptureState({
          status: 'ready',
          bundle,
          error: null
        })
        onStatusMessage?.(`Sim2Real route capture ready: ${capturedFrames.length} frames`)
      } else if (!replayAbortRef.current) {
        onStatusMessage?.('Sim2Real route replay complete')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setConnectionState((current) => ({
        ...current,
        status: 'error',
        error: message
      }))

      if (isCaptureMode) {
        setCaptureState({
          status: 'error',
          bundle: null,
          error: message
        })
        onStatusMessage?.(`Sim2Real route capture error: ${message}`)
      } else {
        onStatusMessage?.(`Sim2Real route replay error: ${message}`)
      }
    } finally {
      replayAbortRef.current = false
      setReplayState({
        active: false,
        index: 0,
        total: routePreviewPoses.length,
        mode: null
      })
    }
  }

  async function replayRoute() {
    await runRouteSequence('replay')
  }

  async function captureRoute() {
    await runRouteSequence('capture')
  }

  useEffect(() => {
    if (!config.enabled) {
      setConnectionState({
        status: 'disabled',
        error: null
      })
      setServerInfo(null)
      setCaptureState({
        status: 'idle',
        bundle: null,
        error: null
      })
      rejectPendingRequest('sim2real socket disabled')
      socketRef.current?.close(1000, 'sim2real disabled')
      socketRef.current = null
      return
    }

    let disposed = false
    const socket = new WebSocket(config.url)
    socketRef.current = socket
    setConnectionState({
      status: 'connecting',
      error: null
    })

    socket.addEventListener('open', () => {
      if (disposed) {
        return
      }

      setConnectionState({
        status: 'connected',
        error: null
      })
      onStatusMessage?.('Sim2Real socket connected')
    })

    socket.addEventListener('message', (event) => {
      if (disposed) {
        return
      }

      try {
        const message = parseSim2realMessage(event.data)

        if (message.type === 'query-ready') {
          setServerInfo(message)
          if (!serverDefaultsHydratedRef.current && message.defaults) {
            setRequestSettings((current) => ({
              ...current,
              width: Number(message.defaults.width ?? current.width),
              height: Number(message.defaults.height ?? current.height),
              fovDegrees: Number(message.defaults.fovDegrees ?? current.fovDegrees),
              nearClip: Number(message.defaults.nearClip ?? current.nearClip),
              farClip: Number(message.defaults.farClip ?? current.farClip),
              pointRadius: Number(message.defaults.pointRadius ?? current.pointRadius)
            }))
            serverDefaultsHydratedRef.current = true
          }
          return
        }

        if (message.type === 'render-result') {
          const pending = pendingRequestRef.current
          if (!pending || pending.expectedResponseType === 'render-result') {
            if (pending) {
              window.clearTimeout(pending.timeoutId)
              pendingRequestRef.current = null
              setRequestPending(false)
              pending.resolve(message)
            }
          }

          setPreviewState((current) => ({
            status: 'ready',
            label: current.label || 'Render Result',
            error: null,
            ...buildSim2realPreviewFromMessage(message)
          }))
          return
        }

        if (message.type === 'localization-image-benchmark-report') {
          const pending = pendingRequestRef.current

          if (pending && pending.expectedResponseType === 'localization-image-benchmark-report') {
            window.clearTimeout(pending.timeoutId)
            pendingRequestRef.current = null
            setRequestPending(false)
            pending.resolve(message)
          }
          return
        }

        if (message.type === 'error') {
          const errorMessage =
            typeof message.error === 'string' && message.error.trim()
              ? message.error
              : 'sim2real query failed'
          rejectPendingRequest(errorMessage)
          setConnectionState({
            status: 'error',
            error: errorMessage
          })
          setPreviewState((current) => ({
            ...current,
            status: 'error',
            error: errorMessage
          }))
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error)
        rejectPendingRequest(errorMessage)
        setConnectionState({
          status: 'error',
          error: errorMessage
        })
      }
    })

    socket.addEventListener('error', () => {
      if (disposed) {
        return
      }

      setConnectionState({
        status: 'error',
        error: `sim2real websocket に接続できません: ${config.url}`
      })
    })

    socket.addEventListener('close', () => {
      if (socketRef.current === socket) {
        socketRef.current = null
      }

      if (disposed) {
        return
      }

      replayAbortRef.current = true
      rejectPendingRequest('sim2real socket closed')
      setReplayState({
        active: false,
        index: 0,
        total: 0,
        mode: null
      })
      setConnectionState((current) => ({
        status: current.status === 'error' ? 'error' : 'closed',
        error: current.error
      }))
    })

    return () => {
      disposed = true
      replayAbortRef.current = true
      rejectPendingRequest('sim2real socket cleanup')
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      socket.close(1000, 'cleanup')
    }
  }, [config.enabled, config.url, onStatusMessage, reconnectNonce])

  if (!config.enabled) {
    return (
      <div className="state-card sim2real-panel">
        <span className="state-label">Sim2Real Panel</span>
        <strong>Disabled</strong>
        <p className="panel-note">
          browser simulator preview を有効化するには
          {' '}
          <code>?sim2real=1</code>
          {' '}
          か
          {' '}
          <code>?sim2realUrl=ws://127.0.0.1:8781/sim2real</code>
          {' '}
          を使います。
        </p>
        <p className="panel-note">default {sim2realDefaultUrl}</p>
      </div>
    )
  }

  return (
    <div className="state-card sim2real-panel">
      <span className="state-label">Sim2Real Panel</span>
      <strong>{connectionLabel}</strong>
      <p className="panel-note">endpoint {config.url}</p>
      <p className="panel-note">
        backend {serverInfo?.renderer ?? 'pending'}
        {serverInfo?.rendererReason ? ` / ${serverInfo.rendererReason}` : ''}
      </p>
      <p className="panel-note">
        route {routeCount} nodes
        {replayState.total
          ? ` / ${replayState.mode === 'capture' ? 'capture' : 'replay'} ${replayState.index}/${replayState.total}`
          : ''}
      </p>
      {connectionState.error ? (
        <p className="panel-note panel-note-error">{connectionState.error}</p>
      ) : null}

      <div className="field-grid-two sim2real-field-grid">
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-width">
            Width
          </label>
          <input
            id="sim2real-width"
            className="manifest-input"
            min="16"
            onChange={(event) =>
              setRequestSettings((current) => ({
                ...current,
                width: Math.max(16, Number(event.target.value) || current.width)
              }))
            }
            type="number"
            value={requestSettings.width}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-height">
            Height
          </label>
          <input
            id="sim2real-height"
            className="manifest-input"
            min="16"
            onChange={(event) =>
              setRequestSettings((current) => ({
                ...current,
                height: Math.max(16, Number(event.target.value) || current.height)
              }))
            }
            type="number"
            value={requestSettings.height}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-fov">
            FOV
          </label>
          <input
            id="sim2real-fov"
            className="manifest-input"
            min="10"
            onChange={(event) =>
              setRequestSettings((current) => ({
                ...current,
                fovDegrees: Math.max(10, Number(event.target.value) || current.fovDegrees)
              }))
            }
            step="1"
            type="number"
            value={requestSettings.fovDegrees}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-route-delay">
            Route Delay
          </label>
          <input
            id="sim2real-route-delay"
            className="manifest-input"
            min="50"
            onChange={(event) =>
              setRequestSettings((current) => ({
                ...current,
                routeDelayMs: Math.max(50, Number(event.target.value) || current.routeDelayMs)
              }))
            }
            step="25"
            type="number"
            value={requestSettings.routeDelayMs}
          />
        </div>
      </div>

      <div className="button-stack">
        <button
          className="primary-button"
          disabled={connectionState.status !== 'connected' || requestPending || replayState.active}
          onClick={renderCurrentPose}
          type="button">
          Render Current Pose
        </button>
        <button
          className="ghost-button"
          disabled={
            connectionState.status !== 'connected' ||
            requestPending ||
            replayState.active ||
            !robotWaypoint?.position
          }
          onClick={renderWaypointPose}
          type="button">
          Render Waypoint
        </button>
        <button
          className="ghost-button"
          disabled={
            connectionState.status !== 'connected' ||
            (replayState.active && replayState.mode !== 'replay') ||
            (!replayState.active && (requestPending || routePreviewPoses.length === 0))
          }
          onClick={replayRoute}
          type="button">
          {replayState.active && replayState.mode === 'replay'
            ? 'Stop Route Replay'
            : 'Replay Route'}
        </button>
        <button
          className="ghost-button"
          disabled={
            connectionState.status !== 'connected' ||
            (replayState.active && replayState.mode !== 'capture') ||
            (!replayState.active && (requestPending || routePreviewPoses.length === 0))
          }
          onClick={captureRoute}
          type="button">
          {replayState.active && replayState.mode === 'capture'
            ? 'Stop Route Capture'
            : 'Capture Route Bundle'}
        </button>
        <button
          className="ghost-button"
          disabled={requestPending}
          onClick={reconnect}
          type="button">
          Reconnect Simulator
        </button>
      </div>

      <div className="sim2real-preview-grid">
        <figure className="sim2real-preview-frame">
          <figcaption>RGB</figcaption>
          {previewState.colorSrc ? (
            <img alt="Sim2Real RGB Preview" src={previewState.colorSrc} />
          ) : (
            <div className="sim2real-preview-empty">No frame yet</div>
          )}
        </figure>
        <figure className="sim2real-preview-frame">
          <figcaption>Depth</figcaption>
          {previewState.depthSrc ? (
            <img alt="Sim2Real Depth Preview" src={previewState.depthSrc} />
          ) : (
            <div className="sim2real-preview-empty">No depth yet</div>
          )}
        </figure>
      </div>

      <p className="panel-note">
        {previewState.label}
        {' / '}
        {previewState.width > 0 && previewState.height > 0
          ? `${previewState.width} x ${previewState.height}`
          : 'idle'}
      </p>
      <p className="panel-note">
        frame {previewState.frameId || serverInfo?.frameId || 'dreamwalker_map'}
        {' / '}
        depth {previewState.depthMin === null
          ? 'n/a'
          : `${previewState.depthMin.toFixed(2)}..${previewState.depthMax.toFixed(2)} m`}
      </p>
      {previewState.error ? (
        <p className="panel-note panel-note-error">{previewState.error}</p>
      ) : null}

      <div className="sim2real-capture-card">
        <span className="state-label">Route Capture Bundle</span>
        <strong>
          {captureState.status === 'ready'
            ? `${captureCount} frames`
            : captureState.status === 'capturing'
              ? 'Capturing'
              : captureState.status === 'error'
                ? 'Error'
                : 'Idle'}
        </strong>
        <p className="panel-note">
          {captureState.bundle
            ? `${captureState.bundle.fragmentId || 'fragment'} / ${captureState.bundle.request.width} x ${captureState.bundle.request.height}`
            : 'route replay の render-result を JSON bundle として保存します。'}
        </p>
        <p className="panel-note">
          {captureState.bundle
            ? `captured ${captureState.bundle.capturedAt} / endpoint ${captureState.bundle.endpoint}`
            : 'RGB JPEG, depth base64, cameraInfo, pose をそのまま保持します。'}
        </p>
        {captureState.error ? (
          <p className="panel-note panel-note-error">{captureState.error}</p>
        ) : null}
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-capture-shelf-label">
            Capture Snapshot Label
          </label>
          <input
            id="sim2real-capture-shelf-label"
            className="manifest-input"
            onChange={(event) => {
              setCaptureShelfLabel(event.target.value)
              setCaptureImportError('')
            }}
            placeholder={`${fragmentLabel || fragmentId || 'Capture'} / ${captureCount || routeCount} frames`}
            type="text"
            value={captureShelfLabel}
          />
        </div>
        <div className="button-stack">
          <button
            className="primary-button"
            disabled={!captureState.bundle}
            onClick={saveCaptureSnapshot}
            type="button">
            Save Capture Snapshot
          </button>
          <button
            className="ghost-button"
            disabled={!captureState.bundle}
            onClick={downloadCaptureBundle}
            type="button">
            Download Capture JSON
          </button>
          <button
            className="ghost-button"
            disabled={!captureState.bundle}
            onClick={clearCaptureBundle}
            type="button">
            Clear Capture
          </button>
          <button
            className="ghost-button"
            onClick={() => captureFileInputRef.current?.click()}
            type="button">
            Import Capture File
          </button>
          <input
            id="sim2real-capture-import"
            ref={captureFileInputRef}
            accept="application/json,.json"
            className="manifest-file-input"
            onChange={handleCaptureFileImport}
            type="file"
          />
          <button
            className="ghost-button"
            disabled={captureShelf.length === 0}
            onClick={clearCaptureShelf}
            type="button">
            Clear Capture Shelf
          </button>
        </div>
        {captureImportError ? (
          <p className="panel-note panel-note-error">
            Shelf Error: {captureImportError}
          </p>
        ) : (
          <p className="panel-note">
            localStorage shelf には大きい route capture を大量保存できません。長い route は download JSON を優先します。
          </p>
        )}
        {captureShelf.length > 0 ? (
          <div className="state-list">
            {captureShelf.map((entry) => (
              <div key={entry.id} className="state-card">
                <div className="status-row">
                  <strong>{entry.label}</strong>
                  <div className="status-row-badges">
                    <span className="chip">
                      {entry.bundle.fragmentLabel || entry.bundle.fragmentId || 'Capture'}
                    </span>
                    <span className="chip">
                      {entry.bundle.captures.length} frames
                    </span>
                  </div>
                </div>
                <p className="panel-note">
                  {entry.bundle.request.width || 'n/a'} x {entry.bundle.request.height || 'n/a'}
                  {' / '}
                  route {Array.isArray(entry.bundle.route) ? entry.bundle.route.length : 0} nodes
                </p>
                <p className="panel-note">
                  captured {entry.bundle.capturedAt} / saved {entry.savedAt}
                </p>
                <div className="button-stack">
                  <button
                    className="primary-button"
                    onClick={() => previewCaptureShelfEntry(entry)}
                    type="button">
                    Preview Last Frame
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => downloadCaptureShelfEntry(entry)}
                    type="button">
                    Download Bundle JSON
                  </button>
                  <button
                    className="ghost-button"
                    onClick={() => removeCaptureShelfEntry(entry.id)}
                    type="button">
                    Remove From Shelf
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      <div className="sim2real-benchmark-card">
        <span className="state-label">Localization Benchmark</span>
        <strong>
          {localizationBenchmark.report
            ? `${localizationBenchmark.report.matching.matchedCount} matched poses`
            : activeLocalizationEstimate
              ? estimateSourceMode === 'live'
                ? 'Live Estimate Ready'
                : estimateSourceMode === 'imported'
                  ? 'Imported Estimate Ready'
                  : 'Estimate Ready'
              : 'Idle'}
        </strong>
        <p className="panel-note">
          ground truth {activeBenchmarkSource?.label || 'none'}
        </p>
        <p className="panel-note">
          {activeLocalizationEstimate
            ? `estimate ${activeLocalizationEstimate.label} / ${activeLocalizationEstimate.poses.length} poses / source ${estimateSourceMode === 'auto' ? (liveLocalizationEstimate ? 'auto-live' : importedLocalizationEstimate ? 'auto-imported' : 'auto') : estimateSourceMode}`
            : 'estimated trajectory JSON を import するか live monitor を接続して、capture bundle と比較しながら ATE / RPE を更新します。'}
        </p>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-benchmark-ground-truth">
            Ground Truth Source
          </label>
          <select
            id="sim2real-benchmark-ground-truth"
            className="manifest-input"
            disabled={benchmarkGroundTruthOptions.length === 0}
            onChange={(event) => setBenchmarkSourceId(event.target.value)}
            value={activeBenchmarkSource?.id || 'current-capture'}>
            {benchmarkGroundTruthOptions.length > 0 ? (
              benchmarkGroundTruthOptions.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.label}
                </option>
              ))
            ) : (
              <option value="current-capture">No capture bundles yet</option>
            )}
          </select>
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-benchmark-estimate-source">
            Estimate Source
          </label>
          <select
            id="sim2real-benchmark-estimate-source"
            className="manifest-input"
            onChange={(event) => setEstimateSourceMode(event.target.value)}
            value={estimateSourceMode}>
            <option value="auto">Auto</option>
            <option value="imported">Imported</option>
            <option value="live">Live</option>
          </select>
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-benchmark-alignment">
            Alignment
          </label>
          <select
            id="sim2real-benchmark-alignment"
            className="manifest-input"
            onChange={(event) => setBenchmarkAlignmentMode(event.target.value)}
            value={benchmarkAlignmentMode}>
            <option value="auto">Auto</option>
            <option value="index">Index</option>
            <option value="timestamp">Timestamp</option>
          </select>
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="sim2real-live-estimate-url">
            Live Estimate Socket
          </label>
          <input
            id="sim2real-live-estimate-url"
            className="manifest-input"
            onChange={(event) =>
              setLiveEstimateMonitorState((current) => ({
                ...current,
                url: event.target.value,
                error: null
              }))
            }
            placeholder={localizationMonitorDefaultUrl}
            type="text"
            value={liveEstimateMonitorState.url}
          />
        </div>
        <div className="button-stack">
          <button
            className="primary-button"
            disabled={benchmarkGroundTruthOptions.length === 0}
            onClick={() => localizationFileInputRef.current?.click()}
            type="button">
            Import Estimate File
          </button>
          <input
            id="sim2real-benchmark-import"
            ref={localizationFileInputRef}
            accept="application/json,.json,text/plain,.txt,.log,.traj,.tum"
            className="manifest-file-input"
            onChange={handleLocalizationEstimateImport}
            type="file"
          />
          <button
            className="ghost-button"
            onClick={connectLiveEstimateMonitor}
            type="button">
            {liveEstimateMonitorState.status === 'connected'
              ? 'Reconnect Live Monitor'
              : 'Connect Live Monitor'}
          </button>
          <button
            className="ghost-button"
            disabled={!liveEstimateMonitorState.enabled && liveEstimateMonitorState.status !== 'connected'}
            onClick={disconnectLiveEstimateMonitor}
            type="button">
            Disconnect Live Monitor
          </button>
          <button
            className="ghost-button"
            disabled={!importedLocalizationEstimate}
            onClick={clearImportedLocalizationEstimate}
            type="button">
            Clear Imported Estimate
          </button>
          <button
            className="ghost-button"
            disabled={!liveLocalizationEstimate}
            onClick={clearLiveLocalizationEstimate}
            type="button">
            Clear Live Estimate
          </button>
          <button
            className="ghost-button"
            disabled={!localizationBenchmark.report}
            onClick={downloadLocalizationBenchmarkReport}
            type="button">
            Download Benchmark Report
          </button>
        </div>
        {localizationImportError ? (
          <p className="panel-note panel-note-error">
            Estimate Error: {localizationImportError}
          </p>
        ) : localizationBenchmark.error ? (
          <p className="panel-note panel-note-error">
            Benchmark Error: {localizationBenchmark.error}
          </p>
        ) : liveEstimateMonitorState.error ? (
          <p className="panel-note panel-note-error">
            Live Monitor Error: {liveEstimateMonitorState.error}
          </p>
        ) : benchmarkGroundTruthOptions.length === 0 ? (
          <p className="panel-note">
            先に route capture か capture shelf entry を用意すると、ここを ground truth source として使えます。
          </p>
        ) : (
          <p className="panel-note">
            route-capture bundle, pose array, trajectory JSON, quaternion orientation を含む pose list, TUM / ORB-SLAM style text trajectory を import できます。live socket は `pose-estimate` と `localization-estimate` JSON を受けます。
          </p>
        )}
        <p className="panel-note">
          live monitor {liveEstimateStatusLabel}
          {' / '}
          {liveEstimateMonitorState.label || 'no stream label'}
          {' / '}
          messages {liveEstimateMonitorState.messageCount}
          {liveEstimateMonitorState.lastMessageAt
            ? ` / last ${liveEstimateMonitorState.lastMessageAt}`
            : ''}
        </p>
        <div className="state-card">
          <span className="state-label">Benchmark Run Shelf</span>
          <strong>
            {localizationRunShelf.length > 0
              ? `${localizationRunShelf.length} saved runs`
              : 'No saved runs'}
          </strong>
          <p className="panel-note">
            current benchmark の estimate と metrics を snapshot として保存します。ground truth は `current-capture` のときだけ bundle を内包し、capture shelf source は参照で持ちます。
          </p>
          <div className="field-group">
            <label className="field-label" htmlFor="sim2real-localization-run-shelf-label">
              Run Snapshot Label
            </label>
            <input
              id="sim2real-localization-run-shelf-label"
              className="manifest-input"
              onChange={(event) => {
                setLocalizationRunShelfLabel(event.target.value)
                setLocalizationRunShelfError('')
              }}
              placeholder={
                activeLocalizationEstimate
                  ? `${activeLocalizationEstimate.label} / ATE ${formatMeters(localizationBenchmark.report?.metrics.ateRmseMeters)}`
                  : 'ORB-SLAM3 Live / ATE 0.000 m'
              }
              type="text"
              value={localizationRunShelfLabel}
            />
          </div>
          <div className="button-stack">
            <button
              className="primary-button"
              disabled={!localizationBenchmark.report || !activeLocalizationEstimate}
              onClick={saveLocalizationRunSnapshot}
              type="button">
              Save Benchmark Run
            </button>
            <button
              className="ghost-button"
              onClick={() => reviewBundleFileInputRef.current?.click()}
              type="button">
              Import Review Bundle
            </button>
            <button
              className="ghost-button"
              disabled={localizationRunShelf.length === 0}
              onClick={() => imageBenchmarkFileInputRef.current?.click()}
              type="button">
              Import Image Benchmark Report
            </button>
            <button
              className="ghost-button"
              disabled={
                localizationRunShelf.length === 0 ||
                localizationRunComparison?.missingImageBenchmarkCount === 0 ||
                connectionState.status !== 'connected' ||
                requestPending ||
                imageBenchmarkBatchState.active
              }
              onClick={() => runLocalizationImageBenchmarkBatch('missing')}
              type="button">
              {imageBenchmarkBatchState.active && imageBenchmarkBatchState.mode === 'missing'
                ? 'Running Missing Benchmarks...'
                : 'Run Missing Image Benchmarks'}
            </button>
            <button
              className="ghost-button"
              disabled={
                localizationRunShelf.length === 0 ||
                connectionState.status !== 'connected' ||
                requestPending ||
                imageBenchmarkBatchState.active
              }
              onClick={() => runLocalizationImageBenchmarkBatch('all')}
              type="button">
              {imageBenchmarkBatchState.active && imageBenchmarkBatchState.mode === 'all'
                ? 'Refreshing All Benchmarks...'
                : 'Refresh All Image Benchmarks'}
            </button>
            <input
              id="sim2real-image-benchmark-import"
              ref={imageBenchmarkFileInputRef}
              accept="application/json,.json"
              className="manifest-file-input"
              onChange={handleLocalizationImageBenchmarkImport}
              type="file"
            />
            <input
              id="sim2real-review-bundle-import"
              ref={reviewBundleFileInputRef}
              accept="application/json,.json"
              className="manifest-file-input"
              onChange={handleLocalizationReviewBundleImport}
              type="file"
            />
            <button
              className="ghost-button"
              disabled={localizationRunShelf.length === 0}
              onClick={clearLocalizationRunShelf}
              type="button">
              Clear Run Shelf
            </button>
          </div>
          <p className="panel-note">
            review bundle import は linked capture, saved runs, baseline compare state をまとめて復元します。
          </p>
          {localizationRunShelfError ? (
            <p className="panel-note panel-note-error">
              Run Shelf Error: {localizationRunShelfError}
            </p>
          ) : (
            <p className="panel-note">
              run shelf には benchmark summary と estimate を優先して保存します。portable JSON export は対応する capture bundle が残っていれば ground truth も含めます。query server が connected なら各 run から直接 image benchmark を実行できます。
            </p>
          )}
          {imageBenchmarkBatchState.active ? (
            <p className="panel-note">
              image benchmark batch {formatImageBenchmarkBatchMode(imageBenchmarkBatchState.mode)}
              {' / '}
              {imageBenchmarkBatchState.completed + imageBenchmarkBatchState.failed}
              {' / '}
              {imageBenchmarkBatchState.total}
              {imageBenchmarkBatchState.currentLabel
                ? ` / current ${imageBenchmarkBatchState.currentLabel}`
                : ''}
            </p>
          ) : localizationRunComparison ? (
            <p className="panel-note">
              image metrics {localizationRunComparison.runsWithImageBenchmarkCount}
              {' / '}
              missing {localizationRunComparison.missingImageBenchmarkCount}
            </p>
          ) : null}
          {localizationRunComparison ? (
            <div className="state-card sim2real-run-compare-card">
              <span className="state-label">Run Compare</span>
              <strong>{localizationRunComparison.rows.length} runs ranked by ATE</strong>
              <div className="field-group">
                <label className="field-label" htmlFor="sim2real-run-compare-baseline">
                  Baseline Run
                </label>
                <select
                  id="sim2real-run-compare-baseline"
                  className="manifest-input"
                  onChange={(event) => setBaselineLocalizationRunId(event.target.value)}
                  value={localizationRunComparison.baselineRun?.id || ''}>
                  {localizationRunComparison.rows.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </div>
              <p className="panel-note">
                best ATE {localizationRunComparison.bestAteRun?.label || 'n/a'}
                {' / '}
                {formatMeters(localizationRunComparison.bestAteRun?.summary.ateRmseMeters)}
                {' / '}
                best yaw {localizationRunComparison.bestYawRun?.label || 'n/a'}
                {' / '}
                {formatDegrees(localizationRunComparison.bestYawRun?.summary.yawRmseDegrees)}
              </p>
              <p className="panel-note">
                latest {localizationRunComparison.latestRun?.label || 'n/a'}
                {localizationRunComparison.ateSpreadMeters !== null
                  ? ` / ate spread ${formatMeters(localizationRunComparison.ateSpreadMeters)}`
                  : ''}
                {' / '}
                best LPIPS {localizationRunComparison.bestLpipsRun?.label || 'n/a'}
                {' / '}
                {formatLpips(localizationRunComparison.bestLpipsRun?.imageBenchmark?.summary?.lpipsMean)}
                {localizationRunComparison.lpipsSpread !== null
                  ? ` / lpips spread ${formatLpips(localizationRunComparison.lpipsSpread)}`
                  : ''}
              </p>
              <p className="panel-note">
                baseline {localizationRunComparison.baselineRun?.label || 'n/a'}
                {' / '}
                delta columns show run minus baseline
              </p>
              <div className="button-stack">
                <button
                  className="ghost-button"
                  onClick={downloadLocalizationRunCompareJson}
                  type="button">
                  Download Compare JSON
                </button>
                <button
                  className="ghost-button"
                  onClick={downloadLocalizationRunCompareCsv}
                  type="button">
                  Download Compare CSV
                </button>
                <button
                  className="ghost-button"
                  onClick={downloadLocalizationRunCompareMarkdown}
                  type="button">
                  Download Compare Markdown
                </button>
                <button
                  className="ghost-button"
                  onClick={downloadLocalizationReviewBundle}
                  type="button">
                  Download Review Bundle
                </button>
              </div>
              <div className="sim2real-run-compare-table" role="table" aria-label="Localization run comparison">
                <div className="sim2real-run-compare-row sim2real-run-compare-header" role="row">
                  <span>#</span>
                  <span>Run</span>
                  <span>ATE</span>
                  <span>ΔATE</span>
                  <span>Yaw</span>
                  <span>ΔYaw</span>
                  <span>LPIPS</span>
                  <span>ΔLPIPS</span>
                  <span>Matched</span>
                  <span>Source</span>
                </div>
                {localizationRunComparison.rows.map((entry) => (
                  <div
                    key={entry.id}
                    className={`sim2real-run-compare-row${entry.isActive ? ' sim2real-run-compare-row-active' : ''}`}
                    role="row">
                    <span className="sim2real-run-compare-rank">#{entry.rank}</span>
                    <span className="sim2real-run-compare-run">
                      {entry.label}
                      <span className="status-row-badges sim2real-run-compare-badges">
                        {entry.isBaseline ? <span className="chip active">Baseline</span> : null}
                        {entry.isBestAte ? <span className="chip">Best ATE</span> : null}
                        {entry.isBestYaw ? <span className="chip">Best Yaw</span> : null}
                        {entry.isBestLpips ? <span className="chip">Best LPIPS</span> : null}
                        {entry.isLatest ? <span className="chip">Latest</span> : null}
                        {entry.isActive ? <span className="chip">Active</span> : null}
                      </span>
                    </span>
                    <span>{formatMeters(entry.summary.ateRmseMeters)}</span>
                    <span
                      className={`sim2real-run-compare-delta sim2real-run-compare-delta-${classifyMetricDelta(entry.ateDeltaMeters)}`}>
                      {formatSignedMetric(entry.ateDeltaMeters, formatMeters, '0.000 m')}
                    </span>
                    <span>{formatDegrees(entry.summary.yawRmseDegrees)}</span>
                    <span
                      className={`sim2real-run-compare-delta sim2real-run-compare-delta-${classifyMetricDelta(entry.yawDeltaDegrees)}`}>
                      {formatSignedMetric(entry.yawDeltaDegrees, formatDegrees, '0.00 deg')}
                    </span>
                    <span>{formatLpips(entry.lpipsMean)}</span>
                    <span
                      className={`sim2real-run-compare-delta sim2real-run-compare-delta-${classifyMetricDelta(entry.lpipsDelta)}`}>
                      {formatSignedMetric(entry.lpipsDelta, formatLpips, '0.000')}
                    </span>
                    <span>{entry.summary.matchedCount}</span>
                    <span>{entry.summary.sourceType}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {imageBenchmarkPreview ? (
            <div className="state-card sim2real-run-image-preview-card">
              <span className="state-label">Worst LPIPS Preview</span>
              <strong>{imageBenchmarkPreview.runLabel}</strong>
              <p className="panel-note">
                LPIPS {formatLpips(imageBenchmarkPreview.value)}
                {' / '}
                frame {imageBenchmarkPreview.frameIndex + 1}
                {' / '}
                gt {imageBenchmarkPreview.groundTruthLabel || 'Ground Truth'}
              </p>
              <p className="panel-note">
                estimate {imageBenchmarkPreview.estimateLabel || 'Rendered Estimate'}
              </p>
              <div className="sim2real-preview-grid">
                <figure className="sim2real-preview-frame">
                  <figcaption>Ground Truth</figcaption>
                  <img alt="Localization Ground Truth Preview" src={imageBenchmarkPreview.groundTruthSrc} />
                </figure>
                <figure className="sim2real-preview-frame">
                  <figcaption>Rendered Estimate</figcaption>
                  <img alt="Localization Rendered Preview" src={imageBenchmarkPreview.renderedSrc} />
                </figure>
              </div>
              <div className="button-stack">
                <button
                  className="ghost-button"
                  onClick={() => setImageBenchmarkPreview(null)}
                  type="button">
                  Clear Image Preview
                </button>
              </div>
            </div>
          ) : null}
          {localizationRunShelf.length > 0 ? (
            <div className="state-list">
              {localizationRunShelf.map((entry) => (
                <div key={entry.id} className="state-card">
                  <div className="status-row">
                    <strong>{entry.label}</strong>
                    <div className="status-row-badges">
                      <span className="chip">{entry.summary.sourceType}</span>
                      <span className="chip">{entry.summary.matchedCount} matched</span>
                    </div>
                  </div>
                  <p className="panel-note">
                    ATE {formatMeters(entry.summary.ateRmseMeters)}
                    {' / '}
                    yaw {formatDegrees(entry.summary.yawRmseDegrees)}
                    {' / '}
                    estimate {entry.summary.estimatePoseCount}
                  </p>
                  <p className="panel-note">
                    gt {entry.summary.groundTruthLabel}
                    {' / '}
                    alignment {entry.summary.alignment}
                    {entry.summary.interpolationMode === 'linear'
                      ? ` / interpolated ${entry.summary.interpolatedCount}`
                      : ''}
                  </p>
                  <p className="panel-note">
                    saved {entry.savedAt}
                    {entry.summary.timeDeltaMeanSeconds !== null
                      ? ` / time delta ${formatSeconds(entry.summary.timeDeltaMeanSeconds)}`
                      : ''}
                  </p>
                  {entry.imageBenchmark ? (
                    <>
                      <p className="panel-note">
                        image lpips {formatLpips(entry.imageBenchmark.summary.lpipsMean)}
                        {' / '}
                        psnr {formatDb(entry.imageBenchmark.summary.psnrMean)}
                        {' / '}
                        ssim {formatLpips(entry.imageBenchmark.summary.ssimMean)}
                      </p>
                      <p className="panel-note">
                        worst lpips {formatLpips(entry.imageBenchmark.highlights.lpips?.value)}
                        {entry.imageBenchmark.highlights.lpips
                          ? ` / frame ${entry.imageBenchmark.highlights.lpips.frameIndex + 1}`
                          : ''}
                      </p>
                    </>
                  ) : (
                    <p className="panel-note">
                      image benchmark report を attach すると LPIPS / PSNR / SSIM を compare に追加できます。
                    </p>
                  )}
                  <div className="button-stack">
                    <button
                      className="primary-button"
                      onClick={() => loadLocalizationRunSnapshot(entry)}
                      type="button">
                      Load Run
                    </button>
                    <button
                      className="ghost-button"
                      disabled={
                        connectionState.status !== 'connected' ||
                        requestPending ||
                        imageBenchmarkBatchState.active
                      }
                      onClick={() => runLocalizationImageBenchmark(entry)}
                      type="button">
                      {imageBenchmarkRequestEntryId === entry.id
                        ? 'Running Image Benchmark...'
                        : 'Run Image Benchmark'}
                    </button>
                    <button
                      className="ghost-button"
                      disabled={!entry.imageBenchmark?.highlights?.lpips}
                      onClick={() => previewLocalizationImageBenchmark(entry, 'lpips')}
                      type="button">
                      Preview Worst LPIPS
                    </button>
                    <button
                      className="ghost-button"
                      disabled={imageBenchmarkBatchState.active}
                      onClick={() => downloadLocalizationRunSnapshot(entry)}
                      type="button">
                      Download Run JSON
                    </button>
                    <button
                      className="ghost-button"
                      disabled={!entry.imageBenchmark}
                      onClick={() => clearLocalizationImageBenchmark(entry.id)}
                      type="button">
                      Clear Image Metrics
                    </button>
                    <button
                      className="ghost-button"
                      disabled={imageBenchmarkBatchState.active}
                      onClick={() => removeLocalizationRunSnapshot(entry.id)}
                      type="button">
                      Remove Run
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
        {localizationBenchmark.report ? (
          <>
            <div className="state-grid sim2real-metric-grid">
              <div className="state-card">
                <span className="state-label">ATE RMSE</span>
                <strong>{formatMeters(localizationBenchmark.report.metrics.ateRmseMeters)}</strong>
                <p className="panel-note">
                  mean {formatMeters(localizationBenchmark.report.metrics.translation?.mean)}
                  {' / '}
                  max {formatMeters(localizationBenchmark.report.metrics.translation?.max)}
                </p>
              </div>
              <div className="state-card">
                <span className="state-label">Yaw RMSE</span>
                <strong>{formatDegrees(localizationBenchmark.report.metrics.yaw?.rmse)}</strong>
                <p className="panel-note">
                  mean {formatDegrees(localizationBenchmark.report.metrics.yaw?.mean)}
                  {' / '}
                  max {formatDegrees(localizationBenchmark.report.metrics.yaw?.max)}
                </p>
              </div>
              <div className="state-card">
                <span className="state-label">RPE Translation</span>
                <strong>
                  {formatMeters(localizationBenchmark.report.metrics.rpeTranslationRmseMeters)}
                </strong>
                <p className="panel-note">
                  segments {localizationBenchmark.report.metrics.rpeTranslation?.max !== undefined
                    ? localizationBenchmark.report.matching.matchedCount - 1
                    : 0}
                </p>
              </div>
              <div className="state-card">
                <span className="state-label">RPE Yaw</span>
                <strong>{formatDegrees(localizationBenchmark.report.metrics.rpeYawRmseDegrees)}</strong>
                <p className="panel-note">
                  alignment {localizationBenchmark.report.alignment}
                  {localizationBenchmark.report.estimate.interpolationMode === 'linear'
                    ? ' / interpolation linear'
                    : ''}
                </p>
              </div>
            </div>
            <p className="panel-note">
              matching {localizationBenchmark.report.matching.matchedCount}
              {' / '}
              gt {localizationBenchmark.report.matching.groundTruthCount}
              {' / '}
              estimate {localizationBenchmark.report.matching.estimateCount}
              {localizationBenchmark.report.matching.groundTruthRemainderCount > 0
                ? ` / gt remainder ${localizationBenchmark.report.matching.groundTruthRemainderCount}`
                : ''}
              {localizationBenchmark.report.matching.estimateRemainderCount > 0
                ? ` / estimate remainder ${localizationBenchmark.report.matching.estimateRemainderCount}`
                : ''}
              {localizationBenchmark.report.matching.interpolatedCount > 0
                ? ` / interpolated ${localizationBenchmark.report.matching.interpolatedCount}`
                : ''}
              {localizationBenchmark.report.matching.clampedCount > 0
                ? ` / clamped ${localizationBenchmark.report.matching.clampedCount}`
                : ''}
            </p>
            {localizationBenchmark.report.metrics.timeDelta ? (
              <p className="panel-note">
                time delta mean {formatSeconds(localizationBenchmark.report.metrics.timeDelta.mean)}
                {' / '}
                max {formatSeconds(localizationBenchmark.report.metrics.timeDelta.max)}
              </p>
            ) : null}
            {largestBenchmarkTranslationSample ? (
              <p className="panel-note">
                worst sample {largestBenchmarkTranslationSample.index + 1}
                {' / '}
                position {formatMeters(largestBenchmarkTranslationSample.translationErrorMeters)}
                {' / '}
                yaw {formatDegrees(largestBenchmarkTranslationSample.yawErrorDegrees)}
              </p>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  )
}
