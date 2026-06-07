const absoluteUrlPattern = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i;

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function normalizeBasePath(value) {
  if (!hasNonEmptyString(value)) {
    return '/';
  }

  const basePath = value.trim();
  if (basePath === '/') {
    return '/';
  }

  return basePath.endsWith('/') ? basePath : `${basePath}/`;
}

export function dreamwalkerPublicBasePath() {
  return normalizeBasePath(import.meta.env?.BASE_URL ?? '/');
}

export function resolvePublicUrl(value, options = {}) {
  if (!hasNonEmptyString(value)) {
    return '';
  }

  const rawUrl = value.trim();
  if (absoluteUrlPattern.test(rawUrl) || !rawUrl.startsWith('/')) {
    return rawUrl;
  }

  const basePath = normalizeBasePath(options.basePath ?? dreamwalkerPublicBasePath());
  const basePathPrefix = basePath.replace(/\/$/, '');
  if (basePath === '/' || rawUrl === basePathPrefix || rawUrl.startsWith(basePath)) {
    return rawUrl;
  }

  return `${basePathPrefix}${rawUrl}`;
}
