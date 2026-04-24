import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MetricType } from '@zilliz/milvus2-sdk-node';
import { getEmbedding, validateEmbedding } from '../embedding-client.mjs';
import { retrieve } from '../rag/retriever.mjs';
import { detectQuestionStrategy } from '../rag5/context-builder.mjs';
import { denoiseMatches } from '../rag4/rag-pipeline-v4.mjs';

const CURRENT_DIR = path.dirname(fileURLToPath(import.meta.url));
const KEYWORD_HINTS_PATH = path.join(CURRENT_DIR, '../rag5/config/keyword-hints.json');
const SECTION_BOOST_CONFIG_PATH = path.join(CURRENT_DIR, '../rag5/config/section-boost-config.json');
const RERANK_CONFIG_PATH = path.join(CURRENT_DIR, '../rag5/config/rerank-config.json');
const PRECISION_CONFIG_PATH = path.join(CURRENT_DIR, '../rag6/config/precision-config.json');
const SELECTION_CONFIG_PATH = path.join(CURRENT_DIR, 'config/selection-config.json');

let keywordHintsCache = null;
let sectionBoostConfigCache = null;
let rerankConfigCache = null;
let precisionConfigCache = null;
let selectionConfigCache = null;

function noop() {}

function getCollectionVectorDim(detail) {
  const dim = detail?.schema?.fields?.find((field) => field.name === 'vector')?.dim;
  return Number(dim || 0);
}

function normalizeText(value) {
  return String(value || '').toLowerCase();
}

async function loadKeywordHintsConfig(noCache = false) {
  if (!noCache && keywordHintsCache) {
    return keywordHintsCache;
  }

  const raw = await fs.readFile(KEYWORD_HINTS_PATH, 'utf8');
  const parsed = JSON.parse(raw);

  if (!noCache) {
    keywordHintsCache = parsed;
  }

  return parsed;
}

async function loadSectionBoostConfig(noCache = false) {
  if (!noCache && sectionBoostConfigCache) {
    return sectionBoostConfigCache;
  }

  const raw = await fs.readFile(SECTION_BOOST_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);

  if (!noCache) {
    sectionBoostConfigCache = parsed;
  }

  return parsed;
}

async function loadRerankConfig(noCache = false) {
  if (!noCache && rerankConfigCache) {
    return rerankConfigCache;
  }

  const raw = await fs.readFile(RERANK_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);

  if (!noCache) {
    rerankConfigCache = parsed;
  }

  return parsed;
}

async function loadPrecisionConfig(noCache = false) {
  if (!noCache && precisionConfigCache) {
    return precisionConfigCache;
  }

  const raw = await fs.readFile(PRECISION_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);

  if (!noCache) {
    precisionConfigCache = parsed;
  }

  return parsed;
}

async function loadSelectionConfig(noCache = false) {
  if (!noCache && selectionConfigCache) {
    return selectionConfigCache;
  }

  const raw = await fs.readFile(SELECTION_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);

  if (!noCache) {
    selectionConfigCache = parsed;
  }

  return parsed;
}

async function getKeywordHints(question, noCache = false) {
  const normalized = normalizeText(question);
  const keywordHintsConfig = await loadKeywordHintsConfig(noCache);
  const result = new Set();

  for (const [trigger, expansions] of Object.entries(keywordHintsConfig)) {
    if (normalized.includes(normalizeText(trigger))) {
      expansions.forEach((item) => result.add(item));
    }
  }

  return [...result];
}

async function extractQuestionKeywords(question, noCache = false) {
  const normalized = normalizeText(question);
  const keywordHintsConfig = await loadKeywordHintsConfig(noCache);
  const keywords = new Set();

  for (const [trigger, expansions] of Object.entries(keywordHintsConfig)) {
    const normalizedTrigger = normalizeText(trigger);

    if (!normalized.includes(normalizedTrigger)) {
      continue;
    }

    keywords.add(normalizedTrigger);
    expansions.forEach((item) => keywords.add(normalizeText(item)));
  }

  return [...keywords];
}

function computeTopicAlignment(item, questionKeywords) {
  const haystack = [
    item.section,
    item.subsection_key,
    item.subsection_title,
    item.entity_type,
    item.content,
    ...(Array.isArray(item.tags) ? item.tags : []),
  ]
    .map((part) => normalizeText(part))
    .join('\n');

  const topicHit = questionKeywords.some((keyword) => haystack.includes(keyword));
  return {
    topicHit,
  };
}

async function scoreSectionBoost(item, strategy, topicHit, noCache = false) {
  const sectionBoostConfig = await loadSectionBoostConfig(noCache);
  const precisionConfig = await loadPrecisionConfig(noCache);
  const strategyConfig = sectionBoostConfig?.[strategy];

  if (!strategyConfig) {
    return 0;
  }

  const sectionConfig = strategyConfig?.[item.section] || strategyConfig?.__default;

  if (!sectionConfig) {
    return 0;
  }

  let baseBoost = Number(sectionConfig.default || 0);

  if (
    String(item.entity_type || item.entityType || '').includes('summary') &&
    typeof sectionConfig.summary === 'number'
  ) {
    baseBoost = sectionConfig.summary;
  }

  const precision = precisionConfig?.[strategy] || precisionConfig?.general || {};

  if (
    (strategy === 'experience' || strategy === 'project') &&
    (item.section === 'projects' || item.section === 'work_experience') &&
    !topicHit
  ) {
    return baseBoost * Number(precision.sectionBoostAttenuationWithoutTopicHit || 0.35);
  }

  return baseBoost;
}

async function scoreKeywordBoost(item, keywordHints, noCache = false) {
  const rerankConfig = await loadRerankConfig(noCache);
  const haystack = [
    item.section,
    item.subsection_key,
    item.subsection_title,
    item.entity_type,
    item.content,
    ...(Array.isArray(item.tags) ? item.tags : []),
  ]
    .map((part) => normalizeText(part))
    .join('\n');

  let boost = 0;
  let matchedHintCount = 0;

  for (const keyword of keywordHints) {
    if (!keyword) continue;
    if (haystack.includes(normalizeText(keyword))) {
      matchedHintCount += 1;
      boost += Number(rerankConfig.keywordBoostPerHit || 0.015);
    }
  }

  return {
    matchedHintCount,
    boost: Math.min(boost, Number(rerankConfig.keywordBoostMax || 0.09)),
  };
}

async function rerankMatches(matches, question, strategy, noCache = false) {
  const keywordHints = await getKeywordHints(question, noCache);
  const questionKeywords = await extractQuestionKeywords(question, noCache);

  const scored = await Promise.all(
    matches.map(async (item, index) => {
      const baseScore = Number(item.score || 0);
      const { topicHit } = computeTopicAlignment(item, questionKeywords);
      const sectionBoost = await scoreSectionBoost(item, strategy, topicHit, noCache);
      const { boost: keywordBoost, matchedHintCount } = await scoreKeywordBoost(item, keywordHints, noCache);
      const rerankScore = baseScore + sectionBoost + keywordBoost;

      return {
        ...item,
        _rawIndex: index,
        _baseScore: baseScore,
        _sectionBoost: sectionBoost,
        _keywordBoost: keywordBoost,
        _matchedHintCount: matchedHintCount,
        _topicHit: topicHit,
        _rerankScore: rerankScore,
      };
    })
  );

  return scored.sort((a, b) => b._rerankScore - a._rerankScore);
}

function formatEvidenceBlock(title, matches) {
  if (!Array.isArray(matches) || matches.length === 0) {
    return `${title}\n（无）`;
  }

  return [
    title,
    ...matches.map(
      (item, index) => `[片段 ${index + 1}]
evidenceTier: ${item._evidenceTier}
section: ${item.section}
subsectionKey: ${item.subsection_key}
subsectionTitle: ${item.subsection_title}
entityType: ${item.entity_type}
content: ${item.content}`
    ),
  ].join('\n\n');
}

function formatContextV7({ primaryMatches, supportMatches, crossSupportMatches, reserveMatches }) {
  return [
    formatEvidenceBlock('【主证据 Primary Evidence】', primaryMatches),
    formatEvidenceBlock('【辅助证据 Support Evidence】', supportMatches),
    formatEvidenceBlock('【跨区补证 Cross-section Support】', crossSupportMatches),
    formatEvidenceBlock('【兜底补充 Reserve Evidence】', reserveMatches),
  ].join('\n\n-----\n\n');
}

function getMatchKey(item) {
  // 注意：Milvus search 结果里不一定总会把主键字段 `id` 带出来。
  //
  // 如果这里直接拿 item.id 去做去重 / 建 Set：
  // - 没有 id 的情况下大家都会变成 undefined
  // - 最后会被误判成“同一条记录”
  // - 于是 primary / support / reserve 可能被错误合并到只剩 1 条
  //
  // 所以版本 7 这里改成：
  // 1. 有 id 就优先用 id
  // 2. 没 id 就退化到“section + subsection + entity + content”的组合键
  return (
    item.id ||
    [
      item.section,
      item.subsection_key,
      item.subsection_title,
      item.entity_type,
      item.content,
    ]
      .map((part) => String(part || ''))
      .join('::')
  );
}

function dedupeByKey(items) {
  const map = new Map();
  items.forEach((item) => {
    map.set(getMatchKey(item), item);
  });
  return [...map.values()];
}

async function selectFinalMatchesV7(matches, question, strategy, topK, noCache = false) {
  const precisionConfig = await loadPrecisionConfig(noCache);
  const sectionBoostConfig = await loadSectionBoostConfig(noCache);
  const rerankConfig = await loadRerankConfig(noCache);
  const selectionConfig = await loadSelectionConfig(noCache);
  const questionKeywords = await extractQuestionKeywords(question, noCache);

  const strategyPrecision = precisionConfig?.[strategy] || precisionConfig?.general || {};
  const strategySelection = selectionConfig?.[strategy] || selectionConfig?.general || {};
  const strategySectionConfig = sectionBoostConfig?.[strategy] || {};

  const preferredSections = Object.entries(strategySectionConfig)
    .filter(([sectionName]) => sectionName !== '__default')
    .filter(([, sectionConfig]) => {
      const defaultBoost = Number(sectionConfig?.default || 0);
      const summaryBoost = Number(sectionConfig?.summary || 0);
      return defaultBoost > 0 || summaryBoost > 0;
    })
    .map(([sectionName]) => sectionName);

  if (preferredSections.length === 0) {
    preferredSections.push('projects', 'work_experience', 'skills', 'core_strengths', 'profile');
  }

  const primaryMinRerankScore = Number(strategyPrecision.primaryMinRerankScore || 0.66);
  const supportMinRerankScore = Number(strategyPrecision.supportMinRerankScore || 0.61);
  const hardDropMinNoiseReasons = Number(strategyPrecision.hardDropMinNoiseReasons || 4);
  const rawScoreNoiseThreshold = Number(rerankConfig.rawScoreNoiseThreshold || 0.5);
  const rerankScoreNoiseThreshold = Number(rerankConfig.rerankScoreNoiseThreshold || 0.6);
  const preferredSectionKeepScore = Number(rerankConfig.preferredSectionKeepScore || 0.63);

  const minFinalCount = Number(strategySelection.minFinalCount || Math.min(topK, 6));
  const maxPrimaryCount = Number(strategySelection.maxPrimaryCount || 4);
  const maxPreferredSupportCount = Number(strategySelection.maxPreferredSupportCount || 2);
  const maxCrossSectionSupportCount = Number(strategySelection.maxCrossSectionSupportCount || 1);
  const crossSectionSupportSections = Array.isArray(strategySelection.crossSectionSupportSections)
    ? strategySelection.crossSectionSupportSections
    : [];
  const crossSectionSupportMinRawScore = Number(strategySelection.crossSectionSupportMinRawScore || 0.62);
  const crossSectionSupportMinRerankScore = Number(strategySelection.crossSectionSupportMinRerankScore || 0.68);
  const crossSectionSupportMinHintCount = Number(strategySelection.crossSectionSupportMinHintCount || 1);
  const reserveMinRerankScore = Number(strategySelection.reserveMinRerankScore || 0.58);

  const { kept, dropped } = denoiseMatches(matches, question, strategy, {
    minKeep: Math.min(topK, 4),
    questionKeywords,
    preferredSections,
    rawScoreNoiseThreshold,
    rerankScoreNoiseThreshold,
    preferredSectionKeepScore,
  });

  const inspected = kept.map((item) => {
    const noiseReasons = [...new Set(item._noiseReasons || [])];
    const isPreferredSection = preferredSections.includes(item.section);
    const rawScore = Number(item._baseScore || 0);
    const rerankScore = Number(item._rerankScore || 0);
    const matchedHintCount = Number(item._matchedHintCount || 0);

    const isPrimary =
      isPreferredSection &&
      (item._topicHit || matchedHintCount > 0 || item.section === 'core_strengths') &&
      rerankScore >= primaryMinRerankScore &&
      noiseReasons.length < hardDropMinNoiseReasons;

    const isPreferredSupport =
      !isPrimary &&
      isPreferredSection &&
      rerankScore >= supportMinRerankScore &&
      noiseReasons.length < hardDropMinNoiseReasons;

    const isCrossSectionSupport =
      !isPrimary &&
      !isPreferredSupport &&
      crossSectionSupportSections.includes(item.section) &&
      item._topicHit &&
      matchedHintCount >= crossSectionSupportMinHintCount &&
      rawScore >= crossSectionSupportMinRawScore &&
      rerankScore >= crossSectionSupportMinRerankScore;

    return {
      ...item,
      _noiseReasons: noiseReasons,
      _isPrimaryEvidence: isPrimary,
      _isPreferredSupportEvidence: isPreferredSupport,
      _isCrossSectionSupportEvidence: isCrossSectionSupport,
    };
  });

  const primaryMatches = inspected
    .filter((item) => item._isPrimaryEvidence)
    .slice(0, maxPrimaryCount)
    .map((item) => ({
      ...item,
      _evidenceTier: 'primary',
    }));

  const primaryIds = new Set(primaryMatches.map((item) => getMatchKey(item)));

  const supportMatches = inspected
    .filter((item) => !primaryIds.has(getMatchKey(item)) && item._isPreferredSupportEvidence)
    .slice(0, maxPreferredSupportCount)
    .map((item) => ({
      ...item,
      _evidenceTier: 'support',
    }));

  const supportIds = new Set(supportMatches.map((item) => getMatchKey(item)));

  const crossSupportMatches = inspected
    .filter(
      (item) =>
        !primaryIds.has(getMatchKey(item)) &&
        !supportIds.has(getMatchKey(item)) &&
        item._isCrossSectionSupportEvidence
    )
    .slice(0, maxCrossSectionSupportCount)
    .map((item) => ({
      ...item,
      _evidenceTier: 'cross_support',
    }));

  const selectedIds = new Set([
    ...primaryMatches.map((item) => getMatchKey(item)),
    ...supportMatches.map((item) => getMatchKey(item)),
    ...crossSupportMatches.map((item) => getMatchKey(item)),
  ]);

  const reserveMatches = inspected
    .filter((item) => !selectedIds.has(getMatchKey(item)))
    .filter((item) => Number(item._rerankScore || 0) >= reserveMinRerankScore)
    .filter((item) => item._topicHit || preferredSections.includes(item.section))
    .map((item) => ({
      ...item,
      _evidenceTier: 'reserve',
    }));

  const finalMatches = dedupeByKey([
    ...primaryMatches,
    ...supportMatches,
    ...crossSupportMatches,
    ...reserveMatches,
  ]).slice(0, Math.max(topK, minFinalCount));

  const finalTrimmedMatches = finalMatches.slice(0, topK);
  const finalIds = new Set(finalTrimmedMatches.map((item) => getMatchKey(item)));
  const droppedBySelection = inspected.filter((item) => !finalIds.has(getMatchKey(item)));

  return {
    primaryMatches,
    supportMatches,
    crossSupportMatches,
    reserveMatches: reserveMatches.filter((item) => finalIds.has(item.id)),
    finalMatches: finalTrimmedMatches,
    droppedMatches: [...dropped, ...droppedBySelection],
  };
}

export async function buildRAGContextV7({
  client,
  collectionName,
  question,
  topK = 8,
  candidateTopK = Math.max(topK * 2, 10),
  filter = '',
  metricType = MetricType.COSINE,
  outputFields,
  strategy = '',
  noCache = false,
  onProgress = noop,
}) {
  // buildRAGContextV7 = 版本 7 的“证据构建层”
  //
  // 1. 识别问题策略
  // 2. 校验集合与向量维度
  // 3. 召回候选片段
  // 4. rerank + denoise
  // 5. 做更细的证据分层，并保证 final 不会过少
  const resolvedStrategy = strategy || detectQuestionStrategy(question);
  onProgress(`1/7 策略识别完成：${resolvedStrategy}`);

  const collectionDetail = await client.describeCollection({
    collection_name: collectionName,
  });

  const collectionVectorDim = getCollectionVectorDim(collectionDetail);
  onProgress(`2/7 已读取集合 schema，向量维度=${collectionVectorDim}`);

  const queryVector = await getEmbedding(question);
  const { nonZeroCount } = validateEmbedding(queryVector, 'query embedding');

  if (queryVector.length !== collectionVectorDim) {
    throw new Error(
      `查询向量维度与集合不一致：集合为 ${collectionVectorDim}，当前 embedding 为 ${queryVector.length}`
    );
  }

  const rawMatches = await retrieve(client, queryVector, {
    collectionName,
    topK: candidateTopK,
    metricType,
    filter,
    outputFields,
  });
  onProgress(`3/7 已完成向量召回：raw candidates=${rawMatches.length}`);

  const rerankedMatches = await rerankMatches(rawMatches, question, resolvedStrategy, noCache);
  onProgress(`4/7 已完成重排：reranked candidates=${rerankedMatches.length}`);

  const {
    primaryMatches,
    supportMatches,
    crossSupportMatches,
    reserveMatches,
    finalMatches,
    droppedMatches,
  } = await selectFinalMatchesV7(rerankedMatches, question, resolvedStrategy, topK, noCache);

  const context = formatContextV7({
    primaryMatches,
    supportMatches,
    crossSupportMatches,
    reserveMatches,
  });

  onProgress(
    `5/7 证据分层完成：primary=${primaryMatches.length}，support=${supportMatches.length}，cross=${crossSupportMatches.length}，final=${finalMatches.length}，dropped=${droppedMatches.length}`
  );

  return {
    question,
    strategy: resolvedStrategy,
    candidateTopK,
    topK,
    filter,
    queryVectorDim: queryVector.length,
    queryNonZeroCount: nonZeroCount,
    rawMatches,
    rerankedMatches,
    primaryMatches,
    supportMatches,
    crossSupportMatches,
    reserveMatches,
    finalMatches,
    droppedMatches,
    context,
  };
}
