import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MetricType } from '@zilliz/milvus2-sdk-node';
import { getEmbedding, validateEmbedding } from '../embedding-client.mjs';
import { retrieveV7 } from './retriever-v7.mjs';

const CURRENT_DIR = path.dirname(fileURLToPath(import.meta.url));

// 版本 7 的配置现在全部收回当前目录。
//
// 这样做不是为了“彻底去重”，而是为了保证：
// - 读 v7 时不需要跨版本跳文件；
// - 后续修改 v7 阈值时不会被其他版本连带影响；
// - 学习者能在一个目录内看完整的检索策略、重排策略与证据选择策略。
const KEYWORD_HINTS_PATH = path.join(CURRENT_DIR, 'config/keyword-hints.json');
const SECTION_BOOST_CONFIG_PATH = path.join(CURRENT_DIR, 'config/section-boost-config.json');
const RERANK_CONFIG_PATH = path.join(CURRENT_DIR, 'config/rerank-config.json');
const PRECISION_CONFIG_PATH = path.join(CURRENT_DIR, 'config/precision-config.json');
const SELECTION_CONFIG_PATH = path.join(CURRENT_DIR, 'config/selection-config.json');

let keywordHintsCache = null;
let sectionBoostConfigCache = null;
let rerankConfigCache = null;
let precisionConfigCache = null;
let selectionConfigCache = null;

function noop() {}

/**
 * 从集合 schema 中读取向量维度。
 *
 * @param {object} detail Milvus collection 详情。
 * @returns {number} 向量维度；若不存在则返回 0。
 */
function getCollectionVectorDim(detail) {
  const dim = detail?.schema?.fields?.find((field) => field.name === 'vector')?.dim;
  return Number(dim || 0);
}

/**
 * 统一做小写归一化，方便后续触发词和文本片段匹配。
 *
 * @param {unknown} value 任意原始值。
 * @returns {string} 归一化后的字符串。
 */
function normalizeText(value) {
  return String(value || '').toLowerCase();
}

/**
 * 识别当前问题属于哪类策略。
 *
 * 版本 7 把策略识别规则直接放回当前目录，目的是让“问题识别 → 召回 → 去噪 → 分层”
 * 这条链路在版本内闭合，阅读时无需再跳转旧实现。
 *
 * @param {string} question 用户问题。
 * @returns {'job_match' | 'skill' | 'project' | 'experience' | 'general'} 策略名。
 */
function detectQuestionStrategy(question) {
  const normalized = normalizeText(question);

  if (/岗位|匹配|胜任|合适吗|适合吗|符合/.test(normalized)) {
    return 'job_match';
  }

  if (/技能|擅长|会什么|技术栈|掌握|熟悉|能力如何/.test(normalized)) {
    return 'skill';
  }

  const hasProject = /项目|作品|案例/.test(normalized);
  const hasExperience = /经验|经历|做过|负责过|实战|落地|主导|参与|开发相关经验/.test(normalized);

  if (hasProject && !hasExperience) {
    return 'project';
  }

  if (hasExperience) {
    return 'experience';
  }

  if (hasProject) {
    return 'project';
  }

  return 'general';
}

/**
 * 读取 v7 的 keyword hints 配置。
 *
 * @param {boolean} [noCache=false] 是否跳过内存缓存。
 * @returns {Promise<Record<string, string[]>>}
 */
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

/**
 * 读取 v7 的 section boost 配置。
 *
 * @param {boolean} [noCache=false] 是否跳过内存缓存。
 * @returns {Promise<Record<string, object>>}
 */
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

/**
 * 读取 v7 的 rerank 配置。
 *
 * @param {boolean} [noCache=false] 是否跳过内存缓存。
 * @returns {Promise<Record<string, number>>}
 */
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

/**
 * 读取 v7 的 precision 配置。
 *
 * @param {boolean} [noCache=false] 是否跳过内存缓存。
 * @returns {Promise<Record<string, object>>}
 */
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

/**
 * 读取 v7 的 selection 配置。
 *
 * @param {boolean} [noCache=false] 是否跳过内存缓存。
 * @returns {Promise<Record<string, object>>}
 */
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

/**
 * 基于问题触发词，展开关键词提示列表。
 *
 * @param {string} question 用户问题。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<string[]>} 去重后的扩展 hints。
 */
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

/**
 * 从问题中提取用于 topic hit 判断的关键词集合。
 *
 * @param {string} question 用户问题。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<string[]>} 归一化后的问题关键词。
 */
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

/**
 * 计算候选片段是否命中当前问题主题。
 *
 * @param {object} item 候选片段。
 * @param {string[]} questionKeywords 问题关键词。
 * @returns {{topicHit: boolean}} topic hit 结果。
 */
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

/**
 * 计算 section boost。
 *
 * `section boost` 的职责不是“绝对裁决”，而是给更符合问题类型的 section 更合理的优先级。
 * 版本 7 继续保留“经验 / 项目类问题下，没命中主题时要衰减 boost”的逻辑，以降低误召回。
 *
 * @param {object} item 候选片段。
 * @param {string} strategy 当前问题策略。
 * @param {boolean} topicHit 是否命中主题。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<number>} section boost 分数。
 */
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

/**
 * 计算关键词命中带来的 rerank boost。
 *
 * @param {object} item 候选片段。
 * @param {string[]} keywordHints 扩展关键词提示。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<{matchedHintCount: number, boost: number}>}
 */
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
    if (!keyword) {
      continue;
    }

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

/**
 * 对召回结果做二次重排。
 *
 * 重排分由三部分构成：
 * - 原始向量分；
 * - section boost；
 * - keyword hint boost。
 *
 * @param {Array<object>} matches 原始候选片段。
 * @param {string} question 用户问题。
 * @param {string} strategy 当前问题策略。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<Array<object>>} 按 rerank 分降序排序后的片段。
 */
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

/**
 * 识别一条候选片段的噪音原因。
 *
 * 注意这里返回的是“原因列表”，不是直接做最终裁决。
 * 版本 7 采用的是：
 * - 先标注噪音原因；
 * - 再结合 hints、preferred section、阈值做保留判断。
 *
 * @param {object} item 当前片段。
 * @param {object} leader 头部片段。
 * @param {string} strategy 当前问题策略。
 * @param {string[]} questionKeywords 问题关键词。
 * @param {{
 *   preferredSections?: string[],
 *   rawScoreNoiseThreshold?: number,
 *   rerankScoreNoiseThreshold?: number,
 * }} [options={}] 检测配置。
 * @returns {string[]} 去重后的噪音原因列表。
 */
function detectNoiseReasonsV7(item, leader, strategy, questionKeywords, options = {}) {
  const reasons = [];
  const preferredSections = Array.isArray(options.preferredSections) ? options.preferredSections : [];
  const isPreferredSection = preferredSections.includes(item.section);
  const hasHints = Number(item._matchedHintCount || 0) > 0;
  const rerankGap = Number(leader?._rerankScore || 0) - Number(item._rerankScore || 0);
  const rawScore = Number(item._baseScore || item.score || 0);
  const rerankScore = Number(item._rerankScore || item.score || 0);
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

  if (!isPreferredSection) {
    reasons.push('非当前问题优先 section');
  }

  if (!hasHints && questionKeywords.length > 0) {
    reasons.push('未命中主题 hint');
  }

  if (!topicHit && questionKeywords.length > 0) {
    reasons.push('与问题主题缺少直接文本关联');
  }

  if (rerankGap > 0.14) {
    reasons.push('与头部结果分差过大');
  }

  if (rawScore < Number(options.rawScoreNoiseThreshold ?? 0.48)) {
    reasons.push('原始向量分偏低');
  }

  if (rerankScore < Number(options.rerankScoreNoiseThreshold ?? 0.6)) {
    reasons.push('重排后分数偏低');
  }

  if ((strategy === 'experience' || strategy === 'project') && !hasHints && item.section !== 'core_strengths') {
    reasons.push('经验类问题下缺少主题证据');
  }

  return [...new Set(reasons)];
}

/**
 * 对重排结果做去噪与最小保留控制。
 *
 * 版本 7 的去噪目标不是“一刀切删干净”，而是：
 * - 尽可能压制明显噪声；
 * - 同时避免 final context 过瘦。
 *
 * 因此这里保留了一个回退机制：
 * - 如果去噪后结果过少，就回退到头部重排结果，保证最小信息量。
 *
 * @param {Array<object>} matches 重排结果。
 * @param {string} question 用户问题。
 * @param {string} strategy 当前问题策略。
 * @param {{
 *   questionKeywords?: string[],
 *   preferredSections?: string[],
 *   rawScoreNoiseThreshold?: number,
 *   rerankScoreNoiseThreshold?: number,
 *   preferredSectionKeepScore?: number,
 *   minKeep?: number,
 * }} [options={}] 去噪参数。
 * @returns {{kept: Array<object>, dropped: Array<object>}} 去噪结果。
 */
function denoiseMatchesV7(matches, question, strategy, options = {}) {
  if (!Array.isArray(matches) || matches.length === 0) {
    return {
      kept: [],
      dropped: [],
    };
  }

  const leader = matches[0];
  const questionKeywords = Array.isArray(options.questionKeywords)
    ? options.questionKeywords
    : [];
  const preferredSections = Array.isArray(options.preferredSections)
    ? options.preferredSections
    : [];
  const minKeep = Number(options.minKeep || 4);

  const inspected = matches.map((item) => {
    const reasons = detectNoiseReasonsV7(item, leader, strategy, questionKeywords, {
      preferredSections,
      rawScoreNoiseThreshold: options.rawScoreNoiseThreshold,
      rerankScoreNoiseThreshold: options.rerankScoreNoiseThreshold,
    });
    const isPreferredSection = preferredSections.includes(item.section);
    const hasHints = Number(item._matchedHintCount || 0) > 0;

    const keep =
      reasons.length <= 2 ||
      hasHints ||
      (
        isPreferredSection &&
        Number(item._rerankScore || 0) >= Number(options.preferredSectionKeepScore ?? 0.63)
      );

    return {
      ...item,
      _noiseReasons: reasons,
      _keptAfterDenoise: keep,
    };
  });

  const kept = inspected.filter((item) => item._keptAfterDenoise);
  const dropped = inspected.filter((item) => !item._keptAfterDenoise);

  if (kept.length < minKeep) {
    return {
      kept: matches.slice(0, Math.max(minKeep, 1)).map((item) => ({
        ...item,
        _noiseReasons: ['去噪后剩余结果过少，回退到重排结果'],
        _keptAfterDenoise: true,
      })),
      dropped: [],
    };
  }

  return {
    kept,
    dropped,
  };
}

/**
 * 把某一层证据格式化为 Prompt context 块。
 *
 * 这里显式保留 `evidenceTier`，是为了让 Prompt 知道：
 * - `cross_support` 是补证，不是抢主位；
 * - `reserve` 是兜底补充，不是默认主证据。
 *
 * @param {string} title 区块标题。
 * @param {Array<object>} matches 证据数组。
 * @returns {string} 格式化后的文本块。
 */
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

/**
 * 组合最终的 Prompt context。
 *
 * @param {{
 *   primaryMatches: Array<object>,
 *   supportMatches: Array<object>,
 *   crossSupportMatches: Array<object>,
 *   reserveMatches: Array<object>,
 * }} tiers 分层后的证据集合。
 * @returns {string} 最终写入 Prompt 的上下文字符串。
 */
function formatContextV7({ primaryMatches, supportMatches, crossSupportMatches, reserveMatches }) {
  return [
    formatEvidenceBlock('【主证据 Primary Evidence】', primaryMatches),
    formatEvidenceBlock('【辅助证据 Support Evidence】', supportMatches),
    formatEvidenceBlock('【跨区补证 Cross-section Support】', crossSupportMatches),
    formatEvidenceBlock('【兜底补充 Reserve Evidence】', reserveMatches),
  ].join('\n\n-----\n\n');
}

/**
 * 获取一条候选片段的稳定去重键。
 *
 * 注意不能只依赖 `id`：
 * - 检索结果未必总带主键；
 * - 没有 `id` 时，所有记录都会退化成 `undefined`；
 * - 这样 primary / support / reserve 可能被误判成同一条数据。
 *
 * 所以版本 7 的策略是：
 * - 有 `id` 时优先用 `id`；
 * - 没有时退化到“section + subsection + entity + content”组合键。
 *
 * @param {object} item 候选片段。
 * @returns {string} 稳定去重键。
 */
function getMatchKey(item) {
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

/**
 * 基于 `getMatchKey()` 做数组去重。
 *
 * @param {Array<object>} items 候选数组。
 * @returns {Array<object>} 去重后数组。
 */
function dedupeByKey(items) {
  const map = new Map();
  items.forEach((item) => {
    map.set(getMatchKey(item), item);
  });
  return [...map.values()];
}

/**
 * 从 section boost 配置推导“当前策略下的优先 section”。
 *
 * @param {Record<string, {default?: number, summary?: number}>} strategySectionConfig 当前策略的 section boost 配置。
 * @returns {string[]} 优先 section 列表。
 */
function resolvePreferredSections(strategySectionConfig) {
  const preferredSections = Object.entries(strategySectionConfig || {})
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

  return preferredSections;
}

/**
 * 统一收口 `selectFinalMatchesV7()` 需要的分层参数。
 *
 * 这里特别补一条版本 7 文档里强调过的语义说明：
 * - `hardDropMinNoiseReasons` 表示“硬丢弃阈值”；
 * - 例如配置为 `4`，意思是噪音原因达到 4 个及以上就直接失去主证据/辅助证据资格；
 * - 也就是说，`0 / 1 / 2 / 3` 个噪音原因仍允许继续参与分层判断。
 *
 * @param {{
 *   strategyPrecision: Record<string, number>,
 *   strategySelection: Record<string, number | string[]>,
 *   rerankConfig: Record<string, number>,
 *   preferredSections: string[],
 * }} config 各类配置。
 * @returns {object} 供分层使用的归一化参数。
 */
function buildSelectionParams({ strategyPrecision, strategySelection, rerankConfig, preferredSections }) {
  return {
    primaryMinRerankScore: Number(strategyPrecision.primaryMinRerankScore || 0.66),
    supportMinRerankScore: Number(strategyPrecision.supportMinRerankScore || 0.61),
    hardDropMinNoiseReasons: Number(strategyPrecision.hardDropMinNoiseReasons || 4),
    rawScoreNoiseThreshold: Number(rerankConfig.rawScoreNoiseThreshold || 0.5),
    rerankScoreNoiseThreshold: Number(rerankConfig.rerankScoreNoiseThreshold || 0.6),
    preferredSectionKeepScore: Number(rerankConfig.preferredSectionKeepScore || 0.63),
    minFinalCount: Number(strategySelection.minFinalCount || 6),
    maxPrimaryCount: Number(strategySelection.maxPrimaryCount || 4),
    maxPreferredSupportCount: Number(strategySelection.maxPreferredSupportCount || 2),
    maxCrossSectionSupportCount: Number(strategySelection.maxCrossSectionSupportCount || 1),
    crossSectionSupportSections: Array.isArray(strategySelection.crossSectionSupportSections)
      ? strategySelection.crossSectionSupportSections
      : [],
    crossSectionSupportMinRawScore: Number(strategySelection.crossSectionSupportMinRawScore || 0.62),
    crossSectionSupportMinRerankScore: Number(strategySelection.crossSectionSupportMinRerankScore || 0.68),
    crossSectionSupportMinHintCount: Number(strategySelection.crossSectionSupportMinHintCount || 1),
    reserveMinRerankScore: Number(strategySelection.reserveMinRerankScore || 0.58),
    preferredSections,
  };
}

/**
 * 判断一条片段属于哪类证据。
 *
 * 注意这里的设计重点是：
 * - `cross_support` 是高相关跨区补证，不是让 skills 抢主位；
 * - `reserve` 不是默认主证据，而是最后兜底补充；
 * - 真正进入哪一层，仍然受噪音阈值、hint 命中和 section 优先级共同影响。
 *
 * @param {object} item 已经带有 noise / rerank 信息的片段。
 * @param {object} params 分层参数。
 * @returns {object} 带有分类标记的片段。
 */
function classifyEvidence(item, params) {
  const noiseReasons = [...new Set(item._noiseReasons || [])];
  const isPreferredSection = params.preferredSections.includes(item.section);
  const rawScore = Number(item._baseScore || 0);
  const rerankScore = Number(item._rerankScore || 0);
  const matchedHintCount = Number(item._matchedHintCount || 0);

  const isPrimary =
    isPreferredSection &&
    (item._topicHit || matchedHintCount > 0 || item.section === 'core_strengths') &&
    rerankScore >= params.primaryMinRerankScore &&
    noiseReasons.length < params.hardDropMinNoiseReasons;

  const isPreferredSupport =
    !isPrimary &&
    isPreferredSection &&
    rerankScore >= params.supportMinRerankScore &&
    noiseReasons.length < params.hardDropMinNoiseReasons;

  const isCrossSectionSupport =
    !isPrimary &&
    !isPreferredSupport &&
    params.crossSectionSupportSections.includes(item.section) &&
    item._topicHit &&
    matchedHintCount >= params.crossSectionSupportMinHintCount &&
    rawScore >= params.crossSectionSupportMinRawScore &&
    rerankScore >= params.crossSectionSupportMinRerankScore;

  return {
    ...item,
    _noiseReasons: noiseReasons,
    _isPrimaryEvidence: isPrimary,
    _isPreferredSupportEvidence: isPreferredSupport,
    _isCrossSectionSupportEvidence: isCrossSectionSupport,
  };
}

/**
 * 按条件选择某一层证据，并补上 `evidenceTier` 标记。
 *
 * @param {Array<object>} items 候选数组。
 * @param {(item: object) => boolean} predicate 选择条件。
 * @param {number} limit 最大数量。
 * @param {string} tierName 层级名。
 * @param {Set<string>} [excludedKeys=new Set()] 需要排除的去重键集合。
 * @returns {Array<object>} 当前 tier 的结果。
 */
function selectTierMatches(items, predicate, limit, tierName, excludedKeys = new Set()) {
  return items
    .filter((item) => !excludedKeys.has(getMatchKey(item)) && predicate(item))
    .slice(0, limit)
    .map((item) => ({
      ...item,
      _evidenceTier: tierName,
    }));
}

/**
 * 从重排结果中选择最终进入 Prompt 的证据。
 *
 * 这里是版本 7 最核心的证据装配逻辑：
 * - 先去噪；
 * - 再分层成 `primary / support / cross_support / reserve`；
 * - 最后按顺序拼装 final matches。
 *
 * @param {Array<object>} matches 重排结果。
 * @param {string} question 用户问题。
 * @param {string} strategy 当前问题策略。
 * @param {number} topK 最终上限。
 * @param {boolean} [noCache=false] 是否跳过配置缓存。
 * @returns {Promise<{
 *   primaryMatches: Array<object>,
 *   supportMatches: Array<object>,
 *   crossSupportMatches: Array<object>,
 *   reserveMatches: Array<object>,
 *   finalMatches: Array<object>,
 *   droppedMatches: Array<object>,
 * }>}
 */
async function selectFinalMatchesV7(matches, question, strategy, topK, noCache = false) {
  const precisionConfig = await loadPrecisionConfig(noCache);
  const sectionBoostConfig = await loadSectionBoostConfig(noCache);
  const rerankConfig = await loadRerankConfig(noCache);
  const selectionConfig = await loadSelectionConfig(noCache);
  const questionKeywords = await extractQuestionKeywords(question, noCache);

  const strategyPrecision = precisionConfig?.[strategy] || precisionConfig?.general || {};
  const strategySelection = selectionConfig?.[strategy] || selectionConfig?.general || {};
  const strategySectionConfig = sectionBoostConfig?.[strategy] || {};

  const preferredSections = resolvePreferredSections(strategySectionConfig);
  const params = buildSelectionParams({
    strategyPrecision,
    strategySelection,
    rerankConfig,
    preferredSections,
  });

  const { kept, dropped } = denoiseMatchesV7(matches, question, strategy, {
    minKeep: Math.min(topK, 4),
    questionKeywords,
    preferredSections: params.preferredSections,
    rawScoreNoiseThreshold: params.rawScoreNoiseThreshold,
    rerankScoreNoiseThreshold: params.rerankScoreNoiseThreshold,
    preferredSectionKeepScore: params.preferredSectionKeepScore,
  });

  const inspected = kept.map((item) => classifyEvidence(item, params));

  const primaryMatches = selectTierMatches(
    inspected,
    (item) => item._isPrimaryEvidence,
    params.maxPrimaryCount,
    'primary'
  );

  const primaryIds = new Set(primaryMatches.map((item) => getMatchKey(item)));

  const supportMatches = selectTierMatches(
    inspected,
    (item) => item._isPreferredSupportEvidence,
    params.maxPreferredSupportCount,
    'support',
    primaryIds
  );

  const supportIds = new Set(supportMatches.map((item) => getMatchKey(item)));

  const crossSupportMatches = selectTierMatches(
    inspected,
    (item) => item._isCrossSectionSupportEvidence,
    params.maxCrossSectionSupportCount,
    'cross_support',
    new Set([...primaryIds, ...supportIds])
  );

  const selectedIds = new Set([
    ...primaryMatches.map((item) => getMatchKey(item)),
    ...supportMatches.map((item) => getMatchKey(item)),
    ...crossSupportMatches.map((item) => getMatchKey(item)),
  ]);

  const reserveMatches = inspected
    .filter((item) => !selectedIds.has(getMatchKey(item)))
    .filter((item) => Number(item._rerankScore || 0) >= params.reserveMinRerankScore)
    .filter((item) => item._topicHit || params.preferredSections.includes(item.section))
    .map((item) => ({
      ...item,
      _evidenceTier: 'reserve',
    }));

  const finalMatches = dedupeByKey([
    ...primaryMatches,
    ...supportMatches,
    ...crossSupportMatches,
    ...reserveMatches,
  ]).slice(0, Math.max(topK, params.minFinalCount));

  const finalTrimmedMatches = finalMatches.slice(0, topK);
  const finalIds = new Set(finalTrimmedMatches.map((item) => getMatchKey(item)));
  const droppedBySelection = inspected.filter((item) => !finalIds.has(getMatchKey(item)));

  return {
    primaryMatches,
    supportMatches,
    crossSupportMatches,
    reserveMatches: reserveMatches.filter((item) => finalIds.has(getMatchKey(item))),
    finalMatches: finalTrimmedMatches,
    droppedMatches: [...dropped, ...droppedBySelection],
  };
}

/**
 * 构建版本 7 的完整 RAG 上下文。
 *
 * 这是 v7 的“证据构建层”，负责：
 * 1. 识别问题策略；
 * 2. 检查集合 schema 与向量维度；
 * 3. 执行召回；
 * 4. 做 rerank 与去噪；
 * 5. 生成分层后的 context 与调试信息。
 *
 * @param {{
 *   client: import('@zilliz/milvus2-sdk-node').MilvusClient,
 *   collectionName: string,
 *   question: string,
 *   topK?: number,
 *   candidateTopK?: number,
 *   filter?: string,
 *   metricType?: string,
 *   outputFields?: string[],
 *   strategy?: string,
 *   noCache?: boolean,
 *   onProgress?: (message: string) => void,
 * }} options 上下文构建参数。
 * @returns {Promise<{
 *   question: string,
 *   strategy: string,
 *   candidateTopK: number,
 *   topK: number,
 *   filter: string,
 *   queryVectorDim: number,
 *   queryNonZeroCount: number,
 *   rawMatches: Array<object>,
 *   rerankedMatches: Array<object>,
 *   primaryMatches: Array<object>,
 *   supportMatches: Array<object>,
 *   crossSupportMatches: Array<object>,
 *   reserveMatches: Array<object>,
 *   finalMatches: Array<object>,
 *   droppedMatches: Array<object>,
 *   context: string,
 * }>}
 * @throws {Error} 当集合向量维度与查询向量不一致时抛错。
 */
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

  const rawMatches = await retrieveV7(client, queryVector, {
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
