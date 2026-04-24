import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MetricType } from '@zilliz/milvus2-sdk-node';
import { getEmbedding, validateEmbedding } from '../embedding-client.mjs';
import { retrieve } from '../rag/retriever.mjs';
import {
  detectQuestionStrategy,
  formatContext,
} from '../rag5/context-builder.mjs';
import { denoiseMatches } from '../rag4/rag-pipeline-v4.mjs';

const CURRENT_DIR = path.dirname(fileURLToPath(import.meta.url));
const KEYWORD_HINTS_PATH = path.join(CURRENT_DIR, '../rag5/config/keyword-hints.json');
const SECTION_BOOST_CONFIG_PATH = path.join(CURRENT_DIR, '../rag5/config/section-boost-config.json');
const RERANK_CONFIG_PATH = path.join(CURRENT_DIR, '../rag5/config/rerank-config.json');
const PRECISION_CONFIG_PATH = path.join(CURRENT_DIR, 'config/precision-config.json');

let keywordHintsCache = null;
let sectionBoostConfigCache = null;
let rerankConfigCache = null;
let precisionConfigCache = null;

function getCollectionVectorDim(detail) {
  // 读取 collection schema 中 vector 字段的 dim。
  // 这个检查非常重要，因为：
  // - 写入时 embedding 模型可能是 A
  // - 查询时 embedding 模型可能被误切成 B
  // 一旦维度不一致，Milvus 查询会直接失败。
  const dim = detail?.schema?.fields?.find((field) => field.name === 'vector')?.dim;
  return Number(dim || 0);
}

function normalizeText(value) {
  // 所有关键词比较前统一小写化，减少大小写差异带来的影响。
  return String(value || '').toLowerCase();
}

async function loadKeywordHintsConfig(noCache = false) {
  // 这里按你的建议，不再额外抽一个通用 loadJson()。
  // 原因是：
  // - 当前只是在读 4 份配置
  // - 通用封装反而把“每份配置在读什么”藏起来了
  // - 对学习阶段来说，直接展开更容易理解
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

async function getKeywordHints(question, noCache = false) {
  // 触发词扩展：
  // 比如问题里出现 "ai"，就自动补入：
  // - agent
  // - prompt
  // - sse
  // - 工作流
  //
  // 这类扩展词现在已经放到 JSON 配置里，方便后续继续实验。
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
 * 版本 6 这里不再单独硬编码一套问题关键词
 * 原因是：
 * - getKeywordHints() 已经从 keyword-hints.json 里读取了一套触发词与扩展词
 * - 如果这里还再手写一套 ai / agent / sse / 工作流，就会形成两套口径
 * - 以后你改了 JSON，主题判断却没同步改，就会出现“配置漂移”
 * 因此这里直接复用同一份配置：
 * - 如果问题命中了某个 trigger
 * - 那么 trigger 本身 + 对应 expansions 一起作为“问题主题锚点”
 * 
 * 这样后面 computeTopicAlignment / denoise 就和 keyword-hints.json 保持统一来源。
 * @param {*} question 
 * @param {*} noCache 
 * @returns 
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

function computeTopicAlignment(item, questionKeywords) {
  // 这里把一条候选记录能用来判断主题的字段拼成一个 haystack，
  // 用于判断：
  // - 它有没有命中问题主题
  // - 它是不是只是“section 上看起来相关”，但内容其实不相关
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
    haystack,
    topicHit,
  };
}

async function scoreSectionBoost(item, strategy, topicHit, noCache = false) {
  // section boost = “业务结构先验”加分。
  //
  // 例如：
  // - 经验类问题里，projects / work_experience 应该更占优
  // - skills 在经验类问题里反而应该降权
  //
  // 但版本 6 关键改进是：
  // 即便属于 projects / work_experience，
  // 如果它和当前问题主题没有对齐，也不能照单全收。
  const sectionBoostConfig = await loadSectionBoostConfig(noCache);
  const precisionConfig = await loadPrecisionConfig(noCache);
  const strategyConfig = sectionBoostConfig?.[strategy];

  if (!strategyConfig) {
    return 0;
  }

  // 这里按你的建议，不再单独抽 getEntityWeight()。
  // 因为这段逻辑只在 scoreSectionBoost() 内部使用一次，
  // 内联后可读性更强，也更符合“不要过度抽象”的原则。
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

  // 版本 6 关键改动：
  // 对经验类/项目类问题下“没有主题对齐的项目/工作经历”，
  // 不再给予满额 section boost，而是做衰减。
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
  // keyword boost = 主题词命中带来的微调分。
  //
  // 它的定位不是“主导排序”，而是：
  // - 在 base score 已经比较接近的情况下，拉开一点业务相关度差异
  // - 强化确实谈到了 AI / Agent / SSE / 工作流 的片段
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
  // rerank 的目标：
  // 在 Milvus 原始相似度分的基础上，加上业务结构偏置和主题关键词偏置。
  //
  // 注意：
  // 这不是 cross-encoder 那种真正的语义重排，
  // 仍然属于“规则重排”。
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

function enrichNoiseReasons(item, strategy) {
  // 这里不是重新做一遍去噪，
  // 而是给现有 noiseReasons 再补充一层更贴近版本 6 目标的解释：
  // “虽然你是项目，但你不一定是当前问题的主证据。”
  const reasons = [...(item._noiseReasons || [])];

  if ((strategy === 'experience' || strategy === 'project') && item.section === 'projects' && !item._topicHit) {
    reasons.push('项目虽属优先 section，但缺少主题对齐');
  }

  return [...new Set(reasons)];
}

/**
 * 版本 6 的核心函数，它不再是“把去噪结果直接 slice(topK)，而是改成：
 * 1. 先从 denoise 后结果里挑出 primary evidence
 * 2. 再挑 support evidence
 * 3. 最后装配 finalMatches
 * 4. 这就是“主证据优先”的核心实现
 */
async function selectFinalMatches(matches, question, strategy, topK, noCache = false) {
  const precisionConfig = await loadPrecisionConfig(noCache);
  const sectionBoostConfig = await loadSectionBoostConfig(noCache);
  const rerankConfig = await loadRerankConfig(noCache);
  const strategyPrecision = precisionConfig?.[strategy] || precisionConfig?.general || {};
  const questionKeywords = await extractQuestionKeywords(question, noCache);
  const strategySectionConfig = sectionBoostConfig?.[strategy] || {};

  // 这里不再额外维护一份 preferredSectionsForStrategy() 的硬编码映射。
  //
  // 而是直接从 section-boost-config.json 动态推导：
  // - 哪个 section 的 default / summary boost > 0
  // - 就说明当前策略下，它是“偏优先”的 section
  //
  // 这样后续如果你只改 JSON 权重，不需要再同步改第二份映射逻辑。
  const preferredSections = Object.entries(strategySectionConfig)
    .filter(([sectionName]) => sectionName !== '__default')
    .filter(([, sectionConfig]) => {
      const defaultBoost = Number(sectionConfig?.default || 0);
      const summaryBoost = Number(sectionConfig?.summary || 0);
      return defaultBoost > 0 || summaryBoost > 0;
    })
    .map(([sectionName]) => sectionName);

  // 如果当前 strategy 没有专门配置（例如 general），
  // 那就回退到一个宽松的通用集合，避免 preferredSections 为空后，
  // 整个去噪阶段把所有记录都当成“非优先 section”。
  if (preferredSections.length === 0) {
    preferredSections.push('projects', 'work_experience', 'skills', 'core_strengths', 'profile');
  }
  const primaryMinRerankScore = Number(strategyPrecision.primaryMinRerankScore || 0.66);
  const supportMinRerankScore = Number(strategyPrecision.supportMinRerankScore || 0.61);
  const maxSupportWithoutTopicHit = Number(strategyPrecision.maxSupportWithoutTopicHit || 1);
  const hardDropMinNoiseReasons = Number(strategyPrecision.hardDropMinNoiseReasons || 4);
  const rawScoreNoiseThreshold = Number(rerankConfig.rawScoreNoiseThreshold || 0.5);
  const rerankScoreNoiseThreshold = Number(rerankConfig.rerankScoreNoiseThreshold || 0.6);
  const preferredSectionKeepScore = Number(rerankConfig.preferredSectionKeepScore || 0.63);

  const { kept, dropped } = denoiseMatches(matches, question, strategy, {
    minKeep: Math.min(topK, 4),
    questionKeywords,
    preferredSections,
    rawScoreNoiseThreshold,
    rerankScoreNoiseThreshold,
    preferredSectionKeepScore,
  });

  const inspected = kept.map((item) => {
    const noiseReasons = enrichNoiseReasons(item, strategy);
    const isPreferredSection = preferredSections.includes(item.section);
    // 主证据要求更严格：
    // - 在优先 section
    // - 命中主题或至少是核心竞争力
    // - rerank 分达到主证据阈值
    // - 噪声原因不能太多
    const isPrimary =
      isPreferredSection &&
      (item._topicHit || Number(item._matchedHintCount || 0) > 0 || item.section === 'core_strengths') &&
      Number(item._rerankScore || 0) >= primaryMinRerankScore &&
      noiseReasons.length < hardDropMinNoiseReasons;

    // 辅助证据稍微放宽一点，
    // 但仍然要求：
    // - 至少是优先 section
    // - rerank 分达到 support 阈值
    const isSupport =
      isPreferredSection &&
      Number(item._rerankScore || 0) >= supportMinRerankScore &&
      noiseReasons.length < hardDropMinNoiseReasons;

    return {
      ...item,
      _noiseReasons: noiseReasons,
      _isPrimaryEvidence: isPrimary,
      _isSupportEvidence: isSupport,
    };
  });

  const primary = inspected.filter((item) => item._isPrimaryEvidence);
  const supportCandidates = inspected.filter((item) => !item._isPrimaryEvidence && item._isSupportEvidence);

  let supportWithoutTopicHitCount = 0;
  const support = supportCandidates.filter((item) => {
    // 有 topicHit / hint 命中的 support，可以直接保留。
    if (item._topicHit || Number(item._matchedHintCount || 0) > 0 || item.section === 'core_strengths') {
      return true;
    }

    // 没有主题对齐的 support，不允许无限混入。
    // 版本 6 在这里做了一个数量限制，避免“无关但高分”的项目继续把 finalMatches 挤满。
    if (supportWithoutTopicHitCount >= maxSupportWithoutTopicHit) {
      return false;
    }

    supportWithoutTopicHitCount += 1;
    return true;
  });

  const finalMatches = [...primary, ...support].slice(0, topK);
  const droppedByPrecision = inspected.filter((item) => !finalMatches.includes(item));

  return {
    primary,
    support,
    finalMatches,
    droppedMatches: [...dropped, ...droppedByPrecision],
  };
}

/**
 * buildRAGContextV6 是版本 6 的“证据构建层”。
 * 它专注做：
    - 向量化
    - 召回
    - 规则重排
    - 主证据 / 辅助证据分层
    - 组装 final context
 *  仍然不负责调用 LLM。
 */
export async function buildRAGContextV6({
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
}) {
  // 1. 策略识别
  const resolvedStrategy = strategy || detectQuestionStrategy(question);
  // 2. 维度校验
  const collectionDetail = await client.describeCollection({
    collection_name: collectionName,
  });
  const collectionVectorDim = getCollectionVectorDim(collectionDetail);
  const queryVector = await getEmbedding(question);
  const { nonZeroCount } = validateEmbedding(queryVector, 'query embedding');
  if (queryVector.length !== collectionVectorDim) {
    throw new Error(
      `查询向量维度与集合不一致：集合为 ${collectionVectorDim}，当前 embedding 为 ${queryVector.length}`
    );
  }
  // 3. 向量召回
  const rawMatches = await retrieve(client, queryVector, {
    collectionName,
    topK: candidateTopK,
    metricType,
    filter,
    outputFields,
  });
  // 4. 重排
  const rerankedMatches = await rerankMatches(rawMatches, question, resolvedStrategy, noCache);
  // 5. 去噪
  const {
    primary,
    support,
    finalMatches,
    droppedMatches,
  } = await selectFinalMatches(rerankedMatches, question, resolvedStrategy, topK, noCache);

  // 6. 截取 + 格式化, 进入模型的 context 只放 finalMatches，
  // 而调试信息（raw/rerank/primary/support/dropped）继续保留在返回结果里供我们观察。
  const context = formatContext(finalMatches);

  // 7. 返回完整快照
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
    primaryMatches: primary,
    supportMatches: support,
    finalMatches,
    droppedMatches,
    context,
  };
}
