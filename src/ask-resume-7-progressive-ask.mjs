import 'dotenv/config';
import { parseArgs as parseNodeArgs } from 'node:util';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { ChatOpenAI } from '@langchain/openai';
import {
  appendSessionTurn,
  loadSession,
  resetSession,
  sessionToPromptHistory,
} from './rag7/history-store-v7.mjs';
import { runRAGv7 } from './rag7/rag-pipeline-v7.mjs';

const COLLECTION_NAME = process.env.RESUME_RAG_COLLECTION || 'resume_profile_chunks';
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS || 'localhost:19530';

/**
 * 解析 `ask:v7` 的命令行参数。
 *
 * 这里继续使用 Node 标准库的 `parseArgs()`，而不是回退到手写 argv for-loop。
 * 原因是版本 7 已经进入“可单独维护”的阶段，CLI 入口更适合：
 * - 规则集中；
 * - 位置参数与长选项分离；
 * - 便于学习者直接看到完整选项定义。
 *
 * @param {string[]} argv CLI 原始参数数组。
 * @returns {{
 *   sessionId: string,
 *   topK: number,
 *   candidateTopK: number,
 *   filter: string,
 *   promptTemplate: string,
 *   resetSession: boolean,
 *   noCache: boolean,
 *   debugMode: boolean,
 *   streamAnswer: boolean,
 *   question: string,
 * }} 归一化后的脚本参数。
 */
function parseArgs(argv) {
  // 某些通过 `pnpm run xxx -- ...` 触发的场景里，脚本实际拿到的 argv
  // 可能会把一个独立的 `--` 一并透传进来。
  //
  // 如果不先清理掉：
  // - `parseArgs()` 会把后续参数视为 positionals；
  // - `--no-stream` 这类选项就可能被误拼进 question。
  //
  // 因此这里先做一次兼容归一化，只移除“单独存在”的分隔符本身。
  const normalizedArgv = argv.filter((item) => item !== '--');

  const defaults = {
    sessionId: 'resume-demo-v7',
    topK: 8,
    candidateTopK: 16,
    filter: '',
    promptTemplate: '',
    resetSession: false,
    noCache: false,
    debugMode: true,
    streamAnswer: true,
  };

  const { values, positionals } = parseNodeArgs({
    args: normalizedArgv,
    allowPositionals: true,
    strict: false,
    options: {
      session: {
        type: 'string',
      },
      topK: {
        type: 'string',
      },
      candidateTopK: {
        type: 'string',
      },
      filter: {
        type: 'string',
      },
      template: {
        type: 'string',
      },
      'reset-session': {
        type: 'boolean',
      },
      'no-cache': {
        type: 'boolean',
      },
      debug: {
        type: 'boolean',
      },
      'no-debug': {
        type: 'boolean',
      },
      'no-stream': {
        type: 'boolean',
      },
    },
  });

  return {
    ...defaults,
    sessionId: values.session || defaults.sessionId,
    topK: Number(values.topK || defaults.topK),
    candidateTopK: Number(values.candidateTopK || defaults.candidateTopK),
    filter: values.filter || defaults.filter,
    promptTemplate: values.template || defaults.promptTemplate,
    resetSession: Boolean(values['reset-session']),
    noCache: Boolean(values['no-cache']),
    debugMode: values['no-debug'] ? false : (values.debug ?? defaults.debugMode),
    streamAnswer: values['no-stream'] ? false : defaults.streamAnswer,
    question:
      positionals.join(' ').trim() || '这个候选人有哪些 AI Agent 开发相关经验？',
  };
}

/**
 * 创建一个“平滑打字机”写入器。
 *
 * 版本 7 的 CLI 层负责最终的交互体验，因此这种输出层节流逻辑应该留在 `ask`，
 * 而不是塞回 pipeline。这样可以保持：
 * - pipeline 只负责编排与数据；
 * - ask 专门负责用户可见输出。
 *
 * @param {{enabled?: boolean, chunkSize?: number, intervalMs?: number}} [options={}] 输出控制项。
 * @returns {{write(text: string): void, drain(): Promise<void>}} 可写入并可等待排空的 writer。
 */
function createSmoothWriter({ enabled = true, chunkSize = 4, intervalMs = 18 } = {}) {
  let buffer = '';
  let pumping = false;
  const idleResolvers = [];

  async function pump() {
    if (pumping) {
      return;
    }

    pumping = true;

    while (buffer.length > 0) {
      const chunk = buffer.slice(0, chunkSize);
      buffer = buffer.slice(chunkSize);
      process.stdout.write(chunk);

      if (buffer.length > 0) {
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }
    }

    pumping = false;

    while (idleResolvers.length > 0) {
      idleResolvers.shift()?.();
    }
  }

  return {
    write(text) {
      if (!text) {
        return;
      }

      const normalized = String(text);

      if (!enabled) {
        process.stdout.write(normalized);
        return;
      }

      buffer += normalized;
      void pump();
    },
    async drain() {
      if (!enabled || (!pumping && buffer.length === 0)) {
        return;
      }

      await new Promise((resolve) => {
        idleResolvers.push(resolve);
      });
    },
  };
}

/**
 * 打印某一组检索结果，供调试模式查看。
 *
 * @param {string} title 标题。
 * @param {Array<object>} matches 结果数组。
 * @returns {void}
 */
function printMatches(title, matches) {
  console.log(`${title}:\n`);

  matches.forEach((item, index) => {
    const rawScore = Number(item._baseScore ?? item.score ?? 0).toFixed(4);
    const rerankScore = Number(item._rerankScore ?? item.score ?? 0).toFixed(4);

    console.log(
      `${index + 1}. [raw=${rawScore} | rerank=${rerankScore}] ${item.section} / ${item.subsection_title} / ${item.entity_type}`
    );
    console.log(
      `   boosts: section=${Number(item._sectionBoost ?? 0).toFixed(3)}, keyword=${Number(item._keywordBoost ?? 0).toFixed(3)}, hints=${Number(item._matchedHintCount ?? 0)}, topicHit=${Boolean(item._topicHit)}`
    );

    if (item._evidenceTier) {
      console.log(`   evidenceTier: ${item._evidenceTier}`);
    }

    if (Array.isArray(item._noiseReasons) && item._noiseReasons.length > 0) {
      console.log(`   noiseCheck: ${item._noiseReasons.join('；')}`);
    }

    console.log(`   subsectionKey: ${item.subsection_key}`);
    console.log(`   tags: ${Array.isArray(item.tags) ? item.tags.join(', ') : ''}`);
    console.log(`   content: ${String(item.content).slice(0, 180)}\n`);
  });
}

/**
 * 仅在结果非空时打印匹配块。
 *
 * 这个小包装是为了避免 `cross_support` / `reserve` 为空时仍输出一整块空标题。
 * 对调试日志来说，空层级静默跳过会更利于聚焦真正出现的证据层。
 *
 * @param {string} title 标题。
 * @param {Array<object>} matches 结果数组。
 * @returns {void}
 */
function printMatchesIfAny(title, matches) {
  if (!Array.isArray(matches) || matches.length === 0) {
    return;
  }

  printMatches(title, matches);
}

/**
 * 在正式回答前打印调试摘要。
 *
 * 版本 7 的调试信息之所以放在回答前，是为了避免“边回答边刷调试”干扰阅读。
 * 调试模式下，用户先看到证据装配结果，再进入回答，更符合学习和分析场景。
 *
 * @param {object} result RAG 结果。
 * @param {string[]} progressLogs 当前已记录的进度日志。
 * @returns {void}
 */
function printDebugSummary(result, progressLogs) {
  console.log(`Question: ${result.question}`);
  console.log(`Question strategy: ${result.strategy}`);
  console.log(`Prompt template: ${result.promptTemplate}`);
  console.log(`Candidate topK: ${result.candidateTopK}`);
  console.log(`Final topK: ${result.topK}`);
  console.log(`Filter: ${result.filter || '(none)'}`);
  console.log(
    `Query embedding check: dim=${result.queryVectorDim}, nonZero=${result.queryNonZeroCount}/${result.queryVectorDim}`
  );
  console.log(`Progress events so far: ${progressLogs.length}\n`);

  printMatches('Raw candidates', result.rawMatches);
  printMatches('Reranked candidates', result.rerankedMatches.slice(0, result.topK));
  printMatches('Primary evidence matches', result.primaryMatches);
  printMatchesIfAny('Support evidence matches', result.supportMatches);
  printMatchesIfAny('Cross-section support matches', result.crossSupportMatches);
  printMatchesIfAny('Reserve evidence matches', result.reserveMatches);
  printMatches('Final matches', result.finalMatches);

  if (result.droppedMatches.length > 0) {
    printMatches('Dropped noisy candidates', result.droppedMatches.slice(0, 8));
  }

  console.log(`Messages count sent to model: ${result.messages.length}\n`);
}

/**
 * `ask:v7` 的 CLI 主入口。
 *
 * 这个函数负责：
 * - 参数解析；
 * - 会话历史加载与重置；
 * - 初始化模型与 Milvus 客户端；
 * - 触发 `runRAGv7()`；
 * - 在回答完成后写回本地 session。
 *
 * 它不负责检索、重排或 Prompt 组装细节，那些逻辑已经收回到 `src/rag7/` 内部模块。
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.resetSession) {
    await resetSession(args.sessionId);
    console.log(`Session reset: ${args.sessionId}`);
    return;
  }

  const session = await loadSession(args.sessionId);
  const history = sessionToPromptHistory(session, 3);

  const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0.2,
    configuration: {
      baseURL: process.env.OPENAI_BASE_URL,
    },
  });

  const client = new MilvusClient({
    address: MILVUS_ADDRESS,
  });

  if (args.debugMode) {
    console.log(`Connecting to Milvus at ${MILVUS_ADDRESS}...`);
  }

  await client.connectPromise;

  if (args.debugMode) {
    console.log('Connected.\n');
    console.log(`Session: ${args.sessionId}`);
    console.log(`Loaded history turns: ${session.turns.length}`);
    console.log(`Prompt history messages: ${history.length}`);
    console.log(`Config cache: ${args.noCache ? 'disabled (--no-cache)' : 'enabled'}`);
    console.log(`Debug mode: on`);
    console.log(`Answer mode: ${args.streamAnswer ? 'stream' : 'invoke'}\n`);
  }

  const progressLogs = [];
  let hasPrintedAnswerHeader = false;
  const writer = createSmoothWriter({
    enabled: args.streamAnswer,
  });

  const result = await runRAGv7({
    client,
    model,
    collectionName: COLLECTION_NAME,
    question: args.question,
    history,
    topK: args.topK,
    candidateTopK: args.candidateTopK,
    filter: args.filter,
    promptTemplate: args.promptTemplate,
    noCache: args.noCache,
    streamAnswer: args.streamAnswer,
    onProgress(message) {
      progressLogs.push(message);

      if (args.debugMode) {
        console.log(`[progress] ${message}`);
      }
    },
    onBeforeAnswer(payload) {
      if (!args.debugMode) {
        return;
      }

      console.log('');
      printDebugSummary(payload, progressLogs);
    },
    onAnswerToken(text) {
      if (!text) {
        return;
      }

      if (args.debugMode && !hasPrintedAnswerHeader) {
        console.log('\nAnswer (stream):\n');
        hasPrintedAnswerHeader = true;
      }

      writer.write(text);
    },
  });

  if (args.streamAnswer) {
    await writer.drain();
    process.stdout.write('\n');
  }

  await appendSessionTurn(args.sessionId, {
    question: args.question,
    answer: result.answer,
    strategy: result.strategy,
    promptTemplate: result.promptTemplate,
    finalMatchCount: result.finalMatches.length,
  });

  if (!args.streamAnswer) {
    if (args.debugMode) {
      console.log('Answer:\n');
    }

    console.log(result.answer);
  }
}

main().catch((error) => {
  console.error('ask-7 failed:', error);
  process.exit(1);
});
