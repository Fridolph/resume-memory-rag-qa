import 'dotenv/config';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { ChatOpenAI } from '@langchain/openai';
import {
  appendSessionTurn,
  loadSession,
  resetSession,
  sessionToPromptHistory,
} from './rag4/history-store.mjs';
import { runRAGv6 } from './rag6/rag-pipeline-v6.mjs';

// v6 入口脚本仍然保持轻量：
// - 读参数
// - 准备 session
// - 初始化 model / client
// - 调用 runRAGv6
// - 打印分层证据结果
const COLLECTION_NAME = process.env.RESUME_RAG_COLLECTION || 'resume_profile_chunks';
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS || 'localhost:19530';

function parseArgs(argv) {
  // CLI 参数约定：
  // --session         当前会话名
  // --topK            最终进入 prompt 的证据条数
  // --candidateTopK   初始候选池大小
  // --filter          Milvus 过滤表达式
  // --template        显式指定 prompt 模板
  // --reset-session   清空当前 session 历史
  const options = {
    sessionId: 'resume-demo-v6',
    topK: 8,
    candidateTopK: 16,
    filter: '',
    promptTemplate: '',
    resetSession: false,
    noCache: false,
  };
  const questionParts = [];

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];

    if (current === '--session') {
      options.sessionId = argv[index + 1] || options.sessionId;
      index += 1;
      continue;
    }

    if (current === '--topK') {
      options.topK = Number(argv[index + 1] || options.topK);
      index += 1;
      continue;
    }

    if (current === '--candidateTopK') {
      options.candidateTopK = Number(argv[index + 1] || options.candidateTopK);
      index += 1;
      continue;
    }

    if (current === '--filter') {
      options.filter = argv[index + 1] || '';
      index += 1;
      continue;
    }

    if (current === '--template') {
      options.promptTemplate = argv[index + 1] || '';
      index += 1;
      continue;
    }

    if (current === '--reset-session') {
      options.resetSession = true;
      continue;
    }

    if (current === '--no-cache') {
      // 开发调试阶段有时会边改 JSON 配置边反复跑脚本。
      // 这里提供一个 --no-cache 开关：
      // - 本次进程里所有配置文件都强制重新读取
      // - 不走内存缓存
      //
      // 这样你改了 keyword-hints / rerank-config / precision-config 后，
      // 不用怀疑是不是“缓存没生效”。
      options.noCache = true;
      continue;
    }

    if (current !== '--') {
      questionParts.push(current);
    }
  }

  return {
    ...options,
    question:
      questionParts.join(' ').trim() || '这个候选人有哪些 AI Agent 开发相关经验？',
  };
}

function printMatches(title, matches) {
  // v6 的打印相比前几版多了一层：
  // - primary evidence
  // - support evidence
  //
  // 目的是让你在运行时直观看到：
  // “哪些是主证据，哪些只是辅助补充。”
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

    if (Array.isArray(item._noiseReasons) && item._noiseReasons.length > 0) {
      console.log(`   noiseCheck: ${item._noiseReasons.join('；')}`);
    }

    // v6 新增字段：
    // _isPrimaryEvidence / _isSupportEvidence
    // 用来标识这一条在最终上下文中的证据层级。
    if (item._isPrimaryEvidence) {
      console.log('   evidenceTier: primary');
    } else if (item._isSupportEvidence) {
      console.log('   evidenceTier: support');
    }

    console.log(`   subsectionKey: ${item.subsection_key}`);
    console.log(`   tags: ${Array.isArray(item.tags) ? item.tags.join(', ') : ''}`);
    console.log(`   content: ${String(item.content).slice(0, 180)}\n`);
  });
}

async function main() {
  // 入口编排保持尽量简单，避免入口脚本本身变成大杂烩。
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

  console.log(`Connecting to Milvus at ${MILVUS_ADDRESS}...`);
  await client.connectPromise;
  console.log('Connected.\n');

  console.log(`Session: ${args.sessionId}`);
  console.log(`Loaded history turns: ${session.turns.length}`);
  console.log(`Prompt history messages: ${history.length}\n`);
  console.log(`Config cache: ${args.noCache ? 'disabled (--no-cache)' : 'enabled'}\n`);

  // 运行版本 6 主流程。
  const result = await runRAGv6({
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
  });

  // 继续沿用历史存储：
  // question + answer 成对保存，方便多轮追问。
  await appendSessionTurn(args.sessionId, {
    question: args.question,
    answer: result.answer,
    strategy: result.strategy,
    promptTemplate: result.promptTemplate,
    finalMatchCount: result.finalMatches.length,
  });

  console.log(`Question: ${result.question}`);
  console.log(`Question strategy: ${result.strategy}`);
  console.log(`Prompt template: ${result.promptTemplate}`);
  console.log(`Candidate topK: ${result.candidateTopK}`);
  console.log(`Final topK: ${result.topK}`);
  console.log(`Filter: ${result.filter || '(none)'}`);
  console.log(
    `Query embedding check: dim=${result.queryVectorDim}, nonZero=${result.queryNonZeroCount}/${result.queryVectorDim}\n`
  );

  printMatches('Raw candidates', result.rawMatches);
  printMatches('Reranked candidates', result.rerankedMatches.slice(0, result.topK));
  printMatches('Primary evidence matches', result.primaryMatches);
  printMatches('Support evidence matches', result.supportMatches);
  printMatches('Final matches', result.finalMatches);

  if (result.droppedMatches.length > 0) {
    // 这里只截前 8 条 dropped，避免日志过长。
    printMatches('Dropped noisy candidates', result.droppedMatches.slice(0, 8));
  }

  console.log(`Messages count sent to model: ${result.messages.length}\n`);
  console.log('Answer:\n');
  console.log(result.answer);
}

main().catch((error) => {
  // CLI demo 统一在入口兜底。
  console.error('ask-6 failed:', error);
  process.exit(1);
});
