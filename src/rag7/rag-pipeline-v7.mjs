import { buildMessagesV7, getTemplateByStrategyV7 } from './prompt-builder-v7.mjs';
import { buildRAGContextV7 } from './context-builder-v7.mjs';

function noop() {}

/**
 * 从流式 chunk 中提取可打印文本。
 *
 * 不同模型 / SDK 的 chunk 结构可能并不完全一致，因此这里做一层兼容提取：
 * - 直接字符串内容；
 * - 数组形式的 content；
 * - 含 `text` 字段的对象片段。
 *
 * @param {any} chunk 模型返回的流式片段。
 * @returns {string} 可写入 stdout 的文本；若没有文本则返回空字符串。
 */
function extractTextFromChunk(chunk) {
  const content = chunk?.content;

  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }

        if (typeof item?.text === 'string') {
          return item.text;
        }

        return '';
      })
      .join('');
  }

  return '';
}

/**
 * 执行版本 7 的完整 RAG 主流程。
 *
 * `runRAGv7()` 的职责是“流程编排”，而不是“检索细节实现”。
 * 它负责：
 * - 调用 `buildRAGContextV7()` 构建证据上下文；
 * - 选择 Prompt 模板；
 * - 组装 Chat messages；
 * - 调用模型进行流式或非流式回答；
 * - 汇总调试数据并返回。
 *
 * 它不负责：
 * - 策略识别规则；
 * - rerank 与去噪；
 * - 证据分层。
 *
 * 这些逻辑已经下沉到 `context-builder-v7`，以保持 pipeline 纯粹。
 *
 * @param {{
 *   client: import('@zilliz/milvus2-sdk-node').MilvusClient,
 *   model: any,
 *   collectionName: string,
 *   question: string,
 *   history?: Array<{role: string, content: string}>,
 *   topK?: number,
 *   candidateTopK?: number,
 *   filter?: string,
 *   outputFields?: string[],
 *   strategy?: string,
 *   promptTemplate?: string,
 *   noCache?: boolean,
 *   streamAnswer?: boolean,
 *   onProgress?: (message: string) => void,
 *   onBeforeAnswer?: (payload: object) => Promise<void> | void,
 *   onAnswerToken?: (text: string) => void,
 * }} options 运行参数。
 * @returns {Promise<object>} 完整的 RAG 运行结果。
 * @throws {Error} 当上下文构建、Prompt 构建或模型调用失败时抛错。
 */
export async function runRAGv7({
  client,
  model,
  collectionName,
  question,
  history = [],
  topK = 8,
  candidateTopK,
  filter = '',
  outputFields,
  strategy = '',
  promptTemplate = '',
  noCache = false,
  streamAnswer = true,
  onProgress = noop,
  onBeforeAnswer = noop,
  onAnswerToken = noop,
}) {
  let ragContext;

  try {
    ragContext = await buildRAGContextV7({
      client,
      collectionName,
      question,
      topK,
      candidateTopK,
      filter,
      outputFields,
      strategy,
      noCache,
      onProgress,
    });
  } catch (error) {
    throw new Error(`RAG 上下文构建失败: ${error.message}`);
  }

  onProgress('6/7 正在构建 Prompt messages…');
  const resolvedTemplate = await getTemplateByStrategyV7(ragContext.strategy, promptTemplate);

  let messages;
  try {
    messages = await buildMessagesV7(resolvedTemplate, {
      context: ragContext.context,
      question,
      history,
    });
  } catch (error) {
    throw new Error(`Prompt messages 构建失败: ${error.message}`);
  }

  await onBeforeAnswer({
    ...ragContext,
    promptTemplate: resolvedTemplate,
    messages,
  });

  let answer = '';

  if (streamAnswer) {
    onProgress('7/7 已开始流式生成回答…');

    try {
      const stream = await model.stream(messages);

      for await (const chunk of stream) {
        const text = extractTextFromChunk(chunk);

        if (!text) {
          continue;
        }

        answer += text;
        onAnswerToken(text);
      }
    } catch (error) {
      throw new Error(`LLM 流式调用失败: ${error.message}`);
    }
  } else {
    onProgress('7/7 正在生成回答…');

    try {
      const response = await model.invoke(messages);
      answer = String(response?.content ?? '');
    } catch (error) {
      throw new Error(`LLM 调用失败: ${error.message}`);
    }
  }

  if (!String(answer).trim()) {
    throw new Error('LLM 返回了空内容');
  }

  return {
    ...ragContext,
    promptTemplate: resolvedTemplate,
    messages,
    answer,
  };
}
