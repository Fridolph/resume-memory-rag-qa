import { buildMessagesV7, getTemplateByStrategyV7 } from './prompt-builder-v7.mjs';
import { buildRAGContextV7 } from './context-builder-v7.mjs';

function noop() {}

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
  onAnswerToken = noop,
}) {
  // runRAGv7 是“生成层”的主流程。
  //
  // 1. 构建分层证据上下文
  // 2. 选择 prompt 模板
  // 3. 构建 chat messages
  // 4. 流式生成回答（默认）
  // 5. 汇总返回调试结果
  //
  // 补充一点架构演进背景：
  // - 早期版本里，pipeline 同时负责策略识别 / rerank / context 组装 / LLM 调用
  // - 到 v6 / v7，这些“证据准备”职责已经下沉到 context-builder
  // - 因此现在的 rag-pipeline 更像一个纯粹的“流程编排器”
  //
  // 它只负责串起：
  // - buildRAGContextV7()
  // - getTemplateByStrategyV7()
  // - buildMessagesV7()
  // - model.stream() / model.invoke()
  //
  // 这样每层职责会更稳定，也更方便后续替换检索层或 Prompt 层实现。
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
