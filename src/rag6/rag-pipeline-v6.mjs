import { buildMessages, getTemplateByStrategy } from '../rag5/prompt-builder-v5.mjs';
import { buildRAGContextV6 } from './context-builder-v6.mjs';

export async function runRAGv6({
  client,
  model,
  collectionName,
  question,
  history = [],
  topK = 8,
  candidateTopK = Math.max(topK * 2, 10),
  filter = '',
  outputFields,
  strategy = '',
  promptTemplate = '',
  noCache = false,
}) {
  // runRAGv6 是“生成层”的主流程。
  //
  // 你可以把它理解为：
  // - buildRAGContextV6() 负责“找证据”
  // - runRAGv6() 负责“拿着证据去问模型”
  //
  // 这样拆分的好处是：
  // - 当你只想调试检索、重排、去噪时，只看 buildRAGContextV6()
  // - 当你想看最终回答链路时，再看 runRAGv6()

  // 1. 构建 RAG 上下文
  //    这一层会完成：
  //    - 策略识别
  //    - 向量维度校验
  //    - Milvus 召回
  //    - rerank / denoise
  //    - 主证据 / 辅助证据分层
  //    - 最终 context 组装
  const ragContext = await buildRAGContextV6({
    client,
    collectionName,
    question,
    topK,
    candidateTopK,
    filter,
    outputFields,
    strategy,
    noCache,
  });

  // 2. 选择 Prompt 模板
  //    规则是：
  //    - 如果调用方显式传了 promptTemplate，就优先用它
  //    - 否则根据 strategy 自动映射到合适模板
  const resolvedTemplate = await getTemplateByStrategy(ragContext.strategy, promptTemplate);

  // 3. 构建 messages
  //    buildMessages() 会把这些内容拼成 LangChain 的消息数组：
  //    - SystemMessage：系统规则 + 当前检索到的 context
  //    - 历史 user / assistant 消息
  //    - 当前问题 HumanMessage(question)
  //
  //    这样模型拿到的就不是一大段裸字符串，
  //    而是更接近真实对话结构的消息序列。
  const messages = await buildMessages(resolvedTemplate, {
    context: ragContext.context,
    question,
    history,
  });

  // 4. 调用模型生成答案
  //    到这里才真正进入 LLM 推理阶段。
  //    前面所有步骤本质上都在做“为这次回答准备高质量证据”。
  const response = await model.invoke(messages);

  // 5. 汇总返回结果
  //    返回值里不仅有最终 answer，
  //    还保留了策略、候选集、证据分层、messages 等调试信息，
  //    方便后续继续观察“回答为什么会这样生成”。
  return {
    ...ragContext,
    promptTemplate: resolvedTemplate,
    messages,
    answer: response.content,
  };
}
