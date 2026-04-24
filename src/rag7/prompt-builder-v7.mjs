import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

const CURRENT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROMPTS_DIR = path.join(CURRENT_DIR, 'prompts');
const STRATEGY_TEMPLATE_MAP_PATH = path.join(CURRENT_DIR, 'config', 'strategy-template-map.json');

let strategyTemplateMapCache = null;

/**
 * 执行一个极简模板替换。
 *
 * 当前 v7 只需要 `{{context}}` 这种轻量插值，因此没有引入额外模板引擎。
 *
 * @param {string} template Prompt 模板原文。
 * @param {Record<string, unknown>} variables 要注入的变量。
 * @returns {string} 渲染后的模板字符串。
 */
function renderTemplate(template, variables) {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key) => String(variables[key] ?? ''));
}

/**
 * 读取“策略 -> 模板名”映射。
 *
 * @returns {Promise<Record<string, string>>}
 */
async function loadStrategyTemplateMap() {
  if (strategyTemplateMapCache) {
    return strategyTemplateMapCache;
  }

  const raw = await fs.readFile(STRATEGY_TEMPLATE_MAP_PATH, 'utf8');
  strategyTemplateMapCache = JSON.parse(raw);
  return strategyTemplateMapCache;
}

/**
 * 读取指定 Prompt 模板原文。
 *
 * @param {string} templateName 模板名，不带扩展名。
 * @returns {Promise<string>} 模板原文。
 */
export async function loadPromptTemplateV7(templateName) {
  const filePath = path.join(PROMPTS_DIR, `${templateName}.md`);
  return fs.readFile(filePath, 'utf8');
}

/**
 * 根据策略解析最终要使用的模板名。
 *
 * 如果调用方显式传入 `preferredTemplate`，则优先使用调用方指定的模板。
 *
 * @param {string} strategy 当前问题策略。
 * @param {string} [preferredTemplate=''] 显式指定的模板名。
 * @returns {Promise<string>} 最终模板名。
 */
export async function getTemplateByStrategyV7(strategy, preferredTemplate = '') {
  if (preferredTemplate) {
    return preferredTemplate;
  }

  const strategyTemplateMap = await loadStrategyTemplateMap();
  return strategyTemplateMap[strategy] ?? 'resume_qa';
}

/**
 * 把历史对话转成 Chat messages。
 *
 * @param {Array<{role?: string, content?: string}>} [history=[]] 历史消息数组。
 * @returns {Array<AIMessage | HumanMessage>} LangChain message 数组。
 */
function historyToMessages(history = []) {
  if (!Array.isArray(history) || history.length === 0) {
    return [];
  }

  return history.map((item) => {
    const content = String(item?.content ?? '').trim();
    const role = String(item?.role ?? 'user').trim();

    if (role === 'assistant') {
      return new AIMessage(content);
    }

    return new HumanMessage(content);
  });
}

/**
 * 组装版本 7 的最终 messages。
 *
 * 结构保持为：
 * - `SystemMessage`：放规则与上下文；
 * - 历史消息：保留最近多轮对话；
 * - `HumanMessage`：放当前问题。
 *
 * @param {string} templateName 模板名。
 * @param {{
 *   question: string,
 *   context?: string,
 *   history?: Array<{role?: string, content?: string}>,
 * }} payload Prompt 负载。
 * @returns {Promise<Array<SystemMessage | AIMessage | HumanMessage>>}
 * @throws {Error} 当 `question` 缺失时抛错。
 */
export async function buildMessagesV7(templateName, payload) {
  if (!payload?.question) {
    throw new Error('[prompt-builder-v7] question 是必填项');
  }

  if (!payload?.context) {
    console.warn('[prompt-builder-v7] context 为空，召回结果可能为空。');
  }

  const template = await loadPromptTemplateV7(templateName);
  const systemPrompt = renderTemplate(template, {
    context: payload.context,
  });

  return [
    new SystemMessage(systemPrompt),
    ...historyToMessages(payload.history),
    new HumanMessage(payload.question),
  ];
}
