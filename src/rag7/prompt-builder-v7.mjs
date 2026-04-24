import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

const CURRENT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROMPTS_DIR = path.join(CURRENT_DIR, 'prompts');
const STRATEGY_TEMPLATE_MAP_PATH = path.join(CURRENT_DIR, 'config', 'strategy-template-map.json');

let strategyTemplateMapCache = null;

function renderTemplate(template, variables) {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key) => String(variables[key] ?? ''));
}

async function loadStrategyTemplateMap() {
  if (strategyTemplateMapCache) {
    return strategyTemplateMapCache;
  }

  const raw = await fs.readFile(STRATEGY_TEMPLATE_MAP_PATH, 'utf8');
  strategyTemplateMapCache = JSON.parse(raw);
  return strategyTemplateMapCache;
}

export async function loadPromptTemplateV7(templateName) {
  const filePath = path.join(PROMPTS_DIR, `${templateName}.md`);
  return fs.readFile(filePath, 'utf8');
}

export async function getTemplateByStrategyV7(strategy, preferredTemplate = '') {
  if (preferredTemplate) {
    return preferredTemplate;
  }

  const strategyTemplateMap = await loadStrategyTemplateMap();
  return strategyTemplateMap[strategy] ?? 'resume_qa';
}

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
