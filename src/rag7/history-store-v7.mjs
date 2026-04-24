import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * 版本 7 的默认历史目录。
 *
 * 注意这里不再沿用旧 demo 时代的 `examples/...` 路径，而是直接落在当前仓库根目录下：
 * - `.runtime/chat-sessions`
 *
 * 这样更符合“v7 版本内闭合”的目标，也避免继续携带已经过时的目录债。
 */
export const DEFAULT_HISTORY_DIR = path.resolve(
  process.cwd(),
  '.runtime/chat-sessions'
);

/**
 * 确保目录存在。
 *
 * @param {string} dirPath 需要创建的目录路径。
 * @returns {Promise<void>}
 */
async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

/**
 * 清洗 sessionId，避免直接把不安全字符串写成文件名。
 *
 * 为什么这里必须做 sanitize：
 * - sessionId 会参与最终的 JSON 文件名拼接；
 * - 如果保留空白、路径分隔符或其他特殊字符，容易引发路径问题或读写失败；
 * - 对学习型 demo 来说，文件名稳定、可读、可预测也更重要。
 *
 * @param {string} sessionId 原始会话 ID。
 * @returns {string} 可安全落盘的会话 ID。
 */
function sanitizeSessionId(sessionId) {
  return String(sessionId || 'default')
    .trim()
    .replace(/[^a-zA-Z0-9-_]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'default';
}

/**
 * 获取某个 session 对应的 JSON 文件路径。
 *
 * @param {string} sessionId 会话 ID。
 * @param {string} [historyDir=DEFAULT_HISTORY_DIR] 历史目录。
 * @returns {string} 会话文件绝对路径。
 */
function getSessionFilePath(sessionId, historyDir = DEFAULT_HISTORY_DIR) {
  const safeSessionId = sanitizeSessionId(sessionId);
  return path.join(historyDir, `${safeSessionId}.json`);
}

/**
 * 加载会话历史。
 *
 * 版本 7 继续沿用“本地 JSON 文件会话存储”方案，而不是写入 Milvus。
 * 原因是：
 * - 会话历史本质上是运行时状态，不是简历知识库的一部分；
 * - 放本地文件更易读、易调试，也更适合学习型仓库；
 * - 避免把临时对话数据混入检索用的向量集合。
 *
 * @param {string} sessionId 会话 ID。
 * @param {string} [historyDir=DEFAULT_HISTORY_DIR] 历史目录。
 * @returns {Promise<{sessionId: string, filePath: string, turns: Array<object>}>}
 * 返回会话对象；若文件不存在，则返回一个空会话。
 */
export async function loadSession(sessionId, historyDir = DEFAULT_HISTORY_DIR) {
  await ensureDir(historyDir);
  const filePath = getSessionFilePath(sessionId, historyDir);

  try {
    const raw = await fs.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw);

    return {
      sessionId: sanitizeSessionId(parsed.sessionId || sessionId),
      filePath,
      turns: Array.isArray(parsed.turns) ? parsed.turns : [],
    };
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }

    return {
      sessionId: sanitizeSessionId(sessionId),
      filePath,
      turns: [],
    };
  }
}

/**
 * 将会话 turns 转成 Prompt 所需的历史消息结构。
 *
 * 注意这里只保留最近若干轮历史，避免上下文无限膨胀。
 * 版本 7 默认继续取最近 3 轮，兼顾：
 * - 多轮连续性；
 * - Token 成本；
 * - 调试时的可控性。
 *
 * @param {{turns?: Array<{question?: string, answer?: string}>}} session 会话对象。
 * @param {number} [maxTurns=3] 最多保留多少轮问答。
 * @returns {Array<{role: 'user' | 'assistant', content: string}>}
 */
export function sessionToPromptHistory(session, maxTurns = 3) {
  const turns = Array.isArray(session?.turns) ? session.turns.slice(-maxTurns) : [];

  return turns.flatMap((turn) => [
    {
      role: 'user',
      content: turn.question,
    },
    {
      role: 'assistant',
      content: turn.answer,
    },
  ]);
}

/**
 * 追加一轮会话到历史文件。
 *
 * 这里会只保留最近若干轮，而不是无限增长。
 * 这样做的原因是：
 * - 学习型 demo 不需要永久保留所有细节；
 * - 历史越长，后续调试越不直观；
 * - 对 Prompt 来说，较近的几轮通常最有价值。
 *
 * @param {string} sessionId 会话 ID。
 * @param {{question?: string, answer?: string, strategy?: string, promptTemplate?: string, finalMatchCount?: number}} turn 当前轮次。
 * @param {{historyDir?: string, maxStoredTurns?: number}} [options={}] 可选参数。
 * @returns {Promise<{sessionId: string, filePath: string, turns: Array<object>}>}
 */
export async function appendSessionTurn(
  sessionId,
  turn,
  {
    historyDir = DEFAULT_HISTORY_DIR,
    maxStoredTurns = 20,
  } = {}
) {
  const session = await loadSession(sessionId, historyDir);
  const nextTurn = {
    createdAt: new Date().toISOString(),
    question: String(turn.question || '').trim(),
    answer: String(turn.answer || '').trim(),
    strategy: String(turn.strategy || '').trim(),
    promptTemplate: String(turn.promptTemplate || '').trim(),
    finalMatchCount: Number(turn.finalMatchCount || 0),
  };

  const turns = [...session.turns, nextTurn].slice(-maxStoredTurns);
  const payload = {
    sessionId: session.sessionId,
    turns,
  };

  await ensureDir(historyDir);
  await fs.writeFile(session.filePath, JSON.stringify(payload, null, 2), 'utf8');

  return {
    ...session,
    turns,
  };
}

/**
 * 重置某个 session 的本地历史。
 *
 * @param {string} sessionId 会话 ID。
 * @param {string} [historyDir=DEFAULT_HISTORY_DIR] 历史目录。
 * @returns {Promise<void>}
 */
export async function resetSession(sessionId, historyDir = DEFAULT_HISTORY_DIR) {
  const session = await loadSession(sessionId, historyDir);

  try {
    await fs.unlink(session.filePath);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }
}
