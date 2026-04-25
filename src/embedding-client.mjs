import 'dotenv/config';
import { OpenAIEmbeddings } from '@langchain/openai';

const embeddingsBaseURL = process.env.EMBEDDINGS_BASE_URL || process.env.OPENAI_BASE_URL;

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDINGS_API_KEY || process.env.OPENAI_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: embeddingsBaseURL,
  },
  // 在生成嵌入时，每次会处理最多 10 个输入数据
  // 这种批量处理可以提高效率，减少 API 请求次数
  batchSize: 10,
});

function normalizeEmbeddingsEndpoint(url) {
  if (!url) return undefined;
  const trimmed = url.trim();
  if (!trimmed) return undefined;

  return trimmed.endsWith('/embeddings')
    ? trimmed
    : `${trimmed.replace(/\/+$/, '')}/embeddings`;
}

function assertSupportedEmbeddingsEndpoint(endpoint) {
  if (!endpoint) return;

  if (endpoint.includes('api.deepseek.com')) {
    throw new Error(
      'DeepSeek 官方 API 目前没有 embeddings 端点，不能把 EMBEDDINGS_URL 配成 https://api.deepseek.com。' +
        '请继续使用写入 Milvus 时的 embedding 服务，或换成支持 embeddings 的模型后重新 ingest。'
    );
  }
}

function resolveEmbeddingsApiKey() {
  // 一旦显式配置了 embedding endpoint/baseURL，就必须显式配置 embedding key。
  // 否则在 Chat 模型切到 DeepSeek 后，很容易误用 OPENAI_API_KEY 去请求 GLM embeddings，
  // 最终得到“令牌不正确”的 401，但真实原因其实是 key 来源错了。
  if (process.env.EMBEDDINGS_URL || process.env.EMBEDDINGS_BASE_URL) {
    if (!process.env.EMBEDDINGS_API_KEY) {
      throw new Error('已配置 EMBEDDINGS_URL/EMBEDDINGS_BASE_URL，请同时设置 EMBEDDINGS_API_KEY，避免误用 OPENAI_API_KEY');
    }

    return {
      apiKey: process.env.EMBEDDINGS_API_KEY,
      source: 'EMBEDDINGS_API_KEY',
    };
  }

  return {
    apiKey: process.env.EMBEDDINGS_API_KEY || process.env.OPENAI_API_KEY,
    source: process.env.EMBEDDINGS_API_KEY ? 'EMBEDDINGS_API_KEY' : 'OPENAI_API_KEY',
  };
}

function getEmbeddingsEndpointCandidates() {
  const explicitEmbeddingsEndpoint = normalizeEmbeddingsEndpoint(process.env.EMBEDDINGS_URL);
  if (explicitEmbeddingsEndpoint) {
    assertSupportedEmbeddingsEndpoint(explicitEmbeddingsEndpoint);
    return [explicitEmbeddingsEndpoint];
  }

  const candidates = [normalizeEmbeddingsEndpoint(embeddingsBaseURL)].filter(Boolean);
  candidates.forEach(assertSupportedEmbeddingsEndpoint);

  return [...new Set(candidates)];
}

export async function getEmbedding(text) {
  const endpoints = getEmbeddingsEndpointCandidates();
  const { apiKey, source: apiKeySource } = resolveEmbeddingsApiKey();

  if (endpoints.length > 0) {
    let lastError;

    for (const endpoint of endpoints) {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: process.env.EMBEDDINGS_MODEL_NAME,
          input: text,
        }),
      });

      if (!response.ok) {
        lastError = `embedding 请求失败: ${response.status} ${await response.text()} (url: ${endpoint}, keySource: ${apiKeySource}, model: ${process.env.EMBEDDINGS_MODEL_NAME})`;
        continue;
      }

      const result = await response.json();
      // 业界通用格式，openai的这里是数组
      const vector = result.data?.[0]?.embedding;

      if (!Array.isArray(vector)) {
        throw new Error(`embedding 响应格式异常: ${JSON.stringify(result)}`);
      }

      return vector;
    }

    throw new Error(lastError || 'embedding 请求失败：没有可用的 embeddings endpoint');
  }

  // LangChain 内部帮你封装了 fetch + 解析 + 提取的全套逻辑
  // 等价于上面手动 fetch 那一大段代码，可理解为兜底
  return embeddings.embedQuery(text);
}

export function validateEmbedding(vector, label = 'embedding') {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new Error(`${label} 为空，请检查 EMBEDDINGS_MODEL_NAME / EMBEDDINGS_URL / EMBEDDINGS_API_KEY`);
  }

  const nonZeroCount = vector.filter((value) => value !== 0).length;

  if (nonZeroCount === 0) {
    throw new Error(`${label} 全部为 0，请检查 EMBEDDINGS_MODEL_NAME / EMBEDDINGS_URL / EMBEDDINGS_API_KEY`);
  }

  return {
    dimension: vector.length,
    nonZeroCount,
  };
}

export async function resolveVectorDim() {
  const probeVector = await getEmbedding('resume vector dimension probe');
  const { dimension } = validateEmbedding(probeVector, 'probe embedding');
  return dimension;
}
