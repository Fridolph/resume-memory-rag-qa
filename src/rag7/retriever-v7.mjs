import { MetricType } from '@zilliz/milvus2-sdk-node';

/**
 * 版本 7 的候选片段召回器。
 *
 * 这个模块在 v7 里被就地收回，目的是让版本 7 从旧的共享目录中闭合出来。
 * 它仍然只做一件事：
 * - 接收已经向量化完成的 queryVector；
 * - 向 Milvus 发起检索；
 * - 返回原始候选结果。
 *
 * 它不负责：
 * - 策略识别；
 * - rerank；
 * - 去噪；
 * - 证据分层。
 *
 * @param {import('@zilliz/milvus2-sdk-node').MilvusClient} client Milvus 客户端。
 * @param {number[]} queryVector 查询向量。
 * @param {{
 *   collectionName: string,
 *   topK?: number,
 *   metricType?: string,
 *   filter?: string,
 *   outputFields?: string[],
 * }} [options={}] 查询选项。
 * @returns {Promise<Array<object>>} 原始候选片段列表。
 */
export async function retrieveV7(client, queryVector, options = {}) {
  const {
    collectionName,
    topK = 5,
    metricType = MetricType.COSINE,
    filter = '',
    outputFields = [
      'source_id',
      'locale',
      'section',
      'subsection_key',
      'subsection_title',
      'entity_type',
      'content',
      'tags',
      'chunk_index',
      'chunk_count',
    ],
  } = options;

  if (!collectionName) {
    throw new Error('[retriever-v7] collectionName 是必填参数');
  }

  const payload = {
    collection_name: collectionName,
    vector: queryVector,
    limit: topK,
    metric_type: metricType,
    output_fields: outputFields,
  };

  if (filter) {
    payload.filter = filter;
  }

  const result = await client.search(payload);
  return result.results || [];
}
