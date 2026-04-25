import { ChatOpenAI } from '@langchain/openai';

/**
 * 创建 DeepSeek v4 专用 Chat 模型。
 *
 * 这个入口只负责回答生成，不负责 embedding。
 * 如果 embedding 仍然使用 GLM，就不要让 DeepSeek 的 key/baseURL 影响向量请求。
 *
 * 注意：DeepSeek v4 是思考模型，temperature 等采样参数在 thinking mode
 * 下会失效或不被支持，所以这个专用入口不传 temperature。
 *
 * @returns {ChatOpenAI}
 */
export function createDeepSeekV4ChatModel() {
  if (!process.env.DEEPSEEK_API_KEY) {
    throw new Error('DeepSeek v4 聊天模型未配置：请设置 DEEPSEEK_API_KEY');
  }

  return new ChatOpenAI({
    model: process.env.DEEPSEEK_MODEL_NAME || 'deepseek-v4-flash',
    apiKey: process.env.DEEPSEEK_API_KEY,
    configuration: {
      baseURL: process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com',
    },
  });
}
