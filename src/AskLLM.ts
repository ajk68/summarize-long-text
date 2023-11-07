import { AI } from "@raycast/api";
import { LLMParams, LLMResponse, ModelSizes } from "./interfaces";
import OpenAI from "openai";

/**
 * Asks a question to the LLM (Language Learning Model) using either OpenAI or Raycast models.
 *
 * @param text - The text or question to ask the model.
 * @param LLMParams - The parameters for choosing and interacting with the model.
 * @returns A promise that resolves to an object containing the response text.
 */
const AskLLM = async (text: string, LLMParams: LLMParams): Promise<LLMResponse> => {
  let responseText = "";
  let promptTokensUsed = 0;
  let responseTokensUsed = 0;

  //console.log("AskLLM: LLMParams: ", LLMParams);

  // Handle OpenAI models
  if (LLMParams.modelName.startsWith("OPENAI-")) {
    const modelName = LLMParams.modelName.replace("OPENAI-", "");
    const openai = new OpenAI({ apiKey: LLMParams.openaiApiToken || undefined });
    try {
      const result = await openai.chat.completions.create({
        model: modelName,
        temperature: LLMParams.creativity,
        messages: [{ role: "user", content: text }],
      });
      responseText = result.choices[0].message?.content || "";
      promptTokensUsed += result.usage?.prompt_tokens || 0;
      responseTokensUsed += result.usage?.completion_tokens || 0;
    } catch (error) {
      // TODO: add toast
      if (error instanceof Error) {
        error.message = "Openai completions: " + error.message;
        error.name = "OPENAI_COMPLETIONS_ERROR";
      }
      throw error;
    }
  }
  // Handle Raycast models
  else if (LLMParams.modelName.startsWith("raycast")) {
    const modelName = LLMParams.modelName.replace("raycast-", "");
    try {
      const result = await AI.ask(text, { model: modelName as AI.Model, creativity: LLMParams.creativity });
      responseText = result || "";
    } catch (error) {
      if (error instanceof Error) {
        error.name = "RAYCASTAI_ASK_ERROR";
        error.message = "AI.ask: " + error.message;
      }
    }
  }
  // Handle unsupported models
  else {
    throw new Error("Unsupported AI model");
  }

  return {
    text: responseText,
    promptTokensUsed: promptTokensUsed,
    responseTokensUsed: responseTokensUsed,
  };
};

export default AskLLM;

/**
 * Function to get the maximum tokens for a specified model.
 *
 * @param {string} modelName - The name of the model.
 * @returns {number} The maximum number of tokens for the model.
 */
export function getModelMaxTokens(modelName: string) {
  const modelSizes: ModelSizes = {
    "raycast-gpt-3.5-turbo": 16000,
    "OPENAI-gpt-3.5": 4000,
    "OPENAI-gpt-3.5-turbo-1106": 16000,
    "OPENAI-gpt-4": 8000,
    "OPENAI-gpt-4-1106-preview": 10000,
  };
  return modelSizes[modelName] || 8000;
}

/**
 * Function to get the usable tokens for a specified model.
 *
 * @param {string} modelName - The name of the model.
 * @returns {number} The usable number of tokens for the model.
 */
export function getModelUsableTokens(modelName: string) {
  // This is a bit of a hack to get the max usable characters for a model
  // Get max token context size and substract 1000 for prompt and system and
  // 1500 tokens for response. Rest is for text to summarize
  let maxUsableTokens = getModelMaxTokens(modelName) - 2500 || 1500;
  // TODO: clean this up after debugging
  if (maxUsableTokens > 5000) { 
    maxUsableTokens = 5000;
  } 
  return maxUsableTokens
}

interface TokenPrices {
  [key: string]: [number, number];
}

/**
 * Function to get the cost of using a specified model for prompt and response tokens.
 *
 * @param {string} modelName - The name of the model.
 * @param {number} promptTokens - The number of prompt tokens.
 * @param {number} responseTokens - The number of response tokens.
 * @returns {number} The total cost.
 */
export function getCost(modelName: string, promptTokens: number, responseTokens: number): number {
  // returns array with two elements price per 1000 tokens for prompt and response
  // source: https://openai.com/pricing
  const tokenPrices: TokenPrices = {
    "raycast-gpt-3.5-turbo": [0, 0],
    "OPENAI-gpt-3.5-turbo": [0.0015, 0.002],
    "OPENAI-gpt-3.5-turbo-1106": [0.001, 0.002],
    "OPENAI-gpt-4": [0.03, 0.06],
    "OPENAI-gpt-4-1106-preview": [0.01, 0.03],
  }

  const [promptPrice, responsePrice] = tokenPrices[modelName] || [0, 0];
  const cost = 0.001 * (promptPrice * promptTokens + responsePrice * responseTokens);
  return cost;
}
