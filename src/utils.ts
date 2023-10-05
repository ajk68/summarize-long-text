import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import AskLLM from "./AskLLM";
import { getBlockSummaryPrompt, getFinalSummaryPrompt } from "./prompts";
import { encodingForModel } from "js-tiktoken";
import { LLMParams } from "./interfaces";
import { getModelUsableTokens } from "./AskLLM";

// Define an interface for the result from the AI
interface AIResult {
  text: string;
  promptTokensUsed: number;
  responseTokensUsed: number;
}

/**
 * Splits a given text into smaller chunks.
 *
 * @param text - The text to be split.
 * @param chunkSize - Maximum size of each chunk in tokens.
 * @param chunkOverlap - Number of overlapping characters between chunks.
 * @returns An array of text chunks.
 */
export const splitText = async (text: string, chunkSize = 4000, chunkOverlap = 50): Promise<string[]> => {
  // Initialize the text splitter
  // Cannot get TokenTextSplitter to work, so converting to characters
  const tokensInChars = Math.round((chunkSize * text.length) / getTokens(text));
  const overlapInChars = Math.round((chunkOverlap * text.length) / getTokens(text));
  //console.log("TOKENS IN CHARS: ", tokensInChars, "OVERLAP ", overlapInChars);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: tokensInChars,
    chunkOverlap: overlapInChars,
  });

  // Perform the text split operation
  const docOutput = await splitter.splitDocuments([new Document({ pageContent: text })]);

  // Return the chunks
  return docOutput.map((doc: Document) => doc.pageContent);
};

/**
 * Splits a given text into smaller chunks, summarizes each chunk, and returns the summaries.
 *
 * @param text - The text to be split and summarized.
 * @param LLMParams - LLM parameters set in Preferences.
 * @returns A object with text, prompt and response tokens used.
 */

export const getBlockSummaries = async (
  text: string,
  LLMParams: LLMParams
): Promise<{
  text: string;
  promptTokensUsed: number;
  responseTokensUsed: number;
}> => {
  // console.log("getBlockSummaries: ", text.length, " tokens: ", getTokens(text), " max usable tokens: ", getModelUsableTokens(LLMParams.modelName));
  const splitTexts = await splitText(text, getModelUsableTokens(LLMParams.modelName), 50);
  // console.log("SplitTexts: ", splitTexts.length);

  // Generate summaries for each text block
  const temporarySummaries = await Promise.all(
    splitTexts.map(async (summaryBlock, i) => {
      const prompt = getBlockSummaryPrompt(i, splitTexts.length, summaryBlock, LLMParams.language);
      const aiResult: AIResult = await AskLLM(prompt, LLMParams);
      return {
        text: aiResult.text,
        promptTokensUsed: aiResult.promptTokensUsed,
        responseTokensUsed: aiResult.responseTokensUsed,
      };
    })
  );

  const totalSummary = temporarySummaries.map((obj) => obj.text).join("\n");
  const totalpromptTokensUsed = temporarySummaries.reduce((acc, obj) => acc + obj.promptTokensUsed, 0);
  const totalresponseTokensUsed = temporarySummaries.reduce((acc, obj) => acc + obj.responseTokensUsed, 0);

  return {
    text: totalSummary,
    promptTokensUsed: totalpromptTokensUsed,
    responseTokensUsed: totalresponseTokensUsed,
  };
};

/**
 * Generates a summary for a given text.
 *
 * @param text - The text to summarize.
 * @param LLMParams - Object containing parameters like maximum number of characters (`maxChars`) and language set in preferences.
 * @returns Object with text of summary, prompt and response tokens used.
 */
export const getSummary = async (
  text: string,
  LLMParams: LLMParams
): Promise<{
  text: string;
  promptTokensUsed: number;
  responseTokensUsed: number;
}> => {
  let promptTokensUsed = 0;
  let responseTokensUsed = 0;
  // If text is too long, we need to split it in smaller chunks
  if (getTokens(text) > getModelUsableTokens(LLMParams.modelName)) {
    const res = await getBlockSummaries(text, LLMParams);
    text = res.text;
    promptTokensUsed += res.promptTokensUsed;
    responseTokensUsed += res.responseTokensUsed;
  }
  const prompt = getFinalSummaryPrompt(text, LLMParams.language);
  const res: AIResult = await AskLLM(prompt, LLMParams);
  promptTokensUsed += res.promptTokensUsed;
  responseTokensUsed += res.responseTokensUsed;
  return {
    text: res.text,
    promptTokensUsed: promptTokensUsed,
    responseTokensUsed: responseTokensUsed,
  };
};

/**
 * Calculates the number tokens used for a text
 *
 * @param text - The text to analyze.
 * @param encoding - Encoding to calculate tokens.
 * @returns number of tokens.
 */
export const getTokens = (text: string): number => {
  const encoding = encodingForModel("gpt-4");
  const tokens = encoding.encode(text).length; // Assumes getEncoding().encode() is a valid function
  return tokens;
};
