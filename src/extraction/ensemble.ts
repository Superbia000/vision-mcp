/**
 * Vision-MCP v8: L4 - Consensus Entropy-Weighted Ensemble Voting
 *
 * v8 improvements:
 * - Edit-distance based consensus entropy scoring
 * - Multiple temperature sampling for diversity
 * - CE-based confidence scoring with review flagging
 * - Lower temperature diversity (0.3-0.5) for OCR tasks
 */

import OpenAI from "openai";
import type { VoteResult, CEFieldResult } from "../config/types.js";
import { CE_VOTES, CE_TEMPERATURE, ENABLE_CE } from "../config/constants.js";

function normalizeForVoting(text: string): string {
  return text
    .trim()
    .replace(/^["'\s]+|["'\s]+$/g, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

/** Simple Levenshtein distance */
function editDistance(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

/** v8: Compute consensus entropy from multiple votes using edit distance */
export function computeConsensusEntropy(votes: string[]): number {
  if (votes.length <= 1) return 0;

  // Compute pairwise edit distances
  let totalDist = 0;
  let pairCount = 0;
  for (let i = 0; i < votes.length; i++) {
    for (let j = i + 1; j < votes.length; j++) {
      const maxLen = Math.max(votes[i].length, votes[j].length, 1);
      totalDist += editDistance(votes[i], votes[j]) / maxLen;
      pairCount++;
    }
  }

  return pairCount > 0 ? totalDist / pairCount : 0;
}

/** v8: CE-based field result with review flagging */
export function analyzeConsensus(
  fieldName: string,
  votes: { value: string; weight: number }[]
): CEFieldResult {
  const values = votes.map((v) => v.value);
  const entropy = computeConsensusEntropy(values);

  // Weighted frequency counting
  const weightedFreq: Record<string, number> = {};
  let totalWeight = 0;
  for (const vote of votes) {
    weightedFreq[vote.value] = (weightedFreq[vote.value] || 0) + vote.weight;
    totalWeight += vote.weight;
  }

  let bestValue = "";
  let bestWeight = 0;
  for (const [v, w] of Object.entries(weightedFreq)) {
    if (w > bestWeight) { bestValue = v; bestWeight = w; }
  }

  const agreement = totalWeight > 0 ? bestWeight / totalWeight : 0;

  // Confidence based on entropy
  let confidence: "high" | "medium" | "low";
  if (entropy < 0.15 && agreement > 0.66) {
    confidence = "high";
  } else if (entropy < 0.4) {
    confidence = "medium";
  } else {
    confidence = "low";
  }

  return {
    fieldName,
    entropy,
    confidence,
    votes: values,
    agreement,
    finalValue: bestValue || values[0] || "",
    needsReview: entropy > 0.5 || agreement < 0.5,
  };
}

/**
 * Weighted self-consistency vote on a single field.
 * v8: Uses multiple temperatures for diversity and CE scoring.
 */
export async function weightedConsistencyVote(
  client: OpenAI,
  model: string,
  imageBase64: string,
  mime: string,
  fieldName: string,
  fieldPrompt: string,
  numVotes: number = CE_VOTES,
  temperature: number = CE_TEMPERATURE
): Promise<VoteResult> {
  // v8: Diverse prompt variations for better consensus
  const variations = [
    fieldPrompt,
    `${fieldPrompt}\nDouble-check your answer carefully. Look closely at each character. Read left to right.`,
    `${fieldPrompt}\nPay extra attention to character shapes, stroke connections, and formatting. Verify against common patterns.`,
    `${fieldPrompt}\nRe-read the field value character by character from left to right. Confirm each character individually.`,
    `${fieldPrompt}\nLook at the entire field region. Consider the context of surrounding text and labels.`,
  ];

  const votes: { value: string; weight: number }[] = [];
  const tempVariations = [temperature, temperature * 1.3, temperature * 0.7, temperature * 1.6]
    .filter((t) => t > 0 && t <= 2);

  for (let i = 0; i < Math.min(numVotes, variations.length); i++) {
    const useTemp = tempVariations[i % tempVariations.length];
    try {
      const r = (await client.chat.completions.create({
        model,
        max_tokens: 256,
        temperature: useTemp,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: `data:${mime};base64,${imageBase64}` },
              },
              { type: "text", text: variations[i] },
            ],
          },
        ],
      })) as any;

      const rawValue = r.choices[0]?.message?.content || "";
      const value = normalizeForVoting(rawValue);

      if (value) {
        const hasUncertainty = /[?\ufffd]|unclear|uncertain|maybe|perhaps/i.test(rawValue);
        const lengthPenalty = Math.min(1, 20 / Math.max(rawValue.length, 1));
        const tempWeight = useTemp < 0.3 ? 1.2 : 1.0; // Lower temp = more reliable
        const weight = (hasUncertainty ? 0.3 * lengthPenalty : 1.0 * lengthPenalty) * tempWeight;
        votes.push({ value, weight });
      }
    } catch (err: any) {
      console.error(`[L4] Vote ${i + 1} for "${fieldName}" failed: ${err.message}`);
    }
  }

  // Weighted frequency counting
  const weightedFreq: Record<string, number> = {};
  let totalWeight = 0;

  for (const vote of votes) {
    weightedFreq[vote.value] = (weightedFreq[vote.value] || 0) + vote.weight;
    totalWeight += vote.weight;
  }

  let bestValue = "";
  let bestWeight = 0;
  for (const [v, w] of Object.entries(weightedFreq)) {
    if (w > bestWeight) { bestValue = v; bestWeight = w; }
  }

  // v8: Also compute CE for logging
  if (ENABLE_CE && votes.length >= 2) {
    const ce = computeConsensusEntropy(votes.map((v) => v.value));
    console.error(`[L4-CE] "${fieldName}": entropy=${ce.toFixed(3)} votes=${votes.length} agreement=${totalWeight > 0 ? (bestWeight / totalWeight).toFixed(2) : "N/A"}`);
  }

  return {
    value: bestValue || votes[0]?.value || "",
    frequency: votes.filter((v) => v.value === bestValue).length,
    totalVotes: votes.length,
    agreement: totalWeight > 0 ? bestWeight / totalWeight : 0,
    isUnanimous: votes.length > 0 && votes.every((v) => v.value === bestValue),
  };
}
