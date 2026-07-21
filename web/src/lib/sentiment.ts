// Turns "neural-net prediction vs. consensus" into a plain-language read for
// the predictions headline: does the model think you'll like this film more
// or less than the critic/member average, and by how much? Shared by both
// datasets -- purely a function of the two numbers, not dataset-specific.

export type Sentiment = "much-more" | "more" | "same" | "less" | "much-less";

// Below this absolute difference, treat the prediction as "about the same as
// the consensus" rather than reading noise as a lean either way.
const NEUTRAL_BAND = 0.05;
// At or above this absolute difference, upgrade to the "a lot more/less" tier.
const STRONG_BAND = 1;

export function sentimentFor(prediction: number, consensus: number): Sentiment {
  const diff = prediction - consensus;
  if (Math.abs(diff) < NEUTRAL_BAND) return "same";
  if (diff >= STRONG_BAND) return "much-more";
  if (diff > 0) return "more";
  if (diff <= -STRONG_BAND) return "much-less";
  return "less";
}

/** `groupNoun` names whoever the consensus is averaged over -- "critics" for
 * Rotten Tomatoes, "members" for Letterboxd. */
export function sentimentText(sentiment: Sentiment, diff: number, groupNoun: string): string {
  const magnitude = Math.abs(diff).toFixed(2);
  switch (sentiment) {
    case "much-more":
      return `You'll like it a lot more than the ${groupNoun} (+${magnitude})`;
    case "more":
      return `You'll like it more than the ${groupNoun} (+${magnitude})`;
    case "much-less":
      return `You'll like it a lot less than the ${groupNoun} (−${magnitude})`;
    case "less":
      return `You'll like it less than the ${groupNoun} (−${magnitude})`;
    case "same":
    default:
      return `About what the ${groupNoun} think`;
  }
}

export function sentimentClass(sentiment: Sentiment): string {
  return `sentiment-${sentiment}`;
}
