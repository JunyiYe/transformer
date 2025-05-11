import math
import numpy as np

from collections import Counter


def blue_stats(hypothesis: list[str], reference: list[str]) -> list[int]:
    """
    Compute BLEU-related statistics between a hypothesis and a reference sentence.

    Args:
        hypothesis: Hypothesis sentence.
        reference: Reference sentence.

    Returns:
        List of integers containing:    
        [len(hyp), len(ref), match_1gram, total_1gram, match_2gram, total_2gram, ..., match_4gram, total_4gram]
    """
    stats = [len(hypothesis), len(reference)]

    for n in range(1, 5):
        # Create n-grams for hypothesis and reference
        hyp_ngrams = Counter(
            tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)
        )

        # Count matches
        match = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())

        stats.append(match)
        stats.append(total)

    return stats


def bleu(stats: list[int]) -> float:
    """
    Calculate BLEU score for a pair of hypothese and reference.

    Args:
        stats: List of BLEU-related statistics.

    Returns:
        BLEU score.
    """
    len_hyp, len_ref = stats[0], stats[1]
    match_ngrams = stats[2:]
    p_numerators = []
    p_denominators = []

    for i in range(0, len(match_ngrams), 2):
        match = match_ngrams[i]
        total = match_ngrams[i + 1]

        if total > 0:
            p_numerators.append(match)
            p_denominators.append(total)

    if not p_numerators:
        return 0.0

    # Calculate precision
    precision = np.prod(
        [match / total for match, total in zip(p_numerators, p_denominators)]
    ) ** (1 / len(p_numerators))

    # Calculate brevity penalty
    brevity_penalty = (
        math.exp(1 - len_ref / len_hyp) if len_hyp < len_ref else 1.0
    )

    return brevity_penalty * precision

def get_bleu(hypotheses: list[list[str]], reference: list[list[str]]) -> float:
    """
    Compute corpus-level BLEU score for a set of hypotheses and references.

    Args:
        hypotheses: List of tokenized hypothesis sentences.
        references: List of tokenized reference sentences.

    Returns:
        BLEU score scaled to 0–100.
    """
    stats = np.zeros(10)
    for hyp, ref in zip(hypotheses, references):
        stats += np.array(bleu_stats(hyp, ref))

    return 100.0 * bleu(stats.tolist())

def idx_to_word(x: List[int], vocab) -> str:
    """
    Convert a list of token indices to a readable string using vocabulary mapping.

    Args:
        x: List of integer token indices.
        vocab: Vocabulary object with `itos` (index-to-string) mapping.

    Returns:
        A whitespace-joined sentence string.
    """
    special_tokens = {'<pad>', '<sos>', '<eos>'}
    words = [vocab.itos[i] for i in x if vocab.itos[i] not in special_tokens]
    return " ".join(words)