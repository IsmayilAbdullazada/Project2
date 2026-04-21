import json
import torch
import torch.nn.functional as F


_TOKEN_COUNTS = None


def batched_sequence_log_likelihood(
    logits,
    vocab_tensor,
    lengths,
    letter2id,
    logit_lengths,
    use_viterbi=False,
):
    device = logits.device
    B, T, V = logits.shape
    W, S = vocab_tensor.shape

    # log p(y_t | x_t)
    log_post = torch.log_softmax(logits, dim=-1)

    # Approximate log p(x_t | y_t) up to a constant via Bayes rule:
    # log p(x|y) = log p(y|x) - log p(y) + const(x)
    global _TOKEN_COUNTS
    if _TOKEN_COUNTS is None:
        with open("data/trn_token_counts.json") as f:
            _TOKEN_COUNTS = json.load(f)
    token_counts = _TOKEN_COUNTS

    prior = torch.ones(V, device=device, dtype=log_post.dtype)
    for tok, idx in letter2id.items():
        if idx < V and tok in token_counts:
            prior[idx] = float(token_counts[tok])

    prior = prior / prior.sum()
    log_prior = torch.log(prior.clamp_min(1e-12))
    log_emissions = log_post - log_prior.view(1, 1, V)

    # Gather emissions for each vocabulary word's letter sequence.
    safe_vocab = vocab_tensor.clamp_min(0)
    expanded_vocab = safe_vocab.view(1, 1, W, S).expand(B, T, W, S)
    gathered = torch.gather(
        log_emissions.unsqueeze(2).expand(B, T, W, V),
        dim=-1,
        index=expanded_vocab,
    )
    gathered = gathered.permute(0, 2, 1, 3).contiguous()  # [B, W, T, S]

    state_mask = (
        torch.arange(S, device=device).view(1, 1, S)
        < lengths.view(1, W, 1)
    )
    neg_inf = torch.finfo(gathered.dtype).min

    # alpha[b, w, s] stores current forward/Viterbi score at frame t.
    alpha = torch.full((B, W, S), neg_inf, device=device, dtype=gathered.dtype)
    alpha[:, :, 0] = gathered[:, :, 0, 0]
    alpha = alpha.masked_fill(~state_mask, neg_inf)

    time_mask = torch.arange(T, device=device).view(1, T) < logit_lengths.view(B, 1)

    for t in range(1, T):
        stay = alpha
        move = torch.roll(alpha, shifts=1, dims=-1)
        move[:, :, 0] = neg_inf

        if use_viterbi:
            trans = torch.maximum(stay, move)
        else:
            trans = torch.logsumexp(torch.stack([stay, move], dim=0), dim=0)

        nxt = trans + gathered[:, :, t, :]
        nxt = nxt.masked_fill(~state_mask, neg_inf)

        # Keep alpha unchanged for sequences already past their valid frame count.
        active = time_mask[:, t].view(B, 1, 1)
        alpha = torch.where(active, nxt, alpha)

    final_pos = (lengths - 1).view(1, W, 1).expand(B, W, 1)
    scores = torch.gather(alpha, dim=-1, index=final_pos).squeeze(-1)

    return scores


def build_vocab_tensor(scr2id, letter2id, device="cpu"):
    vocab = list(scr2id.keys())

    unk = letter2id["<unk>"]

    words = []
    seqs = []

    for w in vocab:
        if w != '<unk>':
            wrapped = f"|{w.lower()}|"
            idxs = [letter2id.get(c, unk) for c in wrapped]

            words.append(w.lower())
            seqs.append(torch.tensor(idxs))

    lengths = torch.tensor([len(s) for s in seqs])
    max_len = lengths.max()

    vocab_size = len(seqs)

    vocab_tensor = torch.full(
        (vocab_size, max_len),
        fill_value=-1,          # padding
        dtype=torch.long
    )

    for i, s in enumerate(seqs):
        vocab_tensor[i, :len(s)] = s

    return words, vocab_tensor.to(device), lengths.to(device)


def decode_batch(logits, logit_lengths, dataset):

    words, vocab_tensor, lengths = build_vocab_tensor(
        dataset.scr2id,
        dataset.letter2id,
        device=logits.device
    )

    scores = batched_sequence_log_likelihood(
        logits, # [b, t, v]: P(y | x_t)
        vocab_tensor, # [w, s]: each row contains the letter indices of the corresponding word, wrapped in '|'. eg. "|money|" → [23, 22, 16, 5, 13, 20, 23]
        lengths, # [w]: length of each word
        dataset.letter2id, # {'a':18, 'b':6, ..., '|':23}
        logit_lengths, # [b]: don't try to decode past the number of frames in each utterance!
        use_viterbi=False,
    )

    if scores is not None:
        best_vocab_idx = scores.argmax(dim=1)
        return best_vocab_idx
