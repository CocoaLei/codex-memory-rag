from types import SimpleNamespace

import pytest

from codex_memory_rag.reranking import Reranker
from codex_memory_rag.schemas import CmrError


def scorer(*, supplementary=True, nan=False):
    # Synthetic tokenizer and local tensor forward; no real model/user source access.
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocabulary = ['[UNK]', '[PAD]', 'doc', 'needle', 'background']
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    model = Reranker.__new__(Reranker)
    model.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]')
    model.config = SimpleNamespace(max_tokens=256, max_candidates=40, batch_size=4,
                                   backend='cpu', supplementary_windows=supplementary)
    class Forward:
        def __call__(self, input_ids, attention_mask, **kwargs):
            values = []
            for ids in input_ids:
                # One occurrence is the query, a second is the supporting passage.
                present = int((ids == 3).sum()) > 1
                values.append(float('nan') if present and nan else (6.0 if present else -6.0))
            return SimpleNamespace(logits=torch.tensor(values))
    model.model = Forward()
    return model


def long_passage():
    return 'doc\n' + 'background ' * 400 + 'needle ' + 'background ' * 30


def test_late_support_is_scored_without_changing_prefix_default():
    passage = long_passage()
    assert scorer(supplementary=False).score('needle', [passage]) == [-6.0]
    assert scorer().score('needle', [passage]) == [6.0]


def test_supplementary_work_has_a_total_pair_budget():
    values = scorer().score('needle', [long_passage()] * 40)
    assert values == [6.0] * 24 + [-6.0] * 16


def test_nonfinite_supplementary_scores_are_not_hidden_by_maximum():
    with pytest.raises(CmrError, match='RERANKER_CONTRACT_MISMATCH'):
        scorer(nan=True).score('needle', [long_passage()])


def test_short_passages_keep_existing_scoring_behavior():
    passages = ['doc\nneedle', 'doc\nbackground']
    assert scorer().score('needle', passages) == scorer(supplementary=False).score('needle', passages) == [6.0, -6.0]
