from test_supplementary_reranking import scorer, long_passage


def test_length_batching_preserves_candidate_order_and_reduces_padding():
    passages = ['doc\n' + ('needle ' if i % 3 == 0 else '') + 'background ' * n
                for i, n in enumerate([180, 5, 160, 8, 190, 4, 150, 6])]
    def run(enabled):
        model = scorer(supplementary=False)
        model.config.length_sorted_batches = enabled
        sizes = []
        original = model.model
        class Counted:
            def __call__(self, **tokens):
                sizes.append(tokens['input_ids'].numel())
                return original(**tokens)
        model.model = Counted()
        return model.score('needle', passages), sum(sizes)
    plain, plain_padding = run(False)
    packed, packed_padding = run(True)
    assert packed == plain == [6.0, -6.0, -6.0, 6.0, -6.0, -6.0, 6.0, -6.0]
    assert packed_padding < plain_padding


def test_length_batching_restores_supplementary_indices_and_empty_pool():
    model = scorer()
    passages = [long_passage(), 'doc\nbackground', 'doc\nneedle'] * 8
    expected = model.score('needle', passages)
    model.config.length_sorted_batches = True
    assert model.score('needle', passages) == expected
    assert model.score('needle', []) == []


def test_length_batching_is_default_off_and_changes_profile_when_enabled():
    from codex_memory_rag.reranking import profile_id
    from codex_memory_rag.schemas import RerankerConfig
    original = RerankerConfig()
    assert original.length_sorted_batches is False
    assert profile_id({}, original) != profile_id({}, original.model_copy(update={'length_sorted_batches': True}))
