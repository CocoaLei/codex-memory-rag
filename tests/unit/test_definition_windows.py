from test_supplementary_reranking import scorer, long_passage


def test_query_window_can_cover_all_64_candidates_without_extra_pairs():
    model = scorer()
    model.config.max_candidates = 64
    model.config.definition_priority = True
    count = 0
    original = model.model
    class Counted:
        def __call__(self, **tokens):
            nonlocal count
            count += len(tokens['input_ids'])
            return original(**tokens)
    model.model = Counted()
    assert model.score('needle', [long_passage()] * 64) == [6.0] * 64
    assert count == 64


def test_protected_40_pool_keeps_prefix_plus_supplementary_contract():
    model = scorer()
    model.config.definition_priority = True
    assert model.score('needle', [long_passage()] * 40) == [6.0] * 24 + [-6.0] * 16
