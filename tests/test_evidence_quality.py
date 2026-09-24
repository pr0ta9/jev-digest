import json

import httpx

from jev_digest import fetch, judge, pipeline, search


async def test_search_preserves_evidence_for_selection(monkeypatch):
    async def get(self, url, **kwargs):
        return httpx.Response(200, request=httpx.Request('GET', url), json={'results': [{
            'url': 'https://agency.example/policy', 'title': 'Current policy',
            'content': 'The revised requirement starts next year.', 'publishedDate': '2026-07-01',
            'engine': 'test', 'engines': ['test', 'other'],
        }]})

    monkeypatch.setattr(httpx.AsyncClient, 'get', get)
    rows, _ = await search.searxng('policy', 10, 'http://search.test')
    assert rows[0].get('snippet') == 'The revised requirement starts next year.'
    assert rows[0].get('published') == '2026-07-01'
    assert rows[0].get('engines') == ['test', 'other']


async def test_selection_rejects_wrong_work_without_rejecting_unknown_sources():
    class Client:
        async def decide(self, state, questions):
            return {str(i): {'choice': role} for i, role in enumerate(['EXCLUDE', 'UNCERTAIN', 'PRIMARY'])} | {f'use__{i}': {'choice': 'POSSIBLE'} for i in range(3)}

    rows = [{'url': f'https://source{i}.example', 'title': title} for i, title in enumerate(
        ['A fan-fiction retelling', 'An unfamiliar research note', 'Original author account'])]
    selected, audit = await search.select_sources(Client(), 'What happened in the original story?', rows)
    assert [r['url'] for r in selected] == [rows[2]['url'], rows[1]['url']]
    assert audit[0]['decision'] == 'EXCLUDE'


async def test_failed_selection_keeps_candidates_and_records_uncertainty():
    class Client:
        async def decide(self, state, questions):
            return None

    rows = [{'url': 'https://new-source.example', 'title': 'Evidence'}]
    selected, audit = await search.select_sources(Client(), 'question', rows)
    assert selected[0]['url'] == rows[0]['url']
    assert selected[0]['source_role'] == 'UNCERTAIN'
    assert audit[0]['selection_failed']


def test_extraction_and_preparation_preserve_section_identity():
    body = ('He returns as a newborn in this different story. ' * 6)
    text, title = fetch.extract(f'<html><title>Character</title><article><h1>Character</h1>'
                                f'<h2>Fanfiction</h2><p>{body}</p></article></html>', 50000)
    prepared = pipeline._prepare({'url': 'https://wiki.example/character', 'text': text, 'title': title}, 0)
    assert prepared['passages']
    assert all('Fanfiction' in p.get('section', '') for p in prepared['passages'])
    assert all('newborn' in p['text'] for p in prepared['passages'])


def test_judging_has_context_to_distinguish_another_authorization():
    page = pipeline._prepare({'url': 'https://travel.example', 'title': 'Travel permits',
                              'text': '# ETIAS\n\n' + ('ETIAS lasts three years. ' * 5) +
                              '\n\n## UK ETA\n\n' + ('This approval lasts two years. ' * 5)}, 0)
    states = [s for s, _, _ in judge.relevance_requests('How long is ETIAS valid?', page, 40)]
    candidate = next(p for s in states for p in s['passages'] if 'two years' in p['text'])
    assert 'UK ETA' in candidate.get('section', '')


def test_source_suitability_precedes_classifier_confidence():
    common = {'aspect': 'T1', 'level': 'DIRECT_ANSWER', 'text': 'An answer.'}
    rows = [dict(common, id='d0p0', confidence=0.99, source_role='UNCERTAIN'),
            dict(common, id='d1p0', confidence=0.8, source_role='PRIMARY')]
    assert judge.group_by_aspect(rows)['T1'][0]['id'] == 'd1p0'


def test_context_opening_exposes_neighboring_passages_without_changing_quote(tmp_path):
    from jev_digest.config import Settings
    cfg = Settings(home=tmp_path)
    folder = cfg.runs_dir / 'example'
    folder.mkdir(parents=True)
    saved = {f'd0p{i}': {'text': text, 'url': 'https://source.example', 'title': 'Evidence', 'section': 'UK ETA'}
             for i, text in enumerate(['This section concerns UK ETA.', 'It lasts two years.', 'ETIAS differs.'])}
    (folder / 'passages.json').write_text(json.dumps(saved), encoding='utf-8')
    opened = pipeline.open_passage('example', 'd0p1', cfg)
    assert opened['text'] == 'It lasts two years.'
    assert [p['text'] for p in opened.get('context', [])] == [p['text'] for p in saved.values()]


async def test_pipeline_screens_before_fetch_and_saves_context(tmp_path, monkeypatch):
    from jev_digest.config import Settings
    calls = []

    async def find(*args):
        return [{'url': 'https://wrong.example'}, {'url': 'https://right.example'}], {}

    async def select(client, query, rows):
        return [dict(rows[1], source_role='PRIMARY')], [{'url': rows[0]['url'], 'decision': 'EXCLUDE'}]

    async def download(urls, *args):
        calls.extend(urls)
        return {u: {'url': u, 'text': '# Official policy\n\nThe authorization is valid for three years or until the passport expires, whichever comes first.',
                    'title': 'Policy', 'status': 'fetched http 200'} for u in urls}

    class Client:
        def __init__(self, *a, **kw):
            self.stats = {}

        async def decide(self, state, questions):
            return {k: {'choice': 'YES' if k == 'page' else 'DIRECT_ANSWER' if k.startswith('level__') else 'T1',
                        'confidence': 1.0} for k in questions}

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, 'search', find)
    monkeypatch.setattr(pipeline, 'select_sources', select)
    monkeypatch.setattr(pipeline, 'http_wave', download)
    monkeypatch.setattr(pipeline, 'JevClient', Client)
    cfg = Settings(home=tmp_path, browser_fallback=False)
    result = await pipeline.run_digest('How long is the authorization valid?', settings=cfg)
    assert calls == ['https://right.example']
    entry = result['blocks']['T1']['representatives'][0]
    assert entry['source_role'] == 'PRIMARY'
    assert 'Official policy' in pipeline.open_passage(result['run_id'], entry['id'], cfg)['section']
    assert result['source_selection'][0]['decision'] == 'EXCLUDE'


def test_source_render_keeps_citation_and_section_next_to_verbatim_evidence():
    from jev_digest import digest
    entry = {'id': 'd0p0', 'level': 'DIRECT_ANSWER', 'text': 'The approval lasts three years.',
             'site': 'agency.example', 'url': 'https://agency.example/policy', 'title': 'Official policy',
             'section': 'Validity', 'source_role': 'PRIMARY'}
    blocks = {'T1': {'aspect': 'validity', 'representatives': [entry], 'adds': [], 'conflicts': [], 'also_reported_by': []}}
    rendered = digest.render_evidence('validity?', {'T1': 'validity', 'T2': 'fee'}, blocks)
    assert 'https://agency.example/policy' in rendered
    assert 'Official policy' in rendered and 'Validity' in rendered
    assert entry['text'] in rendered
    assert 'fee' in rendered and 'No retained evidence' in rendered


async def test_scope_filter_removes_other_timeline_but_keeps_answer_evidence():
    class Client:
        async def decide(self, state, questions):
            assert all(pid in q['instructions'] for pid, q in questions.items())
            return {'d0p0': {'choice': 'IN_SCOPE'}, 'd1p0': {'choice': 'OTHER_CONTEXT'}}

    rows = [{'id': 'd0p0', 'page': 'd0', 'text': 'He died during the failed jump.'},
            {'id': 'd1p0', 'page': 'd1', 'text': 'He lived to old age after the future changed.'}]
    kept, audit = await judge.judge_scope(Client(), 'How did his future self die?', rows, 40)
    assert [p['id'] for p in kept] == ['d0p0']
    assert audit['d1p0'] == 'OTHER_CONTEXT'


def test_representatives_do_not_spend_all_slots_on_identical_copies():
    common = {'aspect': 'T1', 'level': 'DIRECT_ANSWER', 'confidence': 0.9, 'source_role': 'PRIMARY'}
    rows = [dict(common, id=f'd{i}p0', text='The authorization costs twenty euros.') for i in range(3)]
    rows.append(dict(common, id='d3p0', text='Children are exempt from the fee.', confidence=0.8))
    assert 'd3p0' in [p['id'] for p in judge.group_by_aspect(rows)['T1'][:3]]


def test_evidence_preview_limits_repetition_and_keeps_expansion_references():
    from jev_digest import digest
    entries = [{'id': f'd0p{i}', 'level': 'DIRECT_ANSWER', 'text': f'Original evidence number {i}.',
                'site': 'source.example', 'url': 'https://source.example', 'title': 'Evidence',
                'section': 'Scope', 'source_role': 'PRIMARY'} for i in range(12)]
    blocks = {'T1': {'aspect': 'scope', 'representatives': entries[:3], 'adds': entries[3:], 'conflicts': [], 'also_reported_by': []}}
    rendered = digest.render_evidence('scope?', {'T1': 'scope'}, blocks)
    assert sum(e['text'] in rendered for e in entries) == 5
    assert '7 more relevant passages' in rendered
    assert 'browse' in rendered


async def test_readable_secondary_pages_do_not_suppress_primary_source_recovery(tmp_path, monkeypatch):
    from jev_digest.config import Settings
    recovered = []
    urls = ['https://authority.example'] + [f'https://other{i}.example' for i in range(6)]
    text = 'The official authorization lasts three years, subject to passport validity and the stated exemptions.'

    async def find(*args):
        return [{'url': u} for u in urls], {}

    async def select(client, query, rows):
        return [dict(r, source_role='PRIMARY' if i == 0 else 'SECONDARY') for i, r in enumerate(rows)], []

    async def download(*args):
        return {u: {'url': u, 'status': 'insufficient text'} if i == 0 else
                {'url': u, 'text': text, 'status': 'fetched http 200'} for i, u in enumerate(urls)}

    async def browser(requested, *args):
        recovered.extend(requested)
        return {u: {'url': u, 'text': text, 'status': 'fetched via browser'} for u in requested}

    class Client:
        def __init__(self, *a, **kw):
            self.stats = {}

        async def decide(self, state, questions):
            return {k: {'choice': 'YES' if k == 'page' else 'DIRECT_ANSWER' if k.startswith('level__') else 'T1' if k.startswith('aspect__') else 'IN_SCOPE' if 'IN_SCOPE' in v['criteria'] else 'COVERED',
                        'confidence': 1.0} for k, v in questions.items()}

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, 'search', find)
    monkeypatch.setattr(pipeline, 'select_sources', select)
    monkeypatch.setattr(pipeline, 'http_wave', download)
    monkeypatch.setattr(pipeline, 'browser_wave', browser)
    monkeypatch.setattr(pipeline, 'JevClient', Client)
    await pipeline.run_digest('authorization validity', settings=Settings(home=tmp_path, browser_min_pages=6))
    assert recovered == ['https://authority.example']


def test_short_factual_sections_survive_contextual_segmentation():
    from jev_digest.passages import contextual_passages
    result = contextual_passages('# Fees\nAdults pay twenty euros.\n## Exemptions\nChildren do not pay.')
    assert {p['text'] for p in result} == {'Adults pay twenty euros.', 'Children do not pay.'}


async def test_scope_check_preserves_source_identity_from_relevance():
    class Client:
        async def decide(self, state, questions):
            if 'page' in questions:
                return {k: {'choice': 'YES' if k == 'page' else 'DIRECT_ANSWER' if k.startswith('level__') else 'T1', 'confidence': 1.0} for k in questions}
            evidence = state['passages']['d0p0']
            assert evidence['title'] == 'Future timeline'
            assert evidence['url'] == 'https://reference.example/future'
            return {'d0p0': {'choice': 'IN_SCOPE'}}

    page = pipeline._prepare({'url': 'https://reference.example/future', 'title': 'Future timeline',
                              'text': 'He died on returning to the past after a failed time travel spell.'}, 0)
    kept, _ = await judge.judge_relevance(Client(), 'How did the future self die?', [page], 40)
    assert kept
    await judge.judge_scope(Client(), 'How did the future self die?', kept, 40)


async def test_old_plaintext_cache_is_refetched_for_section_context(tmp_path, monkeypatch):
    url = 'https://example.test/article'
    fetch.cache_path(tmp_path, url).write_text(json.dumps({'title': 'Old', 'text': 'No headings'}))
    fetched = []

    async def download(session, requested, *args):
        fetched.append(requested)
        return {'url': requested, 'text': '# Preserved heading', 'title': 'New', 'status': 'fetched http 200'}

    monkeypatch.setattr(fetch, '_fetch_http', download)
    result = await fetch.http_wave([url], tmp_path, 1, 50000, True)
    assert fetched == [url]
    assert result[url]['text'] == '# Preserved heading'
