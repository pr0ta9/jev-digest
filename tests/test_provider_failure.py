import httpx
import pytest

from jev_digest.client import JevClient, JevError, choice


@pytest.mark.parametrize('status', [401, 402, 403])
async def test_account_failure_is_explicit_and_stops_later_requests(status):
    requests = []

    def reject(request):
        requests.append(request)
        return httpx.Response(status, json={'detail': 'Account unavailable'})

    client = JevClient('test-key')
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    try:
        for _ in range(2):
            with pytest.raises(JevError, match=str(status)):
                await client.decide('evidence', {'check': choice('Relevant?', {'YES': 'Yes', 'NO': 'No'})})
        assert len(requests) == 1
        assert client.stats['failed'] == 1
    finally:
        await client.close()
