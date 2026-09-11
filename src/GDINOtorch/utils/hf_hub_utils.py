from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import EntryNotFoundError


def resolve_hub_file(repo_id: str, filename: str) -> str:
    """Return a local path to ``filename`` from a Hub repo, downloading only if needed.

    The cache is checked first so a warm cache works offline and does not ping
    the Hub for the latest revision on every model construction. A file the
    cache has already recorded as absent raises ``EntryNotFoundError`` directly.
    """
    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached
    if cached is _CACHED_NO_EXIST:
        raise EntryNotFoundError(f"{filename} does not exist in {repo_id}")
    return hf_hub_download(repo_id=repo_id, filename=filename)
