# Codex Memory RAG

Use Python 3.12 and the committed uv.lock. Keep changes limited to the requested task.
Default tests use synthetic sources and must not contact real models, registered user data, or deployed services.
Never commit corpus files, model weights, credentials, local policies, indexes, environment snapshots, or operation journals.
Candidate results are related candidates with support_status=not_checked, not verified answers.
Preserve the currently deployed service: editing this repository does not authorize replacing a release, changing MCP configuration, or expanding cloud disclosure.
Run python -m pytest -q and git diff --check for relevant source changes. Live acceptance and deployment require separately scoped authorization.
