"""Generated tier-1 skill 7320039db9ba7e26 (deterministic transpile of the committed steps).

source: (no template)

Element identities live in 7320039db9ba7e26.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api):
    await api.click('next')
    await api.wait(1.0)
    await api.repeat_click('next-2', 10, 1.0)
    await api.wait(1.0)
    await api.click('submit')
