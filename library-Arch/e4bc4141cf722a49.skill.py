"""Generated tier-1 skill e4bc4141cf722a49 (deterministic transpile of the committed steps).

source: (no template)

Element identities live in e4bc4141cf722a49.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api):
    await api.click('settings')
    await api.click('pay-elements')
    await api.click('btn-btnedit')
    await api.wait(1.0)
    await api.click('cancel')
    await api.click('btn-btnedit-2')
    await api.wait(1.0)
    await api.click('cancel-2')
