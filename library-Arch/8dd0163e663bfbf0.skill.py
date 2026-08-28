"""Generated tier-1 skill 8dd0163e663bfbf0 (deterministic transpile of the committed steps).

source: click the + button next to payment, enter 4002 in the amount field, click Save.

Element identities live in 8dd0163e663bfbf0.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, amount='4002'):
    await api.click('click')
    await api.fill('fill', amount)
    await api.press('Enter')
    await api.click('save')
