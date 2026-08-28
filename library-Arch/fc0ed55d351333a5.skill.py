"""Generated tier-1 skill fc0ed55d351333a5 (deterministic transpile of the committed steps).

source: CLick + next to the Add Expenses or benefits, select type Payment on behalf. period to Jun-26, enter the amount 200 and click Save.

Element identities live in fc0ed55d351333a5.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, expense_type='Payment on behalf', period_to='Jun-26', amount='200'):
    await api.click('click')
    await api.select('select', expense_type)
    await api.select('select-2', period_to)
    await api.fill('amount', amount)
    await api.press('Enter')
    await api.click('save')
