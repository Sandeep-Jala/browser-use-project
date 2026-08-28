"""Generated tier-1 skill 8ab11b6ebe2b7f89 (deterministic transpile of the committed steps).

source: Go to the Payroll module, search for and select the business name FOOD LIMITED.

Element identities live in 8ab11b6ebe2b7f89.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, business_name='FOOD LIMITED'):
    await api.click('menus')
    await api.wait(1.0)
    await api.click('payroll')
    await api.fill('search', business_name)
    await api.press('Enter')
    await api.wait(2.0)
    await api.click('food-limited')
