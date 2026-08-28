"""Generated tier-1 skill 5a90660d1df6a541 (deterministic transpile of the committed steps).

source: click the + next tot he Addition or Deductions, select type Deduction, period to Jun-26, Select Name Salary sacrifice, Description Salary sacrifice for childminder, Amount 1000, enter click Save.

Element identities live in 5a90660d1df6a541.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, entry_type='Deduction', period='Jun-26', deduction_name='Salary sacrifice', deduction_description='Salary sacrifice for childminder', deduction_amount='1000'):
    await api.click('click')
    await api.select('select', entry_type)
    await api.select('ao-cb-6', period)
    await api.select('ao-cb-7', deduction_name)
    await api.fill('fill', deduction_description)
    await api.press('Enter')
    await api.fill('fill-2', deduction_amount)
    await api.press('Enter')
    await api.click('save')
