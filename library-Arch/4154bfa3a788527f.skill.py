"""Generated tier-1 skill 4154bfa3a788527f (deterministic transpile of the committed steps).

source: Now Go to Data Request Section. Click + Request, click Payroll Review, click Detailed, change the date combobox to May-26, select only the first 12 employees in the list, and click Save.

Element identities live in 4154bfa3a788527f.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, report_date='May-26'):
    await api.click('data-request')
    await api.click('request')
    await api.click('payroll-review')
    await api.click('detailed')
    await api.wait(1.0)
    await api.click('react-select-15-input')
    await api.select_option(report_date)
    await api.click_indexed('row2168-0-checkbox', 0, 5)
    await api.click('row')
    await api.click_indexed('row2609-6-checkbox', 6, 6)
    await api.click('save')
