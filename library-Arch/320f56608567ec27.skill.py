"""Generated tier-1 skill 320f56608567ec27 (deterministic transpile of the committed steps).

source: Now go to Data Request, click on the staus of the top ref. no., Select the Status Submitted and Add a note, Well done and Save. and click on the top ref. no. link to open the Payroll Review panel, click the employee checkbox to select all the employee and then click Verify all.

Element identities live in 320f56608567ec27.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, note='Well done', status='Submitted'):
    await api.click('sent')
    await api.click('submitted')
    await api.fill('textfield6331', note)
    await api.click('save')
    await api.fill('react-select-26-input', '')
    await api.wait(1.0)
    await api.click('svg-svg-content-collapsed')
    await api.wait(1.0)
    await api.click('row')
    await api.click('all-status')
    await api.select_option(status)
    await api.fill('textfield6331-2', note)
    await api.click('save-2')
    await api.click('react-select-26-input-2')
    await api.select_option(status)
    await api.click('save-3')
    await api.click('pr-01797494-27-cdr084')
    await api.click('header7545-check')
    await api.click('row-2')
    await api.click('verify-all')
