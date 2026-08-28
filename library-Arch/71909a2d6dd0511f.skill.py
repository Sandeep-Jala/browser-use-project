"""Generated tier-1 skill 71909a2d6dd0511f (deterministic transpile of the committed steps).

source: Now go to Data Request, click on the staus of the top ref. no., Select the Status Submitted and Add a note, Well done and Save. and click on the top ref. no. link to open the Payroll Review panel, click the employee checkbox to select all the employee and then click Verify all.

Element identities live in 71909a2d6dd0511f.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, bound_1='PR/01797494/27/CDR079'):
    await api.click('bound-1-target')
    await api.wait(1.0)
    await api.click('react-select-25-input')
    await api.select_option('Submitted')
    await api.fill('textfield7770', 'Well done')
    await api.click('save')
    await api.wait(1.0)
    await api.click('pr-01797494-27-cdr079')
    await api.click('header8650-check')
    await api.click('verify-all')
