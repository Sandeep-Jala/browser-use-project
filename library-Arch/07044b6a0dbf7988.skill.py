"""Generated tier-1 skill 07044b6a0dbf7988 (deterministic transpile of the committed steps).

source: Now go to Data Request, and on the top row (S.No. 1, the newest request) click the ref. no. to open the Payroll Review panel, and click Get OTP, copy the 6 digit number (OTP), close the review panel. Now click on the external link button next to the ref. no, A new tab will be open click Already have an OTP, paste the OTP from the previous step into the first code box and click proceed Securely.

Element identities live in 07044b6a0dbf7988.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, bound_1='PR/01797494/27/CDR079'):
    await api.click('data-request')
    await api.click('pr-01797494-27-cdr079')
    await api.click('get-otp')
    await api.copy('copy', 'otp')
    await api.click('click')
    await api.click('open-payroll-review-request-as-client')
    await api.click('already-have-an-otp')
    await api.paste('please-enter-otp-character-1', api.noted('otp'))
    await api.click('proceed-securely')
