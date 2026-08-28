"""Generated tier-1 skill aa3a76b7c82dcf8b (deterministic transpile of the committed steps).

source: In the Send Email section that opens, click on the dropdown next to from and select the no-reply option, then click Send; if the Send Email section has not closed after clicking Send, click Save again. If the email was not sent, it is saved as Drafted; click the email button next to the Drafted status to reopen the Send Email section, then click Send. if you see a notification of "Why not wish them a great weekend 🙂", click send again.

Element identities live in aa3a76b7c82dcf8b.anchors.json — healing edits THAT file, never this one. Hand edits here are allowed but must stay inside the api.* whitelist (skills.codegen.lint_code).
"""


async def run(api, *, from_address='no-reply'):
    await api.click('react-select-16-input')
    await api.select_option(from_address)
    await api.click('send')
