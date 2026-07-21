"""Login module.

Wraps the original login automation (see git history of main.py) and exposes
a single entry point: `login(playwright, config)`. It launches Chromium with
its CDP (Chrome DevTools Protocol) debugging port open, logs in using plain
Playwright actions, and hands back the `cdp_url`. Browser-Use's `Browser`
then *attaches* to that same running Chromium via `cdp_url` instead of
launching a second browser, so the navigation Agent continues from the
authenticated session login produced.

Uses Playwright's *async* API (not sync) so the whole pipeline -- login,
Groq calls, and the Browser-Use agent -- can share a single asyncio event
loop. Mixing Playwright's sync API (which runs its own event loop via
greenlets) with `asyncio.run()` for the navigation stage is not supported.
"""
import asyncio
import re
import time

from playwright.async_api import Browser, Page, Playwright, TimeoutError as PlaywrightTimeoutError

from automation.config import Config
from automation.browser.error_capture import save_login_error_screenshot
from automation.pipeline.script_compile import REVEAL_CSS_JS


class LoginError(RuntimeError):
    pass


# Accessible-name patterns for the control that dismisses the intermittent post-login
# two-factor-authentication prompt. Ordered most → least specific.
_2FA_SKIP_PATTERNS = [
    re.compile(r"skip", re.I),
    re.compile(r"not now|later|remind me", re.I),
]


async def _fill_verified(page: Page, selector: str, value: str, label: str) -> None:
    """Fill a field and confirm the value stuck, retrying if the SPA reset it.

    The login page hydrates client-side after first paint; a fill that lands mid-hydration
    gets wiped, and the form then submits EMPTY credentials (observed: submit succeeded but
    the page stayed on /Login). Reading the value back catches that race.
    """
    loc = page.locator(selector).first
    for _ in range(3):
        await loc.fill(value)
        if await loc.input_value() == value:
            return
        await asyncio.sleep(0.5)  # hydration reset the field; settle and refill
    raise LoginError(f"{label} field would not hold the typed value (page kept resetting it)")


async def _dismiss_two_factor(page: Page) -> bool:
    """If the intermittent 2FA interstitial is showing, click its skip option.

    Returns True if something was clicked. Only called while the post-login redirect has not
    landed yet (never on /admin), so a stray "skip" elsewhere in the app cannot be hit.
    """
    for pattern in _2FA_SKIP_PATTERNS:
        for role in ("button", "link"):
            loc = page.get_by_role(role, name=pattern).first
            try:
                if await loc.is_visible(timeout=250):
                    label = (await loc.text_content() or "").strip()
                    await loc.click()
                    print(f"[+] 2FA prompt detected -- clicked skip option ({label!r}).")
                    return True
            except Exception:  # noqa: BLE001 - probing; absence is the normal case
                continue
    return False


async def login(playwright: Playwright, config: Config) -> tuple[Browser, Page, str]:
    """Log in and return (browser, page, cdp_url).

    The caller is responsible for keeping `browser` open for as long as the
    navigation Agent needs the session, and for closing it when done.
    """
    if not config.login_email or config.login_email == "your_email_here@example.com":
        raise LoginError("LOGIN_EMAIL is not set in .env")
    if not config.login_password or config.login_password == "your_password_here":
        raise LoginError("LOGIN_PASSWORD is not set in .env")

    print("[*] Starting login automation...")
    print(f"[*] Target URL: {config.login_url}")
    print(f"[*] Login Email: {config.login_email}")
    print(f"[*] Headless Mode: {config.headless}")

    print("[*] Launching Chromium browser (with CDP enabled for handoff)...")
    # No slow_mo: the login flow waits on explicit selectors/URLs below, and a slow_mo delay
    # is charged on EVERY Playwright call here (~10s of dead time before the agent starts).
    browser = await playwright.chromium.launch(
        headless=config.headless,
        args=[f"--remote-debugging-port={config.cdp_port}", "--window-size=1440,900"],
        # Don't let Playwright tear down the browser on Ctrl+C: the Runner installs its own
        # SIGINT handler for the human-in-the-loop pause (see runner._prompt_and_inject), and
        # the browser must survive the pause so the agent can resume against it.
        handle_sigint=False,
    )
    cdp_url = f"http://localhost:{config.cdp_port}"

    # This context is the one browser-use later attaches to over CDP, so ITS viewport owns
    # the geometry of every screenshot the agent sees. Playwright's 1280×720 default crops
    # this app's dense screens; 1440×900 keeps toolbars and table rows on-screen so element
    # indexes map to what the model can actually read (fewer misclicks / "not found").
    context = await browser.new_context(
        viewport={"width": 1440, "height": 900}, device_scale_factor=1.0,
    )
    if config.reveal_hidden_controls:
        # Install the reveal stylesheet into EVERY document this context creates, from
        # birth: hover-revealed/0-size controls otherwise never enter browser-use's
        # snapshot and fail replay's visibility gate. The agent and the replay engine both
        # drive pages of THIS context, so one init script covers authoring and replay
        # across every navigation. (Per-step and per-segment re-asserts back this up for
        # pages created outside the context — see agent_tools.ensure_reveal_css.)
        await context.add_init_script(REVEAL_CSS_JS)
    page = await context.new_page()

    try:
        print(f"[*] Navigating to {config.login_url}...")
        # domcontentloaded is enough: the very next line waits for the email field itself,
        # which is the readiness signal that actually matters (networkidle can add seconds
        # on a page with analytics/streaming requests).
        await page.goto(config.login_url, wait_until="domcontentloaded")
        print("[+] Page loaded.")

        email_selector = "#Input_Email, input[name='Input_Email'], input[id='Input_Email']"
        password_selector = "#Input_Password, input[name='Input_Password'], input[id='Input_Password']"
        await page.wait_for_selector(email_selector, timeout=45000)
        # Let the login page finish hydrating BEFORE typing: values typed mid-hydration get
        # wiped and the form then submits empty. The login page is light enough to reach
        # network idle quickly (unlike the post-login dashboard, which never settles).
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightTimeoutError:
            pass

        landed = False
        for attempt in (1, 2):
            print(f"[*] Filling credentials (attempt {attempt})...")
            await _fill_verified(page, email_selector, config.login_email, "email")
            await _fill_verified(page, password_selector, config.login_password, "password")

            # Hydration can finish AFTER the verified fills and wipe them; re-check right
            # before submitting so an empty form is never sent.
            for _ in range(2):
                if (await page.locator(email_selector).first.input_value() == config.login_email
                        and await page.locator(password_selector).first.input_value()
                        == config.login_password):
                    break
                print("[!] Fields were reset before submit -- refilling.")
                await _fill_verified(page, email_selector, config.login_email, "email")
                await _fill_verified(page, password_selector, config.login_password, "password")

            print("[*] Finding and clicking the login button...")
            submit_btn = None
            for selector in ("button[type='submit']", "input[value='login']"):
                try:
                    locator = page.locator(selector).first
                    if await locator.is_visible(timeout=500):
                        submit_btn = locator
                        print(f"[+] Found login button matching: '{selector}'")
                        break
                except Exception:
                    continue
            if not submit_btn:
                print("[-] No standard submit button found. Trying first visible button...")
                submit_btn = page.get_by_role("button").first
            await submit_btn.click()
            print("[+] Login form submitted.")

            print("[*] Waiting for navigation/login processing...")
            # Race two outcomes: the successful-login redirect (/admin) or the INTERMITTENT
            # 2FA prompt, which has a skip option. Poll for whichever appears first; skipping
            # 2FA loops back to waiting for the redirect.
            deadline = time.monotonic() + 20
            submitted_at = time.monotonic()
            while time.monotonic() < deadline:
                if "/admin" in page.url:
                    landed = True
                    break
                if await _dismiss_two_factor(page):
                    continue  # skip clicked; give the redirect a moment to land
                # Fast-fail: if the login form is back with an EMPTY password after a short
                # grace period, the submit was rejected — retry immediately instead of
                # burning the whole deadline (this was the "slow password" stall).
                if time.monotonic() - submitted_at > 3:
                    try:
                        pwd = page.locator(password_selector).first
                        if await pwd.is_visible() and await pwd.input_value() == "":
                            break
                    except Exception:  # noqa: BLE001 - mid-navigation; keep waiting
                        pass
                await asyncio.sleep(0.5)
            if landed:
                break

            # Still on the login form? The submit was rejected (e.g. hydration wiped a
            # field) — refill and resubmit once rather than handing the agent a login page.
            try:
                still_login = await page.locator(password_selector).first.is_visible(timeout=1000)
            except Exception:
                still_login = False
            if not still_login:
                # Unknown landing page (SSO variant / redirect change): proceed best-effort.
                print("[*] Did not observe the /admin redirect -- continuing after a short settle.")
                await asyncio.sleep(2)
                break
            print("[!] Still on the login page after submit -- retrying with fresh fills.")
        else:
            await save_login_error_screenshot(page, config.errors_dir, "login_rejected")
            raise LoginError(
                "Login was rejected twice (page stayed on the login form). "
                "Check LOGIN_EMAIL/LOGIN_PASSWORD, or see the failure screenshot."
            )
        await asyncio.sleep(1)  # small buffer for the SPA to boot after the redirect

        print(f"[+] Final URL after login: {page.url}")
        print("[+] Login process completed successfully!")
        return browser, page, cdp_url

    except LoginError:
        # Already screenshotted with a specific name; just release the browser.
        await browser.close()
        raise
    except PlaywrightTimeoutError as te:
        await save_login_error_screenshot(page, config.errors_dir, "login_timeout")
        await browser.close()
        raise LoginError(f"Timeout waiting for element during login: {te}") from te
    except Exception as e:
        await save_login_error_screenshot(page, config.errors_dir, "login_failed")
        await browser.close()
        raise LoginError(f"Unexpected error during login: {e}") from e
