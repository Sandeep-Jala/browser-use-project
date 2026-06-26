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

from playwright.async_api import Browser, Page, Playwright, TimeoutError as PlaywrightTimeoutError

from automation.config import Config
from automation.browser.error_capture import save_login_error_screenshot


class LoginError(RuntimeError):
    pass


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
    browser = await playwright.chromium.launch(
        headless=config.headless,
        slow_mo=1000,
        args=[f"--remote-debugging-port={config.cdp_port}"],
    )
    cdp_url = f"http://localhost:{config.cdp_port}"

    context = await browser.new_context()
    page = await context.new_page()

    try:
        print(f"[*] Navigating to {config.login_url}...")
        await page.goto(config.login_url, wait_until="networkidle")
        print("[+] Page loaded.")

        print("[*] Locating email input field...")
        email_selector = "#Input_Email, input[name='Input_Email'], input[id='Input_Email']"
        await page.wait_for_selector(email_selector, timeout=10000)
        await page.locator(email_selector).first.fill(config.login_email)

        print("[*] Locating password input field...")
        password_selector = "#Input_Password, input[name='Input_Password'], input[id='Input_Password']"
        await page.wait_for_selector(password_selector, timeout=10000)
        await page.locator(password_selector).first.fill(config.login_password)

        print("[*] Finding and clicking the login button...")
        button_selectors = ["button[type='submit']", "input[value='login']"]
        submit_btn = None
        for selector in button_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.is_visible(timeout=500):
                    submit_btn = locator
                    print(f"[+] Found login button matching: '{selector}'")
                    break
            except Exception:
                continue

        if not submit_btn:
            print("[-] Could not find standard submit button selector. Trying first visible button...")
            submit_btn = page.get_by_role("button").first

        await submit_btn.click()
        print("[+] Login form submitted.")

        print("[*] Waiting for navigation/login processing...")
        try:
            # Some post-login dashboards (e.g. live charts/polling widgets)
            # never reach true network idle, so this wait is best-effort only
            # and must not be treated as a login failure if it times out.
            await page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightTimeoutError:
            print("[*] Page did not reach network idle (likely live-updating "
                  "widgets) -- continuing anyway.")
        await asyncio.sleep(3)  # short buffer for client-side redirects

        print(f"[+] Final URL after login: {page.url}")
        print("[+] Login process completed successfully!")
        return browser, page, cdp_url

    except PlaywrightTimeoutError as te:
        await save_login_error_screenshot(page, config.errors_dir, "login_timeout")
        await browser.close()
        raise LoginError(f"Timeout waiting for element during login: {te}") from te
    except Exception as e:
        await save_login_error_screenshot(page, config.errors_dir, "login_failed")
        await browser.close()
        raise LoginError(f"Unexpected error during login: {e}") from e
