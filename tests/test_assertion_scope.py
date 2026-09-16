"""Assertion scoping: the health rules judge the app under test, not the whole browser.

The live failure this guards: a helper-tab subtask on fakenamegenerator.com dragged its ad
stack into the run's telemetry — 312 failed requests, 277 console errors, 5 5xx — and failed
every assertion on a run where the app itself was healthy. Scope restricts network rules to
requests whose URL host is the app's, and console rules to entries raised by app pages.
"""
from types import SimpleNamespace

from automation.pipeline import assertions as asserts
from automation.pipeline.assertions import evaluate, scope_hosts_for

APP = "https://test.actingoffice.com/admin"
HELPER = "https://www.fakenamegenerator.com/gen-random-gd-uk.php"


def _req(url, *, status=200, failed=False, page_url=APP):
    cls = "failed" if failed else f"{status // 100}xx"
    return {"step": 1, "url": url, "method": "GET", "status": None if failed else status,
            "failed": failed, "is_error": failed or status >= 400,
            "status_class": cls, "page_url": page_url}


def _entry(text, *, page_url, is_error=True):
    return {"step": 1, "severity": "error" if is_error else "info", "text": text,
            "is_error": is_error, "is_exception": False, "page_url": page_url}


def _spec(**over):
    spec = {"no_5xx": True, "no_failed_requests": {"allow_url_patterns": []},
            "no_console_errors": {"allow_patterns": []},
            "scope": {"hosts": ["actingoffice.com"]}}
    spec.update(over)
    return spec


def _results(collected, spec):
    return {r.name: r for r in evaluate(collected, spec)}


def test_scope_hosts_for_derives_the_registrable_host():
    # 3+ labels -> the parent domain, so api./www. siblings stay in scope.
    assert scope_hosts_for("https://test.actingoffice.com/") == ["actingoffice.com"]
    assert scope_hosts_for("https://app.foo.co.uk/x") == ["foo.co.uk"]
    # 2 labels / bare hosts stay as-is; unparsable -> unscoped.
    assert scope_hosts_for("https://actingoffice.com") == ["actingoffice.com"]
    assert scope_hosts_for("http://localhost:3000") == ["localhost"]
    assert scope_hosts_for("") == []


def test_helper_tab_noise_is_out_of_scope_and_app_errors_still_count():
    collected = {
        "network": {"requests": [
            _req("https://ads.doubleclick.net/x", failed=True, page_url=HELPER),
            _req("https://www.fakenamegenerator.com/img.png", status=502, page_url=HELPER),
            _req("https://test.actingoffice.com/api/list", status=200),
        ]},
        "console": {"entries": [
            _entry("ad blocked", page_url=HELPER),
            _entry("harmless", page_url=APP, is_error=False),
        ]},
    }
    res = _results(collected, _spec())
    assert res["no_5xx"].passed is True
    assert res["no_failed_requests"].passed is True
    assert res["no_console_errors"].passed is True
    assert "[scope: actingoffice.com]" in res["no_5xx"].detail

    # The same shapes ON the app still fail: scope hides noise, never app problems.
    collected["network"]["requests"].append(
        _req("https://api.actingoffice.com/save", status=500))
    collected["console"]["entries"].append(
        _entry("TypeError: x is undefined", page_url=APP))
    res = _results(collected, _spec())
    assert res["no_5xx"].passed is False
    assert res["no_console_errors"].passed is False
    assert res["no_5xx"].evidence[0]["url"] == "https://api.actingoffice.com/save"


def test_scope_matches_subdomains_but_not_lookalike_hosts():
    reqs = [_req("https://evilactingoffice.com/x", status=500),   # suffix, not subdomain
            _req("https://cdn.actingoffice.com/y", status=500)]
    res = _results({"network": {"requests": reqs}, "console": {"entries": []}}, _spec())
    assert res["no_5xx"].passed is False
    assert len(res["no_5xx"].evidence) == 1
    assert res["no_5xx"].evidence[0]["url"].startswith("https://cdn.")


def test_entries_without_page_url_still_count():
    # Pre-scoping recordings carry no page_url — they must not silently vanish.
    entries = [{"step": 0, "severity": "error", "text": "boom",
                "is_error": True, "is_exception": False}]
    res = _results({"network": {"requests": []}, "console": {"entries": entries}}, _spec())
    assert res["no_console_errors"].passed is False


def test_scope_none_keeps_todays_unscoped_behavior():
    reqs = [_req("https://www.fakenamegenerator.com/x", status=500, page_url=HELPER)]
    res = _results({"network": {"requests": reqs}, "console": {"entries": []}},
                   _spec(scope=None))
    assert res["no_5xx"].passed is False
    assert "[scope:" not in res["no_5xx"].detail


def test_default_spec_ships_unscoped_and_merge_can_override():
    assert asserts.DEFAULT_SPEC["scope"] is None
    merged = asserts.merge_spec(asserts.DEFAULT_SPEC, {"scope": {"hosts": ["x.com"]}})
    assert merged["scope"] == {"hosts": ["x.com"]}
    # And `scope` is a modifier, not a rule: it never appears as a result of its own.
    res = evaluate({"network": {"requests": []}, "console": {"entries": []}}, merged)
    assert all(r.name != "scope" for r in res)


# ------------------------------- collector page_url stamping -------------------------------


class _FakePage:
    def __init__(self, url):
        self.url = url

    def on(self, *_a):  # Collector._attach_page registers listeners; irrelevant here
        pass


def test_console_collector_stamps_the_producing_page(tmp_path):
    from automation.collectors.console import ConsoleCollector

    col = ConsoleCollector(SimpleNamespace(pages=[]), tmp_path)
    col._active = True
    page = _FakePage(HELPER)
    msg = SimpleNamespace(type="error", text="ad exploded",
                          location={"url": "https://ads.doubleclick.net/lib.js",
                                    "lineNumber": 3})
    col._on_console(msg, page)
    col._on_page_error(SimpleNamespace(message="boom", stack=None), page)

    entries = col.results()["entries"]
    assert [e["page_url"] for e in entries] == [HELPER, HELPER]
    assert entries[0]["source"] == "https://ads.doubleclick.net/lib.js"


class _FakeRequest:
    """Hashable request double (collector records key on the Request object)."""

    url = "https://ads.doubleclick.net/pixel"
    method = "GET"
    resource_type = "image"
    headers: dict = {}


def test_network_collector_stamps_the_issuing_page(tmp_path):
    from automation.collectors.network import NetworkCollector

    col = NetworkCollector(SimpleNamespace(pages=[]), tmp_path)
    col._active = True
    col._on_request(_FakeRequest(), _FakePage(HELPER))

    record, = col.results()["requests"]
    assert record["page_url"] == HELPER
    assert record["url"] == "https://ads.doubleclick.net/pixel"


# --------------------- capture scoping (the logs themselves) ---------------------
# The user's ask after the assertion scoping shipped: the ARTIFACTS must also hold only
# the app's logs — one live run recorded 3,680 requests of which ~3,600 were helper-tab
# ad-tech. Capture filters by the PRODUCING page's host, with the same suffix matching
# the assertions use; unscoped collectors behave exactly as before.


class _AppRequest:
    """Hashable app-page request double."""

    url = "https://api.actingoffice.com/Payroll/Clients/abc/Employees/"
    method = "POST"
    resource_type = "fetch"
    headers: dict = {}


class _FakeResponse:
    def __init__(self, request, status=200):
        self.request = request
        self.status = status
        self.status_text = "OK"
        self.headers = {}


def _scoped_network(tmp_path):
    from automation.collectors.network import NetworkCollector

    col = NetworkCollector(SimpleNamespace(pages=[]), tmp_path,
                           scope_hosts=["actingoffice.com"])
    col._active = True
    return col


def test_network_capture_scope_drops_helper_page_traffic_stickily(tmp_path):
    col = _scoped_network(tmp_path)
    ad = _FakeRequest()
    col._on_request(ad, _FakePage(HELPER))
    # The drop must be sticky: the ad request's later lifecycle events must not
    # resurrect it as a partial record via _record_for's setdefault.
    col._on_response(_FakeResponse(ad, status=200))
    col._on_request_finished(ad)
    col._on_request_failed(ad)

    out = col.results()
    assert out["requests"] == []
    assert out["summary"]["total"] == 0
    assert out["summary"]["out_of_scope"] == 1


def test_network_capture_scope_keeps_app_page_traffic_and_marker_write(tmp_path):
    col = _scoped_network(tmp_path)
    write = _AppRequest()
    col._on_request(write, _FakePage(APP))   # app page -> api subdomain: kept
    col._on_response(_FakeResponse(write, status=200))

    record, = col.results()["requests"]
    assert record["method"] == "POST" and record["status"] == 200
    assert "Employees" in record["url"]      # the marker gate still sees the create-write
    assert col.results()["summary"]["out_of_scope"] == 0


def test_network_capture_scope_fails_open_on_unreadable_page(tmp_path):
    class _DeadPage:
        @property
        def url(self):
            raise RuntimeError("page closed")

    col = _scoped_network(tmp_path)
    col._on_request(_FakeRequest(), _DeadPage())   # host unknowable -> kept
    col._on_request(_FakeRequest(), None)          # no page handle -> kept
    assert col.results()["summary"]["total"] == 2


def test_unscoped_network_collector_records_everything(tmp_path):
    from automation.collectors.network import NetworkCollector

    col = NetworkCollector(SimpleNamespace(pages=[]), tmp_path)
    col._active = True
    col._on_request(_FakeRequest(), _FakePage(HELPER))
    assert col.results()["summary"]["total"] == 1
    assert col.results()["summary"]["out_of_scope"] == 0


def test_console_capture_scope_drops_helper_page_entries(tmp_path):
    from automation.collectors.console import ConsoleCollector

    col = ConsoleCollector(SimpleNamespace(pages=[]), tmp_path,
                           scope_hosts=["actingoffice.com"])
    col._active = True
    msg = SimpleNamespace(type="error", text="ad exploded",
                          location={"url": "https://ads.doubleclick.net/lib.js",
                                    "lineNumber": 3})
    col._on_console(msg, _FakePage(HELPER))                                # dropped
    col._on_page_error(SimpleNamespace(message="boom", stack=None),
                       _FakePage(HELPER))                                  # dropped
    col._on_console(SimpleNamespace(type="error", text="app broke",
                                    location={}), _FakePage(APP))          # kept
    col._on_console(SimpleNamespace(type="log", text="early", location={}),
                    _FakePage(""))                                         # blank url: kept

    out = col.results()
    assert [e["text"] for e in out["entries"]] == ["app broke", "early"]
    assert out["summary"]["out_of_scope"] == 2
