"""Registry identity + selection tests.

The frozen hashes below were computed from the task prompts BEFORE they moved from
__main__.py into automation/tasks.py. task_store.task_id hashes the prompt to locate a
task's golden script, so if any of these change, that task's recording is orphaned —
a failure here means a prompt was edited (even one word), not that the test is stale.
If a prompt change is intentional, re-author the task and update its hash here.
"""
import pytest

from automation.pipeline import task_store as ts
from automation.tasks import TASKS, TaskSpec, resolve_task, select_tasks

FROZEN_TIDS = {
    "invoice": "b1e80d296d010bfa",
    "credit_notes": "85e1c12969229856",
    "estimates": "a0e85521700c798d",
    "receipt": "e4fbce6bfb7d147f",
    "item": "93f89bb2d795ff03",
    "purchase": "b3c459628fc28924",
    "purchase_credit_notes": "7b7504cbe4dbe443",
    "purchase_po": "e85506234b6bca3a",
    "purchase_payment": "75454b420d6940c2",
    "reimbursements": "bc1ab1aa8bc5da0f",
    "mileage": "7303af666e5e6df7",
    "expense_claims": "eccabb702283079a",
    "refund": "5b1c7c9aabfa88db",
    "journals": "94790614c26b1183",
    "fixed_asset": "84c25eda28e2ffcd",
    "disposed_asset": "1f7c9e24da7e8212",
    "banking": "8f7bc65c2a411131",
    "budget_manager": "3a82e778d765ce63",
    "dividend": "53b429e5f90d0771",
    "invoice_full_creation": "f8511fac5543467a",
    "crm_create_invoice": "c15b853adfd1c42d",
}


def test_no_tasks_lost_or_invented():
    assert set(TASKS) == set(FROZEN_TIDS)


def test_task_ids_stable():
    drifted = {k: ts.task_id(TASKS[k].prompt)
               for k in FROZEN_TIDS
               if ts.task_id(TASKS[k].prompt) != FROZEN_TIDS[k]}
    assert not drifted, f"prompt text drifted (golden scripts orphaned): {drifted}"


def test_every_task_has_a_marker():
    # All 21 registry tasks are write tasks; a None marker would silently disable the
    # ground-truth gate for them.
    missing = [k for k, t in TASKS.items() if not t.marker]
    assert not missing


def test_resolve_known_key_case_insensitive():
    assert resolve_task("  INVOICE ") is TASKS["invoice"]


def test_resolve_free_text_infers_marker():
    spec = resolve_task("add a new invoice for ACME with qty 3")
    assert spec.key == "adhoc"
    assert spec.marker == "Invoices"


def test_resolve_free_text_read_only_gets_no_marker():
    spec = resolve_task("check the bookkeeping dashboard totals")
    assert spec.marker is None


def test_resolve_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown TASK key"):
        resolve_task("not_a_task")


def test_select_all():
    assert len(select_tasks("all")) == len(TASKS)


def test_select_by_tag():
    keys = {t.key for t in select_tasks("tag:sales")}
    assert {"invoice", "receipt", "item"} <= keys
    assert "purchase" not in keys


def test_select_comma_list_preserves_order():
    assert [t.key for t in select_tasks("purchase, invoice")] == ["purchase", "invoice"]


def test_select_unknown_tag_and_key_raise():
    with pytest.raises(ValueError, match="Known tags"):
        select_tasks("tag:nope")
    with pytest.raises(ValueError, match="Unknown task key"):
        select_tasks("invoice,nope")


def test_taskspec_frozen():
    with pytest.raises(Exception):
        TASKS["invoice"].marker = "X"  # type: ignore[misc]


def test_specs_are_taskspecs():
    assert all(isinstance(t, TaskSpec) and t.key == k for k, t in TASKS.items())
