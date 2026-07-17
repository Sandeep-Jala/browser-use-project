"""Registry (tasks.yaml) loading, identity, and selection tests.

FROZEN_TIDS pins subtask_store.task_id for every registry prompt. task_id keys each task's
cached subtask decomposition, so if any hash changes here, that cache is orphaned and the
task is re-decomposed — possibly into a different split that misses its library entries. A
failure here means a prompt was edited (even one word), not that the test is stale. If a
prompt change is intentional, update its hash here.
"""
import pytest

import automation.tasks as tasks_mod
from automation.pipeline import subtask_store as ss
from automation.tasks import SubtaskDecl, TaskSpec, load_tasks, resolve_task

FROZEN_TIDS = {
    "invoice": "b1e80d296d010bfa",
    # credit_notes + invoice_full_creation re-imported verbatim from the source test doc
    # (2026-07-17, registry moved to tasks.yaml) — new ids; every other bookkeeping/CRM
    # prompt kept its pre-YAML id, so their cached decompositions stayed live.
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
    "invoice_full_creation": "00a74cbcea4d9f17",
    "crm_create_invoice": "c15b853adfd1c42d",
    "nps_02_review_for_dropdown": "a1bba1c849c557f8",
    "nps_03_mandatory_review_for": "69b65b3f3a938f6c",
    "nps_04_review_for_tag": "66b9c421c459cdf3",
    "nps_05_cc_bcc": "814b0744e1c47d6f",
    "nps_06_recipient_sync": "e5adb2db95d6d9d0",
    "nps_07_duplicate_requests": "40c9b16b83aac213",
    "nps_08_pending_requests_tag": "9a5753db581dd9b4",
    "nps_09_drafts_sync": "3e9aa1cb01b40cba",
    "nps_10_notification_options": "cc040e8bbb5ebd6b",
    "nps_11_survey_details": "889b903089599369",
    "nps_12_review_conversion": "e3c685b2c70bebe4",
    "nps_13_name_logic": "bcfc688fa3cf3baa",
    "nps_14_expand_collapse": "2b1f3d748f8d2c94",
    "nps_15_review_format": "e31ded7f94a9a596",
    "nps_16_number_reference": "4f31b281d8a87bcd",
    "nps_17_manual_review_visibility": "d4c00b592fff06be",
    "nps_18_form_redirection": "c2755fed3c973a3f",
    "nps_19_review_restriction": "8258b415b4ae249f",
    "nps_20_manual_review_management": "7a5f4b99e35dab46",
    "nps_21_review_for_name_sync": "e141444a67ce4d27",
    "nps_22_mandatory_rating": "4fb76f1098a2fb34",
    "nps_23_feedback_persistence": "cdb254c68bdcc172",
    "nps_24_feedback_privacy": "f9d76715c11ed1f2",
    "nps_25_client_name_sync": "84f579b9f86bbfaa",
    "nps_26_submission_loading": "05b1c3ed5c71d951",
    "nps_27_report_structure": "509b3df53ff1ca88",
    "nps_28_report_privacy_columns": "ba00cfb9bc757dcc",
    "nps_29_report_filters": "62c99162b9e9230c",
    "nps_30_report_columns": "6c6269554ec452a7",
    "nps_31_report_pagination": "6a3f97cd804f4880",
    "nps_32_report_date_accuracy": "8011ab5d93d36926",
    "nps_33_unsubmitted_status": "7a7d8956f8f307ca",
    "nps_34_report_open_form": "7fb700bcae68d219",
    "nps_35_multi_year_collection": "645ec92ed44a8dd6",
    "nps_36_all_in_one_report": "157e209b3541b64b",
}


# ------------------------------- the real registry -------------------------------


def test_no_tasks_lost_or_invented():
    assert set(load_tasks()) == set(FROZEN_TIDS)


def test_task_ids_stable():
    tasks = load_tasks()
    drifted = {k: ss.task_id(tasks[k].prompt)
               for k in FROZEN_TIDS
               if ss.task_id(tasks[k].prompt) != FROZEN_TIDS[k]}
    assert not drifted, f"prompt text drifted (cached decompositions orphaned): {drifted}"


def test_no_login_steps_or_credentials_in_prompts():
    # The framework logs in itself; a prompt must never carry the login step, credentials,
    # or the raw books URL (tasks say "go to Bookkeeping module" instead).
    for key, spec in load_tasks().items():
        low = spec.prompt.lower()
        for forbidden in ("login using", "log in to http", "password", "aoadmin",
                          "welcome1@", "@capsitech.com", "demo.admin@",
                          "actingoffice.com/books"):
            assert forbidden not in low, f"{key}: prompt still carries {forbidden!r}"


def test_markers_only_where_known():
    tasks = load_tasks()
    # Bookkeeping/CRM write tasks keep their verified markers...
    assert tasks["invoice"].marker == "Invoices"
    assert tasks["credit_notes"].marker == "Refunds"
    assert tasks["mileage"].marker == "MileageClaims"
    # ...and every NPS verification task runs with the gate disabled (marker absent) —
    # a guessed marker would force-fail honest successes.
    nps = [t for t in tasks.values() if "nps" in t.tags]
    assert len(nps) == 35
    assert all(t.marker is None for t in nps)


def test_specs_are_frozen_taskspecs():
    tasks = load_tasks()
    assert all(isinstance(t, TaskSpec) and t.key == k for k, t in tasks.items())
    with pytest.raises(Exception):
        tasks["invoice"].marker = "X"  # type: ignore[misc]


# ------------------------------- resolution / selection -------------------------------


def test_resolve_known_key_case_insensitive():
    assert resolve_task("  INVOICE ").key == "invoice"


def test_resolve_free_text_gets_no_marker():
    # No marker inference: an ad-hoc task may fire no create-write at all; the gate is
    # enabled for ad-hoc prompts only explicitly, via --marker.
    spec = resolve_task("add a new invoice for ACME with qty 3")
    assert spec.key == "adhoc"
    assert spec.marker is None


def test_resolve_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown TASK key"):
        resolve_task("not_a_task")


# ------------------------------- loader edge cases -------------------------------


def test_missing_registry_is_empty_and_adhoc_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks_mod, "TASKS_FILE", tmp_path / "tasks.yaml")
    assert load_tasks() == {}
    spec = resolve_task("do a thing on the current page")
    assert spec.key == "adhoc" and spec.marker is None
    with pytest.raises(ValueError, match="Unknown TASK key"):
        resolve_task("invoice")


def test_entry_without_prompt_fails_loud(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("broken:\n  tags: [x]\n")
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        load_tasks(p)


def test_non_mapping_registry_fails_loud(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_tasks(p)


def test_entry_fields_parse(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "My_Task:\n"
        "  prompt: >-\n"
        "    do the thing\n"
        "    across two lines\n"
        "  marker: Things\n"
        "  tags: [a, b]\n"
        "  subtasks:\n"
        "    - prompt: 'open {{page}}'\n"
        "      values: {page: settings}\n"
        "    - prompt: 'save it'\n"
        "      marker: Things\n"
    )
    tasks = load_tasks(p)
    spec = tasks["my_task"]  # keys are lowercased
    assert spec.prompt == "do the thing across two lines"  # block-scalar folding collapsed
    assert spec.marker == "Things" and spec.tags == ("a", "b")
    assert spec.subtasks == (
        SubtaskDecl(prompt="open {{page}}", values={"page": "settings"}),
        SubtaskDecl(prompt="save it", marker="Things"),
    )
