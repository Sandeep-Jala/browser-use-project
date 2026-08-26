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
    # Re-frozen 2026-07-20: prompt edited after the library/decomposition reset, so the
    # orphaned old cache (00a74cbcea4d9f17) was already gone by design. Corrected
    # 2026-07-21: the 07-20 re-freeze (a895eb5badb377c2) hashed a draft wording, not the
    # committed prompt.
    "invoice_full_creation": "4e3bb43616b67254",
    "crm_create_invoice": "c15b853adfd1c42d",
    # Re-frozen 2026-07-21: "select a business name randomly" was pinned to 290 CREW
    # LIMITED (2026-07-20, same edit batch as invoice_full_creation) but never re-frozen;
    # the orphaned old decomposition (a1bba1c849c557f8) was deleted, the new prompt's
    # cache already exists.
    "nps_02_review_for_dropdown": "76568417bebd8569",
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
    # Payroll/CRM tasks. Re-frozen 2026-07-22 after the user retargeted the prompts to
    # Food Limited (and added the crm/pension tasks). The add-employee change is a pure
    # value swap (business name) — decomposition derives, subtask entries still replay.
    # The import and pension prompts CHANGED WORDING beyond values ("The FOOD LIMITED",
    # "and then click save", "The FOOD LIMITED name"), so their navigation/import
    # subtasks re-author on first run instead of replaying the committed skills.
    # Re-frozen 2026-07-29: the fakenamegenerator sentence (shared verbatim by its three
    # carriers: here, payroll_food_limited_e2e, payroll_food_limited_e2e_rti) now opens
    # gen-male-gd-uk.php directly — the URL pre-encodes male/Scottish/UK and generates a
    # fresh identity per load, so the helper-tab step is load -> extract -> done instead
    # of three dropdowns + Generate on an ad-heavy page. Old aux segment cc27ec56df2ab562
    # is orphaned; each carrier records the shared replacement on its first run.
    "payroll_add_generated_employee": "f2139a1401d9b262",
    # Re-frozen 2026-07-22 (second edit round): import/pension retargeted to plain
    # "FOOD LIMITED"; pay_forecast added (matches the live run's decomposition cache).
    "payroll_import_employees_csv": "4ef26eab374400f7",
    "payroll_add_pension_scheme": "f7c69273c96221c1",
    # Re-frozen 2026-07-27 (all three pay-forecast copies). Two changes:
    #  - the search step became a LOOP node ("repeating until you see that the pay rows have
    #    loaded"), so a slow-loading grid is retried live instead of edited blind;
    #  - the net-pay step names the FEATURE the row icon opens ("set the net pay of Feb-27
    #    to 4000 using the Net to Gross option on the Feb-27 row"). Net pay is NOT a grid
    #    field: a live run typed 4000 into the Feb-27 PAY box instead, on top of a value
    #    whose clear had no-opped, leaving 0,000,000,000,000.
    # Clearing is deliberately NOT described in these prompts — it is UI trivia. An earlier
    # attempt to auto-recover it (select-all re-clear inside agent_tools.input) did not work
    # against the real fields and was removed; nothing compensates for a failed clear today.
    # Re-frozen 2026-07-29 (user edit): "combobox search employee" -> "combobox select
    # employee".
    "payroll_pay_forecast": "2c08f433c09a15f1",
    "crm_add_contact": "5f244c1446787a15",
    # Re-frozen 2026-07-24: data-request tail reworded after the 07-23 freeze ("latest
    # sent request"/"Submittec" -> "the top sent request"/"Submitted", "now click" ->
    # "Then click"). The orphaned old decomposition (9202070e2fe707cc) was already gone —
    # the decompositions dir was cleared 2026-07-24, so nothing was lost.
    # Re-frozen 2026-07-24 evening: "the top sent request" -> "the top sent request only"
    # (user edit). The old decomposition (d6642c2001b4bfc4) and its status entry
    # (0817b06f0bd6bc7d, "Note {{remarks}}" tokenization) were archived; the new split's
    # entries (506e646d776bdd2f status, c6cafb1fa01198e6 bound verify) are the live ones,
    # now also shared by payroll_food_limited_e2e's DECLARED subtasks.
    "crm_data_request": "ca42aeecaa638d0c",
    # Combined FOOD LIMITED end-to-end chain (add generated employee + pay-forecast
    # downloads + data-request send/verify). No marker: multi-write chain.
    # Re-frozen 2026-07-23: pay-forecast sentence reworded to "search for employee which
    # we added" (runtime reference — routed to the agent by the consumer-wording net).
    # Re-frozen 2026-07-24: same data-request tail rewording as crm_data_request, plus a
    # new pay-forecast clause ("if there is no data shown, refresh the page and do it
    # again" — the typo "sefresh" was corrected while re-freezing; the old decomposition
    # (f486b7ab4dabbd20) was already gone with the cleared decompositions dir).
    # Re-frozen 2026-07-29: fakenamegenerator direct-URL reword (see the
    # payroll_add_generated_employee note above — same sentence, same replacement segment).
    # Re-frozen again same day (user edit, applied to both e2e copies to keep the shared
    # pay-forecast sentence byte-compatible): "combobox search employee" -> "combobox
    # select employee".
    "payroll_food_limited_e2e": "287ea967fcab1973",
    # RTI payrun-processing chain (Save/Next per employee, min-wage + pop-up branches,
    # employee-8 expenses/addition, 2nd-last FPS submit, date roll, client sort order).
    # Re-frozen 2026-07-27: the employee pass and the advance-to-Struan clause reworded
    # into explicit per-employee loops ("after each click check that the next employee
    # has loaded ... stopping as soon as Owen Millar is the employee shown") — the
    # "check that" phrasing deliberately lands both loops in the JUDGE net so they always
    # run live and are never replayed as a fixed click count (a cached 5-step replay of
    # "Save & Next for all the existing employees" had stopped mid-list, and the old
    # "when we reach employee Name Owen Millar" wording then let the recovery agent jump
    # to Owen by clicking his row, skipping employees). Old decomposition c7485f5b089e1f4d
    # and its loop entry (0b3f702083deb2c0) are orphaned.
    # Re-frozen 2026-07-30: (a) the morning's user edits (Alan Marshall / David Williamson
    # retargets, disabled-button clauses) had already drifted the pin past 5d74633bf54934cd;
    # (b) both date lines disambiguated to "using the period dropdown in the top bar,
    # set/change the period ..." after the wrong-referent runaway — "Set the date to
    # May-26" made the agent type a fabricated 26/05/2026 into the per-employee Pay date
    # field and loop Save & Next across 5 employees under Apr-26 before finding the
    # top-bar dropdown; (c) the SECOND employee pass regained "and after each click check
    # that the next employee has loaded" so it classifies loop again (the 07-29 reword had
    # dropped its judge phrase, silently demoting it to a cacheable fixed-click action).
    # Orphans today's decomposition 853742267f5ce2ca (left on disk, inert).
    # Re-frozen 2026-07-30 PM: (a) the popup conditional gained its explicit false
    # branch ("if no pop up appears, do nothing ... never click other buttons to make
    # it appear") — with the popup suppressed server-side (SkipNotification flag from
    # an earlier session), the agent hunted for "Process" and find_by_text resolved it
    # via tooltip text to the "Reminder to process the payroll" icon button, opening
    # and closing the email compose modal in an endless loop (one Send POST luckily
    # bounced: "Email not sent"); (b) "then click Save & Next" became "once (... do
    # not click it again)" — the pay run completes async, the agent re-clicked while
    # waiting and ran 4 pay runs, committing a 4-click recording for a 1-click step
    # (0725aaf250fbb33f, archived). Orphans decomposition 31eb68f783fe19c8.
    # Re-frozen 2026-08-03: the branch was hard-reset to 4523b58 ("few minor parameter
    # change"), whose tasks.yaml predates the 07-30 PM popup/false-branch rewording this
    # pin captured — the commit itself shipped with this test red. Pinned back to what
    # the CURRENT prompts hash to (the 07-30 evening ids); today's runs already
    # re-decomposed and executed these prompts (decomposition 287ea967fcab1973).
    # Re-frozen 2026-08-05: full imperative rewrite of the RTI flow — per-employee loop
    # steps spelled out (read the name first; explicit stop / minimum-wage / disabled
    # branches; "Keep repeating ... until X is the employee shown"), start month
    # May-26 -> Apr-26, stop employees retargeted (Owen Millar -> Alan Marshall,
    # David Williamson -> Bruce Wright), FPS checkbox rewording, and the "next month"
    # parenthetical corrected June-26 -> May-26 (leftover from the May start). The
    # rewrite dropped every judge phrase, which silently demoted both employee loops to
    # cacheable actions — a frozen Save & Next replay then saved the stop-target
    # employee — until node_kind grew the judge-free loop net (see
    # test_node_kind_loop_detection). Orphans decompositions 1171ff661871481a /
    # 8f5212530e16673c (still usable as derived-match value-swap parents).
    # Re-frozen 2026-08-05 PM (catch-up): the imperative rewrite kept being edited after
    # the freeze above — combobox-named period selection ("in the combobox select Monthly
    # and the combobox next to it select May-26", the ambiguous-referent rule), FPS line
    # now "tick the employees checkbox, tick the checkbox of Alan Marshall", explicit
    # "(May-26)" next-month parenthetical, and a closing "take no further actions"
    # sentence. Run 20260805_123407_334719 executed THIS wording and its decomposition
    # (02498b06d4cd44da) is already cached; no cache ever existed for 98b95b62d114b4c7,
    # so nothing is orphaned.
    # Re-frozen 2026-08-05 later PM, twice-over (user edits between runs): first the
    # expense/FPS target employee moved Alan Marshall -> Aleksander Millar (Alan carries
    # the morning runs' duplicate rows) — run 20260805_142523_388784 executed that
    # wording (6db7d88c88a0e6b1, decomposition cached, now orphaned as a value-swap
    # parent); then, after the FPS-redo forensics, Aleksander Millar -> Alistair Allan
    # and the self-contradictory "change the date to the next month (May-26)" became the
    # explicit "combobox value for period to Jun-26". No cache yet for this id — the
    # next run decomposes fresh.
    # Re-frozen 2026-08-05 evening: further user edits between runs (the FPS header-
    # checkbox line and RTI polish continued to evolve alongside the e2e_rti rewording).
    # Re-frozen 2026-08-10: user retargeted the stop/expense/FPS employee
    # Alistair Allan -> Aran Allan (edited directly in tasks.yaml).
    "payroll_rti_process": "6fcbcf3a74277901",
    # Added 2026-08-05: the pre-rewrite RTI wording, kept deliberately for comparison
    # while the imperative rewrite above beds in (its stop employees were retargeted to
    # Alan Marshall / Bruce Wright and the FPS checkbox rewording applied here too).
    # Re-frozen 2026-08-05 later PM: same employee retargets as the main copy
    # (Alan Marshall -> Aleksander Millar -> Alistair Allan), applied here deliberately
    # to keep the shared wording identity.
    # Re-frozen 2026-08-10: same Alistair Allan -> Aran Allan retarget as the main copy.
    "payroll_rti_process_old": "bb7f800478491cde",
    # Re-frozen 2026-07-27: the data-request tail was rewritten in ALL THREE copies that
    # carry it (here, crm_data_request, payroll_food_limited_e2e) against screenshots of the
    # real UI — the old "click on status sent ... select status Sent/Submitted, Note well
    # done ... select employee and click verify" conflated the list badge with the dialog and
    # named a status ("Sent") the dropdown does not offer (Pending/Submitted/Revoked only).
    # No longer a verbatim concatenation of the two halves (see the note above the tests).
    # payroll_food_limited_e2e + payroll_rti_process concatenated VERBATIM (asserted below),
    # to run the pair as one pass. ~20 subtasks, which is why decompose.MAX_SUBTASKS was
    # raised 15 -> 24; at the old ceiling this split was rejected and the run degraded to
    # whole_prompt_fallback. Re-freeze whenever either half is reworded.
    # Re-frozen 2026-07-29, two edits at once: (a) the user's FPS bulk-upload rewording
    # ("...employee list, unselect all the employee, Select the employee we added earlier,
    # Click on FPS, select document ... and Submit") — the id had already drifted to
    # df1ae025b2ec3d6a before this freeze caught up; (b) the shared fakenamegenerator
    # direct-URL reword (see the payroll_add_generated_employee note above).
    # Re-frozen again 2026-07-29 afternoon (user edits): the FIRST RTI employee pass
    # gained "if the Save & Next button is disabled, Move on to the next employee"; then
    # the SECOND (advance-to-Struan) pass was reworded to "if the Save and next is
    # disabled move on to the next employee" (dropping its "check that each save
    # advances ... never click a name" parenthetical), and the shared pay-forecast
    # sentence became "combobox select employee". Orphans the day's earlier
    # decompositions (73189d7e98b77d8e, 24278f203524c274); the next run re-decomposes.
    # Re-frozen 2026-07-30: same three RTI-half edits as payroll_rti_process (employee-name
    # retargets had already drifted the pin past d96d84af87ab5da5; period-dropdown
    # disambiguation of both date lines; second pass regained its check-that clause ->
    # loop kind), applied to this copy deliberately so the reworded period/loop subtasks
    # keep one shared wording identity across both tasks.
    # Re-frozen 2026-07-30 PM: same two RTI-half edits as payroll_rti_process (popup
    # conditional false branch, single-click Save & Next clause), applied to this copy
    # deliberately to keep the shared wording identity. Orphans 994afb778ccdf6e3.
    # Re-frozen 2026-08-03: same rollback re-pin as payroll_rti_process above.
    # Re-frozen 2026-08-05: two RTI-half edits applied to this copy (its wording stays
    # the OLD style — the imperative rewrite touched only payroll_rti_process): second
    # pass's stop employee David Williamson -> Bruce Wright, and the FPS bulk-upload
    # line reworded to "click on employee checkbox, click on the checkboxof the
    # employee we added earlier" (sic — the missing space is in tasks.yaml).
    # Re-frozen 2026-08-05 later PM: the same Alan Marshall -> Aleksander Millar ->
    # Alistair Allan retargets applied to this copy's RTI half.
    # Re-frozen 2026-08-05 evening: the user rewrote the e2e half into the polished
    # imperative wording (the 16:19 run executed it as 0af549f75a1a352f) and the RTI half
    # carries the Alistair Allan retargets. This entry now also declares its 19-subtask
    # split in tasks.yaml (Tier 1) — declarations do not change the task id, but any
    # prompt reword MUST update the declared slices in lockstep (they are verbatim
    # slices; test_e2e_rti_declared_subtasks_survive_validation enforces it).
    # Re-frozen 2026-08-07: the user reworded the e2e slices at 09:55 (run
    # 20260807_095537 already ran under this id).
    # Re-frozen 2026-08-07 evening: seg-4 slice reworded to name the real Net-to-Gross
    # controls (pencil icon -> 'Salary to take home' popup -> Net amount -> Calculate)
    # and to redo BOTH pay changes after a refresh (a reload wipes client-staged edits) —
    # root-caused from run 20260807_161823's video ("Could not locate Save button").
    # Re-frozen 2026-08-10: expense/bonus slices reworded from run 20260810_092500 —
    # the expense grid shows the NET value (100.00, not the typed Cost 200) and both
    # dialog Saves fire no request, so the old wording made the agent re-add ~6
    # duplicate expenses; the bonus slice now names its real "+" control (the agent
    # opened "Additional Inputs"/btnExtras instead and false-passed on residue rows).
    # Re-frozen 2026-08-10 later (user choice): the verification narration was cut
    # back OUT of the wording — prompts stay natural task language; the neutral
    # no-writes receipt, the window write rule, and the declared verify: checks carry
    # the guarding instead. The bonus slice keeps only the in-bounds "+ next to
    # Additions or Deductions" referent fix.
    # Re-frozen 2026-08-10 (2): the email slice gains the Drafted-recovery sentence —
    # a missed send lands the email as Drafted, and the email button next to the
    # Drafted status reopens the Send Email section (user-described app behavior;
    # in-prompt self-recovery, same shape as the slice's existing Save-again clause).
    # Re-frozen 2026-08-10 (3): the FPS bulk-upload slice now states the header
    # checkbox's semantics (left of 'Employees' = select/unselect ALL employees) and
    # the goal state — unselect everyone first, then tick ONLY Alistair Allan — after
    # the agent misread the terse click list (a header click that SELECTS all followed
    # by the Alistair click submits everyone EXCEPT him; cf. the Aaron Wilson FPS
    # refusals in run 20260810_092500). Also drops the "the the" typo.
    # Re-frozen 2026-08-10 (4): user retargeted Alistair Allan -> Aran Allan in the
    # RTI half (direct tasks.yaml edit; same rename as both payroll_rti_process
    # copies). The find-replace missed ONE occurrence split across a folded line
    # ("until Alistair\n Allan" in the first employee loop) — completed to Aran Allan
    # so the loop's stop condition no longer contradicts its repeat clause.
    # Re-frozen 2026-08-10 (5): both payrun dialog slices gain once-only anti-duplicate
    # wording (user choice — the sanctioned fix from the run-092500 residual-risk note):
    # the expense slice states the silent close + net-value display (grid shows 100.00,
    # not the typed 200) as what a successful add looks like, the bonus slice the silent
    # close; both forbid re-adding.
    # Re-frozen 2026-08-11: the email slice now names the From control ("open the From
    # dropdown and select the no-reply option ... do not type an email address") — run
    # 20260811_102219 forensics: "change the From field to no-reply" under-specified the
    # control, and the agent completed the fragment into a fabricated address
    # (no-reply@actingoffice.co.uk, domain borrowed from the on-page To recipient); the
    # server refused the send ("Email not sent" in a 2xx body) and drafts piled up.
    # Re-frozen 2026-08-11 (2): the user trimmed the belt-and-braces parenthetical
    # ("From is a dropdown; do not type an email address into it") back out — prompts
    # stay natural; "open the From dropdown and select the no-reply option" already
    # names the control and the selection mode.
    # Re-frozen 2026-08-11 PM (counter rewrite, c194d6a9adbcfd1e → a718bbf3184164be):
    # both RTI employee passes went name-based-loop → fixed-count action ("exactly 5
    # clicks" / "exactly 14 more clicks"; subs-10's own Save & Next click saves
    # employee #1, so 1+5+14 = the first 20), the expense lands on "the 7th employee
    # of the pay run" instead of Aran Allan, and the FPS slice ticks the first 20
    # employees after the unselect-all guard. Same edit: subs[9] pinned kind: action
    # (records despite the button named Verify) and subs[11] got the popup probe —
    # neither touches identity; the id drift is the wording alone.
    # Re-frozen again 2026-08-11 evening (a718bbf3184164be → e291dcdf0741eaa2): the
    # user fixed the Add-Employee NI number to the literal AB124557C (was "AB followed
    # by a random 6-digit number and ending with C") — determinism for the recorded
    # run; the slice's values are now fully replayable.
    # Re-frozen 2026-08-12 (e291dcdf0741eaa2 → 541b2091dc0fd192): the user cut the
    # task down to its first half — everything from the Sent-badge slice onward is
    # commented out in tasks.yaml pending re-enable (payroll_detailed_review_fps is
    # the active RTI vehicle) — and extended the email slice's Drafted recovery
    # ("click Send again" + reopen-and-reselect-no-reply).
    # Re-frozen 2026-08-12 PM (541b2091dc0fd192 → 30213b6416455597): the Net-to-Gross
    # slice reworded after the 45-step seg-4 spiral — the user dropped the
    # refresh-first recovery + kind: loop (now a recordable action), and the slice
    # gained the popup facts (Net amount box beside the £ inside the popup, starts at
    # 0, grid cells behind it are NOT the target, remaining-periods checkbox stays
    # unticked, popup closes on Calculate → Feb-27 £4,000.00) plus a reopen-with-the-
    # pencil recovery; refresh is no longer prescribed at all for this slice.
    # Re-frozen again 2026-08-12 (→ 00cddadd6374f2ec): FIXED identity — REVERSED the
    # same day at the user's direction ("get it back to the earlier version where it
    # went to a website"). → 94ae70dcaa115764: the fakenamegenerator aux slice and the
    # noted-details wording are RESTORED verbatim (NI stays the literal AB124557C);
    # the pay-forecast load slice is flattened to a plain action ("wait until the pay
    # rows have loaded" — no repeat cue) per the user's "I need it recorded". With the
    # consumes_noted_data producer-override, the aux slice records/replays again;
    # subs 2-3 are consumers by design (replayable only via runtime bindings).
    # Re-frozen 2026-08-17 (94ae70dcaa115764 → 76b8aeff80b9c11c): the Data Request
    # slice reworded from run 20260817_093555 — "select the employee" named no control,
    # so the agent completed it with the previous slice's combobox recipe and typed the
    # name into the Tax-year react-select, then scroll-hunted the page while the
    # virtualized list (window of ~20 rows of 78) never had the row in DOM. The slice
    # now names the real gesture: scroll inside the employee list itself until the
    # noted employee's name is in view, tick the checkbox on that row (user kept the
    # wording lean — no combobox-forbidding sentence). The virtualization parenthetical
    # says "at once", NOT "at a time" — "at a time" is a _LOOP_REPEAT_RE cue and with
    # "until" it reclassified the slice kind action → loop (never cached). Replaces
    # sid 97bfb8637b9538e3 (segment never committed — no library entry orphaned).
    # Re-frozen 2026-08-17 PM (76b8aeff80b9c11c → bd91fc77296c45e4): user reworded the
    # Pay Forecast slice after run 133135's slow-connection stalls — it now opens with
    # "Refresh the page and then type the noted employee name" (in-prompt recovery for
    # the skeleton-loading Pay Forecast page; still kind=action, no repeat cue). The
    # reword orphans the old seg-3 sid a4f26f734d041598 (library entry unreferenced;
    # the next run authors seg 3 fresh under its new sid).
    # Re-frozen 2026-08-24 (bd91fc77296c45e4 -> 050d2647bf273450): the task is its first
    # SIX slices — 8393188 had uncommented the RTI half (19 slices) and the user then
    # re-commented it, this time also parking the Data Request + Send Email slices, and
    # reworded inside the commented block (Apr-26, "next 6 employees"). Decompositions
    # cached under any earlier id are orphaned; the next run authors the task fresh.
    "payroll_food_limited_e2e_rti": "050d2647bf273450",
    # Added 2026-08-11 (user-dictated): Detailed payroll-review data request for the
    # first 12 employees (May-26) + no-reply send, a Pay Elements bonus-row edit
    # open-and-close, Verify All on the new request, then a Jun-26 payrun pass of
    # 10 + 7 Save & Next clicks and an FPS submission. Declared as 10 verbatim
    # slices; user clarified via options: FOOD LIMITED opener, bonus edit closes
    # unchanged, 10-then-7 is one continuous 17-employee pass, FPS file is the
    # shared New_Employees_List_-_WI_LTD.csv. The Send-Email and popup-guard slices
    # reuse the e2e's proven wording verbatim.
    # Re-frozen 2026-08-24 (09d712c0d271b609 -> cd56d32ebde27ea3): grown to 13 slices —
    # the Verify-All slice was split into the Get OTP / external-tab OTP / per-employee
    # payment-expense-deduction / Verify all sequence, plus the weekend-notification
    # resend, "Bonus" capitalised to match the Pay Elements row, and the first-10 slice
    # picked up a double space. Split into part1-4 below.
    # Re-frozen again 2026-08-24 PM (cd56d32ebde27ea3 -> 10f7611a6d663bd9): the
    # per-employee portal slice was rewritten from "Click Next for rest of the employees"
    # into explicit iteration ("for each ... repeat ... until"). Run 20260824_165824 shows
    # why: the old wording is not a loop signature, so the slice resolved to a plain
    # action, the agent read it as ONE Next click, and it closed the tab on employee 2's
    # freshly loaded (empty) form. Kept byte-identical in the full task and part2.
    # Re-frozen 2026-08-25 (10f7611a6d663bd9 -> 8174b98b5f2bd4b7): TWO edits. The OTP
    # producer slice said "click on the number to copy it", which contains no noting verb
    # — decompose._PRODUCES_NOTED_RE matches note|remember|capture|write, never copy — so
    # produces_noted_data returned False, the capture-required commit gate never fired,
    # and a run that took the line literally would leave no `otp` in run_values for the
    # consumer to bind to. It says "note and remember the OTP" again. The consumer slice
    # carries the paste reword below (both copies edited byte-identically).
    "payroll_detailed_review_fps": "8174b98b5f2bd4b7",
    # The four independently runnable parts of payroll_detailed_review_fps (2026-08-24,
    # user-requested). Parts 2-4 open with a business-selection slice the full task does
    # not have, so their prompts — and ids — are their own; part1 is the full task's first
    # three slices verbatim. See test_detailed_review_fps_parts_partition_the_full_task.
    # part1/part2 re-frozen 2026-08-24 PM: part1's Data-Request slice and part2's OTP
    # slice were reworded while running the parts live, and part2 also carries the
    # portal-loop rewrite above.
    "payroll_detailed_review_fps_part1": "8be15ae6a6323200",
    # Re-frozen 2026-08-25 (a04ea4bbd9a0051b -> f3e746bf3633d102): the final slice
    # dropped "until every employee in that list has been done" for "one click each,
    # and do nothing else on any of them". "Done" implied a body and the agent acted on
    # it — run of 2026-08-24 repeated the PAYMENT for Aarmaan Aman, Aaron Wilson, Aayan
    # Dickson, Aden Graham. The slice steps through the list and nothing more (user,
    # 2026-08-25: "i just want it to click next and nothing else").
    # Re-frozen again 2026-08-25 PM (f3e746bf3633d102 -> 90ca6486695115e1): the OTP
    # consumer slice said "enter the OTP got it from the previous step". "Enter" is the
    # verb for the `input` tool, so the agent typed the six-box code as six 1-character
    # fills — and the whole code then appeared in NO step value, leaving the provenance
    # binder nothing to bind (hybrid refusal #5) and the segment re-authoring at ~107k
    # tokens every run. It now says "paste the OTP from the previous step into the first
    # code box", which paste_text delivers in one action as one bindable value. The
    # producer slice is deliberately untouched: it is the cached, replaying entry
    # 58896a82b8f2d4b7, and copy_text (index-addressed) could not reach an OTP whose
    # capture lands on a bare <form> anyway.
    # Re-frozen a third time 2026-08-25 PM (90ca6486695115e1 -> b4bb372f1bb700eb): the producer
    # slice moved to "copy the 6 digit number (OTP)" while running the new tools live.
    # That reword cost the cached producer 58896a82b8f2d4b7 its sid (each producer reword
    # re-authors at ~140k tokens), and it needed decompose._PRODUCES_NOTED_RE to learn the
    # `copy` verb or the capture-required commit guard would stay switched off for it.
    # Re-frozen 2026-08-25 evening (b4bb372f1bb700eb -> d1f40a8bf70f3b53): the last slice
    # was a LOOP ("keep clicking Next ... until the last employee"), and loop nodes are
    # never cached — hybrid skips their replay and refuses their commit — so it ran live
    # every run at 13 steps / 280s. Reworded to a fixed count the same way the RTI employee
    # passes were on 2026-08-11: "exactly 11 clicks" keeps node_kind=action and feeds
    # repeat_hint_from_wording, so it compiles to one repeat_click. 11 = the request's 12
    # employees minus the one already open.
    # Re-frozen again the same evening (d1f40a8bf70f3b53 -> 5e882a1cb7ace619): the user
    # added a Submit click to the tail of the final slice ("Then click submit, and then
    # close this tab"). Re-checked, not assumed: it stays kind=action and
    # repeat_hint_from_wording still reads 11.
    "payroll_detailed_review_fps_part2": "5e882a1cb7ace619",
    "payroll_detailed_review_fps_part3": "20bbee5816f20709",
    "payroll_detailed_review_fps_part4": "887ef76ffdfef2ae",
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


# payroll_food_limited_e2e_rti STARTED as a byte-exact join of payroll_food_limited_e2e +
# payroll_rti_process, and was pinned to stay one. That guard was dropped 2026-07-27: the
# merged prompt has since been tuned on its own (pay-forecast "clear the ... now set it",
# explicit Cost/Employee contribution fields, no repeated navigation), so it is the refined
# copy and the halves are the standalone ones. They now drift on purpose — when a shared
# step is reworded, update each copy that needs it deliberately.


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


def test_subtask_tab_url_parses_and_rejects_relative(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "aux_task:\n"
        "  subtasks:\n"
        "    - prompt: 'search DuckDuckGo for the thing'\n"
        "      tab_url: https://duckduckgo.com\n"
    )
    spec = load_tasks(p)["aux_task"]
    assert spec.subtasks[0].tab_url == "https://duckduckgo.com"

    p.write_text(
        "aux_task:\n"
        "  subtasks:\n"
        "    - prompt: 'search DuckDuckGo for the thing'\n"
        "      tab_url: duckduckgo.com\n"
    )
    with pytest.raises(ValueError, match="tab_url must be an absolute"):
        load_tasks(p)


def test_entry_fields_parse(tmp_path):
    # A prompt kept alongside subtasks must AGREE with their join (2026-08-05: the
    # slices are the source of truth; the prompt here pins block-scalar folding).
    p = tmp_path / "t.yaml"
    p.write_text(
        "My_Task:\n"
        "  prompt: >-\n"
        "    open settings\n"
        "    save it\n"
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
    assert spec.prompt == "open settings save it"  # block-scalar folding collapsed
    assert spec.marker == "Things" and spec.tags == ("a", "b")
    assert spec.subtasks == (
        SubtaskDecl(prompt="open {{page}}", values={"page": "settings"}),
        SubtaskDecl(prompt="save it", marker="Things"),
    )


async def test_e2e_rti_declared_subtasks_survive_validation(tmp_path, monkeypatch):
    """The declared split of the e2e_rti mega-task must pass Tier 1 validation and never
    degrade to the whole-prompt fallback (the LLM decomposer rejected its own splits 3x
    per run on 2026-08-05 and every run degenerated to one blob). The declared prompts
    are VERBATIM slices whose concatenation reproduces the task prompt exactly — the
    identity contract that keeps shared RTI-half sids aligned with payroll_rti_process."""
    from automation.pipeline import decompose

    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    spec = load_tasks()["payroll_food_limited_e2e_rti"]

    joined = " ".join(" ".join(d.prompt.split()) for d in spec.subtasks)
    assert joined == spec.prompt          # verbatim partition, no reword drift

    subs = await decompose.get_decomposition(spec.prompt, llm=None, spec=spec)
    # 2026-08-24: the task is its first SIX slices — the RTI half went live in 8393188 and
    # the user re-commented it, parking the Data Request + Send Email slices with it (that
    # flow now lives in payroll_detailed_review_fps and its parts). The fakenamegenerator
    # aux flow is the user's chosen shape, and every slice is a recordable action: the
    # forecast-load slice was flattened ("wait until the pay rows have loaded", no repeat
    # cue) and the Net-to-Gross slice lost its refresh-first recovery after the 45-step
    # seg-4 spiral.
    assert len(subs) == 6                 # Tier 1 won; no fallback blob
    assert not any(getattr(s, "fallback", False) for s in subs)
    assert [s.kind for s in subs] == ["action"] * 6
    assert subs[1].tab_url and "fakenamegenerator" in subs[1].tab_url   # aux producer
    # The producer-override keeps the noting slice OUT of the consumer net; the two
    # slices that USE the noted identity stay consumers (bindings-only replay).
    assert not decompose.consumes_noted_data(subs[1].template_prompt)
    assert decompose.consumes_noted_data(subs[2].template_prompt)
    assert decompose.consumes_noted_data(subs[3].template_prompt)
    assert "download" in subs[5].template_prompt.lower()              # download subtask


# --------------------- payroll_detailed_review_fps, split into parts ---------------------
# 2026-08-24 (user-requested): the 13-slice task also exists as four independently runnable
# parts. The full task stays the end-to-end vehicle, so shared slice wording now lives in
# two places and drifts only on purpose — same arrangement as the e2e halves above.
FPS_PART_SLICE_COUNTS = {
    "payroll_detailed_review_fps_part1": 3,   # request + send email (full task's 1-3)
    # The body is split four ways (payment / expense / deduction / step-through) rather
    # than the full task's single portal slice — the user's choice, 2026-08-25.
    "payroll_detailed_review_fps_part2": 8,   # opener + bonus + OTP + portal + body x3 + Next
    "payroll_detailed_review_fps_part3": 2,   # opener + Verify all (8)
    "payroll_detailed_review_fps_part4": 6,   # opener + payrun pass + FPS (9-13)
}
FPS_PART_OPENER = ("Go to the Payroll module, search for and select the business name "
                   "FOOD LIMITED")
# Node kinds the parts resolve to. part2's portal slice is a loop — three dialogs per
# employee across a 12-employee list, so it must act live and can never be cached (a
# replayed loop walks a fixed number of steps and lands anywhere). Its OTP slice is an
# ACTION despite "note and remember": producer wording is not verification, and a replayed
# extract re-reads the live OTP. All wording-driven; see decompose.node_kind.
FPS_PART_SLICE_KINDS = {
    "payroll_detailed_review_fps_part1": ["action"] * 3,
    # All ACTION since 2026-08-25: the final Next-through-employees slice was a loop, and
    # loop nodes are never cached (hybrid skips the replay and refuses the commit), so it
    # ran live every run. Reworded to "exactly 11 clicks" — the same fixed-count move the
    # RTI employee passes took on 2026-08-11 — which keeps it an action and lets it cache.
    # If this flips back to "loop", a reword reintroduced a repeat + stop cue.
    "payroll_detailed_review_fps_part2": ["action"] * 8,
    "payroll_detailed_review_fps_part3": ["action"] * 2,
    "payroll_detailed_review_fps_part4": ["action"] * 6,
}


def test_detailed_review_fps_parts_are_independently_runnable():
    """Each part re-states the business selection so it runs on its own, and its prompt is
    derived from its own slices (one edit surface, no parallel copy).

    This test used to additionally assert the parts rebuild the full task's slice list
    byte-for-byte. That premise is dead as of 2026-08-25 and deliberately so: part2 splits
    the portal body into payment / expense / deduction slices where the full task has one,
    and its final slice now says step through the remaining employees doing NOTHING else,
    while the full task's still says "repeat all of the above for them". The two copies have
    diverged in INTENT, which is the accepted cost of keeping both (see the tasks.yaml
    comment above the parts). Restoring the rebuild assertion would mean reverting a
    deliberate wording choice, so guard what is still true instead."""
    tasks = load_tasks()
    for key in FPS_PART_SLICE_COUNTS:
        spec = tasks[key]
        slices = [" ".join(d.prompt.split()) for d in spec.subtasks]
        assert len(slices) == FPS_PART_SLICE_COUNTS[key], key
        assert slices[0].startswith(FPS_PART_OPENER), key   # or it cannot run alone
        assert spec.tags == ("payroll",), key
        assert spec.prompt == " ".join(slices), key


@pytest.mark.parametrize("key", list(FPS_PART_SLICE_COUNTS))
async def test_detailed_review_fps_parts_survive_validation(key, tmp_path, monkeypatch):
    """Every part must pass Tier 1 on its own and never degrade to the whole-prompt
    fallback blob — the guarantee that makes the parts separately runnable."""
    from automation.pipeline import decompose

    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    spec = load_tasks()[key]

    subs = await decompose.get_decomposition(spec.prompt, llm=None, spec=spec)
    assert len(subs) == FPS_PART_SLICE_COUNTS[key]
    assert not any(getattr(s, "fallback", False) for s in subs)
    assert [s.kind for s in subs] == FPS_PART_SLICE_KINDS[key]


def test_prompt_derived_from_declared_subtasks(tmp_path):
    """A task with `subtasks:` needs no `prompt:` — the spec's prompt is the single-space
    join of the instantiated slices (tokens substituted). One edit surface: the user
    rewords a slice, never a parallel prompt copy."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "derived:\n"
        "  subtasks:\n"
        "    - prompt: 'open the {{page}} page.'\n"
        "      values: {page: settings}\n"
        "    - prompt: 'click save'\n"
    )
    spec = load_tasks(p)["derived"]
    assert spec.prompt == "open the settings page. click save"
    assert spec.subtasks and len(spec.subtasks) == 2


def test_prompt_and_subtasks_disagreement_fails_loud(tmp_path):
    """Keeping BOTH a prompt and subtasks is allowed only while they agree — a mismatch
    is silent drift waiting to poison identities, so the registry load refuses it."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "drifted:\n"
        "  prompt: open the settings page. click SUBMIT\n"
        "  subtasks:\n"
        "    - prompt: 'open the settings page.'\n"
        "    - prompt: 'click save'\n"
    )
    with pytest.raises(ValueError, match="disagree"):
        load_tasks(p)


# ------------------------------- declared verify checks -------------------------------


def test_subtask_verify_parsed_and_substituted():
    """A verify: block parses into Check tuples at load time, with {{tokens}}
    substituted from the slice's own values (load-time, because the downstream
    token grammar is lowercase-only)."""
    from automation.pipeline.checks import Check

    entry = {"subtasks": [
        {"prompt": "Add {{name}} then save.",
         "values": {"name": "Alistair Allan"},
         "verify": [{"text_visible": "{{name}}"},
                    {"write_accepted": "Employees", "timeout_s": 5}]},
    ]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].verify == (
        Check(kind="text_visible", arg="Alistair Allan"),
        Check(kind="write_accepted", arg="Employees", timeout_s=5.0),
    )


def test_subtask_without_verify_is_none():
    entry = {"subtasks": [{"prompt": "Open payroll."}]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].verify is None


def test_subtask_verify_invalid_fails_loud():
    entry = {"subtasks": [{"prompt": "Open payroll.",
                           "verify": [{"nope": "x"}]}]}
    with pytest.raises(ValueError, match="nope"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_verify_unresolved_token_fails_loud():
    """A verify arg has no downstream closure validation, so a leftover {{token}}
    must fail at load rather than probe for the literal braces at runtime."""
    entry = {"subtasks": [{"prompt": "Open payroll.",
                           "verify": [{"text_visible": "{{missing}}"}]}]}
    with pytest.raises(ValueError, match="missing"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_verify_does_not_change_identity():
    """Ids derive from slice WORDING only — adding verify: must never drift a
    task_id (that would orphan decomposition caches and FROZEN_TIDS)."""
    base = {"subtasks": [{"prompt": "Open payroll."}]}
    with_verify = {"subtasks": [{"prompt": "Open payroll.",
                                 "verify": [{"url_contains": "payroll"}]}]}
    a = tasks_mod._spec_from_entry("t", base)
    b = tasks_mod._spec_from_entry("t", with_verify)
    assert a.prompt == b.prompt
    assert ss.task_id(a.prompt) == ss.task_id(b.prompt)


# ------------------------------- declared conditional probe -------------------------------


def test_subtask_probe_parsed_and_substituted():
    """A probe: mapping parses into ONE Check at load time (short default poll — an
    absent popup is a routine outcome, not a failure to wait out), with {{tokens}}
    substituted from the slice's values like verify args."""
    from automation.pipeline.checks import _PROBE_TIMEOUT_S, Check

    entry = {"subtasks": [
        {"prompt": "If a {{thing}} popup appears, dismiss it.",
         "values": {"thing": "Process"},
         "probe": {"text_visible": "Don't show this {{thing}} again"}},
    ]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].probe == Check(
        kind="text_visible", arg="Don't show this Process again",
        timeout_s=_PROBE_TIMEOUT_S)


def test_subtask_without_probe_is_none():
    entry = {"subtasks": [{"prompt": "Open payroll."}]}
    assert tasks_mod._spec_from_entry("t", entry).subtasks[0].probe is None


def test_subtask_probe_list_form_fails_loud():
    entry = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                           "probe": [{"text_visible": "x"}]}]}
    with pytest.raises(ValueError, match="single check mapping"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_probe_unresolved_token_fails_loud():
    entry = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                           "probe": {"text_visible": "{{missing}}"}}]}
    with pytest.raises(ValueError, match="missing"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_probe_does_not_change_identity():
    base = {"subtasks": [{"prompt": "If a popup appears, dismiss it."}]}
    with_probe = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                                "probe": {"text_visible": "popup"}}]}
    a = tasks_mod._spec_from_entry("t", base)
    b = tasks_mod._spec_from_entry("t", with_probe)
    assert ss.task_id(a.prompt) == ss.task_id(b.prompt)
