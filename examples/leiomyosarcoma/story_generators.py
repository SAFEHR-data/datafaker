import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Rows are keyed by days_after_diagnosis relative to this anchor, since the
# CSV has no absolute dates.
_DIAGNOSIS_ANCHOR_DATE = date.today()

# OMOP standard concept for "EHR problem list entry"; the CSV has no
# per-row type information for conditions, so every condition_occurrence
# row generated from it uses this fixed value.
_CONDITION_TYPE_CONCEPT_ID = 32840
_PROCEDURE_TYPE_CONCEPT_ID = 32817

# The CSV has no per-row type information for visits either, so every
# visit_occurrence row generated from it uses this fixed value.
_VISIT_TYPE_CONCEPT_ID = 32817

_CSV_PATH = Path(__file__).parent / "leiomyosarcoma_patient.csv"


def _load_patient_events() -> dict[str, list[dict[str, str]]]:
    """Group the CSV's rows by their (source) person_id."""
    events_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    with _CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            events_by_patient[row["person_id"]].append(row)
    return events_by_patient


def _event_date(event: dict[str, str]) -> date:
    days = event["days_after_diagnosis"]
    offset = int(float(days)) if days else 0
    return _DIAGNOSIS_ANCHOR_DATE + timedelta(days=offset)


# One patient's events are consumed per patient_story() call, so
# num_stories_per_pass in config.yaml must not exceed len(_PATIENT_EVENTS).
_PATIENT_EVENTS = _load_patient_events()
_PATIENT_IDS = iter(_PATIENT_EVENTS)


class Row:
    """Mixin giving the table dataclasses below a dict form for yielding.

    Only fields that have been explicitly set are included, so unset
    fields fall back to whatever the table's row generator produces.
    """

    def to_row(self) -> dict[str, Any]:
        return {
            f.name: value
            for f in fields(self)  # type: ignore[arg-type]
            if (value := getattr(self, f.name)) is not None
        }

@dataclass
class ConditionOccurrence(Row):
    condition_occurrence_id: Optional[int] = None
    condition_concept_id: Optional[int] = None
    condition_start_date: Optional[date] = None
    condition_start_datetime: Optional[datetime] = None
    condition_type_concept_id: Optional[int] = None
    person_id: Optional[int] = None
    visit_occurrence_id: Optional[int] = None


@dataclass
class Person(Row):
    person_id: Optional[int] = None
    ethnicity_concept_id: Optional[int] = None
    gender_concept_id: Optional[int] = None
    race_concept_id: Optional[int] = None
    year_of_birth: Optional[int] = None


@dataclass
class ProcedureOccurrence(Row):
    procedure_occurrence_id: Optional[int] = None
    modifier_concept_id: Optional[int] = None
    modifier_source_value: Optional[str] = None
    person_id: Optional[int] = None
    procedure_concept_id: Optional[int] = None
    procedure_date: Optional[date] = None
    procedure_datetime: Optional[datetime] = None
    # procedure_source_concept_id: Optional[int] = None
    procedure_source_value: Optional[str] = None
    procedure_type_concept_id: Optional[int] = None
    quantity: Optional[int] = None
    visit_occurrence_id: Optional[int] = None


@dataclass
class VisitOccurrence(Row):
    visit_occurrence_id: Optional[int] = None
    person_id: Optional[int] = None
    visit_concept_id: Optional[int] = None
    visit_end_date: Optional[date] = None
    visit_start_date: Optional[date] = None
    visit_start_datetime: Optional[datetime] = None
    visit_type_concept_id: Optional[int] = None


def patient_story():
    """Yield all the data related to a single patient.

    This includes, in order
    * a row for the `person` table
    * possibly a row for `death`, if the patient has died
    * rows for `visit_occurence` and `observation_period`
    * possibly multiple rows, depending on the length of the hospital stay, for
        * `condition_occurrence`
        * `measurement`
        * `device_exposure`
        * `observation`
        * `procedure_occurrence`
        * `specimen`
        * `drug_exposure`
    """
    events = _PATIENT_EVENTS[next(_PATIENT_IDS)]
    conditions = [e for e in events if e["event_type"] == "Condition"]
    visits = [e for e in events if e["event_type"] == "Visit"]
    procedures = [e for e in events if e["event_type"] == "Procedure"]

    person = Person(year_of_birth=1960,
                    gender_concept_id=8507,
                    race_concept_id=8515,
                    ethnicity_concept_id=0)
    person = yield "person", person.to_row()
    logger.info("Generated person row: %s", person)

    for event in conditions:
        condition = ConditionOccurrence(
            person_id=person["person_id"],
            condition_concept_id=int(event["concept_id"]),
            condition_type_concept_id=_CONDITION_TYPE_CONCEPT_ID,
            condition_start_date=_event_date(event),
            visit_occurrence_id=None
        )
        yield "condition_occurrence", condition.to_row()
        logger.info(
            "Generated condition_occurrence row for condition %s",
            event["event_name"],
        )

    for event in visits:
        visit_date = _event_date(event)
        visit = VisitOccurrence(
            person_id=person["person_id"],
            visit_concept_id=int(event["concept_id"]),
            visit_type_concept_id=_VISIT_TYPE_CONCEPT_ID,
            visit_start_date=visit_date,
            visit_end_date=visit_date,
        )
        visit = yield "visit_occurrence", visit.to_row()
        logger.info(
            "Generated visit_occurrence row at days %s",
            event["days_after_diagnosis"],
        )

        same_day_procedures = [
            p for p in procedures
            if p["days_after_diagnosis"] == event["days_after_diagnosis"]
        ]
        for proc_event in same_day_procedures:
            procedure = ProcedureOccurrence(
                person_id=person["person_id"],
                procedure_concept_id=int(proc_event["concept_id"]),
                procedure_date=_event_date(proc_event),
                # procedure_source_concept_id=None,
                visit_occurrence_id=visit["visit_occurrence_id"],
                procedure_type_concept_id=_PROCEDURE_TYPE_CONCEPT_ID
            )
            yield "procedure_occurrence", procedure.to_row()
            logger.info(
                "Generated procedure_occurrence row for procedure %s",
                proc_event["event_name"],
            )