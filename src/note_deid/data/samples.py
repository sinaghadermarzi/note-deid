"""Built-in PHI-free demo notes so the injection pipeline and tests run without any download."""

from __future__ import annotations

from note_deid.schema import Doc

_NOTES: list[tuple[str, str, str]] = [
    (
        "demo-001",
        "discharge_summary",
        "ADMISSION DIAGNOSIS: Community-acquired pneumonia.\n\n"
        "HISTORY OF PRESENT ILLNESS: The patient is a 67-year-old with a three-day history of productive cough, "
        "fever to 38.9, and pleuritic chest pain. The patient was seen in the emergency department of the hospital and "
        "admitted for intravenous antibiotics. Past history is notable for hypertension and type 2 diabetes.\n\n"
        "HOSPITAL COURSE: The patient was started on ceftriaxone and azithromycin with improvement in oxygen "
        "requirement by hospital day two. Blood cultures remained negative. The patient ambulated independently and "
        "tolerated a regular diet.\n\n"
        "DISCHARGE PLAN: Complete a five-day course of oral antibiotics. Follow up with the primary care physician in "
        "one week. Return to the hospital for worsening shortness of breath.",
    ),
    (
        "demo-002",
        "progress_note",
        "SUBJECTIVE: The patient reports improved knee pain after physical therapy. Sleep is adequate. No fevers or "
        "chills.\n\n"
        "OBJECTIVE: Vitals stable. Right knee with mild effusion, full range of motion, no erythema.\n\n"
        "ASSESSMENT AND PLAN: Osteoarthritis of the right knee, improving. Continue home exercise program. The patient "
        "will return to the clinic in six weeks. Discussed weight management and the patient agrees with the plan.",
    ),
    (
        "demo-003",
        "consult_note",
        "REASON FOR CONSULTATION: Evaluation of new-onset atrial fibrillation.\n\n"
        "HISTORY: The patient is a 54-year-old who presented to the hospital with palpitations lasting six hours. No "
        "chest pain. The patient denies alcohol excess. Family history is significant for coronary disease in a "
        "parent.\n\n"
        "EXAMINATION: Irregularly irregular rhythm at 118 beats per minute. Lungs clear. No peripheral edema.\n\n"
        "RECOMMENDATIONS: Rate control with metoprolol. Anticoagulation after shared decision making. Echocardiogram "
        "before discharge. The patient understands the plan and the physician answered all questions.",
    ),
    (
        "demo-004",
        "operative_report",
        "PREOPERATIVE DIAGNOSIS: Symptomatic cholelithiasis.\n\nPOSTOPERATIVE DIAGNOSIS: Same.\n\n"
        "PROCEDURE: Laparoscopic cholecystectomy.\n\n"
        "DESCRIPTION OF PROCEDURE: The patient was brought to the operating room and placed supine. After induction "
        "of general anesthesia, the abdomen was prepped and draped. Pneumoperitoneum was established. The gallbladder "
        "was dissected from the liver bed and removed through the umbilical port. Hemostasis was confirmed. "
        "The patient tolerated the procedure well and was taken to the recovery room in stable condition.",
    ),
    (
        "demo-005",
        "radiology_report",
        "EXAMINATION: Chest radiograph, two views.\n\nINDICATION: Cough and fever.\n\n"
        "FINDINGS: There is a focal opacity in the right lower lobe. No pleural effusion or pneumothorax. Heart size "
        "is normal. Osseous structures are unremarkable.\n\n"
        "IMPRESSION: Right lower lobe opacity, likely pneumonia. Recommend follow-up radiograph in six to eight weeks "
        "to document resolution. Results were discussed with the referring physician.",
    ),
    (
        "demo-006",
        "emergency_note",
        "CHIEF COMPLAINT: Laceration of the left hand.\n\n"
        "HISTORY OF PRESENT ILLNESS: The patient is a 32-year-old who sustained a laceration while working with a "
        "kitchen knife approximately two hours ago. Bleeding controlled with pressure. Tetanus status is up to "
        "date.\n\n"
        "PROCEDURE: The wound was irrigated and closed with five simple interrupted sutures under local "
        "anesthesia.\n\n"
        "DISPOSITION: Discharged home with wound care instructions. The patient will return to the clinic for suture "
        "removal in ten days or sooner for signs of infection.",
    ),
]


def demo_docs() -> list[Doc]:
    return [Doc(i, text, [], {"source": "demo", "note_type": nt}) for i, nt, text in _NOTES]
