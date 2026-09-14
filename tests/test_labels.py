from note_deid.labels import (
    HIPAA_SUBTYPES,
    I2B2_2014_SUBTYPES,
    I2B2_TO_OPF,
    OPENMED_TO_I2B2,
    OPF_LABELS,
    OPF_UNSUPPORTED_SUBTYPES,
    SUBTYPE_TO_CATEGORY,
    bioes_tags,
    frequency_bucket,
    map_label,
    normalize_label,
)


def test_taxonomy_shape():
    assert len(I2B2_2014_SUBTYPES) == 28
    assert len(set(I2B2_2014_SUBTYPES)) == 28
    assert set(SUBTYPE_TO_CATEGORY) == set(I2B2_2014_SUBTYPES)
    assert HIPAA_SUBTYPES <= set(I2B2_2014_SUBTYPES)


def test_opf_mapping_is_total_over_subtypes():
    assert set(I2B2_TO_OPF) == set(I2B2_2014_SUBTYPES)
    assert all(v is None or v in OPF_LABELS for v in I2B2_TO_OPF.values())
    assert {"AGE", "PROFESSION", "HOSPITAL", "ORGANIZATION"} <= OPF_UNSUPPORTED_SUBTYPES
    assert len(bioes_tags(OPF_LABELS)) == 33  # 1 + 8 * 4, as in the OPF classification head


def test_openmed_mapping_targets_are_canonical():
    assert all(v is None or v in I2B2_2014_SUBTYPES for v in OPENMED_TO_I2B2.values())
    assert normalize_label("B-Medical_Record-Number") == "medicalrecordnumber"
    assert map_label("I-medical_record_number", OPENMED_TO_I2B2) == "MEDICALRECORD"
    assert map_label("blood_type", OPENMED_TO_I2B2) is None
    assert map_label("something_new", OPENMED_TO_I2B2) is None


def test_frequency_buckets():
    assert [frequency_bucket(n) for n in (0, 1, 10, 11, 50, 51, 200, 201, 5000)] == [
        "0",
        "1-10",
        "1-10",
        "11-50",
        "11-50",
        "51-200",
        "51-200",
        ">200",
        ">200",
    ]
