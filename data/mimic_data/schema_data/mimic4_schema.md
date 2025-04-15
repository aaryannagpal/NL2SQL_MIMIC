# SQLite Database Schema Report

**Generated**: 2025-04-15 16:46:27
**Database**: `data/mimic_data/mimic4.db`

---

## Table: `admissions`

### Columns

|   cid | name                 | type    |   notnull | dflt_value   |   pk |
|-------|----------------------|---------|-----------|--------------|------|
|     0 | index                | INTEGER |         0 |              |    0 |
|     1 | subject_id           | INTEGER |         0 |              |    0 |
|     2 | hadm_id              | INTEGER |         0 |              |    0 |
|     3 | admittime            | TEXT    |         0 |              |    0 |
|     4 | dischtime            | TEXT    |         0 |              |    0 |
|     5 | deathtime            | TEXT    |         0 |              |    0 |
|     6 | admission_type       | TEXT    |         0 |              |    0 |
|     7 | admit_provider_id    | TEXT    |         0 |              |    0 |
|     8 | admission_location   | TEXT    |         0 |              |    0 |
|     9 | discharge_location   | TEXT    |         0 |              |    0 |
|    10 | insurance            | TEXT    |         0 |              |    0 |
|    11 | language             | TEXT    |         0 |              |    0 |
|    12 | marital_status       | TEXT    |         0 |              |    0 |
|    13 | race                 | TEXT    |         0 |              |    0 |
|    14 | edregtime            | TEXT    |         0 |              |    0 |
|    15 | edouttime            | TEXT    |         0 |              |    0 |
|    16 | hospital_expire_flag | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "admissions" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "admittime" TEXT,
  "dischtime" TEXT,
  "deathtime" TEXT,
  "admission_type" TEXT,
  "admit_provider_id" TEXT,
  "admission_location" TEXT,
  "discharge_location" TEXT,
  "insurance" TEXT,
  "language" TEXT,
  "marital_status" TEXT,
  "race" TEXT,
  "edregtime" TEXT,
  "edouttime" TEXT,
  "hospital_expire_flag" INTEGER
)
```

---

## Table: `caregiver`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | caregiver_id | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "caregiver" (
"index" INTEGER,
  "caregiver_id" INTEGER
)
```

---

## Table: `chartevents`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | subject_id   | INTEGER |         0 |              |    0 |
|     2 | hadm_id      | INTEGER |         0 |              |    0 |
|     3 | stay_id      | INTEGER |         0 |              |    0 |
|     4 | caregiver_id | REAL    |         0 |              |    0 |
|     5 | charttime    | TEXT    |         0 |              |    0 |
|     6 | storetime    | TEXT    |         0 |              |    0 |
|     7 | itemid       | INTEGER |         0 |              |    0 |
|     8 | value        | TEXT    |         0 |              |    0 |
|     9 | valuenum     | REAL    |         0 |              |    0 |
|    10 | valueuom     | TEXT    |         0 |              |    0 |
|    11 | warning      | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "chartevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" REAL,
  "charttime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "value" TEXT,
  "valuenum" REAL,
  "valueuom" TEXT,
  "warning" REAL
)
```

---

## Table: `d_hcpcs`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | code              | TEXT    |         0 |              |    0 |
|     2 | category          | REAL    |         0 |              |    0 |
|     3 | long_description  | TEXT    |         0 |              |    0 |
|     4 | short_description | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "d_hcpcs" (
"index" INTEGER,
  "code" TEXT,
  "category" REAL,
  "long_description" TEXT,
  "short_description" TEXT
)
```

---

## Table: `d_icd_diagnoses`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | icd_code    | TEXT    |         0 |              |    0 |
|     2 | icd_version | INTEGER |         0 |              |    0 |
|     3 | long_title  | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "d_icd_diagnoses" (
"index" INTEGER,
  "icd_code" TEXT,
  "icd_version" INTEGER,
  "long_title" TEXT
)
```

---

## Table: `d_icd_procedures`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | icd_code    | TEXT    |         0 |              |    0 |
|     2 | icd_version | INTEGER |         0 |              |    0 |
|     3 | long_title  | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "d_icd_procedures" (
"index" INTEGER,
  "icd_code" TEXT,
  "icd_version" INTEGER,
  "long_title" TEXT
)
```

---

## Table: `d_items`

### Columns

|   cid | name            | type    |   notnull | dflt_value   |   pk |
|-------|-----------------|---------|-----------|--------------|------|
|     0 | index           | INTEGER |         0 |              |    0 |
|     1 | itemid          | INTEGER |         0 |              |    0 |
|     2 | label           | TEXT    |         0 |              |    0 |
|     3 | abbreviation    | TEXT    |         0 |              |    0 |
|     4 | linksto         | TEXT    |         0 |              |    0 |
|     5 | category        | TEXT    |         0 |              |    0 |
|     6 | unitname        | TEXT    |         0 |              |    0 |
|     7 | param_type      | TEXT    |         0 |              |    0 |
|     8 | lownormalvalue  | REAL    |         0 |              |    0 |
|     9 | highnormalvalue | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "d_items" (
"index" INTEGER,
  "itemid" INTEGER,
  "label" TEXT,
  "abbreviation" TEXT,
  "linksto" TEXT,
  "category" TEXT,
  "unitname" TEXT,
  "param_type" TEXT,
  "lownormalvalue" REAL,
  "highnormalvalue" REAL
)
```

---

## Table: `d_labitems`

### Columns

|   cid | name     | type    |   notnull | dflt_value   |   pk |
|-------|----------|---------|-----------|--------------|------|
|     0 | index    | INTEGER |         0 |              |    0 |
|     1 | itemid   | INTEGER |         0 |              |    0 |
|     2 | label    | TEXT    |         0 |              |    0 |
|     3 | fluid    | TEXT    |         0 |              |    0 |
|     4 | category | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "d_labitems" (
"index" INTEGER,
  "itemid" INTEGER,
  "label" TEXT,
  "fluid" TEXT,
  "category" TEXT
)
```

---

## Table: `datetimeevents`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | subject_id   | INTEGER |         0 |              |    0 |
|     2 | hadm_id      | INTEGER |         0 |              |    0 |
|     3 | stay_id      | INTEGER |         0 |              |    0 |
|     4 | caregiver_id | INTEGER |         0 |              |    0 |
|     5 | charttime    | TEXT    |         0 |              |    0 |
|     6 | storetime    | TEXT    |         0 |              |    0 |
|     7 | itemid       | INTEGER |         0 |              |    0 |
|     8 | value        | TEXT    |         0 |              |    0 |
|     9 | valueuom     | TEXT    |         0 |              |    0 |
|    10 | warning      | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "datetimeevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" INTEGER,
  "charttime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "value" TEXT,
  "valueuom" TEXT,
  "warning" INTEGER
)
```

---

## Table: `diagnoses_icd`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | subject_id  | INTEGER |         0 |              |    0 |
|     2 | hadm_id     | INTEGER |         0 |              |    0 |
|     3 | seq_num     | INTEGER |         0 |              |    0 |
|     4 | icd_code    | TEXT    |         0 |              |    0 |
|     5 | icd_version | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "diagnoses_icd" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "seq_num" INTEGER,
  "icd_code" TEXT,
  "icd_version" INTEGER
)
```

---

## Table: `drgcodes`

### Columns

|   cid | name          | type    |   notnull | dflt_value   |   pk |
|-------|---------------|---------|-----------|--------------|------|
|     0 | index         | INTEGER |         0 |              |    0 |
|     1 | subject_id    | INTEGER |         0 |              |    0 |
|     2 | hadm_id       | INTEGER |         0 |              |    0 |
|     3 | drg_type      | TEXT    |         0 |              |    0 |
|     4 | drg_code      | INTEGER |         0 |              |    0 |
|     5 | description   | TEXT    |         0 |              |    0 |
|     6 | drg_severity  | REAL    |         0 |              |    0 |
|     7 | drg_mortality | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "drgcodes" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "drg_type" TEXT,
  "drg_code" INTEGER,
  "description" TEXT,
  "drg_severity" REAL,
  "drg_mortality" REAL
)
```

---

## Table: `emar`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | hadm_id           | REAL    |         0 |              |    0 |
|     3 | emar_id           | TEXT    |         0 |              |    0 |
|     4 | emar_seq          | INTEGER |         0 |              |    0 |
|     5 | poe_id            | TEXT    |         0 |              |    0 |
|     6 | pharmacy_id       | REAL    |         0 |              |    0 |
|     7 | enter_provider_id | TEXT    |         0 |              |    0 |
|     8 | charttime         | TEXT    |         0 |              |    0 |
|     9 | medication        | TEXT    |         0 |              |    0 |
|    10 | event_txt         | TEXT    |         0 |              |    0 |
|    11 | scheduletime      | TEXT    |         0 |              |    0 |
|    12 | storetime         | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "emar" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" REAL,
  "emar_id" TEXT,
  "emar_seq" INTEGER,
  "poe_id" TEXT,
  "pharmacy_id" REAL,
  "enter_provider_id" TEXT,
  "charttime" TEXT,
  "medication" TEXT,
  "event_txt" TEXT,
  "scheduletime" TEXT,
  "storetime" TEXT
)
```

---

## Table: `emar_detail`

### Columns

|   cid | name                                 | type    |   notnull | dflt_value   |   pk |
|-------|--------------------------------------|---------|-----------|--------------|------|
|     0 | index                                | INTEGER |         0 |              |    0 |
|     1 | subject_id                           | INTEGER |         0 |              |    0 |
|     2 | emar_id                              | TEXT    |         0 |              |    0 |
|     3 | emar_seq                             | INTEGER |         0 |              |    0 |
|     4 | parent_field_ordinal                 | REAL    |         0 |              |    0 |
|     5 | administration_type                  | TEXT    |         0 |              |    0 |
|     6 | pharmacy_id                          | REAL    |         0 |              |    0 |
|     7 | barcode_type                         | TEXT    |         0 |              |    0 |
|     8 | reason_for_no_barcode                | TEXT    |         0 |              |    0 |
|     9 | complete_dose_not_given              | TEXT    |         0 |              |    0 |
|    10 | dose_due                             | TEXT    |         0 |              |    0 |
|    11 | dose_due_unit                        | TEXT    |         0 |              |    0 |
|    12 | dose_given                           | TEXT    |         0 |              |    0 |
|    13 | dose_given_unit                      | TEXT    |         0 |              |    0 |
|    14 | will_remainder_of_dose_be_given      | TEXT    |         0 |              |    0 |
|    15 | product_amount_given                 | TEXT    |         0 |              |    0 |
|    16 | product_unit                         | TEXT    |         0 |              |    0 |
|    17 | product_code                         | TEXT    |         0 |              |    0 |
|    18 | product_description                  | TEXT    |         0 |              |    0 |
|    19 | product_description_other            | TEXT    |         0 |              |    0 |
|    20 | prior_infusion_rate                  | REAL    |         0 |              |    0 |
|    21 | infusion_rate                        | REAL    |         0 |              |    0 |
|    22 | infusion_rate_adjustment             | TEXT    |         0 |              |    0 |
|    23 | infusion_rate_adjustment_amount      | REAL    |         0 |              |    0 |
|    24 | infusion_rate_unit                   | TEXT    |         0 |              |    0 |
|    25 | route                                | TEXT    |         0 |              |    0 |
|    26 | infusion_complete                    | TEXT    |         0 |              |    0 |
|    27 | completion_interval                  | TEXT    |         0 |              |    0 |
|    28 | new_iv_bag_hung                      | TEXT    |         0 |              |    0 |
|    29 | continued_infusion_in_other_location | TEXT    |         0 |              |    0 |
|    30 | restart_interval                     | TEXT    |         0 |              |    0 |
|    31 | side                                 | TEXT    |         0 |              |    0 |
|    32 | site                                 | TEXT    |         0 |              |    0 |
|    33 | non_formulary_visual_verification    | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "emar_detail" (
"index" INTEGER,
  "subject_id" INTEGER,
  "emar_id" TEXT,
  "emar_seq" INTEGER,
  "parent_field_ordinal" REAL,
  "administration_type" TEXT,
  "pharmacy_id" REAL,
  "barcode_type" TEXT,
  "reason_for_no_barcode" TEXT,
  "complete_dose_not_given" TEXT,
  "dose_due" TEXT,
  "dose_due_unit" TEXT,
  "dose_given" TEXT,
  "dose_given_unit" TEXT,
  "will_remainder_of_dose_be_given" TEXT,
  "product_amount_given" TEXT,
  "product_unit" TEXT,
  "product_code" TEXT,
  "product_description" TEXT,
  "product_description_other" TEXT,
  "prior_infusion_rate" REAL,
  "infusion_rate" REAL,
  "infusion_rate_adjustment" TEXT,
  "infusion_rate_adjustment_amount" REAL,
  "infusion_rate_unit" TEXT,
  "route" TEXT,
  "infusion_complete" TEXT,
  "completion_interval" TEXT,
  "new_iv_bag_hung" TEXT,
  "continued_infusion_in_other_location" TEXT,
  "restart_interval" TEXT,
  "side" TEXT,
  "site" TEXT,
  "non_formulary_visual_verification" TEXT
)
```

---

## Table: `hcpcsevents`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | hadm_id           | INTEGER |         0 |              |    0 |
|     3 | chartdate         | TEXT    |         0 |              |    0 |
|     4 | hcpcs_cd          | TEXT    |         0 |              |    0 |
|     5 | seq_num           | INTEGER |         0 |              |    0 |
|     6 | short_description | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "hcpcsevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "chartdate" TEXT,
  "hcpcs_cd" TEXT,
  "seq_num" INTEGER,
  "short_description" TEXT
)
```

---

## Table: `icustays`

### Columns

|   cid | name           | type    |   notnull | dflt_value   |   pk |
|-------|----------------|---------|-----------|--------------|------|
|     0 | index          | INTEGER |         0 |              |    0 |
|     1 | subject_id     | INTEGER |         0 |              |    0 |
|     2 | hadm_id        | INTEGER |         0 |              |    0 |
|     3 | stay_id        | INTEGER |         0 |              |    0 |
|     4 | first_careunit | TEXT    |         0 |              |    0 |
|     5 | last_careunit  | TEXT    |         0 |              |    0 |
|     6 | intime         | TEXT    |         0 |              |    0 |
|     7 | outtime        | TEXT    |         0 |              |    0 |
|     8 | los            | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "icustays" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "first_careunit" TEXT,
  "last_careunit" TEXT,
  "intime" TEXT,
  "outtime" TEXT,
  "los" REAL
)
```

---

## Table: `ingredientevents`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | hadm_id           | INTEGER |         0 |              |    0 |
|     3 | stay_id           | INTEGER |         0 |              |    0 |
|     4 | caregiver_id      | INTEGER |         0 |              |    0 |
|     5 | starttime         | TEXT    |         0 |              |    0 |
|     6 | endtime           | TEXT    |         0 |              |    0 |
|     7 | storetime         | TEXT    |         0 |              |    0 |
|     8 | itemid            | INTEGER |         0 |              |    0 |
|     9 | amount            | REAL    |         0 |              |    0 |
|    10 | amountuom         | TEXT    |         0 |              |    0 |
|    11 | rate              | REAL    |         0 |              |    0 |
|    12 | rateuom           | TEXT    |         0 |              |    0 |
|    13 | orderid           | INTEGER |         0 |              |    0 |
|    14 | linkorderid       | INTEGER |         0 |              |    0 |
|    15 | statusdescription | TEXT    |         0 |              |    0 |
|    16 | originalamount    | INTEGER |         0 |              |    0 |
|    17 | originalrate      | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "ingredientevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" INTEGER,
  "starttime" TEXT,
  "endtime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "amount" REAL,
  "amountuom" TEXT,
  "rate" REAL,
  "rateuom" TEXT,
  "orderid" INTEGER,
  "linkorderid" INTEGER,
  "statusdescription" TEXT,
  "originalamount" INTEGER,
  "originalrate" REAL
)
```

---

## Table: `inputevents`

### Columns

|   cid | name                          | type    |   notnull | dflt_value   |   pk |
|-------|-------------------------------|---------|-----------|--------------|------|
|     0 | index                         | INTEGER |         0 |              |    0 |
|     1 | subject_id                    | INTEGER |         0 |              |    0 |
|     2 | hadm_id                       | INTEGER |         0 |              |    0 |
|     3 | stay_id                       | INTEGER |         0 |              |    0 |
|     4 | caregiver_id                  | INTEGER |         0 |              |    0 |
|     5 | starttime                     | TEXT    |         0 |              |    0 |
|     6 | endtime                       | TEXT    |         0 |              |    0 |
|     7 | storetime                     | TEXT    |         0 |              |    0 |
|     8 | itemid                        | INTEGER |         0 |              |    0 |
|     9 | amount                        | REAL    |         0 |              |    0 |
|    10 | amountuom                     | TEXT    |         0 |              |    0 |
|    11 | rate                          | REAL    |         0 |              |    0 |
|    12 | rateuom                       | TEXT    |         0 |              |    0 |
|    13 | orderid                       | INTEGER |         0 |              |    0 |
|    14 | linkorderid                   | INTEGER |         0 |              |    0 |
|    15 | ordercategoryname             | TEXT    |         0 |              |    0 |
|    16 | secondaryordercategoryname    | TEXT    |         0 |              |    0 |
|    17 | ordercomponenttypedescription | TEXT    |         0 |              |    0 |
|    18 | ordercategorydescription      | TEXT    |         0 |              |    0 |
|    19 | patientweight                 | REAL    |         0 |              |    0 |
|    20 | totalamount                   | REAL    |         0 |              |    0 |
|    21 | totalamountuom                | TEXT    |         0 |              |    0 |
|    22 | isopenbag                     | INTEGER |         0 |              |    0 |
|    23 | continueinnextdept            | INTEGER |         0 |              |    0 |
|    24 | statusdescription             | TEXT    |         0 |              |    0 |
|    25 | originalamount                | REAL    |         0 |              |    0 |
|    26 | originalrate                  | REAL    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "inputevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" INTEGER,
  "starttime" TEXT,
  "endtime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "amount" REAL,
  "amountuom" TEXT,
  "rate" REAL,
  "rateuom" TEXT,
  "orderid" INTEGER,
  "linkorderid" INTEGER,
  "ordercategoryname" TEXT,
  "secondaryordercategoryname" TEXT,
  "ordercomponenttypedescription" TEXT,
  "ordercategorydescription" TEXT,
  "patientweight" REAL,
  "totalamount" REAL,
  "totalamountuom" TEXT,
  "isopenbag" INTEGER,
  "continueinnextdept" INTEGER,
  "statusdescription" TEXT,
  "originalamount" REAL,
  "originalrate" REAL
)
```

---

## Table: `labevents`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | labevent_id       | INTEGER |         0 |              |    0 |
|     2 | subject_id        | INTEGER |         0 |              |    0 |
|     3 | hadm_id           | REAL    |         0 |              |    0 |
|     4 | specimen_id       | INTEGER |         0 |              |    0 |
|     5 | itemid            | INTEGER |         0 |              |    0 |
|     6 | order_provider_id | TEXT    |         0 |              |    0 |
|     7 | charttime         | TEXT    |         0 |              |    0 |
|     8 | storetime         | TEXT    |         0 |              |    0 |
|     9 | value             | TEXT    |         0 |              |    0 |
|    10 | valuenum          | REAL    |         0 |              |    0 |
|    11 | valueuom          | TEXT    |         0 |              |    0 |
|    12 | ref_range_lower   | REAL    |         0 |              |    0 |
|    13 | ref_range_upper   | REAL    |         0 |              |    0 |
|    14 | flag              | TEXT    |         0 |              |    0 |
|    15 | priority          | TEXT    |         0 |              |    0 |
|    16 | comments          | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "labevents" (
"index" INTEGER,
  "labevent_id" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" REAL,
  "specimen_id" INTEGER,
  "itemid" INTEGER,
  "order_provider_id" TEXT,
  "charttime" TEXT,
  "storetime" TEXT,
  "value" TEXT,
  "valuenum" REAL,
  "valueuom" TEXT,
  "ref_range_lower" REAL,
  "ref_range_upper" REAL,
  "flag" TEXT,
  "priority" TEXT,
  "comments" TEXT
)
```

---

## Table: `microbiologyevents`

### Columns

|   cid | name                | type    |   notnull | dflt_value   |   pk |
|-------|---------------------|---------|-----------|--------------|------|
|     0 | index               | INTEGER |         0 |              |    0 |
|     1 | microevent_id       | INTEGER |         0 |              |    0 |
|     2 | subject_id          | INTEGER |         0 |              |    0 |
|     3 | hadm_id             | REAL    |         0 |              |    0 |
|     4 | micro_specimen_id   | INTEGER |         0 |              |    0 |
|     5 | order_provider_id   | TEXT    |         0 |              |    0 |
|     6 | chartdate           | TEXT    |         0 |              |    0 |
|     7 | charttime           | TEXT    |         0 |              |    0 |
|     8 | spec_itemid         | INTEGER |         0 |              |    0 |
|     9 | spec_type_desc      | TEXT    |         0 |              |    0 |
|    10 | test_seq            | INTEGER |         0 |              |    0 |
|    11 | storedate           | TEXT    |         0 |              |    0 |
|    12 | storetime           | TEXT    |         0 |              |    0 |
|    13 | test_itemid         | INTEGER |         0 |              |    0 |
|    14 | test_name           | TEXT    |         0 |              |    0 |
|    15 | org_itemid          | REAL    |         0 |              |    0 |
|    16 | org_name            | TEXT    |         0 |              |    0 |
|    17 | isolate_num         | REAL    |         0 |              |    0 |
|    18 | quantity            | TEXT    |         0 |              |    0 |
|    19 | ab_itemid           | REAL    |         0 |              |    0 |
|    20 | ab_name             | TEXT    |         0 |              |    0 |
|    21 | dilution_text       | TEXT    |         0 |              |    0 |
|    22 | dilution_comparison | TEXT    |         0 |              |    0 |
|    23 | dilution_value      | REAL    |         0 |              |    0 |
|    24 | interpretation      | TEXT    |         0 |              |    0 |
|    25 | comments            | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "microbiologyevents" (
"index" INTEGER,
  "microevent_id" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" REAL,
  "micro_specimen_id" INTEGER,
  "order_provider_id" TEXT,
  "chartdate" TEXT,
  "charttime" TEXT,
  "spec_itemid" INTEGER,
  "spec_type_desc" TEXT,
  "test_seq" INTEGER,
  "storedate" TEXT,
  "storetime" TEXT,
  "test_itemid" INTEGER,
  "test_name" TEXT,
  "org_itemid" REAL,
  "org_name" TEXT,
  "isolate_num" REAL,
  "quantity" TEXT,
  "ab_itemid" REAL,
  "ab_name" TEXT,
  "dilution_text" TEXT,
  "dilution_comparison" TEXT,
  "dilution_value" REAL,
  "interpretation" TEXT,
  "comments" TEXT
)
```

---

## Table: `omr`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | subject_id   | INTEGER |         0 |              |    0 |
|     2 | chartdate    | TEXT    |         0 |              |    0 |
|     3 | seq_num      | INTEGER |         0 |              |    0 |
|     4 | result_name  | TEXT    |         0 |              |    0 |
|     5 | result_value | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "omr" (
"index" INTEGER,
  "subject_id" INTEGER,
  "chartdate" TEXT,
  "seq_num" INTEGER,
  "result_name" TEXT,
  "result_value" TEXT
)
```

---

## Table: `outputevents`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | subject_id   | INTEGER |         0 |              |    0 |
|     2 | hadm_id      | INTEGER |         0 |              |    0 |
|     3 | stay_id      | INTEGER |         0 |              |    0 |
|     4 | caregiver_id | INTEGER |         0 |              |    0 |
|     5 | charttime    | TEXT    |         0 |              |    0 |
|     6 | storetime    | TEXT    |         0 |              |    0 |
|     7 | itemid       | INTEGER |         0 |              |    0 |
|     8 | value        | REAL    |         0 |              |    0 |
|     9 | valueuom     | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "outputevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" INTEGER,
  "charttime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "value" REAL,
  "valueuom" TEXT
)
```

---

## Table: `patients`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | gender            | TEXT    |         0 |              |    0 |
|     3 | anchor_age        | INTEGER |         0 |              |    0 |
|     4 | anchor_year       | INTEGER |         0 |              |    0 |
|     5 | anchor_year_group | TEXT    |         0 |              |    0 |
|     6 | dod               | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "patients" (
"index" INTEGER,
  "subject_id" INTEGER,
  "gender" TEXT,
  "anchor_age" INTEGER,
  "anchor_year" INTEGER,
  "anchor_year_group" TEXT,
  "dod" TEXT
)
```

---

## Table: `pharmacy`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | hadm_id           | INTEGER |         0 |              |    0 |
|     3 | pharmacy_id       | INTEGER |         0 |              |    0 |
|     4 | poe_id            | TEXT    |         0 |              |    0 |
|     5 | starttime         | TEXT    |         0 |              |    0 |
|     6 | stoptime          | TEXT    |         0 |              |    0 |
|     7 | medication        | TEXT    |         0 |              |    0 |
|     8 | proc_type         | TEXT    |         0 |              |    0 |
|     9 | status            | TEXT    |         0 |              |    0 |
|    10 | entertime         | TEXT    |         0 |              |    0 |
|    11 | verifiedtime      | TEXT    |         0 |              |    0 |
|    12 | route             | TEXT    |         0 |              |    0 |
|    13 | frequency         | TEXT    |         0 |              |    0 |
|    14 | disp_sched        | TEXT    |         0 |              |    0 |
|    15 | infusion_type     | TEXT    |         0 |              |    0 |
|    16 | sliding_scale     | TEXT    |         0 |              |    0 |
|    17 | lockout_interval  | TEXT    |         0 |              |    0 |
|    18 | basal_rate        | REAL    |         0 |              |    0 |
|    19 | one_hr_max        | REAL    |         0 |              |    0 |
|    20 | doses_per_24_hrs  | REAL    |         0 |              |    0 |
|    21 | duration          | REAL    |         0 |              |    0 |
|    22 | duration_interval | TEXT    |         0 |              |    0 |
|    23 | expiration_value  | REAL    |         0 |              |    0 |
|    24 | expiration_unit   | TEXT    |         0 |              |    0 |
|    25 | expirationdate    | TEXT    |         0 |              |    0 |
|    26 | dispensation      | TEXT    |         0 |              |    0 |
|    27 | fill_quantity     | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "pharmacy" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "pharmacy_id" INTEGER,
  "poe_id" TEXT,
  "starttime" TEXT,
  "stoptime" TEXT,
  "medication" TEXT,
  "proc_type" TEXT,
  "status" TEXT,
  "entertime" TEXT,
  "verifiedtime" TEXT,
  "route" TEXT,
  "frequency" TEXT,
  "disp_sched" TEXT,
  "infusion_type" TEXT,
  "sliding_scale" TEXT,
  "lockout_interval" TEXT,
  "basal_rate" REAL,
  "one_hr_max" REAL,
  "doses_per_24_hrs" REAL,
  "duration" REAL,
  "duration_interval" TEXT,
  "expiration_value" REAL,
  "expiration_unit" TEXT,
  "expirationdate" TEXT,
  "dispensation" TEXT,
  "fill_quantity" TEXT
)
```

---

## Table: `poe`

### Columns

|   cid | name                   | type    |   notnull | dflt_value   |   pk |
|-------|------------------------|---------|-----------|--------------|------|
|     0 | index                  | INTEGER |         0 |              |    0 |
|     1 | poe_id                 | TEXT    |         0 |              |    0 |
|     2 | poe_seq                | INTEGER |         0 |              |    0 |
|     3 | subject_id             | INTEGER |         0 |              |    0 |
|     4 | hadm_id                | INTEGER |         0 |              |    0 |
|     5 | ordertime              | TEXT    |         0 |              |    0 |
|     6 | order_type             | TEXT    |         0 |              |    0 |
|     7 | order_subtype          | TEXT    |         0 |              |    0 |
|     8 | transaction_type       | TEXT    |         0 |              |    0 |
|     9 | discontinue_of_poe_id  | TEXT    |         0 |              |    0 |
|    10 | discontinued_by_poe_id | TEXT    |         0 |              |    0 |
|    11 | order_provider_id      | TEXT    |         0 |              |    0 |
|    12 | order_status           | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "poe" (
"index" INTEGER,
  "poe_id" TEXT,
  "poe_seq" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "ordertime" TEXT,
  "order_type" TEXT,
  "order_subtype" TEXT,
  "transaction_type" TEXT,
  "discontinue_of_poe_id" TEXT,
  "discontinued_by_poe_id" TEXT,
  "order_provider_id" TEXT,
  "order_status" TEXT
)
```

---

## Table: `poe_detail`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | poe_id      | TEXT    |         0 |              |    0 |
|     2 | poe_seq     | INTEGER |         0 |              |    0 |
|     3 | subject_id  | INTEGER |         0 |              |    0 |
|     4 | field_name  | TEXT    |         0 |              |    0 |
|     5 | field_value | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "poe_detail" (
"index" INTEGER,
  "poe_id" TEXT,
  "poe_seq" INTEGER,
  "subject_id" INTEGER,
  "field_name" TEXT,
  "field_value" TEXT
)
```

---

## Table: `prescriptions`

### Columns

|   cid | name              | type    |   notnull | dflt_value   |   pk |
|-------|-------------------|---------|-----------|--------------|------|
|     0 | index             | INTEGER |         0 |              |    0 |
|     1 | subject_id        | INTEGER |         0 |              |    0 |
|     2 | hadm_id           | INTEGER |         0 |              |    0 |
|     3 | pharmacy_id       | INTEGER |         0 |              |    0 |
|     4 | poe_id            | TEXT    |         0 |              |    0 |
|     5 | poe_seq           | REAL    |         0 |              |    0 |
|     6 | order_provider_id | TEXT    |         0 |              |    0 |
|     7 | starttime         | TEXT    |         0 |              |    0 |
|     8 | stoptime          | TEXT    |         0 |              |    0 |
|     9 | drug_type         | TEXT    |         0 |              |    0 |
|    10 | drug              | TEXT    |         0 |              |    0 |
|    11 | formulary_drug_cd | TEXT    |         0 |              |    0 |
|    12 | gsn               | TEXT    |         0 |              |    0 |
|    13 | ndc               | REAL    |         0 |              |    0 |
|    14 | prod_strength     | TEXT    |         0 |              |    0 |
|    15 | form_rx           | TEXT    |         0 |              |    0 |
|    16 | dose_val_rx       | TEXT    |         0 |              |    0 |
|    17 | dose_unit_rx      | TEXT    |         0 |              |    0 |
|    18 | form_val_disp     | TEXT    |         0 |              |    0 |
|    19 | form_unit_disp    | TEXT    |         0 |              |    0 |
|    20 | doses_per_24_hrs  | REAL    |         0 |              |    0 |
|    21 | route             | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "prescriptions" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "pharmacy_id" INTEGER,
  "poe_id" TEXT,
  "poe_seq" REAL,
  "order_provider_id" TEXT,
  "starttime" TEXT,
  "stoptime" TEXT,
  "drug_type" TEXT,
  "drug" TEXT,
  "formulary_drug_cd" TEXT,
  "gsn" TEXT,
  "ndc" REAL,
  "prod_strength" TEXT,
  "form_rx" TEXT,
  "dose_val_rx" TEXT,
  "dose_unit_rx" TEXT,
  "form_val_disp" TEXT,
  "form_unit_disp" TEXT,
  "doses_per_24_hrs" REAL,
  "route" TEXT
)
```

---

## Table: `procedureevents`

### Columns

|   cid | name                     | type    |   notnull | dflt_value   |   pk |
|-------|--------------------------|---------|-----------|--------------|------|
|     0 | index                    | INTEGER |         0 |              |    0 |
|     1 | subject_id               | INTEGER |         0 |              |    0 |
|     2 | hadm_id                  | INTEGER |         0 |              |    0 |
|     3 | stay_id                  | INTEGER |         0 |              |    0 |
|     4 | caregiver_id             | REAL    |         0 |              |    0 |
|     5 | starttime                | TEXT    |         0 |              |    0 |
|     6 | endtime                  | TEXT    |         0 |              |    0 |
|     7 | storetime                | TEXT    |         0 |              |    0 |
|     8 | itemid                   | INTEGER |         0 |              |    0 |
|     9 | value                    | REAL    |         0 |              |    0 |
|    10 | valueuom                 | TEXT    |         0 |              |    0 |
|    11 | location                 | TEXT    |         0 |              |    0 |
|    12 | locationcategory         | TEXT    |         0 |              |    0 |
|    13 | orderid                  | INTEGER |         0 |              |    0 |
|    14 | linkorderid              | INTEGER |         0 |              |    0 |
|    15 | ordercategoryname        | TEXT    |         0 |              |    0 |
|    16 | ordercategorydescription | TEXT    |         0 |              |    0 |
|    17 | patientweight            | REAL    |         0 |              |    0 |
|    18 | isopenbag                | INTEGER |         0 |              |    0 |
|    19 | continueinnextdept       | INTEGER |         0 |              |    0 |
|    20 | statusdescription        | TEXT    |         0 |              |    0 |
|    21 | originalamount           | REAL    |         0 |              |    0 |
|    22 | originalrate             | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "procedureevents" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "stay_id" INTEGER,
  "caregiver_id" REAL,
  "starttime" TEXT,
  "endtime" TEXT,
  "storetime" TEXT,
  "itemid" INTEGER,
  "value" REAL,
  "valueuom" TEXT,
  "location" TEXT,
  "locationcategory" TEXT,
  "orderid" INTEGER,
  "linkorderid" INTEGER,
  "ordercategoryname" TEXT,
  "ordercategorydescription" TEXT,
  "patientweight" REAL,
  "isopenbag" INTEGER,
  "continueinnextdept" INTEGER,
  "statusdescription" TEXT,
  "originalamount" REAL,
  "originalrate" INTEGER
)
```

---

## Table: `procedures_icd`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | subject_id  | INTEGER |         0 |              |    0 |
|     2 | hadm_id     | INTEGER |         0 |              |    0 |
|     3 | seq_num     | INTEGER |         0 |              |    0 |
|     4 | chartdate   | TEXT    |         0 |              |    0 |
|     5 | icd_code    | TEXT    |         0 |              |    0 |
|     6 | icd_version | INTEGER |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "procedures_icd" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "seq_num" INTEGER,
  "chartdate" TEXT,
  "icd_code" TEXT,
  "icd_version" INTEGER
)
```

---

## Table: `provider`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | provider_id | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "provider" (
"index" INTEGER,
  "provider_id" TEXT
)
```

---

## Table: `services`

### Columns

|   cid | name         | type    |   notnull | dflt_value   |   pk |
|-------|--------------|---------|-----------|--------------|------|
|     0 | index        | INTEGER |         0 |              |    0 |
|     1 | subject_id   | INTEGER |         0 |              |    0 |
|     2 | hadm_id      | INTEGER |         0 |              |    0 |
|     3 | transfertime | TEXT    |         0 |              |    0 |
|     4 | prev_service | TEXT    |         0 |              |    0 |
|     5 | curr_service | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "services" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" INTEGER,
  "transfertime" TEXT,
  "prev_service" TEXT,
  "curr_service" TEXT
)
```

---

## Table: `transfers`

### Columns

|   cid | name        | type    |   notnull | dflt_value   |   pk |
|-------|-------------|---------|-----------|--------------|------|
|     0 | index       | INTEGER |         0 |              |    0 |
|     1 | subject_id  | INTEGER |         0 |              |    0 |
|     2 | hadm_id     | REAL    |         0 |              |    0 |
|     3 | transfer_id | INTEGER |         0 |              |    0 |
|     4 | eventtype   | TEXT    |         0 |              |    0 |
|     5 | careunit    | TEXT    |         0 |              |    0 |
|     6 | intime      | TEXT    |         0 |              |    0 |
|     7 | outtime     | TEXT    |         0 |              |    0 |

### Creation SQL

```sql
CREATE TABLE "transfers" (
"index" INTEGER,
  "subject_id" INTEGER,
  "hadm_id" REAL,
  "transfer_id" INTEGER,
  "eventtype" TEXT,
  "careunit" TEXT,
  "intime" TEXT,
  "outtime" TEXT
)
```

---

