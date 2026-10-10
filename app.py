import json
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import streamlit as st
import torch

from PIL import Image
from transformers.models.donut.processing_donut import DonutProcessor
from transformers import VisionEncoderDecoderModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "splitsnap_model"
MAX_LENGTH = 768


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SplitSnap",
    page_icon="🧾",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    processor = DonutProcessor.from_pretrained(
        MODEL_PATH
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_PATH
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


processor, model, device = load_model()


# ============================================================
# MODEL INFERENCE
# ============================================================

def extract_receipt(image):

    image = image.convert("RGB")

    pixel_values = (
        processor.image_processor(
            images=image,
            return_tensors="pt",
        )
        .pixel_values
        .to(device)
    )

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=MAX_LENGTH,
            num_beams=1,
            do_sample=False,
        )

    prediction = processor.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    return prediction


# ============================================================
# JSON PARSING
# ============================================================

def try_parse_json(text):

    try:
        return json.loads(text), None

    except json.JSONDecodeError as first_error:

        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            candidate = text[start:end + 1]

            try:
                return json.loads(candidate), None

            except json.JSONDecodeError:
                pass

        return None, str(first_error)


# ============================================================
# NORMALIZE MODEL OUTPUT
# ============================================================

def normalize_bill(bill):

    if not isinstance(bill, dict):
        bill = {}

    bill.setdefault("items", [])
    bill.setdefault("charges", [])
    bill.setdefault("subtotal", None)
    bill.setdefault("grand_total", None)

    if not isinstance(
        bill["items"],
        list
    ):
        bill["items"] = []

    if not isinstance(
        bill["charges"],
        list
    ):
        bill["charges"] = []

    return bill


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def optional_float(value):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


# ============================================================
# MONEY HELPERS
# ============================================================

def money(value):

    return Decimal(
        str(value)
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def format_money(value):

    if value is None:
        return "—"

    value = float(value)

    if value < 0:

        return (
            f"-₹{abs(value):,.2f}"
        )

    return f"₹{value:,.2f}"


# ============================================================
# AUTOMATIC BILL CONSISTENCY CHECK
# ============================================================

def analyze_bill_balance(
    bill,
    tolerance=0.02,
):

    subtotal = optional_float(
        bill.get("subtotal")
    )

    grand_total = optional_float(
        bill.get("grand_total")
    )

    # --------------------------------------------------------
    # Prefer printed subtotal if one exists.
    # Otherwise reconstruct it from items.
    # --------------------------------------------------------

    if (
        subtotal is not None
        and subtotal > 0
    ):

        base_amount = subtotal
        base_source = "subtotal"

    else:

        base_amount = 0.0
        base_source = "items"

        for item in bill.get(
            "items",
            []
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            total = optional_float(
                item.get("total")
            )

            if total is None:
                continue

            effective_total = total

            discount = optional_float(
                item.get("discount")
            )

            if discount is not None:

                effective_total += (
                    discount
                )

            base_amount += (
                effective_total
            )


    # --------------------------------------------------------
    # Add bill-level tax / service / discounts
    # --------------------------------------------------------

    charge_total = 0.0

    for charge in bill.get(
        "charges",
        []
    ):

        if not isinstance(
            charge,
            dict
        ):
            continue

        amount = optional_float(
            charge.get("amount")
        )

        if amount is not None:

            charge_total += amount


    calculated_total = (
        base_amount
        + charge_total
    )


    if grand_total is None:

        return {

            "balanced": None,

            "base_source":
                base_source,

            "base_amount":
                round(
                    base_amount,
                    2
                ),

            "charge_total":
                round(
                    charge_total,
                    2
                ),

            "calculated_total":
                round(
                    calculated_total,
                    2
                ),

            "grand_total":
                None,

            "difference":
                None,
        }


    difference = (
        grand_total
        - calculated_total
    )


    return {

        "balanced":
            abs(difference)
            <= tolerance,

        "base_source":
            base_source,

        "base_amount":
            round(
                base_amount,
                2
            ),

        "charge_total":
            round(
                charge_total,
                2
            ),

        "calculated_total":
            round(
                calculated_total,
                2
            ),

        "grand_total":
            round(
                grand_total,
                2
            ),

        "difference":
            round(
                difference,
                2
            ),
    }


# ============================================================
# SPLITTING ENGINE
# ============================================================

def split_bill(
    bill,
    assignments,
):

    people_totals = {}
    item_breakdown = {}


    # --------------------------------------------------------
    # 1. SPLIT ITEMS
    # --------------------------------------------------------

    for item_index, item in enumerate(
        bill.get("items", [])
    ):

        item_total = optional_float(
            item.get("total")
        )

        if item_total is None:
            continue


        assigned_people = (
            assignments.get(
                item_index,
                []
            )
        )

        if not assigned_people:
            continue


        item_total = money(
            item_total
        )


        # Item-level discount
        item_discount = optional_float(
            item.get("discount")
        )


        if item_discount is None:

            item_discount = (
                Decimal("0.00")
            )

        else:

            item_discount = money(
                item_discount
            )


        effective_total = (
            item_total
            + item_discount
        )


        raw_share = (
            effective_total
            / Decimal(
                len(
                    assigned_people
                )
            )
        )


        remaining = (
            effective_total
        )


        for position, person in enumerate(
            assigned_people
        ):

            # Final person receives the remainder
            # so rounding always reconciles exactly.
            if (
                position
                == len(
                    assigned_people
                ) - 1
            ):

                share = remaining

            else:

                share = money(
                    raw_share
                )

                remaining -= share


            people_totals.setdefault(
                person,
                Decimal("0.00"),
            )

            item_breakdown.setdefault(
                person,
                [],
            )


            people_totals[
                person
            ] += share


            item_breakdown[
                person
            ].append(
                {
                    "item":
                        item.get(
                            "name",
                            f"Item {item_index + 1}",
                        ),

                    "share":
                        share,
                }
            )


    # --------------------------------------------------------
    # 2. ASSIGNED SUBTOTAL
    # --------------------------------------------------------

    assigned_subtotal = sum(
        people_totals.values(),
        Decimal("0.00"),
    )


    # --------------------------------------------------------
    # 3. CHARGE BREAKDOWN
    # --------------------------------------------------------

    charge_breakdown = {

        person: []

        for person
        in people_totals
    }


    # --------------------------------------------------------
    # 4. SPLIT BILL-LEVEL CHARGES PROPORTIONALLY
    # --------------------------------------------------------

    if assigned_subtotal > 0:

        people = list(
            people_totals.keys()
        )


        for charge in bill.get(
            "charges",
            []
        ):

            amount = optional_float(
                charge.get("amount")
            )

            if amount is None:
                continue


            amount = money(
                amount
            )

            remaining_charge = (
                amount
            )


            for position, person in enumerate(
                people
            ):

                if (
                    position
                    == len(people) - 1
                ):

                    person_charge = (
                        remaining_charge
                    )

                else:

                    proportion = (
                        people_totals[
                            person
                        ]
                        / assigned_subtotal
                    )


                    person_charge = money(
                        amount
                        * proportion
                    )


                    remaining_charge -= (
                        person_charge
                    )


                charge_breakdown[
                    person
                ].append(
                    {
                        "type":
                            charge.get(
                                "type",
                                "Charge",
                            ),

                        "amount":
                            person_charge,
                    }
                )


    # --------------------------------------------------------
    # 5. FINAL TOTALS
    # --------------------------------------------------------

    final_totals = {}


    for person, base_total in (
        people_totals.items()
    ):

        charge_total = sum(
            (
                charge["amount"]

                for charge

                in charge_breakdown[
                    person
                ]
            ),

            Decimal("0.00"),
        )


        final_totals[
            person
        ] = money(
            base_total
            + charge_total
        )


    # --------------------------------------------------------
    # 6. CONVERT DECIMAL -> FLOAT
    # --------------------------------------------------------

    return {

        "item_totals": {

            person:
                float(value)

            for person, value
            in people_totals.items()
        },


        "items": {

            person: [

                {
                    "item":
                        item["item"],

                    "share":
                        float(
                            item["share"]
                        ),
                }

                for item
                in items
            ]

            for person, items
            in item_breakdown.items()
        },


        "charges": {

            person: [

                {
                    "type":
                        charge["type"],

                    "amount":
                        float(
                            charge["amount"]
                        ),
                }

                for charge
                in charges
            ]

            for person, charges
            in charge_breakdown.items()
        },


        "final_totals": {

            person:
                float(value)

            for person, value
            in final_totals.items()
        },
    }


# ============================================================
# VERIFY FINAL SPLIT
# ============================================================

def validate_split(
    bill,
    result,
    tolerance=0.02,
):

    calculated_total = sum(
        result[
            "final_totals"
        ].values()
    )


    expected_total = optional_float(
        bill.get(
            "grand_total"
        )
    )


    if expected_total is None:

        return {

            "valid": True,

            "calculated_total":
                calculated_total,

            "expected_total":
                None,

            "difference":
                None,
        }


    difference = (
        calculated_total
        - expected_total
    )


    return {

        "valid":
            abs(difference)
            <= tolerance,

        "calculated_total":
            round(
                calculated_total,
                2
            ),

        "expected_total":
            round(
                expected_total,
                2
            ),

        "difference":
            round(
                difference,
                2
            ),
    }


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "bill":
        None,

    "raw_prediction":
        None,

    "json_error":
        None,

    "extraction_complete":
        False,

    "split_result":
        None,

    "uploaded_signature":
        None,
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧾 SplitSnap"
)

st.write(
    "Upload a receipt, correct the AI extraction, "
    "assign items, and split the bill fairly."
)

st.caption(
    f"Fine-tuned Donut model • "
    f"Running on {device.upper()}"
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a receipt",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Reset when a different receipt is uploaded
    # --------------------------------------------------------

    signature = (
        f"{uploaded_file.name}-"
        f"{uploaded_file.size}"
    )


    if (
        st.session_state.uploaded_signature
        != signature
    ):

        st.session_state.uploaded_signature = (
            signature
        )

        st.session_state.bill = None

        st.session_state.raw_prediction = (
            None
        )

        st.session_state.json_error = (
            None
        )

        st.session_state.extraction_complete = (
            False
        )

        st.session_state.split_result = (
            None
        )


    image = Image.open(
        uploaded_file
    )


    left, right = st.columns(
        [1, 1]
    )


    # ========================================================
    # RECEIPT IMAGE
    # ========================================================

    with left:

        st.subheader(
            "1. Receipt"
        )

        st.image(
            image,
            use_container_width=True,
        )


    # ========================================================
    # AI EXTRACTION
    # ========================================================

    with right:

        st.subheader(
            "2. AI Extraction"
        )


        if st.button(
            "Extract Receipt",
            type="primary",
        ):

            with st.spinner(
                "Reading receipt..."
            ):

                prediction = (
                    extract_receipt(
                        image
                    )
                )


            bill, error = (
                try_parse_json(
                    prediction
                )
            )


            st.session_state.raw_prediction = (
                prediction
            )

            st.session_state.json_error = (
                error
            )

            st.session_state.extraction_complete = (
                True
            )

            st.session_state.split_result = (
                None
            )


            if bill is not None:

                st.session_state.bill = (
                    normalize_bill(
                        bill
                    )
                )

            else:

                # Invalid AI output does not stop
                # the workflow.
                st.session_state.bill = {

                    "items": [],

                    "charges": [],

                    "subtotal": None,

                    "grand_total": None,
                }


        # ----------------------------------------------------
        # RAW MODEL OUTPUT
        # ----------------------------------------------------

        if (
            st.session_state.raw_prediction
            is not None
        ):

            with st.expander(
                "Show raw model output"
            ):

                st.code(
                    st.session_state.raw_prediction,
                    language="json",
                )


        # ----------------------------------------------------
        # EXTRACTION STATUS
        # ----------------------------------------------------

        if (
            st.session_state.extraction_complete
        ):

            if (
                st.session_state.json_error
                is None
            ):

                st.success(
                    "Receipt extracted successfully."
                )

            else:

                st.warning(
                    "The model output was not valid JSON. "
                    "Correct the receipt manually below."
                )

                st.caption(
                    st.session_state.json_error
                )


# ============================================================
# REVIEW + CORRECTION
# ============================================================

if st.session_state.bill is not None:

    st.divider()

    st.header(
        "3. Review & Correct"
    )

    st.caption(
        "AI extraction is a draft. "
        "Fix anything incorrect before splitting."
    )


    bill = (
        st.session_state.bill
    )


    # ========================================================
    # ITEMS EDITOR
    # ========================================================

    st.subheader(
        "Items"
    )


    item_columns = [

        "name",

        "qty",

        "unit_price",

        "total",

        "discount",
    ]


    items_df = pd.DataFrame(
        bill.get(
            "items",
            [],
        )
    )


    for column in item_columns:

        if column not in items_df.columns:

            items_df[
                column
            ] = None


    items_df = items_df[
        item_columns
    ]


    edited_items = st.data_editor(
        items_df,
        num_rows="dynamic",
        use_container_width=True,
        key="items_editor",

        column_config={

            "name":
                st.column_config.TextColumn(
                    "Item Name"
                ),

            "qty":
                st.column_config.NumberColumn(
                    "Quantity",
                    min_value=0.0,
                ),

            "unit_price":
                st.column_config.NumberColumn(
                    "Unit Price",
                    format="%.2f",
                ),

            "total":
                st.column_config.NumberColumn(
                    "Item Total",
                    format="%.2f",
                ),

            "discount":
                st.column_config.NumberColumn(
                    "Item Discount",
                    format="%.2f",
                    help=(
                        "Use a negative "
                        "number for discounts."
                    ),
                ),
        },
    )


    # ========================================================
    # CHARGES EDITOR
    # ========================================================

    st.subheader(
        "Taxes / Fees / Discounts"
    )


    charge_columns = [
        "type",
        "amount",
    ]


    charges_df = pd.DataFrame(
        bill.get(
            "charges",
            [],
        )
    )


    for column in charge_columns:

        if column not in charges_df.columns:

            charges_df[
                column
            ] = None


    charges_df = charges_df[
        charge_columns
    ]


    edited_charges = st.data_editor(
        charges_df,
        num_rows="dynamic",
        use_container_width=True,
        key="charges_editor",

        column_config={

            "type":
                st.column_config.SelectboxColumn(
                    "Charge Type",
                    options=[
                        "Tax",
                        "Service Charge",
                        "Discount",
                        "Other Adjustment",
                    ],
                ),

            "amount":
                st.column_config.NumberColumn(
                    "Amount",
                    format="%.2f",
                    help=(
                        "Discounts should normally "
                        "be negative."
                    ),
                ),
        },
    )


    # ========================================================
    # TOTALS
    # ========================================================

    st.subheader(
        "Bill Totals"
    )


    total_col1, total_col2 = (
        st.columns(2)
    )


    subtotal_value = (
        0.0

        if bill.get(
            "subtotal"
        ) is None

        else float(
            bill[
                "subtotal"
            ]
        )
    )


    grand_total_value = (
        0.0

        if bill.get(
            "grand_total"
        ) is None

        else float(
            bill[
                "grand_total"
            ]
        )
    )


    with total_col1:

        edited_subtotal = (
            st.number_input(
                "Subtotal",
                value=subtotal_value,
                step=0.01,
                format="%.2f",
            )
        )


    with total_col2:

        edited_grand_total = (
            st.number_input(
                "Grand Total",
                value=grand_total_value,
                step=0.01,
                format="%.2f",
            )
        )


    # ========================================================
    # AUTOMATIC CONSISTENCY CHECK
    # ========================================================

    preview_bill = {

        "items":
            edited_items.to_dict(
                orient="records"
            ),

        "charges":
            edited_charges.to_dict(
                orient="records"
            ),

        "subtotal":
            edited_subtotal,

        "grand_total":
            edited_grand_total,
    }


    balance = analyze_bill_balance(
        preview_bill
    )


    if balance[
        "balanced"
    ] is True:

        st.success(
            "✓ Extracted bill is mathematically consistent."
        )


    elif balance[
        "balanced"
    ] is False:

        difference = (
            balance[
                "difference"
            ]
        )


        st.warning(
            "⚠️ The extracted values do not add up "
            "to the printed grand total."
        )


        balance_col1, balance_col2, balance_col3 = (
            st.columns(3)
        )


        with balance_col1:

            st.metric(
                "Calculated",
                format_money(
                    balance[
                        "calculated_total"
                    ]
                )
            )


        with balance_col2:

            st.metric(
                "Receipt Total",
                format_money(
                    balance[
                        "grand_total"
                    ]
                )
            )


        with balance_col3:

            st.metric(
                "Unexplained",
                format_money(
                    abs(
                        difference
                    )
                )
            )


        # ----------------------------------------------------
        # Missing negative adjustment
        # ----------------------------------------------------

        if difference < 0:

            st.error(
                "Possible missing discount "
                "or negative adjustment: "
                f"{format_money(difference)}"
            )

            st.caption(
                "Check the receipt for a discount, "
                "coupon, promotion, cashback, "
                "or another deduction."
            )


        # ----------------------------------------------------
        # Missing positive adjustment
        # ----------------------------------------------------

        elif difference > 0:

            st.error(
                "Possible missing tax, fee, "
                "or positive adjustment: "
                f"+{format_money(difference)}"
            )

            st.caption(
                "Check the receipt for tax, "
                "service charge, delivery fee, "
                "tip, or another surcharge."
            )


    # ========================================================
    # SAVE CORRECTIONS
    # ========================================================

    if st.button(
        "Save Corrections",
        type="primary",
    ):

        clean_items = []


        for record in (
            edited_items.to_dict(
                orient="records"
            )
        ):

            name = record.get(
                "name"
            )


            if (
                name is None
                or str(
                    name
                ).strip() == ""
            ):

                continue


            clean_items.append(
                {

                    "name":
                        str(
                            name
                        ).strip(),

                    "qty":
                        optional_float(
                            record.get(
                                "qty"
                            )
                        ),

                    "unit_price":
                        optional_float(
                            record.get(
                                "unit_price"
                            )
                        ),

                    "total":
                        optional_float(
                            record.get(
                                "total"
                            )
                        ),

                    "discount":
                        optional_float(
                            record.get(
                                "discount"
                            )
                        ),
                }
            )


        clean_charges = []


        for record in (
            edited_charges.to_dict(
                orient="records"
            )
        ):

            charge_type = (
                record.get(
                    "type"
                )
            )


            amount = (
                optional_float(
                    record.get(
                        "amount"
                    )
                )
            )


            if (
                charge_type is None
                and amount is None
            ):

                continue


            clean_charges.append(
                {

                    "type":
                        charge_type
                        or "Other Adjustment",

                    "amount":
                        amount,
                }
            )


        st.session_state.bill = {

            "items":
                clean_items,

            "charges":
                clean_charges,

            "subtotal":
                float(
                    edited_subtotal
                ),

            "grand_total":
                float(
                    edited_grand_total
                ),
        }


        st.session_state.split_result = (
            None
        )


        st.success(
            "Corrections saved."
        )


# ============================================================
# PEOPLE + ITEM ASSIGNMENTS
# ============================================================

if st.session_state.bill is not None:

    bill = (
        st.session_state.bill
    )


    st.divider()

    st.header(
        "4. Who's Paying?"
    )


    people_text = st.text_input(
        "Enter names separated by commas",
        placeholder=(
            "Alice, Bob, Charlie"
        ),
    )


    people = []


    for raw_name in (
        people_text.split(",")
    ):

        name = (
            raw_name.strip()
        )


        if (
            name
            and name not in people
        ):

            people.append(
                name
            )


    if len(people) >= 2:

        st.success(
            f"{len(people)} people added: "
            + ", ".join(
                people
            )
        )


        st.subheader(
            "5. Assign Items"
        )

        st.caption(
            "Select everyone who shared each item. "
            "Shared items are divided equally."
        )


        assignments = {}


        for item_index, item in enumerate(
            bill.get(
                "items",
                []
            )
        ):

            item_name = (
                item.get(
                    "name",
                    f"Item {item_index + 1}",
                )
            )


            item_total = (
                item.get(
                    "total"
                )
            )


            if item_total is None:

                display_total = (
                    "No total"
                )

            else:

                display_total = (
                    format_money(
                        item_total
                    )
                )


            key = (
                f"assignment_"
                f"{item_index}"
            )


            if key in st.session_state:

                st.session_state[
                    key
                ] = [

                    person

                    for person

                    in st.session_state[
                        key
                    ]

                    if person
                    in people
                ]


            selected_people = (
                st.multiselect(

                    f"{item_name} — "
                    f"{display_total}",

                    options=
                        people,

                    key=
                        key,
                )
            )


            assignments[
                item_index
            ] = selected_people


        # ====================================================
        # CALCULATE SPLIT
        # ====================================================

        if st.button(
            "Calculate Split",
            type="primary",
        ):

            missing_totals = []

            missing_assignments = []


            for item_index, item in enumerate(
                bill.get(
                    "items",
                    []
                )
            ):

                if (
                    optional_float(
                        item.get(
                            "total"
                        )
                    )
                    is None
                ):

                    missing_totals.append(
                        item.get(
                            "name",
                            f"Item {item_index + 1}",
                        )
                    )


                if not assignments.get(
                    item_index
                ):

                    missing_assignments.append(
                        item.get(
                            "name",
                            f"Item {item_index + 1}",
                        )
                    )


            if missing_totals:

                st.error(
                    "These items have no total: "
                    + ", ".join(
                        missing_totals
                    )
                    + ". Fix them above first."
                )


            elif missing_assignments:

                st.error(
                    "Assign these items first: "
                    + ", ".join(
                        missing_assignments
                    )
                )


            else:

                result = split_bill(
                    bill,
                    assignments,
                )


                st.session_state.split_result = (
                    result
                )


                # Initialize override values
                for person, amount in (
                    result[
                        "final_totals"
                    ].items()
                ):

                    st.session_state[
                        f"override_{person}"
                    ] = float(
                        amount
                    )


    elif people_text:

        st.info(
            "Enter at least two people."
        )


# ============================================================
# FINAL RESULTS
# ============================================================

if (
    st.session_state.split_result
    is not None
):

    result = (
        st.session_state.split_result
    )

    bill = (
        st.session_state.bill
    )


    st.divider()

    st.header(
        "6. Final Split"
    )


    # ========================================================
    # TOTAL CARDS
    # ========================================================

    people_results = list(
        result[
            "final_totals"
        ].items()
    )


    columns = st.columns(
        min(
            len(
                people_results
            ),
            4,
        )
    )


    for index, (
        person,
        total,
    ) in enumerate(
        people_results
    ):

        with columns[
            index
            % len(
                columns
            )
        ]:

            st.metric(
                person,
                format_money(
                    total
                ),
            )


    # ========================================================
    # VERIFY SPLIT
    # ========================================================

    verification = validate_split(
        bill,
        result,
    )


    if verification[
        "valid"
    ]:

        st.success(
            "✓ Split matches the bill total."
        )


    else:

        st.warning(
            "The split does not match the "
            "receipt grand total."
        )


        verify_col1, verify_col2, verify_col3 = (
            st.columns(3)
        )


        with verify_col1:

            st.metric(
                "Calculated",
                format_money(
                    verification[
                        "calculated_total"
                    ]
                )
            )


        with verify_col2:

            st.metric(
                "Receipt Total",
                format_money(
                    verification[
                        "expected_total"
                    ]
                )
            )


        with verify_col3:

            st.metric(
                "Difference",
                format_money(
                    verification[
                        "difference"
                    ]
                )
            )


        st.caption(
            "Review the extracted items, "
            "discounts and charges above."
        )


    # ========================================================
    # EXPLANATIONS
    # ========================================================

    st.subheader(
        "7. How was this calculated?"
    )


    assigned_subtotal = sum(
        result[
            "item_totals"
        ].values()
    )


    for person, final_total in (
        result[
            "final_totals"
        ].items()
    ):

        with st.expander(
            f"{person} — "
            f"{format_money(final_total)}",
            expanded=True,
        ):

            st.write(
                "**Items**"
            )


            for item in (
                result[
                    "items"
                ].get(
                    person,
                    [],
                )
            ):

                st.write(
                    f"• {item['item']}: "
                    f"{format_money(item['share'])}"
                )


            item_total = (
                result[
                    "item_totals"
                ][
                    person
                ]
            )


            st.write(
                "**Item subtotal:** "
                f"{format_money(item_total)}"
            )


            if assigned_subtotal > 0:

                percentage = (
                    item_total
                    / assigned_subtotal
                    * 100
                )


                st.caption(
                    f"{person} consumed "
                    f"{percentage:.1f}% "
                    "of the assigned item value. "
                    "Shared bill-level charges and "
                    "discounts are distributed "
                    "using that proportion."
                )


            person_charges = (
                result[
                    "charges"
                ].get(
                    person,
                    [],
                )
            )


            if person_charges:

                st.write(
                    "**Shared charges / discounts**"
                )


                for charge in (
                    person_charges
                ):

                    st.write(
                        f"• {charge['type']}: "
                        f"{format_money(charge['amount'])}"
                    )


            st.write(
                "### Total: "
                f"{format_money(final_total)}"
            )


    # ========================================================
    # MANUAL OVERRIDE
    # ========================================================

    st.subheader(
        "8. Manual Override"
    )

    st.caption(
        "Optional: change a final amount "
        "if the group agrees on a different split."
    )


    override_totals = {}


    for person, calculated in (
        result[
            "final_totals"
        ].items()
    ):

        key = (
            f"override_{person}"
        )


        if key not in st.session_state:

            st.session_state[
                key
            ] = float(
                calculated
            )


        override_totals[
            person
        ] = st.number_input(

            f"{person} final amount",

            step=0.01,

            format="%.2f",

            key=key,
        )


    override_sum = sum(
        override_totals.values()
    )


    st.write(
        "**Final overridden total:**",
        format_money(
            override_sum
        ),
    )


    expected = optional_float(
        bill.get(
            "grand_total"
        )
    )


    if expected is not None:

        override_difference = (
            override_sum
            - expected
        )


        if (
            abs(
                override_difference
            )
            <= 0.02
        ):

            st.success(
                "✓ Overrides still match "
                "the bill total."
            )


        else:

            st.warning(
                "Overrides differ from the bill by "
                f"{format_money(override_difference)}."
            )


    # ========================================================
    # STRUCTURED RESULT
    # ========================================================

    with st.expander(
        "View split JSON"
    ):

        st.json(
            {

                "calculated":
                    result[
                        "final_totals"
                    ],

                "manual_overrides":
                    override_totals,

                "breakdown":
                    result,
            }
        )