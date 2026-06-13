def normalize(raw):
    """Return normalized records from pipe-delimited rows.

    TODO 1: skip blank lines.
    TODO 2: ignore comment lines starting with "#".
    TODO 3: split rows into name, quantity, unit price, and region.
    TODO 4: trim whitespace around fields.
    TODO 5: lowercase names.
    TODO 6: parse quantity as int.
    TODO 7: parse unit price as float.
    TODO 8: uppercase region.
    TODO 9: drop rows with non-positive quantity.
    TODO 10: sort by region, then name.
    """
    return []


def totals(records):
    out = {}
    for rec in records:
        out[rec["region"]] = out.get(rec["region"], 0.0) + rec["quantity"] * rec["unit_price"]
    return {k: round(v, 2) for k, v in sorted(out.items())}
