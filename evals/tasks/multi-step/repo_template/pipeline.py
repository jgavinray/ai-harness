from textutil import slugify


def make_id(title, seq):
    return f"{slugify(title)}-{seq:03d}"
