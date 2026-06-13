from pipeline import normalize, totals


RAW = """
# name | qty | unit | region
Widget | 2 | 3.50 | west
Gadget | 0 | 9.00 | east
Bolt | 5 | 1.20 | west
Clamp | 3 | 4 | east
 widget | 1 | 3.50 | east

"""


records = normalize(RAW)
assert records == [
    {"name": "clamp", "quantity": 3, "unit_price": 4.0, "region": "EAST"},
    {"name": "widget", "quantity": 1, "unit_price": 3.5, "region": "EAST"},
    {"name": "bolt", "quantity": 5, "unit_price": 1.2, "region": "WEST"},
    {"name": "widget", "quantity": 2, "unit_price": 3.5, "region": "WEST"},
]
assert totals(records) == {"EAST": 15.5, "WEST": 13.0}
print("long-horizon task passed")
