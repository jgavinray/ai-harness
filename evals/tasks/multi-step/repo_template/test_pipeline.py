from pipeline import make_id

assert make_id("Hello World", 7) == "hello-world-007"
assert make_id("  Spaces  And--Dashes ", 1) == "spaces-and-dashes-001"
assert make_id("MiXeD CaSe!", 42) == "mixed-case-042"
print("all tests passed")
