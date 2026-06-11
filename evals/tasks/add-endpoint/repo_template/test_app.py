from app import handle

assert handle("/")["status"] == 200
assert handle("/version")["body"] == "1.0.0"
assert handle("/health") == {"status": 200, "body": "ok"}, "GET /health must return status 200 body 'ok'"
print("all tests passed")
