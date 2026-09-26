"""Exercise the offline Ch06 page in local headless Chrome over DevTools."""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import requests
import websockets


CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8766/"
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


async def verify():
    with tempfile.TemporaryDirectory(prefix="ch06-browser-", dir=TEMP_ROOT, ignore_cleanup_errors=True) as profile:
        assert Path(profile).resolve().is_relative_to(TEMP_ROOT)
        process = subprocess.Popen([
            str(CHROME), "--headless=new", "--no-first-run", "--no-default-browser-check",
            "--remote-debugging-port=0", "--remote-allow-origins=*",
            f"--user-data-dir={profile}", URL,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            for _ in range(40):
                if not port_file.exists():
                    await asyncio.sleep(0.25)
                    continue
                try:
                    port = int(port_file.read_text(encoding="utf-8").splitlines()[0])
                    tabs = requests.get(f"http://127.0.0.1:{port}/json", timeout=1).json()
                    page = next(item for item in tabs if item.get("type") == "page" and item.get("url", "").startswith(URL))
                    break
                except (requests.RequestException, StopIteration):
                    await asyncio.sleep(0.25)
            else:
                raise AssertionError("Chrome DevTools did not start")
            async with websockets.connect(page["webSocketDebuggerUrl"], max_size=4_000_000) as ws:
                next_id = 0

                async def evaluate(expression):
                    nonlocal next_id
                    next_id += 1
                    ident = next_id
                    await ws.send(json.dumps({"id": ident, "method": "Runtime.evaluate", "params": {
                        "expression": expression, "returnByValue": True, "awaitPromise": True,
                    }}))
                    while True:
                        message = json.loads(await ws.recv())
                        if message.get("id") != ident:
                            continue
                        if "exceptionDetails" in message.get("result", {}):
                            raise AssertionError(message["result"]["exceptionDetails"])
                        return message["result"]["result"].get("value")

                async def until(expression, expected=True):
                    for _ in range(60):
                        if await evaluate(expression) == expected:
                            return
                        await asyncio.sleep(0.2)
                    raise AssertionError(f"UI condition not reached: {expression}")

                await until("document.readyState === 'complete'")
                await until("document.querySelector('#demoOrders') !== null")
                start_writes = sum(x["path"] == "/api/actions/create-refund"
                                   for x in requests.get(URL + "_calls").json())
                await evaluate("document.querySelector('#demoOrders').click()")
                assert await evaluate("localStorage.getItem('petshop.user_id')") == "demo-user"
                await evaluate("document.querySelector('#message').value='我要退款'; document.querySelector('#chatForm').requestSubmit()")
                await until("document.querySelectorAll('.order-choice').length === 1")
                assert await evaluate("document.querySelector('#send').disabled") is True
                assert await evaluate("document.querySelector('.order-choice').textContent.includes('1001')") is True
                assert await evaluate("document.querySelector('.order-choice').textContent.includes('2001')") is False
                assert sum(x["path"] == "/api/actions/create-refund"
                           for x in requests.get(URL + "_calls").json()) == start_writes

                await evaluate("document.querySelector('.order-choice').click()")
                await until("[...document.querySelectorAll('.actions button')].some(b => b.textContent === '申请退款')")
                await asyncio.sleep(0.3)
                assert await evaluate("document.querySelectorAll('.bubble.error').length") == 0
                await evaluate("[...document.querySelectorAll('.actions button')].find(b => b.textContent === '申请退款').click()")
                await until("!document.querySelector('.action-form').hidden")
                await evaluate("[...document.querySelectorAll('.action-form button')].find(b => b.textContent === '取消').click()")
                assert sum(x["path"] == "/api/actions/create-refund"
                           for x in requests.get(URL + "_calls").json()) == start_writes

                await evaluate("[...document.querySelectorAll('.actions button')].find(b => b.textContent === '申请退款').click()")
                await evaluate("document.querySelector('.action-form button.primary').click()")
                await until("document.querySelector('.action-note').textContent.includes('R-DEMO-1')")
                assert await evaluate("[...document.querySelectorAll('.actions button')].some(b => b.textContent === '申请 R-DEMO-1' && b.disabled)")
                writes = [x for x in requests.get(URL + "_calls").json() if x["path"] == "/api/actions/create-refund"][start_writes:]
                assert len(writes) == 1, writes
                assert writes[0]["payload"]["user_id"] == "demo-user"
                assert writes[0]["payload"]["order_id"] == "1001"
                assert writes[0]["payload"]["reason"] == "质量问题"
                print("PASS: order choice, resume, cancel no write, confirm once, own order only")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)


if __name__ == "__main__":
    asyncio.run(verify())
