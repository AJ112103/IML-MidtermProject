import os
import time
import json
import asyncio
import aiohttp

async def fetch_orderbook_binance_async(session, symbol="BTCUSDT", limit=100):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    async with session.get(url) as response:
        if response.status != 200:
            text = await response.text()
            raise ValueError(f"Error fetching data for {symbol}: {text}")
        data = await response.json()
    return data

def save_raw_orderbook(data, symbol, folder="data/raw"):
    ts = int(time.time() * 1000)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{symbol}_orderbook_{ts}.json")
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Saved {symbol} orderbook to {filename}")

async def collect_all_asset_async(assets=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"], folder="data/raw"):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in assets:
            task = asyncio.create_task(fetch_orderbook_binance_async(session, symbol))
            tasks.append((symbol, task))
        for symbol, task in tasks:
            try:
                data = await task
                save_raw_orderbook(data, symbol, folder)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

async def main():
    assets_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
    while True:
        await collect_all_asset_async(assets=assets_list)
        # Wait for 60 seconds before fetching the next set of snapshots (adjust as needed)
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
