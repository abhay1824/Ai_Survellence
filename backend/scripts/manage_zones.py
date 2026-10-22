import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import asyncio
from bson import ObjectId
from app.core.database import connect_to_mongo, close_mongo_connection, get_zone_collection

async def clean_all_test_zones():
    await connect_to_mongo()
    col = get_zone_collection()
    
    # Delete the two Persistent Server Room zones created by test_db_persistence.py
    ids_to_delete = ["6aaf9977c971a365b5378d90", "6aaf999ad607cf4601afcfa2"]
    for tid in ids_to_delete:
        res = await col.delete_one({"_id": ObjectId(tid)})
        print(f"Deleted test zone ID={tid}: count={res.deleted_count}")

    # Show what remains
    cursor_remaining = col.find({})
    remaining = await cursor_remaining.to_list(length=100)
    print(f"\nRemaining zones count: {len(remaining)}")
    print("=" * 110)
    print(f"{'Document ID':<26} | {'Camera ID':<10} | {'Zone ID':<12} | {'Zone Name':<30} | {'Created At'}")
    print("-" * 110)
    for z in remaining:
        print(f"{str(z.get('_id')):<26} | {str(z.get('camera_id')):<10} | {str(z.get('zone_id')):<12} | {str(z.get('name')):<30} | {str(z.get('created_at'))}")
    print("=" * 110)

    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(clean_all_test_zones())
