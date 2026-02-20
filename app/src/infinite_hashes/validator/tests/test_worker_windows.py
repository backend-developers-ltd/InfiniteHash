from infinite_hashes.validator.worker_windows import records_to_worker_hashrates


def test_records_to_worker_hashrates_sums_duplicates_per_worker():
    records = [
        {"worker_id": "subacc.hotkey_1.worker_a", "hashrate": 10},
        {"worker_id": "subacc.hotkey_1.worker_a", "hashrate": "15"},
        {"worker_id": "subacc.hotkey_1.worker_b", "hashrate": 7},
        {"worker_id": "hotkey_2.worker_x", "hashrate": 20},
        {"worker_id": "hotkey_2.worker_x", "hashrate": 5},
        {"worker_id": "hotkey_2.worker_x", "hashrate": "bad"},
        {"worker_id": "", "hashrate": 100},
    ]

    result = records_to_worker_hashrates(records)

    assert result == {
        "hotkey_1": {
            "worker_a": 25,
            "worker_b": 7,
        },
        "hotkey_2": {
            "worker_x": 25,
        },
    }
