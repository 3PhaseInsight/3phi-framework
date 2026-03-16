import xxhash

# For new data storage layout
N_SHARDS = 3


def v1_get_shard_for_meter_id(meter_id: str):
    shard = xxhash.xxh64(meter_id).intdigest() % N_SHARDS
    return shard
