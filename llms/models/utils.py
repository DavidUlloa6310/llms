import flax
import flax.serialization as flax_serialization


def save_model(state, path):
    state_dict = flax.serialization.to_state_dict(state)
    with open(path, "wb") as f:
        f.write(flax_serialization.msgpack_serialize(state_dict))


def load_model(state, path):
    with open(path, "rb") as f:
        loaded_state_dict = flax_serialization.msgpack_restore(f.read())
    return flax.serialization.from_state_dict(state, loaded_state_dict)