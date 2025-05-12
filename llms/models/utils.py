import flax
import flax.serialization as flax_serialization


def save_model(state, path):
    state_dict = flax.serialization.to_state_dict(state)
    with open(path, "wb") as f:
        f.write(flax_serialization.msgpack_serialize(state_dict))