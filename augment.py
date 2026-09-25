import itertools

NO_AUGMENT = {
    "delay": 0,
    "pitch": 0,
    "volume": 1,
}

ALL_VARIANTS = {
    "delay": [0, 0.25, 0.5, 0.75, 1],
    "volume": [0.1, 0.16, 0.25, 0.40, 0.63, 1],
    "pitch": [0, 1, -1, 2, -2, 3, -3],
}

BEST_VARIANTS = {
    "delay": [0, 0.25, 0.5, 0.75],
    "volume": [0.25, 0.40, 0.63, 1],
    "pitch": [0, 1, -1, 2, -2],
}

def get_augment_meshgrid(variant_dict=ALL_VARIANTS):

    keys = variant_dict.keys()
    variations = [dict(zip(keys, values)) for values in itertools.product(*variant_dict.values())]

    return variations

def is_variant(file, variant: dict[str, float | int]):
    return f"d={variant['delay']}_v={variant['volume']}_p={variant['pitch']}" in file

def is_no_augment(file):
    return is_variant(file, NO_AUGMENT)

def in_variant_mesh(file, variant_mesh):
    return any([is_variant(file, variant) for variant in variant_mesh])

def test_augment():

    variants = [
        {
            "name": "aug_delay_0.5s",
            "delay": [0, 0.5],
            "volume": [1],
            "pitch": [0],
        },
        {
            "name": "aug_delay_0.75s",
            "delay": [0, 0.75],
            "volume": [1],
            "pitch": [0],
        },
        {
            "name": "aug_delay_cumulate_0.5s",
            "delay": [0, 0.25, 0.5],
            "volume": [1],
            "pitch": [0],
        },
        {
            "name": "aug_delay_cumulate_0.75s",
            "delay": [0, 0.25, 0.5, 0.75],
            "volume": [1],
            "pitch": [0],
        }
    ]

    import train

    for variant in variants:

        filtered_variant = {k: variant[k] for k in variant.keys() - {"name"}}
        aug_mesh = get_augment_meshgrid(filtered_variant)

        results = train.trial(
            n_datasets=None, augment_data=True, adjust_target=True,
            augment_mesh=aug_mesh, trials=5
        )

        train.save_results(results, variant["name"])

if __name__ == "__main__":

    test_augment()
