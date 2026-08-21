import csv
import os
from typing import Optional
import numpy as np
import pandas as pd
from detectron2.data import MetadataCatalog

def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def make_colors(num, seed=1, ctype=1) -> list[tuple[int,int,int]]:
    """Return `num` number of unique colors in a list,
    where colors are [r,g,b] lists."""
    rng_gen = np.random.default_rng(seed)
    colors = []

    def random_unique_color(colors, ctype, rng_gen):
        """
        ctype=0: random high saturation colors
        ctype=1: completely random
        ctype=2: red random
        ctype=3: blue random
        ctype=4: green random
        ctype=5: yellow random
        """
        if ctype == 0:
            import colorsys
            h = rng_gen.random()
            s = rng_gen.uniform(0.8, 1.0)
            v = rng_gen.uniform(0.7, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color = "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )
        elif ctype == 1:
            color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
            while color in colors:
                color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
        elif ctype == 2:
            color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 4:  # green
            color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 3:  # blue
            color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 5:  # yellow
            h = rng_gen.integers(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
            while color in colors:
                h = rng_gen.integers(0xAA, 0xFF)
                color = "#%02x%02x00" % (h, h)
        else:
            raise ValueError("Unrecognized color type %s" % (str(ctype)))
        return color

    while len(colors) < num:
        colors.append(hex_to_rgb(random_unique_color(colors,ctype=ctype,rng_gen=rng_gen)))
    return colors

# HSSD-HAB's own native per-object category scheme (hssd-hab_semantic_lexicon.json)
CLASS_LABELS_HSSD400 = [
    "unknown", "air_conditioner", "air_duct", "animal", "appliance", "aquarium", "arcade_game",
    "art_frame", "art_stand", "awning", "baby_changing_station", "bag", "balcony",
    "balcony_railing", "bar", "bar_cabinet", "barbecue", "barrel", "basin", "basket",
    "bath_sink", "bathrobe", "bathroom_accessory", "bathroom_shelf", "bathroom_utensil",
    "bathtub", "beam", "bed", "bed_small", "bed_table", "bedframe", "bell", "bench", "bicycle",
    "bidet", "bin", "binder", "blanket", "blinds", "board", "boiler", "book", "book_rack",
    "bottle", "bottles_of_wine", "bowl", "box", "box_of_fruit", "box_of_tissue", "breadbin",
    "bridge", "broom", "brush", "bucket", "bush", "bust", "button", "cabinet", "cake_stand",
    "calendar", "camera", "can", "candle", "car", "carpet", "carpet_roll", "cart", "case",
    "casket", "cat", "ceiling_fan", "chair_stand", "chaise", "chandelier", "chest", "chimney",
    "christmas_tree", "clock", "cloth", "clothes", "clothes_hanger", "clothes_hanger_rod",
    "clothes_rack", "clothing", "coat_hanger", "coffee_machine", "coffee_maker", "computer",
    "computer_desk", "computer_equipment", "container", "cooker", "cooker_unit", "cosmetic",
    "couch", "counter", "cover", "credenza", "crib", "cup", "curtain", "curtain_rail",
    "curtain_rod", "dartboard", "decor", "decoration", "decorative_bowl", "decorative_plate",
    "decorative_quilt", "desk", "desk_and_chairs", "desk_clutter", "dinner_table", "dinnerware",
    "dishwasher", "display_cabinet", "display_table", "dj_table", "dog", "dog_bed", "door",
    "doormat", "drawer", "drawer_desk", "dresser", "dressing_table", "drinkware", "drum",
    "dumbbell", "electric_outlet", "exercise_equipment", "fan", "fence", "fencing",
    "file_cabinet", "fire_dish", "fire_extinguisher", "fire_pit", "fireplace",
    "fireplace_utensil", "fireplace_wall", "firewood_holder", "floor", "floor__outside",
    "floor_mat", "flower", "flower_stand", "flowerbed", "flowerpot", "food", "food_tray",
    "fountain", "frame", "freezer", "fridge", "fruit", "fruit_bowl", "furnace", "garage_door",
    "garden_bower", "gate", "gift", "globe", "grill", "guitar", "gym_equipment", "hammock",
    "hand_wash", "handle", "hanger", "hanging_clothes", "hat", "heater", "hedge", "hose",
    "ironing_board", "jacuzzi", "jar", "jewelry_box", "jug", "kettle", "kitchen_appliance",
    "kitchen_counter", "kitchen_island", "kitchen_lower_cabinet", "kitchen_shelf",
    "kitchen_utensil", "kitchen_wall", "knife", "knife_holder", "knob", "ladder", "lamp",
    "lamp_stand", "lamp_table", "laptop", "lattice", "laundry_basket", "led_tv", "light",
    "light_switch", "liquid_soap", "locker", "machine", "magazine", "mailbox", "media_console",
    "microwave", "mirror", "mixer", "monitor", "mortar", "motorcycle", "musical_instrument",
    "newspaper", "newspaper_basket", "nightstand", "object", "object__outside",
    "office_utensil", "oil_lamp", "oven", "oven_vent", "painting", "painting_frame", "pan",
    "panel", "panel_screen", "paper_towel_dispenser", "partition", "perfume", "person", "phone",
    "photo", "photo_mount", "piano", "picture", "picture_frame", "pillar", "pillow",
    "ping_pong_table", "pipe", "pitcher", "plant", "plant_art", "plate", "platform", "platter",
    "playground", "playground_element", "plush_toy", "pool", "pool_table", "post", "pot",
    "printer", "projector", "projector_screen", "rack", "radiator", "radio", "rail", "railing",
    "range_hood", "record_player", "rock", "rocking_horse", "rod", "rolling_cart", "roof",
    "safe", "sauna_heater", "scale", "screen", "sculpture", "seat", "security_camera",
    "semi_chair", "set_of_armchairs", "sewing_machine", "shack", "shampoo", "shelf",
    "shelf_cabinet", "shirt", "shoe", "shoes", "shovel", "shower", "shower_door",
    "shower_glass", "shower_hose", "shower_mat", "shower_soap_shelf", "shower_tap",
    "shower_wall", "showerhead", "sink", "sink_basin", "sink_cabinet", "sitting_area",
    "skateboard", "skirting_board", "slide", "sliding_door", "smoke_alarm", "soap",
    "soap_bottle", "soap_dish", "soap_dispenser", "soap_dispenser_shelf_in_shower", "soapbox",
    "socket", "sofa_set", "speaker", "spice_rack", "stack_of_papers", "stair_step", "staircase",
    "stairs", "stand", "statue", "step", "stone", "stool", "storage", "storage_box",
    "storage_cabinet", "storage_space", "stove", "stovetop", "stuffed_animal", "sunbed",
    "swimming_pool", "swing", "switch", "table", "tablecloth", "tablet", "tap", "teapot",
    "telescope", "tent", "thermostat", "throw_blanket", "tile", "tissue", "tissue_box",
    "toaster", "toilet", "toilet_bin", "toilet_brush", "toilet_brush_holder", "toilet_cleaner",
    "toilet_paper", "toilet_sink", "toilet_stall", "toiletry", "tool", "towel", "towel_basket",
    "towel_holder", "towel_rack", "towel_ring", "toy", "tray", "treadmill", "tree", "tripod",
    "tv", "tv_stand", "umbrella", "urinal", "utensil", "vacuum_cleaner", "vanity", "vase",
    "vent", "ventilation_hood", "wall", "wall__outside", "wall_board", "wall_cubby",
    "wall_desk", "wall_panel", "wardrobe", "wash_cabinet", "washbasin", "washer_dryer",
    "washing_machine_and_dryer", "water_dispenser", "watering_can", "weight", "wheelbarrow",
    "window", "window_frame", "window_shade", "window_shutter", "wine_cabinet", "wine_rack",
    "wood", "wooden_house", "workout_bike", "workstation", "wreath", "yard",
]

CLASS_LABELS_HSSD80 = [
    "alarm_clock", "animal", "bathtub", "bed", "bench", "bicycle", "blender", "book",
    "bottle", "bowl", "breadbin", "cabinet", "camera", "candle", "car", "carpet", 
    "ceiling_lamp", "chair", "chest_of_drawers", "clothing", "coffee_maker", "colander", 
    "couch", "counter", "curtain", "cushion", "dishwasher", "door", "drinkware", "earphone", 
    "exercise_bike", "eyeglasses", "filing_cabinet", "fireplace", "floor_lamp", "flower", "fridge", 
    "grandfather_clock", "kettle", "kitchen_scale", "laptop", "mantel_clock", "microwave", "mirror", 
    "mixing_bowl", "mobile_phone", "motorcycle", "oven", "pan", "pathway_light", "person", "phone", 
    "picture", "picture_frame", "plate", "plush_toy", "pot", "potted_plant", "printer", "range_hood", 
    "shelves", "shoes", "shower", "sink", "soap_dispenser", "spicemill", "stand", "stool", "table", 
    "table_lamp", "teapot", "toaster", "toilet", "toiletry", "trashcan", "tray", "treadmill", 
    "tree", "tv", "vase", "wall_clock", "wall_lamp", "wardrobe", "washer_dryer", "window"
]

CLASS_LABELS_SCANNET200 = [
    "alarm clock", "armchair", "backpack", "bag", "ball", "bar", "basket", 
    "bathroom cabinet", "bathroom counter", "bathroom stall", 
    "bathroom stall door", "bathroom vanity", 
    "bathtub", "bed", "bench", "bicycle", "bin", "blackboard", "blanket", 
    "blinds", "board", "book", "bookshelf", "bottle", "bowl", "box", "broom", "bucket", 
    "bulletin board", "cabinet", "calendar", "candle", "cart", "case of water bottles", 
    "cd case", "ceiling", "ceiling light", "chair", "clock", "closet", 
    "closet door", "closet rod", "closet wall", "clothes", "clothes dryer", 
    "coat rack", "coffee kettle", "coffee maker", "coffee table", 
    "column", "computer tower", "container", "copier", "couch", "counter", 
    "crate", "cup", "curtain", "cushion", "decoration", "desk", "dining table", 
    "dish rack", "dishwasher", "divider", "door", "doorframe", "dresser", 
    "dumbbell", "dustpan", "end table", "fan", "file cabinet", 
    "fire alarm", "fire extinguisher", "fireplace", "floor", "folded chair", 
    "furniture", "guitar", "guitar case", "hair dryer", "handicap bar", 
    "hat", "headphones", "ironing board", "jacket", "keyboard", "keyboard piano", 
    "kitchen cabinet", "kitchen counter", "ladder", "lamp", "laptop", 
    "laundry basket", "laundry detergent", "laundry hamper", 
    "ledge", "light", "light switch", "luggage", "machine", "mailbox", 
    "mat", "mattress", "microwave", "mini fridge", "mirror", "monitor", 
    "mouse", "music stand", "nightstand", "object", "office chair", 
    "ottoman", "oven", "paper", "paper bag", "paper cutter", 
    "paper towel dispenser", "paper towel roll", 
    "person", "piano", "picture", "pillar", "pillow", "pipe", "plant", "plate", 
    "plunger", "poster", "potted plant", "power outlet", 
    "power strip", "printer", "projector", "projector screen", 
    "purse", "rack", "radiator", "rail", "range hood", "recycling bin", 
    "refrigerator", "scale", "seat", "shelf", "shoe", "shower", "shower curtain", 
    "shower curtain rod", "shower door", "shower floor", 
    "shower head", "shower wall", "sign", "sink", "soap dish", "soap dispenser", 
    "sofa chair", "speaker", "stair rail", "stairs", "stand", "stool", "storage bin", 
    "storage container", "storage organizer", 
    "stove", "structure", "stuffed animal", "suitcase", "table", "telephone", 
    "tissue box", "toaster", "toaster oven", "toilet", "toilet paper", 
    "toilet paper dispenser", "toilet paper holder", 
    "toilet seat cover dispenser", 
    "towel", "trash bin", "trash can", "tray", "tube", "tv", "tv stand", 
    "vacuum cleaner", "vent", "wall", "wardrobe", "washing machine", 
    "water bottle", "water cooler", "water pitcher", 
    "whiteboard", "window", "windowsill"
]

CLASS_LABELS_NYU40 = [
    "bag", "bathtub", "bed", "blinds", "books", "bookshelf", "box", "cabinet", 
    "ceiling", "chair", "clothes", "counter", "curtain", "desk", "door", "dresser", 
    "floor", "floor_mat", "lamp", "mirror", "night_stand", "otherfurniture", 
    "otherprop", "otherstructure", "paper", "picture", "pillow", "refridgerator", 
    "shelves", "shower_curtain", "sink", "sofa", "table", "television", 
    "toilet", "towel","wall", "whiteboardperson", "window", 
]

CLASS_LABELS_MPCAT40 = [
    "appliances", "bathtub", "beam", "bed", "blinds", "board_panel", 
    "cabinet", "ceiling", "chair", "chest_of_drawers", 
    "clothes", "column", "counter", "curtain", "cushion", "door", "fireplace", 
    "floor", "furniture", "gym_equipment", "lighting", "mirror", "misc", 
    "objects", "picture", "plant", "railing", "seating", "shelving", 
    "shower", "sink", "sofa", "stairs", "stool", "table", "toilet", "towel", "tv_monitor", 
    "wall", "window", 
]

CLASS_LABELS_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

def generate_mappings(source_vocab: list[str], target_vocab: list[str]):
    import sys
    import torch

    detic_root = os.path.join(os.environ["BASE_DIR"], "third_party/Detic")
    sys.path.insert(0, detic_root)
    from detic.modeling.text.text_encoder import build_text_encoder  # type: ignore

    def get_clip_embeddings(texts: list[str], text_encoder) -> torch.Tensor:
        return text_encoder(texts).detach().permute(1, 0).contiguous().cpu()

    def cosine_similarity(vec1, vec2):
        return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()

    prompts = ["a {} can be seen"]

    def ensemble_embeddings(vocab):
        all_embs = [get_clip_embeddings([prompt.format(c.replace("_", " ")) for c in vocab], text_encoder).permute(1, 0) for prompt in prompts]
        stacked = torch.stack(all_embs, dim=0)
        emb = torch.mean(stacked, dim=0)
        return emb / torch.norm(emb, dim=1, keepdim=True)

    source_embeddings = ensemble_embeddings(source_vocab)
    target_embeddings = ensemble_embeddings(target_vocab)

    def find_closest(idx_label, source_embeddings, target_embeddings):
        source_emb = source_embeddings[idx_label]
        best_label_idx = None
        best_sim = -1

        for i, target_emb in enumerate(target_embeddings):
            sim = cosine_similarity(source_emb, target_emb)
            if sim > best_sim:
                best_sim = sim.item()
                best_label_idx = i

        return best_label_idx, best_sim

    mapping_source_to_target = []

    for i in range(len(source_vocab)):
        closest_label_idx, closest_label_proximity = find_closest(i, source_embeddings, target_embeddings)
        mapping_source_to_target.append((closest_label_idx, closest_label_proximity))

    return mapping_source_to_target


HSSD400_VOCAB_REGISTRY: list[tuple[str, list[str]]] = [
    ("HSSD80",     CLASS_LABELS_HSSD80),
    ("SCANNET200", CLASS_LABELS_SCANNET200),
    ("NYU40",      CLASS_LABELS_NYU40),
    ("MPCAT40",    CLASS_LABELS_MPCAT40),
    ("COCO80",     CLASS_LABELS_COCO80),
]
BASE_DIR = os.environ["BASE_DIR"]
HSSD400_MAPPING_OUTPUT_PATH = os.path.join(BASE_DIR, "common", "env_utils", "hssd400_cross_vocab_mapping.csv")
THRESHOLD = 0.875
AUTO_ACCEPT_THRESHOLD = 0.925  # >= this, the match is confident enough to auto-accept

def create_hssd400_cross_vocab_mapping_csv(output_path: str = HSSD400_MAPPING_OUTPUT_PATH) -> None:
    mappings = {
        target_name: generate_mappings(CLASS_LABELS_HSSD400, target_labels)
        for target_name, target_labels in HSSD400_VOCAB_REGISTRY
    }

    columns = ["HSSD400"] + [
        y for name, _ in HSSD400_VOCAB_REGISTRY for y in [name, name + "_proximity", name + "_reject"]
    ]
    header = {
        "HSSD400": f"classes ({len(CLASS_LABELS_HSSD400)}): " + " | ".join(CLASS_LABELS_HSSD400),
        **{name: f"classes ({len(labels)}): " + " | ".join(labels) for name, labels in HSSD400_VOCAB_REGISTRY},
    }

    rows = []
    for idx, label in enumerate(CLASS_LABELS_HSSD400):
        row = {"HSSD400": label}
        for target_name, target_labels in HSSD400_VOCAB_REGISTRY:
            target_idx, target_prox = mappings[target_name][idx]
            row[target_name] = target_labels[target_idx]
            row[target_name + "_proximity"] = str(round(target_prox, 4))
            # True below THRESHOLD (clear reject), False above AUTO_ACCEPT_THRESHOLD
            # (clear accept), blank in between - fill those in by hand one by one.
            if target_prox < THRESHOLD:
                row[target_name + "_reject"] = True
            elif target_prox > AUTO_ACCEPT_THRESHOLD:
                row[target_name + "_reject"] = False
            else:
                row[target_name + "_reject"] = ""
        rows.append(row)

    df = pd.concat(
        [
            pd.DataFrame([header],                  columns=columns),
            pd.DataFrame([{c: "" for c in columns}], columns=columns),  # blank separator
            pd.DataFrame(rows,                       columns=columns),
        ],
        ignore_index=True,
    )
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    create_hssd400_cross_vocab_mapping_csv(HSSD400_MAPPING_OUTPUT_PATH)
    
_df = pd.read_csv(HSSD400_MAPPING_OUTPUT_PATH).iloc[2:].reset_index(drop=True)

# register the vocabularies in detectron2's MetadataCatalog
HSSD400_TO_VOCAB: dict[str, dict[str, str]] = {}

for vocab_name, _ in HSSD400_VOCAB_REGISTRY:
    mapping = _df.set_index("HSSD400")[vocab_name].fillna("unknown").to_dict()

    reject = _df.set_index("HSSD400")[vocab_name + "_reject"].to_dict()
    HSSD400_TO_VOCAB[vocab_name] = {
        k: (v if str(reject.get(k)).strip() == "False" else "unknown") for k, v in mapping.items()
    }

VOCABULARIES: dict[str, tuple[list[str], Optional[dict[str, str]], list[tuple[int, int, int]]]] = {
    "HSSD400": (CLASS_LABELS_HSSD400, None, make_colors(len(CLASS_LABELS_HSSD400), seed=0, ctype=0)),
}
for vocab_name, labels in HSSD400_VOCAB_REGISTRY:
    VOCABULARIES[vocab_name] = (labels, HSSD400_TO_VOCAB[vocab_name], make_colors(len(labels), seed=0, ctype=0))

for vocab_name, (class_labels, _, colors) in VOCABULARIES.items():
    meta = MetadataCatalog.get(vocab_name)
    meta.thing_classes = class_labels
    meta.thing_colors = colors
