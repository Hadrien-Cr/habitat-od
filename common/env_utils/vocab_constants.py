import csv
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
        ctype=1: completely random
        ctype=2: red random
        ctype=3: blue random
        ctype=4: green random
        ctype=5: yellow random
        """
        if ctype == 1:
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


CLASS_LABELS_HSSD500 = [
    "air_conditioner", "air_hockey_table", "alarm_clock", 
    "aquarium", "armchair", "armoire", "armrest", "ashcan", "ashtray", 
    "audio_system", "bag", "balcony", "ball", "ball_chair", "bandsaw", 
    "bar", "bar_stool", "barbecue", "barrow", "base_cabinet", 
    "basket", "bath_mat", "bath_towel", "bathrobe", "bathroom_scale", 
    "bathtub", "beam", "beanbag_chair", "bed", "bedclothes", "bench", "bench_grinder", 
    "bi-fold_door", "bicycle", "bicycle_rack", "bidet", "binder", "birdcage", 
    "birdhouse", "blackboard", "blanket", "blanket_chest", 
    "blender", "board_game", "bolster", "book", "bookcase", "bookend", 
    "bottle", "bottle_opener", "bouquet", "bowl", "bread-bin", "bridge", 
    "broom", "bucket", "buffet", "bulletin_board", "bunk_bed", "butter_dish", 
    "cabin", "cabinet", "caddy", "cafeteria_tray", "cage", "cake", "cake_stand", 
    "camcorder", "camera", "camp_chair", "candelabrum", "candle", "candlestick", 
    "canister", "car", "carafe", "casserole", "cat", "ceiling_fan", 
    "ceiling_lamp", "cellular_telephone", "chain_saw", "chair", 
    "chaise_longue", "chandelier", "chess", "chest", "chest_of_drawers", 
    "chicken", "china_cabinet", "chopping_board", "christmas_stocking", 
    "christmas_tree", "clock", "clothes_dryer", "clothes_tree", 
    "clothing_rack", "club_chair", "coaster", "coatrack", "cocktail_shaker", 
    "coffee_maker", "coffee_table", "coffeepot", "colander", 
    "computer_screen", "computer_work_area", 
    "conference_table", "console_table", "cooker", "cookie_sheet", 
    "countertop", "cradle", "crate", "credenza", "crib", "cruet", "cup", "curtain", 
    "curtain_rod", "cushion", "dartboard", "darts", "daybed", "desk", "desk_calendar", 
    "desk_organizer", "dining_area", "dining_table", 
    "dish_rack", "dishwasher", "dog", "dollhouse", "door", "doorbell", 
    "doormat", "double_bed", "double_door", "drawer", "drawer_unit", 
    "dresser", "dressing_table", "drum", "drum_set", "dryer", "drying_rack", 
    "dvd_player", "eames_chair", "earphone", "easel", "easy_chair", 
    "electric_fan", "electric_frying_pan", 
    "elevator", "elevator_door", "end_table", "espresso_maker", 
    "exercise_bike", "fan", "faucet", "fence", "file", "fire_extinguisher", 
    "firepit", "fireplace", "flat_bench", "floor_lamp", "floor_mirror", 
    "flower", "flower_in_vase", "folding_chair", "foosball_table", 
    "football", "footstool", "frying_pan", "game_table", "garage_door", 
    "gate", "gazebo", "gift_box", "glass", "globe", "grab_bar", "grandfather_clock", 
    "greenhouse", "greeting_card", "guitar", "gym_equipment", 
    "hall_tree", "hammock", "hamper", "hand_glass", "handcart", "handle", 
    "hanging_cabinet", "hatbox", "headboard", "heating_system", 
    "hedge", "highchair", "hobby", "hook", "horse", "hot_tub", "hourglass", 
    "interior_barn_door", "ironing_board", "jar", "jewelry_box", 
    "jug", "kettle", "king_bed", "kitchen_appliance", 
    "kitchen_cabinet", "kitchen_island", "kitchen_scale", 
    "kitchen_timer", "knife", "knocker", "l-shaped_couch", 
    "ladder", "ladder_bookcase", "lamp", "lantern", "laptop", "laundry_bag", 
    "lawn_mower", "lectern", "letter", "loudspeaker", "luggage_rack", 
    "magazine", "magazine_rack", "magnet", "mailbox", "makeup_mirror", 
    "mantel", "mantel_clock", "mat", "mattress", "measuring_cup", 
    "media_player", "medicine_chest", "microphone", "microwave", 
    "mirror", "mixer", "mixing_bowl", "mobile", "monitor", "motorcycle", 
    "mousepad", "music_stand", "napkin", "net", "nightstand", "notebook", 
    "notepad", "ottoman", "oven", "overnighter", "pan", "paper_organizer", 
    "paper_towel", "paperweight", "parrot", "pathway_light", 
    "pedestal_sink", "pendant_lamp", "penguin", "pepper_mill", 
    "person", "pestle", "pet_bed", "pet_bowl", "pet_house", "piano", "picnic_rug", 
    "picnic_table", "picture_frame", "piggy_bank", "pinball_machine", 
    "pitcher", "place_mat", "place_setting", "plant", "plant_stand", 
    "planter", "plate", "play_area", "playhouse", "playpen", "plaything", 
    "plush_toy", "pond", "pool_table", "postbox", "poster", "potholder", 
    "potted_plant", "power_saw", "printer", "projector", "punch_bowl", 
    "punching_bag", "purse", "quilt", "rack", "radiator", "radio_receiver", 
    "railing", "range_hood", "reamer", "recliner", "record_player", 
    "refrigerator", "revolving_door", "roaster", "rock", "rocking_chair", 
    "roof", "room_divider", "round_daybed", "rug", "safe", "salver", "saucepan", 
    "sauna", "scooter", "screen", "sculpture", "seat_cushion", 
    "serving_cart", "sewing_machine", "shed", "sheep", "shelving", 
    "shoe", "shoe_rack", "shoebox", "shopping_bag", "shot_glass", 
    "shower_caddy", "shower_curtain", "shower_door", "shower_faucet", 
    "shower_pan", "shower_stall", "showerhead", "side_table", 
    "single_bed", "sink", "sink_cabinet", "sink_stand", "skateboard", 
    "slide", "sliding_door", "smoke_detector", "soap", "soap_dish", 
    "soap_dispenser", "soccer_ball", "socket", "soda_can", "sofa", "spade", 
    "spectacles", "spice_holder", "spice_rack", "spicemill", 
    "spoon", "stairway", "step_ladder", "step_stool", "stool", "storage_bench", 
    "storage_box", "stove", "straight_chair", "strainer", "streetlight", 
    "string_lights", "subwoofer", "sugar_bowl", "surfboard", 
    "swimming_pool", "swing", "swing_bench", "swing_chair", 
    "switch", "swivel_chair", "table", "table-tennis_table", 
    "table_lamp", "table_mirror", "table_runner", "tablet_computer", 
    "tapestry", "teapot", "telephone", "telescope", "television_receiver", 
    "tent", "thermos", "throw", "throw_pillow", "timer", "tissue_box", 
    "toaster", "toaster_oven", "toilet", "toilet_bag", "toilet_brush", 
    "toilet_flush_plate", "toilet_paper_holder", 
    "toilet_tissue", "toiletry", "towel", "towel_rack", "towel_rail", 
    "towel_ring", "toy_box", "track_lighting", "trailer", "trampoline", 
    "tray", "treadmill", "tree", "trellis", "trunk", "tumbler", "tureen", "tv_stand", 
    "umbrella", "umbrella_stand", "urinal", "utensil", 
    "vacuum", "valve", "vase", "video_game_console", 
    "videodisk", "wall_art", "wall_calendar", "wall_clock", 
    "wall_decor", "wall_hook", "wall_hook_rack", "wall_lamp", 
    "wall_mirror", "wall_organizer", "wall_panel", "wall_shelf", 
    "wall_shelving", "wall_sign", "wall_socket", "wall_sticker", 
    "wall_unit", "wardrobe", "washbasin", "washer", "water_scooter", 
    "watering_can", "weight", "whiteboard", "wind_chime", "window", 
    "window_blind", "window_curtain", "window_shade", 
    "wine_bottle", "wine_bucket", "wine_rack", "wineglass", 
    "wok", "workbench", "wreath",
]

CLASS_LABELS_HSSD40 = [
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

def generate_mappings(source_vocab: list[str], target_vocab: list[str], threshold: float):
    from common.vision.clip import get_clip_embeddings, cosine_similarity

    source_embeddings = get_clip_embeddings(source_vocab, prompt = "indoor photo of a ") .permute(1,0)
    target_embeddings = get_clip_embeddings(target_vocab, prompt = "indoor photo of a ") .permute(1,0)

    def find_closest(idx_label, source_embeddings, target_embeddings, threshold=0.9):
        source_emb = source_embeddings[idx_label]
        best_label_idx = None
        best_sim = -1

        for i, target_emb in enumerate(target_embeddings):
            sim = cosine_similarity(source_emb, target_emb)
            if sim > best_sim:
                best_sim = sim
                best_label_idx = i

        if best_sim < threshold:
            return None

        return best_label_idx

    mapping_source_to_target = []
    mapping_target_to_source = [[] for _ in target_vocab]

    for i, label in enumerate(source_vocab):
        closest_label_idx = find_closest(i, source_embeddings, target_embeddings, threshold)
        
        if closest_label_idx is None:
            closest_label_idx = -1

        mapping_source_to_target.append(closest_label_idx)
        mapping_target_to_source[closest_label_idx].append(i)

    return mapping_source_to_target, mapping_target_to_source


VOCAB_REGISTRY: list[tuple[str, list[str]]] = [
    ("SCANNET200", CLASS_LABELS_SCANNET200),
    ("NYU40",      CLASS_LABELS_NYU40),
    ("MPCAT40",    CLASS_LABELS_MPCAT40),
    ("COCO80",     CLASS_LABELS_COCO80),
]
OUTPUT_PATH = "common/env_utils/hssd500_cross_vocab_mapping.csv"

def create_cross_vocab_mapping_csv(output_path: str = OUTPUT_PATH) -> None:
    mappings = {
        target_name: generate_mappings(
            CLASS_LABELS_HSSD500, target_labels, threshold=0.87
        )[0]
        for target_name, target_labels in VOCAB_REGISTRY
    }

    columns = ["HSSD500"] + [name for name, _ in VOCAB_REGISTRY]
    
    header = {
        "HSSD500": f"classes ({len(CLASS_LABELS_HSSD500)}): "
                   + " | ".join(CLASS_LABELS_HSSD500),
        **{
            name: f"classes ({len(labels)}): " + " | ".join(labels)
            for name, labels in VOCAB_REGISTRY
        },
    }

    # 4. Mapping rows
    rows = []
    for hssd_idx, hssd_label in enumerate(CLASS_LABELS_HSSD500):
        row = {"HSSD500": hssd_label}
        for target_name, target_labels in VOCAB_REGISTRY:
            target_idx = mappings[target_name][hssd_idx]
            row[target_name] = (
                target_labels[target_idx] if target_idx >= 0 else None
            )
        rows.append(row)

    # 5. Assemble — all DataFrames share the same explicit column list
    df = pd.concat(
        [
            pd.DataFrame([header],                  columns=columns),
            pd.DataFrame([{c: "" for c in columns}], columns=columns),  # blank separator
            pd.DataFrame(rows,                       columns=columns),
        ],
        ignore_index=True,
    )

    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

df = pd.read_csv(OUTPUT_PATH)
df = df.iloc[2:].reset_index(drop=True)

VOCABULARIES = {
    "HSSD500": (CLASS_LABELS_HSSD500, None, make_colors(len(CLASS_LABELS_HSSD500), seed=0, ctype=1)),
}

for name, labels in VOCAB_REGISTRY:
    VOCABULARIES[name] = (labels, df.set_index("HSSD500")[name].fillna("undefined").to_dict(), make_colors(len(labels), seed=0, ctype=1))

for vocab_name, (class_labels, mapping_500_to_target, colors) in VOCABULARIES.items():
    meta = MetadataCatalog.get(vocab_name)
    meta.thing_classes = class_labels
    meta.thing_colors = colors
