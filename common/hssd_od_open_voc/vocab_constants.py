from enum import Enum


CLASS_LABELS_HSSD500 = [
    'air_conditioner', 'air_hockey_table', 'alarm_clock', 
    'aquarium', 'armchair', 'armoire', 'armrest', 'ashcan', 'ashtray', 
    'audio_system', 'bag', 'balcony', 'ball', 'ball_chair', 'bandsaw', 
    'bar', 'bar_stool', 'barbecue', 'barrow', 'base_cabinet', 
    'basket', 'bath_mat', 'bath_towel', 'bathrobe', 'bathroom_scale', 
    'bathtub', 'beam', 'beanbag_chair', 'bed', 'bedclothes', 'bench', 'bench_grinder', 
    'bi-fold_door', 'bicycle', 'bicycle_rack', 'bidet', 'binder', 'birdcage', 
    'birdhouse', 'blackboard', 'blanket', 'blanket_chest', 
    'blender', 'board_game', 'bolster', 'book', 'bookcase', 'bookend', 
    'bottle', 'bottle_opener', 'bouquet', 'bowl', 'bread-bin', 'bridge', 
    'broom', 'bucket', 'buffet', 'bulletin_board', 'bunk_bed', 'butter_dish', 
    'cabin', 'cabinet', 'caddy', 'cafeteria_tray', 'cage', 'cake', 'cake_stand', 
    'camcorder', 'camera', 'camp_chair', 'candelabrum', 'candle', 'candlestick', 
    'canister', 'car', 'carafe', 'casserole', 'cat', 'ceiling_fan', 
    'ceiling_lamp', 'cellular_telephone', 'chain_saw', 'chair', 
    'chaise_longue', 'chandelier', 'chess', 'chest', 'chest_of_drawers', 
    'chicken', 'china_cabinet', 'chopping_board', 'christmas_stocking', 
    'christmas_tree', 'clock', 'clothes_dryer', 'clothes_tree', 
    'clothing_rack', 'club_chair', 'coaster', 'coatrack', 'cocktail_shaker', 
    'coffee_maker', 'coffee_table', 'coffeepot', 'colander', 
    'computer_screen', 'computer_work_area', 
    'conference_table', 'console_table', 'cooker', 'cookie_sheet', 
    'countertop', 'cradle', 'crate', 'credenza', 'crib', 'cruet', 'cup', 'curtain', 
    'curtain_rod', 'cushion', 'dartboard', 'darts', 'daybed', 'desk', 'desk_calendar', 
    'desk_organizer', 'dining_area', 'dining_table', 
    'dish_rack', 'dishwasher', 'dog', 'dollhouse', 'door', 'doorbell', 
    'doormat', 'double_bed', 'double_door', 'drawer', 'drawer_unit', 
    'dresser', 'dressing_table', 'drum', 'drum_set', 'dryer', 'drying_rack', 
    'dvd_player', 'eames_chair', 'earphone', 'easel', 'easy_chair', 
    'electric_fan', 'electric_frying_pan', 
    'elevator', 'elevator_door', 'end_table', 'espresso_maker', 
    'exercise_bike', 'fan', 'faucet', 'fence', 'file', 'fire_extinguisher', 
    'firepit', 'fireplace', 'flat_bench', 'floor_lamp', 'floor_mirror', 
    'flower', 'flower_in_vase', 'folding_chair', 'foosball_table', 
    'football', 'footstool', 'frying_pan', 'game_table', 'garage_door', 
    'gate', 'gazebo', 'gift_box', 'glass', 'globe', 'grab_bar', 'grandfather_clock', 
    'greenhouse', 'greeting_card', 'guitar', 'gym_equipment', 
    'hall_tree', 'hammock', 'hamper', 'hand_glass', 'handcart', 'handle', 
    'hanging_cabinet', 'hatbox', 'headboard', 'heating_system', 
    'hedge', 'highchair', 'hobby', 'hook', 'horse', 'hot_tub', 'hourglass', 
    'interior_barn_door', 'ironing_board', 'jar', 'jewelry_box', 
    'jug', 'kettle', 'king_bed', 'kitchen_appliance', 
    'kitchen_cabinet', 'kitchen_island', 'kitchen_scale', 
    'kitchen_timer', 'knife', 'knocker', 'l-shaped_couch', 
    'ladder', 'ladder_bookcase', 'lamp', 'lantern', 'laptop', 'laundry_bag', 
    'lawn_mower', 'lectern', 'letter', 'loudspeaker', 'luggage_rack', 
    'magazine', 'magazine_rack', 'magnet', 'mailbox', 'makeup_mirror', 
    'mantel', 'mantel_clock', 'mat', 'mattress', 'measuring_cup', 
    'media_player', 'medicine_chest', 'microphone', 'microwave', 
    'mirror', 'mixer', 'mixing_bowl', 'mobile', 'monitor', 'motorcycle', 
    'mousepad', 'music_stand', 'napkin', 'net', 'nightstand', 'notebook', 
    'notepad', 'ottoman', 'oven', 'overnighter', 'pan', 'paper_organizer', 
    'paper_towel', 'paperweight', 'parrot', 'pathway_light', 
    'pedestal_sink', 'pendant_lamp', 'penguin', 'pepper_mill', 
    'person', 'pestle', 'pet_bed', 'pet_bowl', 'pet_house', 'piano', 'picnic_rug', 
    'picnic_table', 'picture_frame', 'piggy_bank', 'pinball_machine', 
    'pitcher', 'place_mat', 'place_setting', 'plant', 'plant_stand', 
    'planter', 'plate', 'play_area', 'playhouse', 'playpen', 'plaything', 
    'plush_toy', 'pond', 'pool_table', 'postbox', 'poster', 'potholder', 
    'potted_plant', 'power_saw', 'printer', 'projector', 'punch_bowl', 
    'punching_bag', 'purse', 'quilt', 'rack', 'radiator', 'radio_receiver', 
    'railing', 'range_hood', 'reamer', 'recliner', 'record_player', 
    'refrigerator', 'revolving_door', 'roaster', 'rock', 'rocking_chair', 
    'roof', 'room_divider', 'round_daybed', 'rug', 'safe', 'salver', 'saucepan', 
    'sauna', 'scooter', 'screen', 'sculpture', 'seat_cushion', 
    'serving_cart', 'sewing_machine', 'shed', 'sheep', 'shelving', 
    'shoe', 'shoe_rack', 'shoebox', 'shopping_bag', 'shot_glass', 
    'shower_caddy', 'shower_curtain', 'shower_door', 'shower_faucet', 
    'shower_pan', 'shower_stall', 'showerhead', 'side_table', 
    'single_bed', 'sink', 'sink_cabinet', 'sink_stand', 'skateboard', 
    'slide', 'sliding_door', 'smoke_detector', 'soap', 'soap_dish', 
    'soap_dispenser', 'soccer_ball', 'socket', 'soda_can', 'sofa', 'spade', 
    'spectacles', 'spice_holder', 'spice_rack', 'spicemill', 
    'spoon', 'stairway', 'step_ladder', 'step_stool', 'stool', 'storage_bench', 
    'storage_box', 'stove', 'straight_chair', 'strainer', 'streetlight', 
    'string_lights', 'subwoofer', 'sugar_bowl', 'surfboard', 
    'swimming_pool', 'swing', 'swing_bench', 'swing_chair', 
    'switch', 'swivel_chair', 'table', 'table-tennis_table', 
    'table_lamp', 'table_mirror', 'table_runner', 'tablet_computer', 
    'tapestry', 'teapot', 'telephone', 'telescope', 'television_receiver', 
    'tent', 'thermos', 'throw', 'throw_pillow', 'timer', 'tissue_box', 
    'toaster', 'toaster_oven', 'toilet', 'toilet_bag', 'toilet_brush', 
    'toilet_flush_plate', 'toilet_paper_holder', 
    'toilet_tissue', 'toiletry', 'towel', 'towel_rack', 'towel_rail', 
    'towel_ring', 'toy_box', 'track_lighting', 'trailer', 'trampoline', 
    'tray', 'treadmill', 'tree', 'trellis', 'trunk', 'tumbler', 'tureen', 'tv_stand', 
    'umbrella', 'umbrella_stand', 'unknown', 'urinal', 'utensil', 
    'vacuum', 'valve', 'vase', 'video_game_console', 
    'videodisk', 'wall_art', 'wall_calendar', 'wall_clock', 
    'wall_decor', 'wall_hook', 'wall_hook_rack', 'wall_lamp', 
    'wall_mirror', 'wall_organizer', 'wall_panel', 'wall_shelf', 
    'wall_shelving', 'wall_sign', 'wall_socket', 'wall_sticker', 
    'wall_unit', 'wardrobe', 'washbasin', 'washer', 'water_scooter', 
    'watering_can', 'weight', 'whiteboard', 'wind_chime', 'window', 
    'window_blind', 'window_curtain', 'window_shade', 
    'wine_bottle', 'wine_bucket', 'wine_rack', 'wineglass', 
    'wok', 'workbench', 'wreath',
]

CLASS_LABELS_HSSD40 = [
    'alarm_clock', 'animal', 'bathtub', 'bed', 'bench', 'bicycle', 'blender', 'book', 
    'bottle', 'bowl', 'breadbin', 'cabinet', 'camera', 'candle', 'car', 'carpet', 
    'ceiling_lamp', 'chair', 'chest_of_drawers', 'clothing', 'coffee_maker', 'colander', 
    'couch', 'counter', 'curtain', 'cushion', 'dishwasher', 'door', 'drinkware', 'earphone', 
    'exercise_bike', 'eyeglasses', 'filing_cabinet', 'fireplace', 'floor_lamp', 'flower', 'fridge', 
    'grandfather_clock', 'kettle', 'kitchen_scale', 'laptop', 'mantel_clock', 'microwave', 'mirror', 
    'mixing_bowl', 'mobile_phone', 'motorcycle', 'oven', 'pan', 'pathway_light', 'person', 'phone', 
    'picture', 'picture_frame', 'plate', 'plush_toy', 'pot', 'potted_plant', 'printer', 'range_hood', 
    'shelves', 'shoes', 'shower', 'sink', 'soap_dispenser', 'spicemill', 'stand', 'stool', 'table', 
    'table_lamp', 'teapot', 'toaster', 'toilet', 'toiletry', 'trashcan', 'tray', 'treadmill', 
    'tree', 'tv', 'unknown', 'vase', 'wall_clock', 'wall_lamp', 'wardrobe', 'washer_dryer', 'window'
]

CLASS_LABELS_SCANNET200 = [
    'alarm clock', 'armchair', 'backpack', 'bag', 'ball', 'bar', 'basket', 
    'bathroom cabinet', 'bathroom counter', 'bathroom stall', 
    'bathroom stall door', 'bathroom vanity', 
    'bathtub', 'bed', 'bench', 'bicycle', 'bin', 'blackboard', 'blanket', 
    'blinds', 'board', 'book', 'bookshelf', 'bottle', 'bowl', 'box', 'broom', 'bucket', 
    'bulletin board', 'cabinet', 'calendar', 'candle', 'cart', 'case of water bottles', 
    'cd case', 'ceiling', 'ceiling light', 'chair', 'clock', 'closet', 
    'closet door', 'closet rod', 'closet wall', 'clothes', 'clothes dryer', 
    'coat rack', 'coffee kettle', 'coffee maker', 'coffee table', 
    'column', 'computer tower', 'container', 'copier', 'couch', 'counter', 
    'crate', 'cup', 'curtain', 'cushion', 'decoration', 'desk', 'dining table', 
    'dish rack', 'dishwasher', 'divider', 'door', 'doorframe', 'dresser', 
    'dumbbell', 'dustpan', 'end table', 'fan', 'file cabinet', 
    'fire alarm', 'fire extinguisher', 'fireplace', 'floor', 'folded chair', 
    'furniture', 'guitar', 'guitar case', 'hair dryer', 'handicap bar', 
    'hat', 'headphones', 'ironing board', 'jacket', 'keyboard', 'keyboard piano', 
    'kitchen cabinet', 'kitchen counter', 'ladder', 'lamp', 'laptop', 
    'laundry basket', 'laundry detergent', 'laundry hamper', 
    'ledge', 'light', 'light switch', 'luggage', 'machine', 'mailbox', 
    'mat', 'mattress', 'microwave', 'mini fridge', 'mirror', 'monitor', 
    'mouse', 'music stand', 'nightstand', 'object', 'office chair', 
    'ottoman', 'oven', 'paper', 'paper bag', 'paper cutter', 
    'paper towel dispenser', 'paper towel roll', 
    'person', 'piano', 'picture', 'pillar', 'pillow', 'pipe', 'plant', 'plate', 
    'plunger', 'poster', 'potted plant', 'power outlet', 
    'power strip', 'printer', 'projector', 'projector screen', 
    'purse', 'rack', 'radiator', 'rail', 'range hood', 'recycling bin', 
    'refrigerator', 'scale', 'seat', 'shelf', 'shoe', 'shower', 'shower curtain', 
    'shower curtain rod', 'shower door', 'shower floor', 
    'shower head', 'shower wall', 'sign', 'sink', 'soap dish', 'soap dispenser', 
    'sofa chair', 'speaker', 'stair rail', 'stairs', 'stand', 'stool', 'storage bin', 
    'storage container', 'storage organizer', 
    'stove', 'structure', 'stuffed animal', 'suitcase', 'table', 'telephone', 
    'tissue box', 'toaster', 'toaster oven', 'toilet', 'toilet paper', 
    'toilet paper dispenser', 'toilet paper holder', 
    'toilet seat cover dispenser', 
    'towel', 'trash bin', 'trash can', 'tray', 'tube', 'tv', 'tv stand', 
    'unknown', 'vacuum cleaner', 'vent', 'wall', 'wardrobe', 'washing machine', 
    'water bottle', 'water cooler', 'water pitcher', 
    'whiteboard', 'window', 'windowsill'
]
MAPPING_HSSD500_TO_SCANNET200 = [     
    191, 189, 0, 189, 1, 193, 145, 112, 24, 160, 3, 189, 4, 37, 189, 5, 164, 123, 32, 29, 6, 
    189, 182, 189, 144, 12, 98, 189, 13, 13, 14, 189, 189, 15, 15, 177, 21, 189, 189, 17, 
    18, 18, 47, 20, 125, 21, 22, 22, 23, 23, 127, 24, 16, 169, 26, 27, 123, 28, 13, 157, 29, 
    29, 189, 185, 169, 128, 189, 189, 123, 37, 189, 31, 31, 51, 101, 197, 189, 109, 35, 
    36, 173, 189, 37, 37, 36, 123, 25, 67, 121, 29, 189, 189, 189, 38, 44, 43, 43, 37, 128, 
    45, 189, 47, 48, 46, 189, 187, 60, 172, 70, 115, 189, 90, 123, 55, 189, 123, 123, 56, 
    57, 57, 58, 123, 123, 13, 60, 30, 167, 61, 61, 62, 63, 123, 189, 65, 65, 189, 13, 65, 67, 
    67, 67, 60, 24, 189, 44, 189, 187, 189, 84, 110, 37, 71, 189, 189, 65, 70, 47, 15, 71, 
    156, 189, 123, 74, 75, 75, 14, 92, 107, 127, 189, 77, 189, 4, 164, 123, 172, 65, 65, 
    189, 25, 56, 4, 189, 38, 189, 123, 79, 101, 189, 189, 6, 189, 32, 126, 29, 83, 13, 189, 
    189, 37, 123, 123, 101, 12, 38, 189, 85, 23, 25, 197, 46, 13, 90, 89, 90, 144, 90, 123, 
    189, 53, 91, 91, 92, 92, 93, 3, 189, 110, 116, 160, 189, 21, 21, 101, 102, 107, 146, 
    38, 103, 104, 56, 123, 123, 160, 105, 107, 101, 24, 101, 108, 15, 189, 110, 116, 
    103, 111, 21, 116, 114, 115, 123, 123, 167, 178, 189, 127, 189, 156, 36, 189, 
    189, 121, 189, 13, 24, 123, 122, 189, 172, 123, 189, 189, 197, 128, 61, 127, 127, 
    127, 128, 123, 169, 123, 123, 170, 189, 172, 102, 130, 189, 131, 123, 134, 135, 
    24, 189, 137, 18, 138, 139, 189, 161, 141, 189, 1, 21, 143, 65, 101, 4, 37, 169, 64, 
    13, 76, 123, 128, 24, 189, 189, 123, 59, 58, 32, 189, 169, 109, 146, 147, 147, 25, 3, 
    56, 148, 149, 151, 148, 148, 148, 153, 70, 13, 156, 156, 156, 20, 123, 65, 73, 157, 
    157, 158, 4, 132, 56, 53, 26, 189, 189, 189, 123, 24, 162, 91, 164, 164, 14, 25, 168, 
    37, 189, 98, 98, 160, 24, 20, 189, 189, 14, 37, 123, 37, 172, 189, 92, 107, 172, 93, 
    189, 46, 173, 189, 187, 3, 195, 4, 125, 38, 174, 175, 176, 177, 3, 177, 177, 180, 
    178, 177, 182, 189, 189, 189, 25, 98, 123, 189, 185, 189, 127, 189, 189, 56, 24, 
    188, 189, 189, 189, 177, 112, 190, 126, 23, 187, 187, 189, 30, 38, 189, 189, 189, 
    92, 107, 189, 189, 146, 146, 189, 132, 189, 22, 193, 156, 194, 189, 27, 101, 198, 
    189, 199, 19, 57, 199, 23, 27, 189, 56, 189, 60, 189, 
]
MAPPING_HSSD40_TO_SCANNET200 = [
    0, 101, 12, 13, 14, 15, 189, 21, 23, 24, 189, 29, 189, 31, 189, 189, 36, 37, 189, 43, 47, 
    189, 53, 54, 57, 58, 63, 65, 56, 84, 189, 189, 72, 75, 189, 127, 143, 189, 46, 189, 93, 
    38, 105, 107, 24, 173, 15, 115, 189, 189, 121, 173, 123, 189, 128, 189, 24, 131, 
    134, 141, 146, 147, 148, 156, 158, 189, 163, 164, 172, 189, 189, 175, 177, 177, 
    184, 185, 189, 189, 187, 189, 189, 38, 189, 193, 194, 199, 
]

CLASS_LABELS_NYU40 = [
    'bag', 'bathtub', 'bed', 'blinds', 'books', 'bookshelf', 'box', 'cabinet', 
    'ceiling', 'chair', 'clothes', 'counter', 'curtain', 'desk', 'door', 'dresser', 
    'floor', 'floor_mat', 'lamp', 'mirror', 'night_stand', 'otherfurniture', 
    'otherprop', 'otherstructure', 'paper', 'picture', 'pillow', 'refridgerator', 
    'shelves', 'shower_curtain', 'sink', 'sofa', 'table', 'television', 
    'toilet', 'towel', 'unknown','wall', 'whiteboardperson', 'window', 
]
MAPPING_HSSD500_TO_NYU40 = [
    25, 36, 25, 36, 9, 7, 9, 25, 36, 25, 0, 36, 25, 9, 36, 11, 36, 25, 36, 7, 6, 17, 35, 36, 34, 1, 25, 36, 
    2, 2, 9, 36, 36, 25, 36, 34, 4, 36, 36, 36, 35, 36, 36, 25, 26, 4, 5, 5, 25, 36, 36, 32, 36, 25, 36, 6, 
    25, 36, 2, 36, 7, 7, 36, 32, 36, 25, 36, 36, 25, 9, 36, 18, 36, 36, 25, 36, 36, 25, 8, 8, 25, 36, 9, 9, 
    36, 25, 6, 15, 25, 7, 36, 36, 36, 25, 10, 10, 10, 9, 36, 36, 36, 36, 32, 36, 36, 33, 13, 32, 36, 
    11, 36, 11, 25, 6, 36, 25, 25, 25, 12, 12, 26, 25, 25, 2, 13, 13, 36, 32, 32, 36, 36, 25, 36, 14, 
    14, 17, 2, 14, 15, 15, 15, 13, 25, 36, 36, 36, 33, 36, 36, 36, 9, 36, 36, 36, 14, 32, 36, 36, 25, 
    30, 36, 25, 36, 36, 36, 36, 18, 19, 25, 36, 9, 36, 25, 9, 25, 32, 14, 14, 36, 6, 19, 36, 36, 36, 
    36, 25, 10, 10, 36, 36, 6, 36, 36, 36, 7, 6, 2, 36, 36, 9, 25, 25, 25, 1, 36, 36, 36, 36, 6, 0, 36, 2, 
    36, 7, 36, 36, 36, 25, 36, 31, 28, 5, 18, 18, 33, 0, 36, 36, 24, 33, 36, 4, 4, 25, 6, 19, 28, 36, 25, 
    2, 36, 25, 25, 36, 33, 19, 36, 36, 25, 33, 25, 36, 36, 24, 25, 13, 4, 24, 36, 33, 25, 25, 24, 35, 
    36, 36, 36, 30, 18, 36, 36, 25, 36, 2, 36, 25, 36, 36, 32, 25, 36, 36, 36, 25, 32, 25, 36, 36, 
    25, 25, 6, 25, 25, 25, 36, 32, 36, 25, 36, 36, 25, 24, 33, 36, 36, 0, 36, 36, 36, 36, 36, 36, 36, 
    9, 4, 27, 14, 25, 25, 9, 8, 36, 2, 17, 25, 25, 36, 36, 36, 25, 25, 26, 36, 36, 6, 36, 28, 10, 28, 6, 0, 
    25, 36, 29, 14, 36, 36, 36, 36, 32, 2, 30, 30, 30, 36, 25, 14, 36, 36, 36, 36, 36, 25, 36, 31, 
    36, 36, 36, 36, 25, 36, 36, 36, 36, 9, 13, 6, 36, 9, 36, 18, 36, 36, 25, 36, 36, 36, 36, 9, 25, 9, 
    32, 36, 18, 19, 32, 36, 36, 36, 33, 36, 33, 0, 36, 25, 26, 25, 6, 33, 36, 34, 0, 34, 34, 34, 34, 
    34, 35, 36, 36, 36, 6, 36, 25, 36, 32, 36, 36, 36, 36, 36, 36, 33, 36, 36, 36, 34, 25, 25, 36, 
    36, 33, 4, 36, 36, 36, 36, 36, 36, 18, 19, 36, 36, 28, 28, 36, 36, 36, 5, 10, 30, 10, 36, 36, 25, 
    38, 36, 39, 3, 12, 39, 25, 36, 36, 36, 36, 13, 36, 
]
MAPPING_HSSD40_TO_NYU40 = [
    36, 25, 1, 2, 36, 36, 36, 4, 36, 36, 36, 7, 36, 36, 36, 36, 8, 9, 36, 10, 36, 36, 31, 11, 12, 26, 36, 
    14, 36, 36, 36, 36, 36, 36, 36, 36, 27, 36, 36, 36, 36, 36, 36, 19, 36, 36, 36, 36, 36, 36, 25, 
    36, 25, 36, 36, 36, 36, 36, 36, 36, 28, 36, 36, 30, 36, 36, 36, 36, 32, 36, 36, 36, 34, 34, 36, 
    36, 36, 36, 33, 36, 36, 36, 36, 10, 36, 39, 
]

CLASS_LABELS_MPCAT40 = [
    'appliances', 'bathtub', 'beam', 'bed', 'blinds', 'board_panel', 
    'cabinet', 'ceiling', 'chair', 'chest_of_drawers', 
    'clothes', 'column', 'counter', 'curtain', 'cushion', 'door', 'fireplace', 
    'floor', 'furniture', 'gym_equipment', 'lighting', 'mirror', 'misc', 
    'objects', 'picture', 'plant', 'railing', 'seating', 'shelving', 
    'shower', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'towel', 'tv_monitor', 
    'unknown', 'wall', 'window', 
]
MAPPING_HSSD500_TO_MPCAT40 = [
    22, 38, 24, 38, 8, 6, 27, 22, 38, 24, 24, 26, 22, 8, 38, 12, 33, 24, 38, 6, 38, 38, 36, 38, 35, 1, 2, 
    38, 3, 3, 27, 38, 38, 24, 38, 35, 38, 38, 38, 38, 36, 38, 38, 5, 14, 24, 28, 38, 24, 38, 25, 34, 
    38, 24, 38, 35, 24, 5, 3, 38, 6, 6, 38, 34, 38, 24, 38, 38, 24, 8, 38, 22, 38, 38, 22, 38, 38, 22, 7, 
    7, 24, 38, 8, 8, 38, 24, 10, 9, 22, 6, 38, 38, 38, 24, 10, 10, 10, 8, 38, 38, 38, 38, 34, 38, 38, 37, 
    38, 34, 38, 0, 38, 12, 24, 38, 38, 24, 24, 22, 13, 13, 14, 24, 24, 3, 34, 38, 38, 34, 34, 38, 38, 
    22, 38, 15, 15, 38, 3, 15, 9, 9, 6, 34, 24, 38, 38, 38, 37, 38, 38, 38, 8, 38, 38, 38, 15, 34, 38, 
    19, 24, 30, 38, 24, 38, 16, 16, 38, 38, 21, 25, 38, 8, 38, 24, 33, 22, 34, 15, 15, 38, 24, 21, 
    38, 38, 38, 38, 24, 10, 19, 38, 38, 38, 38, 38, 38, 6, 38, 3, 38, 38, 8, 24, 24, 24, 1, 38, 38, 38, 
    38, 24, 38, 38, 3, 0, 6, 38, 38, 38, 22, 38, 31, 32, 38, 20, 38, 34, 10, 38, 38, 24, 38, 38, 24, 
    38, 24, 38, 21, 16, 38, 24, 3, 38, 24, 24, 38, 0, 21, 38, 38, 24, 37, 22, 38, 38, 38, 24, 34, 24, 
    24, 38, 0, 24, 22, 38, 36, 38, 25, 38, 30, 38, 38, 38, 24, 38, 3, 38, 24, 38, 38, 34, 24, 38, 38, 
    38, 24, 34, 25, 25, 25, 24, 24, 18, 24, 24, 24, 38, 34, 38, 24, 38, 25, 24, 22, 38, 38, 38, 38, 
    38, 38, 38, 38, 26, 38, 38, 8, 24, 0, 15, 24, 24, 8, 7, 38, 3, 17, 24, 24, 38, 38, 38, 24, 24, 14, 
    38, 38, 24, 38, 28, 22, 28, 24, 38, 24, 29, 38, 29, 29, 29, 29, 29, 34, 3, 30, 30, 30, 38, 24, 
    15, 38, 38, 38, 38, 38, 24, 38, 31, 38, 38, 38, 38, 22, 38, 32, 32, 33, 33, 18, 24, 0, 8, 38, 20, 
    20, 38, 24, 38, 38, 38, 38, 8, 22, 8, 34, 38, 34, 21, 34, 38, 38, 38, 24, 38, 37, 38, 38, 24, 14, 
    24, 38, 38, 0, 35, 35, 35, 35, 35, 35, 35, 36, 38, 38, 38, 24, 20, 24, 38, 34, 38, 25, 38, 38, 
    38, 38, 37, 38, 38, 38, 35, 22, 22, 38, 38, 22, 22, 38, 38, 38, 38, 38, 38, 38, 21, 38, 38, 28, 
    28, 39, 39, 38, 28, 10, 30, 10, 38, 38, 24, 38, 38, 40, 4, 13, 40, 24, 38, 38, 38, 38, 34, 38, 
]
MAPPING_HSSD40_TO_MPCAT40 = [
    38, 24, 1, 3, 38, 38, 38, 24, 38, 38, 38, 6, 38, 38, 22, 38, 7, 8, 9, 10, 38, 38, 31, 12, 13, 14, 38, 
    15, 38, 38, 38, 38, 38, 16, 38, 25, 38, 38, 38, 38, 38, 38, 38, 21, 38, 38, 38, 38, 38, 38, 24, 
    38, 24, 38, 38, 38, 38, 38, 38, 38, 28, 38, 29, 30, 38, 38, 38, 33, 34, 38, 38, 38, 35, 35, 38, 
    38, 38, 38, 37, 38, 38, 38, 38, 10, 38, 40,
]


def generate_mappings(source_vocab: list[str], target_vocab: list[str], threshold: float):
    from common.vision.clip import get_clip_embeddings

    source_embeddings = get_clip_embeddings(source_vocab).permute(1,0)
    target_embeddings = get_clip_embeddings(target_vocab).permute(1,0)

    def find_closest(idx_label, source_embeddings, target_embeddings, threshold=0.8):
        source_emb = source_embeddings[idx_label]
        best_label_idx = None
        best_sim = -1

        for i, target_emb in enumerate(target_embeddings):
            sim = (source_emb @ target_emb) / (source_emb.norm() * target_emb.norm())
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
            closest_label_idx = target_vocab.index("unknown")

        mapping_source_to_target.append(closest_label_idx)
        mapping_target_to_source[closest_label_idx].append(i)

    return mapping_source_to_target, mapping_target_to_source


# for (source_labels, target_labels, threshold) in [
#     (CLASS_LABELS_HSSD500, CLASS_LABELS_SCANNET200, 0.85),
#     (CLASS_LABELS_HSSD40, CLASS_LABELS_SCANNET200, 0.9),
#     (CLASS_LABELS_HSSD500, CLASS_LABELS_NYU40, 0.85),
#     (CLASS_LABELS_HSSD40, CLASS_LABELS_NYU40, 0.9),
#     (CLASS_LABELS_HSSD500, CLASS_LABELS_MPCAT40, 0.85),
#     (CLASS_LABELS_HSSD40, CLASS_LABELS_MPCAT40, 0.9),
# ]:
#     source_to_target, target_to_source = generate_mappings(source_labels, target_labels, threshold)

#     stack = []
#     out = "     "
#     list_to_print = source_to_target

#     for x in list_to_print:
#         if len("".join(stack)) + len(str(x)) + 1 >= 50:
#             out += "\n      "
#             stack = []
#         stack.append(str(x))
#         out += str(x) + ", "

#     print(out)

VOCABULARIES = {
    "HSSD500": (CLASS_LABELS_HSSD500,  None, None),
    "HSSD40": (CLASS_LABELS_HSSD40,  None, None),
    "SCANNET200": (CLASS_LABELS_SCANNET200, MAPPING_HSSD500_TO_SCANNET200, MAPPING_HSSD40_TO_SCANNET200),
    "NYU40": (CLASS_LABELS_NYU40, MAPPING_HSSD500_TO_NYU40, MAPPING_HSSD40_TO_NYU40),
    "MPCAT40": (CLASS_LABELS_MPCAT40, MAPPING_HSSD500_TO_MPCAT40, MAPPING_HSSD40_TO_MPCAT40),
}
