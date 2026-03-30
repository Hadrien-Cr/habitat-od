PROCTHOR_DS = None

SCENE_TYPES = ["kitchen", "living_room", "bathroom", "bedroom", "procthor-train", "procthor-test"]

def enumerate_scenes(scene_type: str) -> list[str]:
    if scene_type == "procthor-train":
        return [f"HousePlan{i}" for i in range(0, 100)]
    elif scene_type == "procthor-test":
        return [f"HousePlan{i}" for i in range(100, 200)]
    else:
        return [f"FloorPlan{100 * s + i}" for i in range(1, 31) for s in [0,2,3,4] if get_scene_type(f"FloorPlan{100 * s + i}") == scene_type]


def get_scene_type(scene_name: str) -> str:
    if scene_name.startswith("FloorPlan"):
        s = int(int(scene_name.replace("FloorPlan",""))//100)
        try:
            return SCENE_TYPES[[0,None,1,2,3][s]]
        except:
            raise ValueError(f"Unknown scene name: {scene_name}")
        
    elif scene_name.startswith("HousePlan"):
        if int(scene_name.replace("HousePlan","")) < 100:
            return "procthor-train"
        else:
            return "procthor-test"
    else:
        raise ValueError(f"Unknown scene name: {scene_name}")


def scene_name_to_scene_spec(scene_name: str) -> dict:
    global PROCTHOR_DS

    assert scene_name.startswith(
        "HousePlan"
    ), "Only ProcTHOR scenes need to be converted."

    if PROCTHOR_DS is None:
        import prior

        PROCTHOR_DS = prior.load_dataset(
            "procthor-10k", revision="439193522244720b86d8c81cde2e51e3a4d150cf"
        )

    if get_scene_type(scene_name) == "procthor-train":
        i = int(int(scene_name.replace("HousePlan","")))
        return PROCTHOR_DS["train"][i]

    if get_scene_type(scene_name) == "procthor-train":
        i = int(int(scene_name.replace("HousePlan","")) - 100)
        return PROCTHOR_DS["test"][i]
    
    else:
        raise ValueError
