PROCTHOR_DS = None

def enumerate_scenes(scene_type: str):
    if scene_type == "procthor-train":
        return [f"procthor-train{i}" for i in range(100)]
    if scene_type == "procthor-val":
        return [f"procthor-val{i}" for i in range(100)]
    if scene_type == "procthor-test":
        return [f"procthor-test{i}" for i in range(100)]
    raise ValueError(f"Unknown scene type: {scene_type}")


def get_scene_type(scene_name: str) -> str:
    if scene_name.startswith("procthor-train"):
        return "procthor-train"
    if scene_name.startswith("procthor-val"):
        return "procthor-val"
    if scene_name.startswith("procthor-test"):
        return "procthor-test"
    raise ValueError(f"Unknown scene name: {scene_name}")

def scene_name_to_scene_spec(scene_name: str) -> dict:
    global PROCTHOR_DS

    assert scene_name.startswith(
        "procthor"
    ), "Only ProcTHOR scenes need to be converted."

    if PROCTHOR_DS is None:
        import prior

        PROCTHOR_DS = prior.load_dataset(
            "procthor-10k", revision="439193522244720b86d8c81cde2e51e3a4d150cf"
        )

    if get_scene_type(scene_name) == "procthor-train":
        i = int(int(scene_name.replace("procthor-train","")))
        return PROCTHOR_DS["train"][i]
    
    if get_scene_type(scene_name) == "procthor-val":
        i = int(int(scene_name.replace("procthor-val","")))
        return PROCTHOR_DS["train"][-i]

    if get_scene_type(scene_name) == "procthor-test":
        i = int(int(scene_name.replace("procthor-test","")))
        return PROCTHOR_DS["test"][i]
    
    else:
        raise ValueError
