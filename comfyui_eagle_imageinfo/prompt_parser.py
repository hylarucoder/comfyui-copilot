def parse_prompt_token(prompt):
    return [_.replace("\n", "").strip() for _ in prompt.split(",")]


def parse_KSampler(
    class_type,
    inputs,
    **kwargs,
):
    annotations = []
    # TODO: tags
    tags = []
    data = kwargs.get("data")
    for input_key, input_value in inputs.items():
        if isinstance(input_value, (int, float)):
            annotations.append(f"{class_type}:{input_key}:{input_value}")
        if isinstance(input_value, list):
            for v in input_value:
                if v not in data:
                    continue
                if data[v]["class_type"] not in [
                    "CLIPTextEncode",
                    "CheckpointLoaderSimple",
                ]:
                    continue

                if "positive" in input_key.lower():
                    p = data[v]["inputs"]["text"]
                    annotations.append(f"{class_type}:{input_key.capitalize()}:{p}")
                    continue
                if "negative" in input_key.lower():
                    p = data[v]["inputs"]["text"]
                    annotations.append(f"{class_type}:{input_key.capitalize()}:{p}")
                    continue
                if "model" in input_key.lower():
                    p = data[v]["inputs"].get("ckpt_name", None)
                    assert p
                    annotations.append(f"{class_type}:{input_key.capitalize()}:{p}")
                    continue
                annotations.append(f"{class_type}:{input_key.capitalize()}:{p}")
    return annotations


def parse_CLIPTextEncode(class_type, inputs):
    annotations = []
    for input_key, input_value in inputs.items():
        if isinstance(input_value, str):
            if class_type == "CLIPTextEncode":
                for line in input_value.split(","):
                    line = line.strip()
                    if line:
                        annotations.append(f"{class_type}:{line}")
            else:
                annotations(f"{class_type}:{input_key}:{input_value}")


def parse_prompt_info(prompt_info):
    annotations_list = []
    for key, value in prompt_info.items():
        class_type = value.get("class_type")
        inputs = value["inputs"]
        if class_type == "KSampler":
            annotations_list.append(
                parse_KSampler(class_type, inputs, data=prompt_info)
            )
            continue
        # if class_type == "CLIPTextEncode":
        #     parse_CLIPTextEncode(class_type, inputs)
        #     continue
        if class_type not in [
            "CheckpointLoaderSimple",
        ]:
            continue
    import itertools

    a = list(set(itertools.chain.from_iterable(annotations_list)))
    a.sort()
    return a


import json


def trace_inputs(node, workflow, path=None, tags=None):
    if tags is None:
        tags = []

    class_type = node["class_type"]
    inputs = node["inputs"]

    # 添加类类型到路径中
    if path is None:
        path = [class_type]
    else:
        path.append(class_type)

    for key, value in inputs.items():
        new_path = path + [key]
        if isinstance(value, list):
            ref_id, _ = value
            ref_node = workflow[str(ref_id)]
            trace_inputs(ref_node, workflow, new_path, tags)
        else:
            tag = f"{'->'.join(new_path[-4:])}::{value}"
            tags.append(tag)

    return tags


def parse_workflow(workflow):
    # TODO: only parse from eagle node!!!
    image_nodes = [
        node for node in workflow.values() if node["class_type"] == "PreviewImage"
    ]
    image_node = image_nodes[0]
    tags = trace_inputs(image_node, workflow)
    print("Image Node Tags:")
    tags = list(set(tags))
    tags.sort()
    # // raw info
    return tags
