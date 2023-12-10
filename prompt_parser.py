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
