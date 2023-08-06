json_data = {
    "3": {
        "inputs": {
            "seed": 979537337409677,
            "steps": 30,
            "cfg": 8.0,
            "sampler_name": "ddim",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
        "class_type": "KSampler",
    },
    "4": {
        "inputs": {"ckpt_name": "AWPainting 1.1_v1.1.safetensors"},
        "class_type": "CheckpointLoaderSimple",
    },
    "5": {
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "class_type": "EmptyLatentImage",
    },
    "6": {
        "inputs": {
            "text": "masterpiece, best quality,\n1girl, solo, long hair, black hair, from behind",
            "clip": ["4", 1],
        },
        "class_type": "CLIPTextEncode",
    },
    "7": {
        "inputs": {
            "text": "(worst quality, low quality:1.4), EasyNegative, ng_deepnegative_v1_75t, bad_prompt_version2,",
            "clip": ["4", 1],
        },
        "class_type": "CLIPTextEncode",
    },
    "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
    "9": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["11", 0]},
        "class_type": "SaveImage",
    },
    "11": {
        "inputs": {
            "int_field": 0,
            "float_field": 1.0,
            "print_to_screen": "enable",
            "string_field": "Hello World!",
            "image": ["8", 0],
        },
        "class_type": "EagleImageNode",
    },
}


def parse_prompt_token(prompt):
    return [_.replace("\n", "").strip() for _ in prompt.split(",")]


def parse_KSampler(
    class_type,
    inputs,
    **kwargs,
):
    data = kwargs.get("data")
    for input_key, input_value in inputs.items():
        if isinstance(input_value, (int, float)):
            print(f"{class_type}:{input_key.capitalize()}:{input_value}")
        if isinstance(input_value, list):
            for v in input_value:
                if v not in data:
                    continue
                if data[v]["class_type"] != "CLIPTextEncode":
                    continue

                if "positive" in input_key.lower():
                    p = data[v]["inputs"]["text"]
                    print(f"{class_type}:{input_key.capitalize()}:{p}")
                    continue
                if "negative" in input_key.lower():
                    p = data[v]["inputs"]["text"]
                    print(f"{class_type}:{input_key.capitalize()}:{p}")
                    continue
                print(f"{class_type}:{input_key.capitalize()}:{p}")


def parse_CLIPTextEncode(class_type, inputs):
    for input_key, input_value in inputs.items():
        if isinstance(input_value, str):
            if class_type == "CLIPTextEncode":
                for line in input_value.split(","):
                    line = line.strip()
                    if line:
                        print(f"{class_type}:{line}")
            else:
                print(f"{class_type}:{input_key}:{input_value}")


def extract_info(data):
    for key, value in data.items():
        class_type = value.get("class_type")
        inputs = value["inputs"]
        if class_type == "KSampler":
            parse_KSampler(class_type, inputs, data=data)
            continue
        # if class_type == "CLIPTextEncode":
        #     parse_CLIPTextEncode(class_type, inputs)
        #     continue
        if class_type not in [
            "CheckpointLoaderSimple",
        ]:
            continue


extract_info(json_data)


class EagleImageNode:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 64,  # Slider's step
                    },
                ),
                "float_field": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "print_to_screen": (["enable", "disable"],),
                "string_field": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "do_saving"

    # OUTPUT_NODE = False

    CATEGORY = "Example"

    def do_saving(
        self,
        image,
        string_field,
        int_field,
        float_field,
        print_to_screen,
        prompt=None,
        extra_pnginfo=None,
    ):
        print("--->imgae", "prompt", prompt, "extra", extra_pnginfo)
        if print_to_screen == "enable":
            print(
                f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """
            )
        # do some processing on the image, in this example I just invert it
        image = image
        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"EagleImageNode": EagleImageNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"EagleImageNode": "Eagle Image Node for PNGInfo"}
