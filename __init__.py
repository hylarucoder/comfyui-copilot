from .sdxl_prompt_styler import SDXLPromptStyler, SDXLPromptStylerAdvanced
from .prompt_parser import parse_workflow
import typing as t


class BaseNode:
    CATEGORY = "copilot"

    @classmethod
    def INPUT_TYPES(cls):
        pass


class EagleImageNode(BaseNode):
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
                        "multiline": False,
                        # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "do_saving"

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
        a = parse_workflow(prompt)
        print(a)


TResolution = t.Literal[
    "Square (1024x1024)",
    "Cinematic (1536x640)",
    "Cinematic (640x1536)",
    "Widescreen (1344x768)",
    "Widescreen (768x1344)",
    "Photo (1216x832)",
    "Photo (832x1216)",
    "Portrait (1152x896)",
]


class SDXLResolutionPresets(BaseNode):
    RESOLUTIONS: list[TResolution] = [
        "Square (1024x1024)",
        "Cinematic (1536x640)",
        "Cinematic (640x1536)",
        "Widescreen (1344x768)",
        "Widescreen (768x1344)",
        "Photo (1216x832)",
        "Photo (832x1216)",
        "Portrait (1152x896)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.RESOLUTIONS, {"default": "Square (1024x1024)"}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_value"

    def get_value(self, resolution: TResolution, ) -> tuple[int, int]:
        if resolution == "Cinematic (1536x640)":
            return 1536, 640
        if resolution == "Cinematic (640x1536)":
            return 640, 1536
        if resolution == "Widescreen (1344x768)":
            return 1344, 768
        if resolution == "Widescreen (768x1344)":
            return 768, 1344
        if resolution == "Photo (1216x832)":
            return 1216, 832
        if resolution == "Photo (832x1216)":
            return 832, 1216
        if resolution == "Portrait (1152x896)":
            return 1152, 896
        if resolution == "Portrait (896x1152)":
            return 896, 1152
        if resolution == "Square (1024x1024)":
            return 1024, 1024


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "EagleImageNode": EagleImageNode,
    "SDXLResolutionPresets": SDXLResolutionPresets,
    "SDXLPromptStyler": SDXLPromptStyler,
    "SDXLPromptStylerAdvanced": SDXLPromptStylerAdvanced,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleImageNode": "Eagle Image Node for PNGInfo",
    "SDXLResolutionPresets": "SDXL Resolution Presets (ws)",
    "SDXLPromptStyler": "SDXL Prompt Styler",
    "SDXLPromptStylerAdvanced": "SDXL Prompt Styler Advanced",
}
