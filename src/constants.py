IGNORE_INDEX = -100

DEFAULT_START_TOKEN = "<start_of_turn>"
DEFAULT_END_TOKEN = "<end_of_turn>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_TOKEN = "<image_soft_token>"
VISION_START_TOKEN = "<start_of_image>"
VISION_END_TOKEN = "<end_of_image>"

SYSTEM_MESSAGE="You are a professional medical coder. Your task is to provide the exact ICD-10-CM code for the clinical description provided by the user."\
               "As a medical coder, you first think about the reasoning behind the code and then provide the code."\
               "The reasoning process and the ICD-10-CM code are enclosed within <think>...</think> and <code>...</code> tags, respectively, i.e., <think> reasoning process here </think> <code> ICD-10-CM code here </code>."

MULTIMODAL_KEYWORDS = ["pixel_values", "token_type_ids"]