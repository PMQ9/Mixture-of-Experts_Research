from dataclasses import fields
import ast

# **************** Default Params ****************
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCH = 800
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CUTMIX_ALPHA = 0.4
DEFAULT_CUTMIX_PROB = 0.2
DEFAULT_TEST_START_EPOCH = 50
DEFAULT_TEST_FREQUENCY = 2
DEFAULT_WARMUP_EPOCH = 10
DEFAULT_LABEL_SMOOTHING = 0.1

# **************** Normalization Values ****************
NORM_MEAN_R_GTSRB = 0.3432482055626116
NORM_MEAN_G_GTSRB = 0.31312152061376486
NORM_MEAN_B_GTSRB = 0.32248030768500435
NORM_STD_R_GTSRB = 0.27380229614172485
NORM_STD_G_GTSRB = 0.26033050034131744
NORM_STD_B_GTSRB = 0.2660272789537349

NORM_MEAN_R_PTSD = 0.42227414577051153
NORM_MEAN_G_PTSD = 0.40389899174730964
NORM_MEAN_B_PTSD = 0.42392441068660547
NORM_STD_R_PTSD = 0.2550717671385188
NORM_STD_G_PTSD = 0.2273784047793104
NORM_STD_B_PTSD = 0.22533597220675006

NORM_MEAN_R_TSRD = 0.4312231187340167
NORM_MEAN_G_TSRD = 0.4171651016894004
NORM_MEAN_B_TSRD = 0.4256379578039229
NORM_STD_R_TSRD = 0.22851532470436997
NORM_STD_G_TSRD = 0.21587419120059484
NORM_STD_B_TSRD = 0.23286749563231549

NORM_MEAN_R_CIFAR10 = 0.4914
NORM_MEAN_G_CIFAR10 = 0.4822
NORM_MEAN_B_CIFAR10 = 0.4465
NORM_STD_R_CIFAR10 = 0.247
NORM_STD_G_CIFAR10 = 0.243
NORM_STD_B_CIFAR10 = 0.261

# **************** Unified Normalization Values ****************
NORM_MEAN_R_UNIFIED = 0.3432482081311601
NORM_MEAN_G_UNIFIED = 0.313121524381566
NORM_MEAN_B_UNIFIED = 0.3224803091795953
NORM_STD_R_UNIFIED = 0.27380229608003775
NORM_STD_G_UNIFIED = 0.2603304975369596
NORM_STD_B_UNIFIED = 0.26602727458940584

# **************** Overide Default Config Params ****************
def apply_config_overrides(config, overrides_str):
    if not overrides_str:
        return
    overrides = overrides_str.split(',')
    for override in overrides:
        if '=' in override:
            key, value = override.split('=', 1)
            if hasattr(config, key):
                field = next((f for f in fields(config) if f.name == key), None)
                if field:
                    try:
                        parsed_value = ast.literal_eval(value)
                        if isinstance(parsed_value, field.type):
                            setattr(config, key, parsed_value)
                        else:
                            print(f"Type mismatch for {key}: expected {field.type}, got {type(parsed_value)}")
                    except ValueError:
                        print(f"Invalid value for {key}: {value}")
                else:
                    print(f"Unknown config parameter: {key}")
            else:
                print(f"Invalid override format: {override}")