from dataclasses import fields
import ast

# **************** Default Params ****************
DEFAULT_PARAMS = {
    'batch_size': 128,
    'epoch': 800,
    'learning_rate': 1e-3,
    'cutmix_alpha': 0.4,
    'cutmix_prob': 0.2,
    'warmup_epoch': 10,
    'label_smoothing': 0.1,
    'test_start_epoch': 50,
    'test_frequency': 2
}

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

NORM_MEAN_R_BTSD = 0.40434199015299477
NORM_MEAN_G_BTSD = 0.3790186805933551
NORM_MEAN_B_BTSD = 0.39015360347560196
NORM_STD_R_BTSD = 0.26260338559456875
NORM_STD_G_BTSD = 0.25895359572117344
NORM_STD_B_BTSD = 0.27221993697931895

NORM_MEAN_R_ETSD = 0.46754800898205023
NORM_MEAN_G_ETSD = 0.4284103203224121
NORM_MEAN_B_ETSD = 0.4867949374597146
NORM_STD_R_ETSD = 0.2571963667971265
NORM_STD_G_ETSD = 0.21638702829615006
NORM_STD_B_ETSD = 0.23543424770475022

NORM_MEAN_R_CIFAR10 = 0.4914
NORM_MEAN_G_CIFAR10 = 0.4822
NORM_MEAN_B_CIFAR10 = 0.4465
NORM_STD_R_CIFAR10 = 0.247
NORM_STD_G_CIFAR10 = 0.243
NORM_STD_B_CIFAR10 = 0.261

NORM_MEAN_R_UNIFIED = 0.38879402463614293
NORM_MEAN_G_UNIFIED = 0.36229729920994996
NORM_MEAN_B_UNIFIED = 0.37985949886524545
NORM_STD_R_UNIFIED = 0.2671561389170561
NORM_STD_G_UNIFIED = 0.2490427395905567
NORM_STD_B_UNIFIED = 0.25780709084169384

GTSRB_NORM = {'mean': (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB),'std': (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)}
PTSD_NORM = {'mean': (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD),'std': (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)}
TSRD_NORM = {'mean': (NORM_MEAN_R_TSRD, NORM_MEAN_G_TSRD, NORM_MEAN_B_TSRD),'std': (NORM_STD_R_TSRD, NORM_STD_G_TSRD, NORM_STD_B_TSRD)}
BTSD_NORM = {'mean': (NORM_MEAN_R_BTSD, NORM_MEAN_G_BTSD, NORM_MEAN_B_BTSD),'std': (NORM_STD_R_BTSD, NORM_STD_G_BTSD, NORM_STD_B_BTSD)}
ETSD_NORM = {'mean': (NORM_MEAN_R_ETSD, NORM_MEAN_G_ETSD, NORM_MEAN_B_ETSD),'std': (NORM_STD_R_ETSD, NORM_STD_G_ETSD, NORM_STD_B_ETSD)}
CIFAR10_NORM = {'mean': (NORM_MEAN_R_CIFAR10, NORM_MEAN_G_CIFAR10, NORM_MEAN_B_CIFAR10),'std': (NORM_STD_R_CIFAR10, NORM_STD_G_CIFAR10, NORM_STD_B_CIFAR10)}
UNIFIED_NORM = {'mean': (NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED),'std': (NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED)}

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