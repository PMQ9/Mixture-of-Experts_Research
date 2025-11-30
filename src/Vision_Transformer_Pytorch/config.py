from dataclasses import fields
import ast

# **************** Default Params ****************
DEFAULT_PARAMS = {
    'batch_size': 128,
    'epoch': 200,
    'learning_rate': 1e-3,
    'cutmix_alpha': 0.4,
    'cutmix_prob': 0.2,
    'warmup_epoch': 10,
    'label_smoothing': 0.1,
    'test_start_epoch': 0,
    'test_frequency': 2,
    'gpd_attack_epsilon_default': 0.1,
    'gpd_attack_epsilon_cifar_gtsrb': 0.031,
    'gpd_attack_epsilon_mnist': 0.031,
    'gpd_attack_iter': 20,
    'gpd_attack_iter_default': 10
}

# **************** Normalization Values ****************
NORM_MEAN_R_GTSRB = 0.3432482081311601
NORM_MEAN_G_GTSRB = 0.313121524381566
NORM_MEAN_B_GTSRB = 0.3224803091795953
NORM_STD_R_GTSRB = 0.27380229608003775
NORM_STD_G_GTSRB = 0.2603304975369596
NORM_STD_B_GTSRB = 0.26602727458940584

NORM_MEAN_R_PTSD = 0.4222741427251093
NORM_MEAN_G_PTSD = 0.4038989949251206
NORM_MEAN_B_PTSD = 0.42392439757813516
NORM_STD_R_PTSD = 0.2550717800965526
NORM_STD_G_PTSD = 0.22737840524891367
NORM_STD_B_PTSD = 0.22533599466418203

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

NORM_MEAN_R_CIFAR10 = 0.49139968910217285
NORM_MEAN_G_CIFAR10 = 0.4821584191894531
NORM_MEAN_B_CIFAR10 = 0.44653092010498047
NORM_STD_R_CIFAR10 = 0.2470322410602002
NORM_STD_G_CIFAR10 = 0.24348513156455537
NORM_STD_B_CIFAR10 = 0.26158785261491585

NORM_MEAN_R_MNIST = 0.1309002565383911
NORM_MEAN_G_MNIST = 0.1309002565383911
NORM_MEAN_B_MNIST = 0.1309002565383911
NORM_STD_R_MNIST = 0.2892882413884404
NORM_STD_G_MNIST = 0.2892882413884404
NORM_STD_B_MNIST = 0.2892882413884404

# Unified normalization for CIFAR-10 and MNIST (recalculated 2025-10-20)
NORM_MEAN_R_UNIFIED = 0.2947636386351152
NORM_MEAN_G_UNIFIED = 0.29056305959874934
NORM_MEAN_B_UNIFIED = 0.2743687416596846
NORM_STD_R_UNIFIED = 0.32497365569210473
NORM_STD_G_UNIFIED = 0.3212261072335478
NORM_STD_B_UNIFIED = 0.31851437012140965

GTSRB_NORM = {'mean': (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB),'std': (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)}
PTSD_NORM = {'mean': (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD),'std': (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)}
TSRD_NORM = {'mean': (NORM_MEAN_R_TSRD, NORM_MEAN_G_TSRD, NORM_MEAN_B_TSRD),'std': (NORM_STD_R_TSRD, NORM_STD_G_TSRD, NORM_STD_B_TSRD)}
BTSD_NORM = {'mean': (NORM_MEAN_R_BTSD, NORM_MEAN_G_BTSD, NORM_MEAN_B_BTSD),'std': (NORM_STD_R_BTSD, NORM_STD_G_BTSD, NORM_STD_B_BTSD)}
ETSD_NORM = {'mean': (NORM_MEAN_R_ETSD, NORM_MEAN_G_ETSD, NORM_MEAN_B_ETSD),'std': (NORM_STD_R_ETSD, NORM_STD_G_ETSD, NORM_STD_B_ETSD)}
CIFAR10_NORM = {'mean': (NORM_MEAN_R_CIFAR10, NORM_MEAN_G_CIFAR10, NORM_MEAN_B_CIFAR10),'std': (NORM_STD_R_CIFAR10, NORM_STD_G_CIFAR10, NORM_STD_B_CIFAR10)}
MNIST_NORM = {'mean': (NORM_MEAN_R_MNIST, NORM_MEAN_G_MNIST, NORM_MEAN_B_MNIST),'std': (NORM_STD_R_MNIST, NORM_STD_G_MNIST, NORM_STD_B_MNIST)}
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