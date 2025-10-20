%% Verify MetaMoE Router (MetaGatingNet) using NNV
% This script verifies that the router maintains correct routing decisions
% under input perturbations (epsilon-ball).
%
% Router verification ensures:
%   - Given an input x from dataset D (e.g., GTSRB)
%   - For all x' in B_epsilon(x) (epsilon-ball around x)
%   - Router(x') still selects expert for dataset D
%
% This is critical for MetaMoE robustness - even if experts are robust,
% misrouting can cause failures.
%
% Prerequisites:
%   1. NNV installed (run startup_nnv.m)
%   2. Router exported to ONNX (use export_router_to_onnx.py)
%   3. Test images available
%
% Usage:
%   Set parameters below and run in MATLAB

%% ======================== CONFIGURATION ========================

% Path to NNV root
nnv_root = fullfile('..', '..', 'modules', 'nnv_moe', 'code', 'nnv');

% Path to exported router ONNX model
router_onnx_path = fullfile('..', '..', 'artifacts', 'nnv_models', ...
    'meta_moe_small_cnn_best_router_logits.onnx');

% Dataset configuration
dataset_name = 'GTSRB';  % Options: GTSRB, CIFAR10, MNIST
data_root = fullfile('..', '..', 'data');

% Meta-class mapping (must match training)
% 0: GTSRB, 1: CIFAR10, 2: MNIST
meta_class_map = containers.Map({'GTSRB', 'CIFAR10', 'MNIST'}, {0, 1, 2});
true_meta_class = meta_class_map(dataset_name);

% Verification parameters
epsilon = 8/255;  % L-infinity perturbation (can be larger for router)
reachMethod = 'approx-star';  % Options: 'exact-star', 'approx-star', 'abs-dom'

% Test configuration
test_image_idx = 1;
num_test_images = 10;  % Number of images to test

%% ======================== INITIALIZATION ========================

% Add NNV to path
if exist(fullfile(nnv_root, 'startup_nnv.m'), 'file')
    cd(nnv_root);
    startup_nnv;
    cd(fileparts(mfilename('fullpath')));
else
    error('NNV not found. Please check nnv_root path.');
end

fprintf('\n========================================\n');
fprintf('Router Verification with NNV\n');
fprintf('========================================\n\n');

%% ======================== LOAD ROUTER ========================

fprintf('Loading router ONNX model: %s\n', router_onnx_path);

try
    router_net = onnx2nnv(router_onnx_path);
    fprintf('✓ Router loaded successfully!\n');
    fprintf('Number of layers: %d\n', length(router_net.Layers));
    fprintf('Output size (num experts): %d\n', router_net.OutputSize);
catch ME
    error('Failed to load router ONNX model: %s', ME.message);
end

%% ======================== LOAD DATASET ========================

fprintf('\nLoading %s dataset...\n', dataset_name);

switch dataset_name
    case 'GTSRB'
        dataset_path = fullfile(data_root, 'GTSRB', 'Test', 'Images');
        if ~exist(dataset_path, 'dir')
            error('GTSRB test images not found at: %s', dataset_path);
        end
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.3337, 0.3064, 0.3171];
        stdNorm = [0.2672, 0.2564, 0.2629];

    case 'CIFAR10'
        dataset_path = fullfile(data_root, 'CIFAR10', 'Test');
        if ~exist(dataset_path, 'dir')
            error('CIFAR-10 test images not found at: %s', dataset_path);
        end
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.4914, 0.4822, 0.4465];
        stdNorm = [0.2023, 0.1994, 0.2010];

    case 'MNIST'
        dataset_path = fullfile(data_root, 'MNIST', 'Test');
        if ~exist(dataset_path, 'dir')
            error('MNIST test images not found at: %s', dataset_path);
        end
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [28 28];
        numChannels = 1;
        meanNorm = 0.1307;
        stdNorm = 0.3081;

    otherwise
        error('Unknown dataset: %s', dataset_name);
end

imds.ReadFcn = @(loc) imresize(imread(loc), inputSize);
fprintf('Dataset loaded: %d images\n', length(imds.Files));
fprintf('True meta-class for %s: %d\n', dataset_name, true_meta_class);

%% ======================== VERIFY ROUTING STABILITY ========================

fprintf('\n========================================\n');
fprintf('VERIFYING ROUTING STABILITY\n');
fprintf('========================================\n');
fprintf('Testing %d images with epsilon = %.4f\n\n', num_test_images, epsilon);

% Results storage
results = struct();
results.dataset = dataset_name;
results.true_meta_class = true_meta_class;
results.epsilon = epsilon;
results.method = reachMethod;
results.image_results = cell(num_test_images, 1);

% Statistics
num_verified_robust = 0;
num_not_robust = 0;
num_unknown = 0;
total_time = 0;

for img_idx = 1:min(num_test_images, length(imds.Files))
    fprintf('\n--- Image %d/%d ---\n', img_idx, num_test_images);

    % Load and preprocess image
    [img, fileInfo] = readimage(imds, img_idx);
    img = single(img) / 255.0;

    % Apply normalization
    if numChannels == 3
        for c = 1:3
            img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
        end
    else
        img = (img - meanNorm) / stdNorm;
    end

    % Evaluate clean image
    clean_output = router_net.evaluate(img);
    [~, clean_prediction] = max(clean_output);
    clean_prediction = clean_prediction - 1;  % Convert to 0-indexed

    fprintf('Clean routing: Expert %d (true: %d) %s\n', ...
        clean_prediction, true_meta_class, ...
        string(clean_prediction == true_meta_class));

    % Only verify if clean routing is correct
    if clean_prediction ~= true_meta_class
        fprintf('⚠ Skipping: Clean routing is already incorrect\n');
        results.image_results{img_idx}.verified = false;
        results.image_results{img_idx}.reason = 'clean_incorrect';
        num_not_robust = num_not_robust + 1;
        continue;
    end

    % Create input set
    disturbance = epsilon * ones(size(img), 'single');
    lb = img - disturbance;
    ub = img + disturbance;
    IS = ImageStar(lb, ub);

    % Verify routing stability
    reachOptions = struct;
    reachOptions.reachMethod = reachMethod;

    tic;
    % Note: true_meta_class + 1 because MATLAB is 1-indexed
    res = router_net.verify_robustness(IS, reachOptions, true_meta_class + 1);
    verify_time = toc;
    total_time = total_time + verify_time;

    % Store results
    results.image_results{img_idx}.image_idx = img_idx;
    results.image_results{img_idx}.clean_prediction = clean_prediction;
    results.image_results{img_idx}.verification_result = res;
    results.image_results{img_idx}.time = verify_time;

    % Interpret result
    if res == 1
        fprintf('✓ VERIFIED ROBUST (%.2f sec)\n', verify_time);
        num_verified_robust = num_verified_robust + 1;
    elseif res == 0
        fprintf('✗ NOT ROBUST (%.2f sec)\n', verify_time);
        num_not_robust = num_not_robust + 1;
    else
        fprintf('? UNKNOWN (%.2f sec)\n', verify_time);
        num_unknown = num_unknown + 1;
    end
end

%% ======================== SUMMARY STATISTICS ========================

fprintf('\n========================================\n');
fprintf('VERIFICATION SUMMARY\n');
fprintf('========================================\n');
fprintf('Dataset: %s (meta-class %d)\n', dataset_name, true_meta_class);
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Method: %s\n', reachMethod);
fprintf('Images tested: %d\n', num_test_images);
fprintf('----------------------------------------\n');
fprintf('✓ Verified robust: %d (%.1f%%)\n', num_verified_robust, ...
    100 * num_verified_robust / num_test_images);
fprintf('✗ Not robust: %d (%.1f%%)\n', num_not_robust, ...
    100 * num_not_robust / num_test_images);
fprintf('? Unknown: %d (%.1f%%)\n', num_unknown, ...
    100 * num_unknown / num_test_images);
fprintf('----------------------------------------\n');
fprintf('Average time per image: %.2f sec\n', total_time / num_test_images);
fprintf('Total time: %.2f sec\n', total_time);
fprintf('========================================\n');

% Certified routing accuracy
certified_routing_accuracy = num_verified_robust / num_test_images;
fprintf('\nCertified Routing Accuracy: %.2f%%\n', certified_routing_accuracy * 100);

%% ======================== VISUALIZE SINGLE IMAGE ========================

fprintf('\nVisualizing detailed results for image %d...\n', test_image_idx);

% Load test image
[img, ~] = readimage(imds, test_image_idx);
img_display = img;
img = single(img) / 255.0;

% Normalize
if numChannels == 3
    for c = 1:3
        img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
    end
else
    img = (img - meanNorm) / stdNorm;
end

% Create input set
disturbance = epsilon * ones(size(img), 'single');
lb = img - disturbance;
ub = img + disturbance;
IS = ImageStar(lb, ub);

% Verify
reachOptions.reachMethod = reachMethod;
res = router_net.verify_robustness(IS, reachOptions, true_meta_class + 1);

% Get output reachable set
R = router_net.reachSet{end};

% Get ranges
if strcmp(reachMethod, 'exact-star')
    lb_out = inf * ones(router_net.OutputSize, 1);
    ub_out = -inf * ones(router_net.OutputSize, 1);
    for i = 1:length(R)
        [lb_temp, ub_temp] = R(i).getRanges;
        lb_temp = squeeze(lb_temp);
        ub_temp = squeeze(ub_temp);
        lb_out = min(lb_out, lb_temp);
        ub_out = max(ub_out, ub_temp);
    end
else
    [lb_out, ub_out] = R.getRanges;
    lb_out = squeeze(lb_out);
    ub_out = squeeze(ub_out);
end

% Evaluate clean image
clean_output = router_net.evaluate(img);

% Visualize
figure('Position', [100, 100, 1200, 500]);

% Plot 1: Input image
subplot(1, 3, 1);
imshow(img_display);
title(sprintf('Test Image (Dataset: %s)', dataset_name));

% Plot 2: Router output ranges
subplot(1, 3, 2);
x = 0:(router_net.OutputSize - 1);
mid_range = (lb_out + ub_out) / 2;
range_size = ub_out - mid_range;

errorbar(x, mid_range, range_size, 'bo-', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
scatter(x, clean_output, 100, 'rx', 'LineWidth', 2);
xline(true_meta_class, 'g--', 'LineWidth', 2, 'Label', 'True Expert');
xlabel('Expert Index');
ylabel('Router Output (Logits)');
title(sprintf('Router Output Ranges (\\epsilon=%.3f)', epsilon));
legend('Reachable range', 'Clean prediction', 'Location', 'best');
grid on;
set(gca, 'XTick', x);

% Plot 3: Decision stability
subplot(1, 3, 3);
bar(x, [lb_out, ub_out], 'grouped');
hold on;
xline(true_meta_class, 'g--', 'LineWidth', 3, 'Label', 'True Expert');
xlabel('Expert Index');
ylabel('Output Range');
title('Lower and Upper Bounds');
legend('Lower bound', 'Upper bound', 'True expert', 'Location', 'best');
grid on;
set(gca, 'XTick', x);

sgtitle(sprintf('Router Verification: %s (Result: %s)', ...
    dataset_name, ...
    string(res == 1) + " ROBUST" + string(res ~= 1)), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
output_fig = fullfile(fileparts(mfilename('fullpath')), ...
    sprintf('router_verification_%s_img%d.png', dataset_name, test_image_idx));
saveas(gcf, output_fig);
fprintf('Saved visualization to: %s\n', output_fig);

fprintf('\nRouter verification script completed!\n');
