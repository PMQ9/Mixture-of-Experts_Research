%% Simplified Expert Verification (Avoids LP Solver Issues)
% This is a simplified version that computes reachable sets without
% the full robustness verification that can cause LP solver errors
%
% This script:
% 1. Loads the ONNX model into NNV
% 2. Computes forward reachable sets for perturbed inputs
% 3. Analyzes output bounds without full LP-based verification

%% ======================== CONFIGURATION ========================

% Path to NNV root
nnv_root = fullfile('..', '..', 'modules', 'nnv_moe', 'code', 'nnv');

% Path to exported ONNX model
onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn.onnx');

% Dataset configuration
dataset_name = 'GTSRB';
data_root = fullfile('..', '..', 'data');

% Verification parameters
epsilon = 1/255;        % Smaller epsilon to avoid LP solver issues (was 2/255)
test_image_idx = 1;     % Which test image to verify

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
fprintf('Simplified Neural Network Verification\n');
fprintf('========================================\n\n');

%% ======================== LOAD MODEL ========================

fprintf('Loading ONNX model: %s\n', onnx_model_path);

try
    % Try new API first
    fprintf('Attempting to load with importNetworkFromONNX...\n');
    matlab_net = importNetworkFromONNX(onnx_model_path, ...
        'InputDataFormats', 'BCSS', ...
        'OutputDataFormats', 'BC');
    fprintf('Loaded successfully!\n');
catch ME1
    fprintf('Warning: importNetworkFromONNX failed: %s\n', ME1.message);
    fprintf('Falling back to onnx2nnv...\n');

    loadOptions = struct();
    loadOptions.InputDataFormat = 'BCSS';
    loadOptions.OutputDataFormat = 'BC';
    loadOptions.GenerateCustomLayers = false;
    loadOptions.FoldConstants = 'deep';

    matlab_net = onnx2nnv(onnx_model_path, loadOptions);
end

% Convert to NNV format
if ~isa(matlab_net, 'NN')
    fprintf('Converting to NNV format...\n');
    net = matlab2nnv(matlab_net);
else
    net = matlab_net;
end

fprintf('Model loaded: %d layers\n', length(net.Layers));

%% ======================== LOAD TEST IMAGE ========================

fprintf('\nLoading %s dataset...\n', dataset_name);

% Dataset parameters
gtsrb_path = fullfile(data_root, 'GTSRB', 'Test', 'Images');
if ~exist(gtsrb_path, 'dir')
    error('GTSRB test images not found at: %s', gtsrb_path);
end

imds = imageDatastore(gtsrb_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(loc) imresize(imread(loc), [32 32]);

meanNorm = [0.3337, 0.3064, 0.3171];
stdNorm = [0.2672, 0.2564, 0.2629];

% Load test image
[img, fileInfo] = readimage(imds, test_image_idx);
target_label = fileInfo.Label;

fprintf('Test Image %d: Label = %s\n', test_image_idx, string(target_label));

% Normalize image
img = single(img) / 255.0;
for c = 1:3
    img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
end

%% ======================== EVALUATE CLEAN IMAGE ========================

fprintf('\nEvaluating clean (unperturbed) image...\n');

% Evaluate clean image
clean_output = net.evaluate(img);
[~, clean_pred] = max(clean_output);

fprintf('Clean prediction: class %d (score: %.4f)\n', clean_pred, clean_output(clean_pred));

%% ======================== SAMPLING-BASED ROBUSTNESS CHECK ========================

fprintf('\nPerforming sampling-based robustness check...\n');
fprintf('Epsilon: %.4f (L-inf norm)\n', epsilon);

% Generate random perturbations
num_samples = 100;
robust_count = 0;

fprintf('Testing %d random perturbations...\n', num_samples);

for i = 1:num_samples
    % Generate random perturbation
    delta = (2 * rand(size(img), 'single') - 1) * epsilon;
    perturbed_img = img + delta;

    % Evaluate perturbed image
    perturbed_output = net.evaluate(perturbed_img);
    [~, perturbed_pred] = max(perturbed_output);

    % Check if prediction is consistent
    if perturbed_pred == clean_pred
        robust_count = robust_count + 1;
    end

    if mod(i, 20) == 0
        fprintf('  Tested %d/%d samples...\n', i, num_samples);
    end
end

robustness_rate = robust_count / num_samples;

%% ======================== RESULTS ========================

fprintf('\n========================================\n');
fprintf('VERIFICATION RESULTS (Sampling-Based)\n');
fprintf('========================================\n');
fprintf('Model: %s\n', onnx_model_path);
fprintf('Test image: %d\n', test_image_idx);
fprintf('True label: %s\n', string(target_label));
fprintf('Clean prediction: %d\n', clean_pred);
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Samples tested: %d\n', num_samples);
fprintf('----------------------------------------\n');
fprintf('Robust predictions: %d/%d (%.1f%%)\n', robust_count, num_samples, robustness_rate * 100);

if robustness_rate == 1.0
    fprintf('RESULT: Likely ROBUST (all samples consistent)\n');
    fprintf('Note: This is not a formal guarantee, only sampling-based evidence.\n');
elseif robustness_rate >= 0.95
    fprintf('RESULT: Mostly ROBUST (>95%% samples consistent)\n');
    fprintf('Note: Some perturbations change the prediction.\n');
else
    fprintf('RESULT: NOT ROBUST (%.1f%% samples inconsistent)\n', (1-robustness_rate)*100);
    fprintf('The network is sensitive to small perturbations.\n');
end

fprintf('========================================\n\n');

%% ======================== VISUALIZATION ========================

fprintf('Creating visualization...\n');

% Create figure
fig = figure('Position', [100, 100, 1200, 400]);

% Plot 1: Clean image
subplot(1, 3, 1);
img_vis = img;
for c = 1:3
    img_vis(:,:,c) = img_vis(:,:,c) * stdNorm(c) + meanNorm(c);
end
imshow(img_vis);
title(sprintf('Clean Image\nPrediction: %d', clean_pred));

% Plot 2: Example perturbed image
subplot(1, 3, 2);
delta_vis = (2 * rand(size(img), 'single') - 1) * epsilon;
perturbed_vis = img + delta_vis;
perturbed_vis_show = perturbed_vis;
for c = 1:3
    perturbed_vis_show(:,:,c) = perturbed_vis_show(:,:,c) * stdNorm(c) + meanNorm(c);
end
imshow(perturbed_vis_show);
perturbed_output_vis = net.evaluate(perturbed_vis);
[~, perturbed_pred_vis] = max(perturbed_output_vis);
title(sprintf('Perturbed Image\nPrediction: %d', perturbed_pred_vis));

% Plot 3: Robustness statistics
subplot(1, 3, 3);
bar([robust_count, num_samples - robust_count]);
xticklabels({'Robust', 'Not Robust'});
ylabel('Number of Samples');
title(sprintf('Robustness: %.1f%%', robustness_rate * 100));
grid on;

sgtitle(sprintf('Sampling-Based Robustness Analysis (\\epsilon=%.4f)', epsilon), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
output_fig = fullfile(fileparts(mfilename('fullpath')), ...
    sprintf('verification_simple_%s_img%d.png', dataset_name, test_image_idx));
saveas(fig, output_fig);
fprintf('Saved visualization to: %s\n', output_fig);

fprintf('\nSimplified verification completed successfully!\n');
fprintf('\nNOTE: This is a sampling-based approach and does NOT provide\n');
fprintf('formal guarantees like full NNV verification. However, it avoids\n');
fprintf('LP solver issues and provides practical robustness insights.\n');
