%% Verify Expert Model Robustness using NNV
% This script demonstrates how to verify the robustness of individual expert
% models (MicroExpertCNN, TinyExpertCNN, or SmallExpertCNN) using NNV
%
% Prerequisites:
%   1. NNV must be installed and initialized (run startup_nnv.m)
%   2. Expert model exported to ONNX format
%   3. Dataset images available
%
% Usage:
%   Set the parameters below and run the script in MATLAB

%% ======================== CONFIGURATION ========================

% Path to NNV root (adjust to your installation)
nnv_root = fullfile('..', '..', 'modules', 'nnv_moe', 'code', 'nnv');

% Path to exported ONNX model
onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn_best.onnx');

% Dataset configuration
dataset_name = 'GTSRB'; % Options: GTSRB, CIFAR10, MNIST
data_root = fullfile('..', '..', 'data');

% Verification parameters
epsilon = 2/255;        % L-infinity perturbation bound (normalized)
reachMethod = 'approx-star'; % Options: 'exact-star', 'approx-star', 'abs-dom'

% Image selection
test_image_idx = 1;     % Which test image to verify

%% ======================== INITIALIZATION ========================

% Add NNV to path if not already added
if exist(fullfile(nnv_root, 'startup_nnv.m'), 'file')
    cd(nnv_root);
    startup_nnv;
    cd(fileparts(mfilename('fullpath')));
else
    error('NNV not found. Please check nnv_root path.');
end

fprintf('\n========================================\n');
fprintf('Neural Network Verification with NNV\n');
fprintf('========================================\n\n');

%% ======================== LOAD MODEL ========================

fprintf('Loading ONNX model: %s\n', onnx_model_path);

% Load ONNX model into NNV
try
    net = onnx2nnv(onnx_model_path);
    fprintf('Model loaded successfully!\n');
    fprintf('Number of layers: %d\n', length(net.Layers));
catch ME
    error('Failed to load ONNX model: %s', ME.message);
end

%% ======================== LOAD DATASET ========================

fprintf('\nLoading %s dataset...\n', dataset_name);

switch dataset_name
    case 'GTSRB'
        % Load GTSRB dataset
        gtsrb_path = fullfile(data_root, 'GTSRB', 'Test', 'Images');
        if ~exist(gtsrb_path, 'dir')
            error('GTSRB test images not found at: %s', gtsrb_path);
        end
        imds = imageDatastore(gtsrb_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.3337, 0.3064, 0.3171];
        stdNorm = [0.2672, 0.2564, 0.2629];

    case 'CIFAR10'
        % Load CIFAR-10 dataset
        cifar_path = fullfile(data_root, 'CIFAR10', 'Test');
        if ~exist(cifar_path, 'dir')
            error('CIFAR-10 test images not found at: %s', cifar_path);
        end
        imds = imageDatastore(cifar_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.4914, 0.4822, 0.4465];
        stdNorm = [0.2023, 0.1994, 0.2010];

    case 'MNIST'
        % Load MNIST dataset
        mnist_path = fullfile(data_root, 'MNIST', 'Test');
        if ~exist(mnist_path, 'dir')
            error('MNIST test images not found at: %s', mnist_path);
        end
        imds = imageDatastore(mnist_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        inputSize = [28 28];
        numChannels = 1;
        meanNorm = 0.1307;
        stdNorm = 0.3081;

    otherwise
        error('Unknown dataset: %s', dataset_name);
end

% Ensure images are resized
imds.ReadFcn = @(loc) imresize(imread(loc), inputSize);

fprintf('Dataset loaded: %d images\n', length(imds.Files));

%% ======================== LOAD TEST IMAGE ========================

% Read image and label
[img, fileInfo] = readimage(imds, test_image_idx);
target_label = fileInfo.Label;

fprintf('\nTest Image %d:\n', test_image_idx);
fprintf('  True label: %s\n', string(target_label));
fprintf('  Image size: %dx%dx%d\n', size(img, 1), size(img, 2), size(img, 3));

% Convert to single precision and normalize to [0, 1]
img = single(img) / 255.0;

% Apply dataset normalization
if numChannels == 3
    for c = 1:3
        img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
    end
else
    img = (img - meanNorm) / stdNorm;
end

%% ======================== CREATE INPUT SET ========================

fprintf('\nCreating input set with epsilon = %.4f...\n', epsilon);

% Define perturbation bounds
disturbance = epsilon * ones(size(img), 'single');

% Create input set (normalized space)
lb = img - disturbance;
ub = img + disturbance;

% Create ImageStar input set
IS = ImageStar(lb, ub);

fprintf('Input set created successfully\n');

%% ======================== VERIFY ROBUSTNESS ========================

fprintf('\nVerifying robustness using %s method...\n', reachMethod);

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = reachMethod;

% Perform verification
target_idx = double(target_label); % Convert label to numeric
tic;
res = net.verify_robustness(IS, reachOptions, target_idx);
verify_time = toc;

fprintf('\nVerification completed in %.2f seconds\n', verify_time);

%% ======================== DISPLAY RESULTS ========================

fprintf('\n========================================\n');
fprintf('VERIFICATION RESULTS\n');
fprintf('========================================\n');
fprintf('Model: %s\n', onnx_model_path);
fprintf('Dataset: %s\n', dataset_name);
fprintf('Test image: %d\n', test_image_idx);
fprintf('True label: %s\n', string(target_label));
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Method: %s\n', reachMethod);
fprintf('----------------------------------------\n');

if res == 1
    fprintf('RESULT: ✓ Network is VERIFIED ROBUST\n');
    fprintf('All inputs within epsilon ball are correctly classified.\n');
elseif res == 0
    fprintf('RESULT: ✗ Network is NOT ROBUST\n');
    fprintf('Found counterexample within epsilon ball.\n');
else
    fprintf('RESULT: ? UNKNOWN\n');
    fprintf('Could not determine robustness (possibly timeout or approximation).\n');
end

fprintf('========================================\n\n');

%% ======================== VISUALIZE OUTPUT RANGES ========================

fprintf('Visualizing output ranges...\n');

% Get output reachable set
R = net.reachSet{end};

% Get ranges for each output class
if strcmp(reachMethod, 'exact-star')
    % For exact method, multiple stars may exist
    lb_out = inf * ones(net.OutputSize, 1);
    ub_out = -inf * ones(net.OutputSize, 1);
    for i = 1:length(R)
        [lb_temp, ub_temp] = R(i).getRanges;
        lb_temp = squeeze(lb_temp);
        ub_temp = squeeze(ub_temp);
        lb_out = min(lb_out, lb_temp);
        ub_out = max(ub_out, ub_temp);
    end
else
    % For approximate methods
    [lb_out, ub_out] = R.getRanges;
    lb_out = squeeze(lb_out);
    ub_out = squeeze(ub_out);
end

% Compute midpoint and range size
mid_range = (lb_out + ub_out) / 2;
range_size = ub_out - mid_range;

% Evaluate the original (unperturbed) image
Y_original = net.evaluate(img);

% Create visualization
figure('Position', [100, 100, 1000, 600]);

% Plot 1: Output ranges
subplot(1, 2, 1);
x = 1:net.OutputSize;
errorbar(x, mid_range, range_size, 'b.', 'MarkerSize', 15, 'LineWidth', 1.5);
hold on;
scatter(x, Y_original, 100, 'rx', 'LineWidth', 2);
xlabel('Output Class Index');
ylabel('Output Value');
title(sprintf('Output Reachable Set (%s)', reachMethod));
legend('Reachable range', 'Original prediction', 'Location', 'best');
grid on;

% Highlight the true class
if target_idx <= net.OutputSize
    xline(target_idx, 'g--', 'LineWidth', 2, 'Label', 'True Class');
end

% Plot 2: Original image
subplot(1, 2, 2);
if numChannels == 3
    % Denormalize for visualization
    img_vis = img;
    for c = 1:3
        img_vis(:,:,c) = img_vis(:,:,c) * stdNorm(c) + meanNorm(c);
    end
    imshow(img_vis);
else
    img_vis = img * stdNorm + meanNorm;
    imshow(img_vis, []);
end
title(sprintf('Test Image (Label: %s)', string(target_label)));

sgtitle(sprintf('NNV Verification Results - %s', dataset_name), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
output_fig = fullfile(fileparts(mfilename('fullpath')), ...
    sprintf('verification_result_%s_img%d.png', dataset_name, test_image_idx));
saveas(gcf, output_fig);
fprintf('Saved visualization to: %s\n', output_fig);

fprintf('\nVerification script completed successfully!\n');
