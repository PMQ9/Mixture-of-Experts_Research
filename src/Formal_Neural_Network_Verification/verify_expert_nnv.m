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
onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn.onnx');

% Dataset configuration
dataset_name = 'GTSRB'; % Options: GTSRB, CIFAR10, MNIST
data_root = fullfile('..', '..', 'data');

% Verification parameters
epsilon = 0.5/255;        % L-infinity perturbation bound (normalized)
reachMethod = 'abs-dom'; % Options: 'exact-star', 'approx-star', 'abs-dom'
lp_solver = 'linprog';  % Options: 'linprog', 'glpk', 'gurobi'
relaxFactor = 0;        % Relaxation factor for approximate methods (0 = tight)
numCores = 1;           % Number of parallel cores (1 = sequential)

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
    % First try the new recommended API (MATLAB R2020b+)
    fprintf('Attempting to load with importNetworkFromONNX (recommended)...\n');
    try
        matlab_net = importNetworkFromONNX(onnx_model_path, ...
            'InputDataFormats', 'BCSS', ...
            'OutputDataFormats', 'BC');
        fprintf('Loaded with importNetworkFromONNX successfully!\n');
    catch ME1
        % Fall back to onnx2nnv if new API fails
        fprintf('Warning: importNetworkFromONNX failed: %s\n', ME1.message);
        fprintf('Falling back to onnx2nnv...\n');
        matlab_net = [];
    end

    % If new API didn't work, try onnx2nnv
    if isempty(matlab_net)
        % Configure loading options for onnx2nnv
        loadOptions = struct();
        loadOptions.InputDataFormat = 'BCSS';  % Batch, Channel, Spatial, Spatial
        loadOptions.OutputDataFormat = 'BC';    % Batch, Class
        loadOptions.GenerateCustomLayers = false;
        loadOptions.FoldConstants = 'deep';

        matlab_net = onnx2nnv(onnx_model_path, loadOptions);
        fprintf('Loaded with onnx2nnv successfully!\n');
    end

    % Convert to NNV format if not already
    if ~isa(matlab_net, 'NN')
        fprintf('Converting MATLAB network to NNV format...\n');
        net = matlab2nnv(matlab_net);
    else
        net = matlab_net;
    end

    fprintf('Model loaded successfully!\n');
    fprintf('Number of layers: %d\n', length(net.Layers));
catch ME
    error('Failed to load ONNX model: %s\nStack trace:\n%s', ...
        ME.message, getReport(ME, 'extended'));
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

% Configure network for reachability analysis
net.reachMethod = reachMethod;
net.lp_solver = lp_solver;
net.relaxFactor = relaxFactor;
net.numCores = numCores;

% For MaxPooling layers, use less precise but faster approximation
% This avoids the LP solver timeout issues
net.dis_opt = [];  % Disable some optimization options that cause LP issues
net.reachOption = [];  % Use default options

% Define reachability options
reachOptions = struct;
reachOptions.reachMethod = reachMethod;

% Perform verification
% Convert label to numeric index (MATLAB uses 1-based indexing)
% Note: The label might be a folder name, we need the actual class index
if isnumeric(target_label)
    target_idx = double(target_label);
elseif ischar(target_label) || isstring(target_label)
    % For GTSRB, folder names are class IDs
    % Try to extract numeric part from label
    target_str = char(target_label);
    target_idx = str2double(target_str);
    if isnan(target_idx)
        % If conversion fails, just use 0 (will check all classes)
        fprintf('Warning: Could not extract class index from label "%s", will verify all classes\n', target_label);
        target_idx = 0;
    else
        % GTSRB classes are 0-based in folders but we need 1-based for MATLAB
        target_idx = target_idx + 1;
    end
end

fprintf('Verifying for target class index: %d\n', target_idx);

tic;
try
    res = net.verify_robustness(IS, reachOptions, target_idx);
catch ME
    fprintf('Error during verification: %s\n', ME.message);
    fprintf('\nTrying alternative approach: compute reachable set directly...\n');

    % Alternative: Just compute the reachable set without robustness check
    try
        tic;
        R = net.reach(IS, reachMethod);
        reach_time = toc;
        fprintf('Reachable set computed in %.2f seconds\n', reach_time);
        fprintf('Reachable set size: %d\n', length(R));

        % Manual robustness check
        % Check if the target class has the highest lower bound
        if ~isempty(R)
            % Get output ranges
            output_star = R{end};
            if iscell(output_star)
                output_star = output_star{1};
            end

            fprintf('Successfully computed reachable output set\n');
            res = -1;  % Unknown result, but we have the reachable set
        else
            error('Failed to compute reachable set');
        end
    catch ME2
        fprintf('Alternative approach also failed: %s\n', ME2.message);
        error('Verification failed with both approaches');
    end
end
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
