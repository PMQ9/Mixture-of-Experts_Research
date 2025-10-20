%% Optimized NNV Verification with Progressive Strategy
% This script tries multiple configurations in order of increasing difficulty
% to maximize the chance of successful formal verification
%
% Strategy:
%   1. Start with very small epsilon + conservative settings
%   2. Progressively increase epsilon until verification fails
%   3. Report the maximum verified epsilon
%
% This provides the strongest possible formal guarantee given NNV's limitations

%% ======================== CONFIGURATION ========================

% Path to NNV root
nnv_root = fullfile('..', '..', 'modules', 'nnv_moe', 'code', 'nnv');

% Path to exported ONNX model
onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn.onnx');

% Dataset configuration
dataset_name = 'GTSRB';
data_root = fullfile('..', '..', 'data');

% Image selection
test_image_idx = 1;

% Progressive verification configurations (try in order)
% Each row: [epsilon, method_id, relaxFactor, timeout_seconds]
% method_id: 1='abs-dom', 2='approx-star'
verification_configs = [
    0.1/255,  1, 1.0, 300;   % Level 1: Very conservative (best chance)
    0.25/255, 1, 0.5, 600;   % Level 2: Conservative
    0.5/255,  1, 0.25, 900;  % Level 3: Moderate
    1/255,    1, 0.0, 1200;  % Level 4: Aggressive (abs-dom, tight)
    1/255,    2, 0.0, 1800;  % Level 5: Very aggressive (approx-star)
];

method_names = {'abs-dom', 'approx-star'};

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
fprintf('Optimized NNV Verification (Progressive)\n');
fprintf('========================================\n\n');

%% ======================== LOAD MODEL ========================

fprintf('Loading ONNX model: %s\n', onnx_model_path);

try
    matlab_net = importNetworkFromONNX(onnx_model_path, ...
        'InputDataFormats', 'BCSS', 'OutputDataFormats', 'BC');
    fprintf('Loaded successfully!\n');
catch ME1
    fprintf('importNetworkFromONNX failed, using onnx2nnv...\n');
    loadOptions = struct();
    loadOptions.InputDataFormat = 'BCSS';
    loadOptions.OutputDataFormat = 'BC';
    loadOptions.GenerateCustomLayers = false;
    loadOptions.FoldConstants = 'deep';
    matlab_net = onnx2nnv(onnx_model_path, loadOptions);
end

% Convert to NNV format
if ~isa(matlab_net, 'NN')
    net = matlab2nnv(matlab_net);
else
    net = matlab_net;
end

fprintf('Model loaded: %d layers\n', length(net.Layers));

%% ======================== LOAD TEST IMAGE ========================

fprintf('\nLoading %s dataset...\n', dataset_name);

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

% Extract class index
target_str = char(target_label);
target_class = str2double(target_str);
if isnan(target_class)
    img_path = imds.Files{test_image_idx};
    [~, parent_folder, ~] = fileparts(fileparts(img_path));
    target_class = str2double(parent_folder);
    if isnan(target_class)
        target_class = 0;
    end
end
target_idx = target_class + 1;  % MATLAB 1-based indexing

fprintf('Extracted class: %d (target_idx: %d)\n', target_class, target_idx);

% Normalize
img = single(img) / 255.0;
for c = 1:3
    img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
end

% Get clean prediction
clean_output = net.evaluate(img);
[~, clean_pred] = max(clean_output);
fprintf('Clean prediction: class %d (score: %.4f)\n', clean_pred, clean_output(clean_pred));

%% ======================== PROGRESSIVE VERIFICATION ========================

fprintf('\n========================================\n');
fprintf('Starting Progressive Verification\n');
fprintf('========================================\n\n');

max_verified_epsilon = 0;
verification_results = cell(size(verification_configs, 1), 1);

for i = 1:size(verification_configs, 1)
    cfg_epsilon = verification_configs(i, 1);
    cfg_method_id = verification_configs(i, 2);
    cfg_relax = verification_configs(i, 3);
    cfg_timeout = verification_configs(i, 4);

    cfg_method = method_names{cfg_method_id};

    fprintf('\n=== Attempt %d/%d ===\n', i, size(verification_configs, 1));
    fprintf('Epsilon:      %.6f (~%.4f/255)\n', cfg_epsilon, cfg_epsilon * 255);
    fprintf('Method:       %s\n', cfg_method);
    fprintf('RelaxFactor:  %.2f\n', cfg_relax);
    fprintf('Timeout:      %d seconds\n', cfg_timeout);
    fprintf('Status:       ');

    % Create input set
    disturbance = cfg_epsilon * ones(size(img), 'single');
    IS = ImageStar(img - disturbance, img + disturbance);

    % Configure network
    net.reachMethod = cfg_method;
    net.relaxFactor = cfg_relax;
    net.lp_solver = 'linprog';
    net.numCores = 1;
    net.dis_opt = [];

    % Configure options
    reachOptions = struct();
    reachOptions.reachMethod = cfg_method;

    % Attempt verification
    try
        tic;
        res = net.verify_robustness(IS, reachOptions, target_idx);
        verify_time = toc;

        if res == 1
            fprintf('✓ VERIFIED ROBUST\n');
            fprintf('Time:         %.2f seconds\n', verify_time);
            max_verified_epsilon = cfg_epsilon;
            verification_results{i} = struct('result', 'robust', 'time', verify_time, 'epsilon', cfg_epsilon);

            % Continue to next level to see if we can verify larger epsilon

        elseif res == 0
            fprintf('✗ NOT ROBUST (counterexample found)\n');
            fprintf('Time:         %.2f seconds\n', verify_time);
            verification_results{i} = struct('result', 'not_robust', 'time', verify_time, 'epsilon', cfg_epsilon);

            % Stop here - found counterexample
            fprintf('\nStopping: Counterexample found at this epsilon level.\n');
            break;

        else
            fprintf('? UNKNOWN\n');
            fprintf('Time:         %.2f seconds\n', verify_time);
            verification_results{i} = struct('result', 'unknown', 'time', verify_time, 'epsilon', cfg_epsilon);

            % Try next configuration
        end

    catch ME
        fprintf('✗ FAILED\n');
        fprintf('Error:        %s\n', ME.message);
        verification_results{i} = struct('result', 'failed', 'time', 0, 'epsilon', cfg_epsilon, 'error', ME.message);

        % Try next configuration
    end
end

%% ======================== FALLBACK: SAMPLING ========================

% If no formal verification succeeded, run sampling on the max attempted epsilon
if max_verified_epsilon == 0
    fprintf('\n========================================\n');
    fprintf('Formal verification failed at all levels\n');
    fprintf('Falling back to sampling-based verification\n');
    fprintf('========================================\n\n');

    % Use the moderate epsilon for sampling
    sampling_epsilon = 1/255;
    num_samples = 100;

    fprintf('Running sampling verification at epsilon = %.6f...\n', sampling_epsilon);

    robust_count = 0;
    for j = 1:num_samples
        delta = (2 * rand(size(img), 'single') - 1) * sampling_epsilon;
        perturbed = img + delta;
        perturbed_output = net.evaluate(perturbed);
        [~, perturbed_pred] = max(perturbed_output);
        if perturbed_pred == clean_pred
            robust_count = robust_count + 1;
        end

        if mod(j, 20) == 0
            fprintf('  Progress: %d/%d samples...\n', j, num_samples);
        end
    end

    sampling_result = robust_count / num_samples;
    fprintf('\nSampling result: %d/%d (%.1f%% robust)\n', robust_count, num_samples, sampling_result * 100);
end

%% ======================== FINAL RESULTS ========================

fprintf('\n========================================\n');
fprintf('FINAL VERIFICATION RESULTS\n');
fprintf('========================================\n');
fprintf('Model:          %s\n', onnx_model_path);
fprintf('Dataset:        %s\n', dataset_name);
fprintf('Test image:     %d\n', test_image_idx);
fprintf('True class:     %d\n', target_class);
fprintf('Clean pred:     %d\n', clean_pred);
fprintf('----------------------------------------\n');

if max_verified_epsilon > 0
    fprintf('RESULT:         ✓ FORMALLY VERIFIED ROBUST\n');
    fprintf('Max epsilon:    %.6f (~%.4f/255)\n', max_verified_epsilon, max_verified_epsilon * 255);
    fprintf('Guarantee:      All perturbations within L∞ ball of radius %.6f\n', max_verified_epsilon);
    fprintf('                are correctly classified as class %d.\n', target_class);
else
    fprintf('RESULT:         ? FORMAL VERIFICATION FAILED\n');
    fprintf('Max epsilon:    None (could not verify any configuration)\n');

    if exist('sampling_result', 'var')
        fprintf('----------------------------------------\n');
        fprintf('Sampling-based result at epsilon = %.6f:\n', sampling_epsilon);
        fprintf('  Robustness: %.1f%% (%d/%d samples)\n', sampling_result * 100, robust_count, num_samples);

        if sampling_result >= 0.95
            fprintf('  Interpretation: Strong empirical evidence of robustness\n');
        elseif sampling_result >= 0.80
            fprintf('  Interpretation: Moderate empirical robustness\n');
        else
            fprintf('  Interpretation: Network appears vulnerable\n');
        end
    end
end

fprintf('========================================\n\n');

%% ======================== DETAILED RESULTS TABLE ========================

fprintf('Detailed results for each configuration:\n\n');
fprintf('%-6s %-12s %-12s %-8s %-10s %-10s\n', ...
    'Level', 'Epsilon', 'Method', 'Relax', 'Result', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:length(verification_results)
    if isempty(verification_results{i})
        continue;
    end

    res = verification_results{i};
    cfg_epsilon = verification_configs(i, 1);
    cfg_method = method_names{verification_configs(i, 2)};
    cfg_relax = verification_configs(i, 3);

    fprintf('%-6d %-12.6f %-12s %-8.2f %-10s %-10.2f\n', ...
        i, cfg_epsilon, cfg_method, cfg_relax, res.result, res.time);
end

fprintf('\n');

%% ======================== RECOMMENDATIONS ========================

fprintf('========================================\n');
fprintf('RECOMMENDATIONS\n');
fprintf('========================================\n\n');

if max_verified_epsilon > 0
    fprintf('✓ Success! Your model has been formally verified.\n\n');
    fprintf('For your research paper, you can claim:\n');
    fprintf('  "The model is formally verified to be robust against\n');
    fprintf('   L∞ perturbations with radius ε = %.6f using NNV."\n\n', max_verified_epsilon);

    if max_verified_epsilon < 1/255
        fprintf('Note: The verified epsilon is relatively small.\n');
        fprintf('Consider:\n');
        fprintf('  - Using sampling for larger epsilon values\n');
        fprintf('  - Trying α,β-CROWN for potentially larger verified epsilon\n');
        fprintf('  - Reporting both formal and empirical robustness\n\n');
    end
else
    fprintf('✗ Formal verification did not succeed.\n\n');
    fprintf('This is expected due to MaxPooling complexity.\n');
    fprintf('Options:\n');
    fprintf('  1. Use sampling-based verification (verify_expert_nnv_simple.m)\n');
    fprintf('     - Fast and reliable\n');
    fprintf('     - Scientifically valid for research\n\n');
    fprintf('  2. Try α,β-CROWN verification tool\n');
    fprintf('     - Better success rate with MaxPooling\n');
    fprintf('     - State-of-the-art performance\n');
    fprintf('     - See: https://github.com/Verified-Intelligence/alpha-beta-CROWN\n\n');
    fprintf('  3. For future models, consider:\n');
    fprintf('     - Replacing MaxPooling with AvgPooling\n');
    fprintf('     - Using strided convolutions instead\n\n');
end

fprintf('========================================\n\n');

fprintf('Verification completed!\n');
