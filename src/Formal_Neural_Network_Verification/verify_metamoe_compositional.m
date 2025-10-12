%% Compositional Verification of MetaMoE System
% This script performs end-to-end verification of the MetaMoE system by
% combining router and expert verification results.
%
% Compositional Verification Approach:
%   1. Verify router maintains correct routing: Router(x') = correct_expert
%   2. Verify expert classifies correctly: Expert(x') = correct_class
%   3. Combine: If both hold, MetaMoE(x') is correct
%
% This provides stronger guarantees than testing individual components.
%
% Prerequisites:
%   1. NNV installed
%   2. Router exported to ONNX
%   3. Relevant expert exported to ONNX
%   4. Both models trained
%
% Usage:
%   Configure paths and parameters below, then run in MATLAB

%% ======================== CONFIGURATION ========================

% Paths
nnv_root = fullfile('..', '..', 'modules', 'nnv_moe', 'code', 'nnv');
router_onnx = fullfile('..', '..', 'artifacts', 'nnv_models', ...
    'meta_moe_small_cnn_best_router_logits.onnx');
expert_onnx = fullfile('..', '..', 'artifacts', 'nnv_models', ...
    'gtsrb_micro_cnn_best.onnx');

% Dataset configuration
dataset_name = 'GTSRB';
data_root = fullfile('..', '..', 'data');

% Meta-class mapping
meta_class_map = containers.Map({'GTSRB', 'CIFAR10', 'MNIST'}, {0, 1, 2});
true_meta_class = meta_class_map(dataset_name);

% Verification parameters
epsilon = 2/255;  % L-infinity perturbation
reachMethod = 'approx-star';

% Test configuration
test_image_idx = 1;
num_test_images = 10;

%% ======================== INITIALIZATION ========================

if exist(fullfile(nnv_root, 'startup_nnv.m'), 'file')
    cd(nnv_root);
    startup_nnv;
    cd(fileparts(mfilename('fullpath')));
else
    error('NNV not found.');
end

fprintf('\n========================================\n');
fprintf('Compositional MetaMoE Verification\n');
fprintf('========================================\n\n');

%% ======================== LOAD MODELS ========================

fprintf('Loading router...\n');
try
    router_net = onnx2nnv(router_onnx);
    fprintf('✓ Router loaded (%d experts)\n', router_net.OutputSize);
catch ME
    error('Failed to load router: %s', ME.message);
end

fprintf('Loading expert...\n');
try
    expert_net = onnx2nnv(expert_onnx);
    fprintf('✓ Expert loaded (%d classes)\n', expert_net.OutputSize);
catch ME
    error('Failed to load expert: %s', ME.message);
end

%% ======================== LOAD DATASET ========================

fprintf('\nLoading %s dataset...\n', dataset_name);

switch dataset_name
    case 'GTSRB'
        dataset_path = fullfile(data_root, 'GTSRB', 'Test', 'Images');
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.3337, 0.3064, 0.3171];
        stdNorm = [0.2672, 0.2564, 0.2629];

    case 'CIFAR10'
        dataset_path = fullfile(data_root, 'CIFAR10', 'Test');
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');
        inputSize = [32 32];
        numChannels = 3;
        meanNorm = [0.4914, 0.4822, 0.4465];
        stdNorm = [0.2023, 0.1994, 0.2010];

    case 'MNIST'
        dataset_path = fullfile(data_root, 'MNIST', 'Test');
        imds = imageDatastore(dataset_path, 'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');
        inputSize = [28 28];
        numChannels = 1;
        meanNorm = 0.1307;
        stdNorm = 0.3081;
end

imds.ReadFcn = @(loc) imresize(imread(loc), inputSize);
fprintf('Dataset loaded: %d images\n', length(imds.Files));

%% ======================== COMPOSITIONAL VERIFICATION ========================

fprintf('\n========================================\n');
fprintf('COMPOSITIONAL VERIFICATION\n');
fprintf('========================================\n');
fprintf('Dataset: %s (meta-class %d)\n', dataset_name, true_meta_class);
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Method: %s\n', reachMethod);
fprintf('Testing %d images...\n\n', num_test_images);

% Results storage
results = struct();
results.router_robust = 0;
results.expert_robust = 0;
results.both_robust = 0;
results.neither_robust = 0;
results.router_only = 0;
results.expert_only = 0;
results.total_tested = 0;
results.details = cell(num_test_images, 1);

reachOptions = struct;
reachOptions.reachMethod = reachMethod;

for img_idx = 1:min(num_test_images, length(imds.Files))
    fprintf('\n--- Image %d/%d ---\n', img_idx, num_test_images);

    % Load and preprocess
    [img, fileInfo] = readimage(imds, img_idx);
    true_class = double(fileInfo.Label);
    img = single(img) / 255.0;

    if numChannels == 3
        for c = 1:3
            img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
        end
    else
        img = (img - meanNorm) / stdNorm;
    end

    % Evaluate clean predictions
    router_clean = router_net.evaluate(img);
    [~, router_pred] = max(router_clean);
    router_pred = router_pred - 1;  % 0-indexed

    expert_clean = expert_net.evaluate(img);
    [~, expert_pred] = max(expert_clean);

    fprintf('Clean predictions:\n');
    fprintf('  Router: Expert %d (true: %d) %s\n', ...
        router_pred, true_meta_class, ...
        string(router_pred == true_meta_class));
    fprintf('  Expert: Class %d (true: %d) %s\n', ...
        expert_pred, true_class, ...
        string(expert_pred == true_class));

    % Skip if clean predictions are wrong
    if router_pred ~= true_meta_class || expert_pred ~= true_class
        fprintf('⚠ Skipping: Clean prediction(s) incorrect\n');
        continue;
    end

    results.total_tested = results.total_tested + 1;

    % Create input set
    disturbance = epsilon * ones(size(img), 'single');
    lb = img - disturbance;
    ub = img + disturbance;
    IS = ImageStar(lb, ub);

    % Verify router
    fprintf('Verifying router...\n');
    tic;
    router_result = router_net.verify_robustness(IS, reachOptions, true_meta_class + 1);
    router_time = toc;

    % Verify expert
    fprintf('Verifying expert...\n');
    tic;
    expert_result = expert_net.verify_robustness(IS, reachOptions, true_class);
    expert_time = toc;

    % Analyze results
    router_robust = (router_result == 1);
    expert_robust = (expert_result == 1);
    both_robust = router_robust && expert_robust;

    fprintf('Results:\n');
    fprintf('  Router: %s (%.2f sec)\n', ...
        string(router_robust) + " robust" + string(~router_robust && router_result ~= 1), ...
        router_time);
    fprintf('  Expert: %s (%.2f sec)\n', ...
        string(expert_robust) + " robust" + string(~expert_robust && expert_result ~= 1), ...
        expert_time);

    if both_robust
        fprintf('  ✓ END-TO-END ROBUST: MetaMoE verified!\n');
        results.both_robust = results.both_robust + 1;
    elseif router_robust && ~expert_robust
        fprintf('  ⚠ Router robust but expert not robust\n');
        results.router_only = results.router_only + 1;
    elseif ~router_robust && expert_robust
        fprintf('  ⚠ Expert robust but router not robust\n');
        results.expert_only = results.expert_only + 1;
    else
        fprintf('  ✗ Neither component verified robust\n');
        results.neither_robust = results.neither_robust + 1;
    end

    % Update statistics
    if router_robust
        results.router_robust = results.router_robust + 1;
    end
    if expert_robust
        results.expert_robust = results.expert_robust + 1;
    end

    % Store details
    results.details{img_idx}.image_idx = img_idx;
    results.details{img_idx}.router_result = router_result;
    results.details{img_idx}.expert_result = expert_result;
    results.details{img_idx}.end_to_end_robust = both_robust;
    results.details{img_idx}.router_time = router_time;
    results.details{img_idx}.expert_time = expert_time;
end

%% ======================== FINAL SUMMARY ========================

fprintf('\n========================================\n');
fprintf('COMPOSITIONAL VERIFICATION SUMMARY\n');
fprintf('========================================\n');
fprintf('Dataset: %s\n', dataset_name);
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Method: %s\n', reachMethod);
fprintf('Images tested: %d (with correct clean predictions)\n', results.total_tested);
fprintf('----------------------------------------\n');
fprintf('Component Analysis:\n');
fprintf('  Router robust: %d (%.1f%%)\n', ...
    results.router_robust, 100 * results.router_robust / max(results.total_tested, 1));
fprintf('  Expert robust: %d (%.1f%%)\n', ...
    results.expert_robust, 100 * results.expert_robust / max(results.total_tested, 1));
fprintf('----------------------------------------\n');
fprintf('Combined Analysis:\n');
fprintf('  ✓ Both robust (END-TO-END): %d (%.1f%%)\n', ...
    results.both_robust, 100 * results.both_robust / max(results.total_tested, 1));
fprintf('  ⚠ Router only: %d (%.1f%%)\n', ...
    results.router_only, 100 * results.router_only / max(results.total_tested, 1));
fprintf('  ⚠ Expert only: %d (%.1f%%)\n', ...
    results.expert_only, 100 * results.expert_only / max(results.total_tested, 1));
fprintf('  ✗ Neither robust: %d (%.1f%%)\n', ...
    results.neither_robust, 100 * results.neither_robust / max(results.total_tested, 1));
fprintf('========================================\n');

% Certified accuracy
certified_metamoe_accuracy = results.both_robust / max(results.total_tested, 1);
fprintf('\n🎯 Certified MetaMoE Accuracy: %.2f%%\n', certified_metamoe_accuracy * 100);

% Interpretation
fprintf('\nInterpretation:\n');
if certified_metamoe_accuracy > 0.8
    fprintf('✓ Strong: MetaMoE system shows high certified robustness\n');
elseif certified_metamoe_accuracy > 0.5
    fprintf('⚠ Moderate: MetaMoE has some robustness but room for improvement\n');
else
    fprintf('✗ Weak: MetaMoE needs significant robustness improvements\n');
end

if results.router_only > results.expert_only
    fprintf('→ Bottleneck: Expert is the weakest link\n');
    fprintf('  Recommendation: Train expert with adversarial training\n');
elseif results.expert_only > results.router_only
    fprintf('→ Bottleneck: Router is the weakest link\n');
    fprintf('  Recommendation: Train router with adversarial gating\n');
else
    fprintf('→ Both components need improvement equally\n');
end

%% ======================== VISUALIZATION ========================

fprintf('\nGenerating visualization...\n');

figure('Position', [100, 100, 1000, 600]);

% Plot 1: Component comparison
subplot(2, 2, 1);
component_data = [results.router_robust, results.expert_robust];
component_labels = {'Router', 'Expert'};
bar(component_data);
set(gca, 'XTickLabel', component_labels);
ylabel('Number Verified Robust');
title('Component Verification Results');
ylim([0, results.total_tested]);
grid on;

% Plot 2: Combined analysis
subplot(2, 2, 2);
combined_data = [results.both_robust, results.router_only, ...
                results.expert_only, results.neither_robust];
combined_labels = {'Both', 'Router Only', 'Expert Only', 'Neither'};
bar(combined_data);
set(gca, 'XTickLabel', combined_labels);
xtickangle(45);
ylabel('Number of Images');
title('Compositional Analysis');
ylim([0, results.total_tested]);
grid on;

% Plot 3: Pie chart of end-to-end results
subplot(2, 2, 3);
pie_data = [results.both_robust, results.total_tested - results.both_robust];
pie_labels = {sprintf('Robust (%.1f%%)', 100*certified_metamoe_accuracy), ...
              sprintf('Not Robust (%.1f%%)', 100*(1-certified_metamoe_accuracy))};
pie(pie_data, pie_labels);
title('End-to-End Robustness');

% Plot 4: Bottleneck analysis
subplot(2, 2, 4);
failure_data = [results.router_only, results.expert_only, results.neither_robust];
failure_labels = {'Expert Failed', 'Router Failed', 'Both Failed'};
if sum(failure_data) > 0
    pie(failure_data, failure_labels);
    title('Failure Analysis (Non-Robust Cases)');
else
    text(0.5, 0.5, 'All Tests Passed!', 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold');
    axis off;
    title('Failure Analysis');
end

sgtitle(sprintf('MetaMoE Compositional Verification (%s, \\epsilon=%.3f)', ...
    dataset_name, epsilon), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
output_fig = fullfile(fileparts(mfilename('fullpath')), ...
    sprintf('metamoe_compositional_%s_eps%.3f.png', dataset_name, epsilon));
saveas(gcf, output_fig);
fprintf('Saved visualization to: %s\n', output_fig);

fprintf('\nCompositional verification completed!\n');
