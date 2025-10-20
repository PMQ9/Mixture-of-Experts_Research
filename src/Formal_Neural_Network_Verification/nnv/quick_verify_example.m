%% Quick NNV Verification Example
% Minimal script to quickly test NNV verification on an expert model
%
% Usage:
%   1. Update paths below
%   2. Run: quick_verify_example

%% Setup
clear; clc;

% Paths
nnv_path = 'd:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv';
onnx_model = 'd:\Mixture-of-Experts_Research\artifacts\nnv_models\gtsrb_micro_cnn_best.onnx';

% Add NNV to path
cd(nnv_path);
startup_nnv;
cd(fileparts(mfilename('fullpath')));

fprintf('\n=== Quick NNV Verification Example ===\n\n');

%% Load Model
fprintf('Loading ONNX model...\n');
net = onnx2nnv(onnx_model);
fprintf('✓ Model loaded: %d layers\n\n', length(net.Layers));

%% Create Synthetic Input (32x32x3 image)
fprintf('Creating synthetic input...\n');
img = rand(32, 32, 3, 'single');

% Normalize (GTSRB normalization)
mean_vals = reshape([0.3337, 0.3064, 0.3171], [1, 1, 3]);
std_vals = reshape([0.2672, 0.2564, 0.2629], [1, 1, 3]);
img = (img - mean_vals) ./ std_vals;

%% Evaluate Clean Image
fprintf('Evaluating clean image...\n');
clean_output = net.evaluate(img);
[~, pred_class] = max(clean_output);
fprintf('Predicted class: %d\n', pred_class);
fprintf('Output values (first 5): [%.3f, %.3f, %.3f, %.3f, %.3f]\n\n', ...
    clean_output(1), clean_output(2), clean_output(3), clean_output(4), clean_output(5));

%% Create Input Set
epsilon = 2/255;
fprintf('Creating input set (epsilon = %.4f)...\n', epsilon);

lb = img - epsilon;
ub = img + epsilon;
IS = ImageStar(lb, ub);
fprintf('✓ Input set created\n\n');

%% Verify Robustness (Approximate)
fprintf('Verifying robustness (approx-star method)...\n');
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';

tic;
res = net.verify_robustness(IS, reachOptions, pred_class);
elapsed = toc;

fprintf('Verification time: %.2f seconds\n\n', elapsed);

%% Results
fprintf('=== RESULTS ===\n');
if res == 1
    fprintf('✓ VERIFIED ROBUST: Network is robust within epsilon-ball\n');
elseif res == 0
    fprintf('✗ NOT ROBUST: Found counterexample\n');
else
    fprintf('? UNKNOWN: Could not determine robustness\n');
end

%% Visualize Output Ranges
fprintf('\nVisualizing output ranges...\n');

R = net.reachSet{end};
[lb_out, ub_out] = R.getRanges;
lb_out = squeeze(lb_out);
ub_out = squeeze(ub_out);

mid_range = (lb_out + ub_out) / 2;
range_size = ub_out - mid_range;

figure('Position', [100, 100, 800, 400]);
x = 1:length(mid_range);
errorbar(x, mid_range, range_size, 'b.', 'MarkerSize', 12);
hold on;
scatter(x, clean_output, 50, 'rx', 'LineWidth', 2);
xline(pred_class, 'g--', 'LineWidth', 2);
xlabel('Class Index');
ylabel('Output Value');
title('Output Reachable Set (Approximate)');
legend('Reachable range', 'Clean prediction', 'Predicted class');
grid on;

fprintf('✓ Visualization complete\n');
fprintf('\n=== Verification Complete ===\n');
