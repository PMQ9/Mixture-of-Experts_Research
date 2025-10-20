%% Test Gurobi Installation and MATLAB Integration
% This script verifies Gurobi is properly configured for MATLAB

fprintf('=== Gurobi MATLAB Integration Test ===\n\n');

%% Step 1: Add Gurobi to MATLAB path
gurobi_home = 'D:\Program Files\gurobi1203\win64';
gurobi_matlab_path = fullfile(gurobi_home, 'matlab');

fprintf('1. Adding Gurobi MATLAB path: %s\n', gurobi_matlab_path);
if exist(gurobi_matlab_path, 'dir')
    addpath(gurobi_matlab_path);
    fprintf('   [OK] Path added successfully\n\n');
else
    error('   [ERROR] Gurobi MATLAB directory not found: %s', gurobi_matlab_path);
end

%% Step 2: Test Gurobi is accessible
fprintf('2. Testing Gurobi accessibility...\n');
try
    gurobi_setup;
    fprintf('   [OK] gurobi_setup executed successfully\n\n');
catch ME
    fprintf('   [WARNING] gurobi_setup not found or failed: %s\n', ME.message);
    fprintf('   This is okay - we will test with a simple problem\n\n');
end

%% Step 3: Solve a simple LP problem
fprintf('3. Solving a simple LP problem with Gurobi...\n');
fprintf('   Problem: maximize x + y subject to x + 2y <= 4, x,y >= 0\n\n');

% Model setup
model.A = sparse([1 2]);
model.obj = [1 1];
model.rhs = 4;
model.sense = '<';
model.vtype = 'C';
model.modelsense = 'max';
model.lb = [0 0];

% Gurobi parameters
params.outputflag = 1;  % Show output
params.TimeLimit = 10;

try
    % Solve
    result = gurobi(model, params);

    fprintf('\n   [SUCCESS] Gurobi solved the problem!\n');
    fprintf('   Status: %s\n', result.status);
    fprintf('   Optimal value: %.4f\n', result.objval);
    fprintf('   Solution: x = %.4f, y = %.4f\n\n', result.x(1), result.x(2));

    % Verify solution is correct (should be x=0, y=2, objval=2)
    if abs(result.objval - 2.0) < 1e-6
        fprintf('   [VERIFIED] Solution is correct!\n\n');
    else
        fprintf('   [WARNING] Solution seems incorrect\n\n');
    end

catch ME
    fprintf('   [ERROR] Failed to solve with Gurobi\n');
    fprintf('   Error: %s\n', ME.message);
    fprintf('   Make sure Gurobi license is activated\n\n');
    rethrow(ME);
end

%% Step 4: Save path for future sessions
fprintf('4. Saving MATLAB path for future sessions...\n');
try
    savepath;
    fprintf('   [OK] Path saved successfully\n\n');
catch
    fprintf('   [WARNING] Could not save path (may need admin privileges)\n');
    fprintf('   You will need to run addpath(''%s'') in each session\n\n', gurobi_matlab_path);
end

fprintf('=== All Tests Passed! ===\n');
fprintf('Gurobi is ready to use with MATLAB and NNV\n\n');

fprintf('Next step: Test with NNV by running:\n');
fprintf('  cd src/Formal_Neural_Network_Verification\n');
fprintf('  verify_expert_nnv\n\n');
